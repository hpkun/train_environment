"""F22-only MAV failure audit.  No F16 surrogate logic anywhere in this file.

Analyses why the F22 MAV dies: control/flight-dynamics issues,
blue tactical pressure, initial geometry disadvantage, or all three.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
from uav_env import make_env

# ── helpers reused from eval infrastructure ──────────────────────────────
def _resolve_checkpoint(checkpoint_arg: str) -> Path:
    p = Path(checkpoint_arg)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():
        return p / "model.pt"
    return p


def _load_meta(model_path: Path) -> dict:
    meta_path = model_path.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _build_policy(meta: dict, device: torch.device):
    pa = meta.get("policy_arch", "flat")
    if pa == "hetero_entity_recurrent":
        validate_entity_policy_meta(meta)
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta.get("entity_dim", 21)),
            action_dim=3,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
    raise ValueError(f"unsupported policy_arch for audit: {pa}")


def _team_done(terminated, truncated):
    return bool(terminated and all(terminated.values())) or bool(truncated and all(truncated.values()))


# ── config validation ────────────────────────────────────────────────────
def _validate_f22_config(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    atp = cfg.get("aircraft_type_params", {})
    mav_model = atp.get("mav", {}).get("aircraft_model", "")
    if mav_model != "f22":
        raise ValueError(
            f"F22 MAV audit requires aircraft_type_params.mav.aircraft_model == 'f22'; "
            f"got {mav_model!r} from {config_path}"
        )


# ── episode runner ───────────────────────────────────────────────────────
def run_episodes(policy, device, config_path, args, adapter, mav_mode, opp_mode,
                 n_eps, seed):
    """Run n_eps episodes and return list of per-episode records."""
    records = []
    for ep in range(n_eps):
        env = make_env(config_path, env_type="jsbsim_hetero")
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)
        opponent = OpponentPolicy(mode=opp_mode, seed=ep_seed + 33)
        _rnn_hidden = np.zeros((len(env.red_ids), getattr(policy, "rnn_hidden_size", 0)),
                              dtype=np.float32) if getattr(policy, "rnn_hidden_size", 0) else None

        # Initial distances: MAV (red_0) -> each blue
        init_mav_to_blue = {}
        mav_pos = env.red_planes["red_0"].get_position().copy()
        for bid in env.blue_ids:
            b_pos = env.blue_planes[bid].get_position()
            init_mav_to_blue[bid] = float(np.linalg.norm(mav_pos - b_pos))

        ep_data = {"episode": ep, "mav_mode": mav_mode, "opp_mode": opp_mode,
                   "init_mav_to_blue_m": init_mav_to_blue}
        # Per-agent tracking: red_0, red_1, red_2
        agent_alt = {rid: [] for rid in env.red_ids}
        agent_spd = {rid: [] for rid in env.red_ids}
        agent_roll = {rid: [] for rid in env.red_ids}
        agent_pitch = {rid: [] for rid in env.red_ids}
        agent_act = {rid: [] for rid in env.red_ids}  # raw actions [a0,a1,a2]
        red_fired, blue_fired, red_hits, blue_hits = 0, 0, 0, 0
        prev_hits = {"red": 0, "blue": 0}
        death_events = []
        mav_death_step, mav_death_reason = -1, ""
        first_death = ""
        blue_launch_targets = defaultdict(int)
        blue_hit_targets = defaultdict(int)
        blue_first = None

        step = 0
        while step < args.max_steps:
            adapted = adapter.adapt_all(obs, info=info,
                                        red_ids=env.red_ids, blue_ids=env.blue_ids)
            active = np.ones(len(env.red_ids), dtype=np.float32)
            for i, rid in enumerate(env.red_ids):
                ai = (info or {}).get(rid, {})
                active[i] = 1.0 if ai.get("alive", True) else 0.0
            active_rows = active > 0.5

            at = adapted["actor_entity_tokens"].copy()
            ak = adapted["actor_keep_mask"].copy()
            ct = adapted["critic_entity_tokens"]
            ck = adapted["critic_keep_mask"]
            cc = adapted.get("critic_counts", np.zeros(4, dtype=np.float32))
            at[~active_rows] = 0.0; ak[~active_rows] = 0.0; ak[~active_rows, 0] = 1.0

            act_kw = {}
            if _rnn_hidden is not None:
                from algorithms.happo.rollout_safety import zero_inactive_hidden
                _rnn_hidden = zero_inactive_hidden(_rnn_hidden, active)
                act_kw["rnn_hidden"] = torch.as_tensor(_rnn_hidden, device=device)
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(at, device=device), torch.as_tensor(ak, device=device),
                    torch.as_tensor(adapted["role_ids"], device=device),
                    torch.as_tensor(ct, device=device), torch.as_tensor(ck, device=device),
                    deterministic=True,
                    critic_counts=torch.as_tensor(cc, device=device), **act_kw)

            raw_act = out["action"].cpu().numpy()

            # MAV action override for fixed-mav ablation modes
            if mav_mode == "fixed_mav_zero":
                raw_act[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            elif mav_mode == "fixed_mav_speed_0_3":
                raw_act[0] = np.array([0.0, 0.0, 0.3], dtype=np.float32)
            elif mav_mode == "mav_action_scale_0_3":
                raw_act[0] = raw_act[0] * 0.3
            elif mav_mode == "mav_action_scale_0_1":
                raw_act[0] = raw_act[0] * 0.1

            from algorithms.happo.rollout_safety import zero_inactive_actions, zero_inactive_hidden
            actions = zero_inactive_actions(raw_act, active)
            if _rnn_hidden is not None and "rnn_hidden" in out:
                _rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)

            ad = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            ad.update(opponent.act(obs, env.blue_ids, env=env))

            # Track ALL red agents state + actions
            for i, rid in enumerate(env.red_ids):
                sim = env.red_planes.get(rid)
                if sim is not None and sim.is_alive:
                    rpy = sim.get_rpy(); vel = sim.get_velocity()
                    agent_alt[rid].append(sim.get_geodetic()[2])
                    agent_spd[rid].append(float(np.linalg.norm(vel)))
                    agent_roll[rid].append(float(np.rad2deg(rpy[0])))
                    agent_pitch[rid].append(float(np.rad2deg(rpy[1])))
                    agent_act[rid].append(raw_act[i].copy())

            obs, rewards, terminated, truncated, info = env.step(ad)
            step += 1

            # Missile tracking
            for aid, ai in (info or {}).items():
                if isinstance(ai, dict):
                    f = int(ai.get("missiles_fired_this_step", 0))
                    if aid.startswith("red_"): red_fired += f
                    else: blue_fired += f
            mt = info.get("__missile_term__", {}) if isinstance(info, dict) else {}
            if isinstance(mt, dict):
                red_hits += max(int(mt.get("red", {}).get("hit", 0)) - prev_hits["red"], 0)
                blue_hits += max(int(mt.get("blue", {}).get("hit", 0)) - prev_hits["blue"], 0)
                prev_hits["red"] = int(mt.get("red", {}).get("hit", 0))
                prev_hits["blue"] = int(mt.get("blue", {}).get("hit", 0))

            # Launch target tracking
            lqr = getattr(env, "_launch_quality_step_records", None) or []
            for rec in lqr:
                if rec.get("shooter_id", "").startswith("blue_"):
                    blue_launch_targets[rec.get("target_id", "?")] += 1
                    if blue_first is None:
                        blue_first = {"step": step, "target": rec.get("target_id", "?"),
                                      "range": rec.get("range_m", 0)}

            # Death events
            for de in info.get("death_events", []) if isinstance(info, dict) else []:
                if not isinstance(de, dict): continue
                death_events.append(dict(de))
                aid = de.get("agent_id", "")
                if aid == "red_0" and mav_death_step < 0:
                    mav_death_step = step
                    mav_death_reason = de.get("death_reason", "unknown")
                if not first_death:
                    first_death = aid
                if de.get("death_reason") == "missile_hit":
                    if de.get("shooter_id", "").startswith("blue_"):
                        blue_hit_targets[de.get("agent_id", "?")] += 1

            if _team_done(terminated, truncated):
                break

        env.close()

        # Episode summary
        red_alive = sum(1 for rid in env.red_ids if env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive)
        blue_alive = sum(1 for bid in env.blue_ids if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive)
        winner = "draw"
        if blue_alive == 0 and red_alive > 0: winner = "red"
        elif red_alive == 0 and blue_alive > 0: winner = "blue"

        # Per-agent aggregates
        per_agent = {}
        for rid in env.red_ids:
            aa = np.array(agent_act[rid]) if agent_act[rid] else np.zeros((0, 3))
            per_agent[rid] = {
                "max_abs_pitch_deg": float(np.max(np.abs(agent_pitch[rid]))) if agent_pitch[rid] else 0,
                "max_abs_roll_deg": float(np.max(np.abs(agent_roll[rid]))) if agent_roll[rid] else 0,
                "min_altitude_m": float(np.min(agent_alt[rid])) if agent_alt[rid] else 0,
                "mean_altitude_m": float(np.mean(agent_alt[rid])) if agent_alt[rid] else 0,
                "min_speed_mps": float(np.min(agent_spd[rid])) if agent_spd[rid] else 0,
                "mean_speed_mps": float(np.mean(agent_spd[rid])) if agent_spd[rid] else 0,
                "action_pitch_mean": float(np.mean(aa[:, 0])) if aa.size else 0,
                "action_pitch_min": float(np.min(aa[:, 0])) if aa.size else 0,
                "action_pitch_max": float(np.max(aa[:, 0])) if aa.size else 0,
                "action_heading_mean": float(np.mean(aa[:, 1])) if aa.size else 0,
                "action_heading_min": float(np.min(aa[:, 1])) if aa.size else 0,
                "action_heading_max": float(np.max(aa[:, 1])) if aa.size else 0,
                "action_speed_mean": float(np.mean(aa[:, 2])) if aa.size else 0,
                "action_speed_min": float(np.min(aa[:, 2])) if aa.size else 0,
                "action_speed_max": float(np.max(aa[:, 2])) if aa.size else 0,
            }

        ep_data.update({
            "ep_len": step,
            "winner": winner,
            "red_alive": red_alive, "blue_alive": blue_alive,
            "mav_alive": mav_death_step < 0,
            "mav_death_step": mav_death_step,
            "mav_death_reason": mav_death_reason,
            "first_death": first_death,
            "red_fired": red_fired, "blue_fired": blue_fired,
            "red_hits": red_hits, "blue_hits": blue_hits,
            "blue_launch_targets": dict(blue_launch_targets),
            "blue_hit_targets": dict(blue_hit_targets),
            "blue_first_launch": blue_first,
            "death_events": death_events,
            "per_agent": per_agent,
        })
        records.append(ep_data)
    return records


# ── report generator ─────────────────────────────────────────────────────
def _generate_report(records, output_dir):
    lines = ["# F22 MAV Failure Audit Report", ""]
    n = len(records)
    # Aggregate stats
    mav_dead = sum(1 for r in records if not r["mav_alive"])
    mav_first = sum(1 for r in records if r["first_death"] == "red_0")
    reasons = defaultdict(int)
    for r in records:
        reasons[r["mav_death_reason"] or "survived"] += 1
    lines.append(f"## Summary ({n} episodes)")
    lines.append(f"- MAV died: {mav_dead}/{n} ({mav_dead/n*100:.0f}%)")
    lines.append(f"- MAV first-death: {mav_first}/{n} ({mav_first/n*100:.0f}%)")
    lines.append(f"- Death reasons: {dict(reasons)}")
    lines.append("")
    # By mode
    modes = defaultdict(list)
    for r in records:
        modes[(r["mav_mode"], r["opp_mode"])].append(r)
    lines.append("## Per-mode summary")
    lines.append(f"| mav_mode opp_mode | episodes | mav_died | mav_first | top_reason | red_win | blue_win | mav_mean_alt | mav_max_pitch | red_hits |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (mm, om), recs in sorted(modes.items()):
        died = sum(1 for r in recs if not r["mav_alive"])
        first_d = sum(1 for r in recs if r["first_death"] == "red_0")
        reason = Counter(r["mav_death_reason"] or "survived" for r in recs).most_common(1)[0][0]
        rw = sum(1 for r in recs if r["winner"] == "red")
        bw = sum(1 for r in recs if r["winner"] == "blue")
        pa0 = [r.get("per_agent", {}).get("red_0", {}) for r in recs]
        alt_m = np.mean([a.get("mean_altitude_m", 0) for a in pa0])
        pitch_m = np.mean([a.get("max_abs_pitch_deg", 0) for a in pa0])
        rh = sum(r.get("red_hits", 0) for r in recs)
        lines.append(f"| {mm} {om} | {len(recs)} | {died} | {first_d} | {reason} | {rw} | {bw} | {alt_m:.0f} | {pitch_m:.0f} | {rh} |")
    lines.append("")

    # Initial distance analysis
    mav_nearest = sum(1 for r in records
                      if r["init_mav_to_blue_m"] and
                      min(r["init_mav_to_blue_m"].values()) == r["init_mav_to_blue_m"].get("blue_0", 1e9) or
                      min(r["init_mav_to_blue_m"].values()) == r["init_mav_to_blue_m"].get("blue_1", 1e9))
    lines.append("## Initial geometry")
    lines.append(f"- MAV nearest to a blue: {mav_nearest}/{n} episodes")
    lines.append("")

    # Root cause decomposition
    # Per-agent comparison (learned_all only)
    learned = [r for r in records if r["mav_mode"] == "learned_all"]
    if learned:
        lines.append("## Per-agent state & action ranges (learned_all)")
        lines.append("| agent | max_pitch | max_roll | min_alt | mean_alt | min_spd | mean_spd | act_pitch [min/mean/max] | act_speed [min/mean/max] |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rid in ["red_0", "red_1", "red_2"]:
            pa = {}
            for r in learned:
                a = r.get("per_agent", {}).get(rid, {})
                for k, v in a.items():
                    pa.setdefault(k, []).append(v)
            if not pa: continue
            lines.append(
                f"| {rid} | {np.mean(pa['max_abs_pitch_deg']):.0f} | {np.mean(pa['max_abs_roll_deg']):.0f} | "
                f"{np.mean(pa['min_altitude_m']):.0f} | {np.mean(pa['mean_altitude_m']):.0f} | "
                f"{np.mean(pa['min_speed_mps']):.0f} | {np.mean(pa['mean_speed_mps']):.0f} | "
                f"{np.mean(pa['action_pitch_min']):.2f}/{np.mean(pa['action_pitch_mean']):.2f}/{np.mean(pa['action_pitch_max']):.2f} | "
                f"{np.mean(pa['action_speed_min']):.2f}/{np.mean(pa['action_speed_mean']):.2f}/{np.mean(pa['action_speed_max']):.2f} |")
        # red_0 target pitch
        red0_ap = [r.get("per_agent", {}).get("red_0", {}).get("action_pitch_mean", 0) for r in learned]
        red0_ap_min = [r.get("per_agent", {}).get("red_0", {}).get("action_pitch_min", 0) for r in learned]
        lines.append(f"- red_0 target_pitch_deg: mean={np.mean(red0_ap)*90:.1f} min={np.mean(red0_ap_min)*90:.1f}")
        lines.append(f"- red_0 action_pitch <= 0: {np.mean(red0_ap) <= 0} ({np.mean(red0_ap)*90:.1f} deg)")
        lines.append("")

    lines.append("## Root cause assessment")
    pa_recs = [r.get("per_agent", {}).get("red_0", {}) for r in records if r["mav_mode"] == "learned_all"]
    alt_mins = [a.get("min_altitude_m",0) for a in pa_recs]
    alt_maxs = [a.get("mean_altitude_m",0) for a in pa_recs]
    pitch_maxs = [a.get("max_abs_pitch_deg",0) for a in pa_recs]
    lines.append(f"- **Control/flight-dynamics**: MAV pitch max {np.mean(pitch_maxs):.0f} deg, altitude {np.mean(alt_mins):.0f}-{np.mean(alt_maxs):.0f} m")
    lines.append(f"- **Blue tactical pressure**: blue fired {sum(r['blue_fired'] for r in records)} missiles, hit {sum(r['blue_hits'] for r in records)} targets")
    lines.append(f"- **MAV first-death rate {mav_first/n*100:.0f}%** — MAV is priority target")
    lines.append("")
    lines.append("## Ablation comparison")
    learned = [r for r in records if r["mav_mode"] == "learned_all"]
    fixed = [r for r in records if r["mav_mode"].startswith("fixed_mav")]
    if learned and fixed:
        l_died = sum(1 for r in learned if not r["mav_alive"])
        f_died = sum(1 for r in fixed if not r["mav_alive"])
        lines.append(f"- Learned MAV died: {l_died}/{len(learned)}")
        lines.append(f"- Fixed MAV died: {f_died}/{len(fixed)}")
    blue_zero = [r for r in records if r["opp_mode"] == "zero"]
    blue_rule = [r for r in records if r["opp_mode"] == "brma_rule"]
    if blue_zero and blue_rule:
        bz_died = sum(1 for r in blue_zero if not r["mav_alive"])
        br_died = sum(1 for r in blue_rule if not r["mav_alive"])
        lines.append(f"- Blue zero (no attack): MAV died {bz_died}/{len(blue_zero)} ({bz_died/max(len(blue_zero),1)*100:.0f}%)")
        lines.append(f"- Blue brma_rule: MAV died {br_died}/{len(blue_rule)} ({br_died/max(len(blue_rule),1)*100:.0f}%)")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",
                        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--output-dir", default="outputs/f22_mav_failure_audit")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    _validate_f22_config(args.config)

    ckpt = _resolve_checkpoint(args.checkpoint)
    meta = _load_meta(ckpt)
    device = torch.device(args.device)
    policy = _build_policy(meta, device)
    policy.load(str(ckpt), map_location=device)
    policy.eval()
    adapter = HeteroEntitySetAdapter()

    os.makedirs(args.output_dir, exist_ok=True)

    all_records = []

    # Ablation cases
    runs = [
        ("learned_all", args.opponent_policy),
        ("fixed_mav_zero", args.opponent_policy),
        ("fixed_mav_speed_0_3", args.opponent_policy),
        ("mav_action_scale_0_3", args.opponent_policy),
        ("mav_action_scale_0_1", args.opponent_policy),
    ]
    # Also run with blue zero opponent
    if args.opponent_policy != "zero":
        runs += [
            ("learned_all", "zero"),
            ("fixed_mav_zero", "zero"),
            ("fixed_mav_speed_0_3", "zero"),
        ]

    for mav_mode, opp_mode in runs:
        label = f"mav={mav_mode}_opp={opp_mode}"
        print(f"=== {label} ===", flush=True)
        records = run_episodes(policy, device, args.config, args, adapter,
                               mav_mode, opp_mode, args.episodes, args.seed)
        for r in records:
            r["mav_mode"] = mav_mode
            r["opp_mode"] = opp_mode
        all_records.extend(records)

    # Write CSV
    csv_path = os.path.join(args.output_dir, "episodes.csv")
    fieldnames = ["episode", "mav_mode", "opp_mode", "ep_len", "winner",
                  "mav_alive", "mav_death_step", "mav_death_reason", "first_death",
                  "red_fired", "blue_fired", "red_hits", "blue_hits",
                  "blue_first_target", "blue_first_step", "blue_first_range"]
    for rid in ["red_0", "red_1", "red_2"]:
        fieldnames += [f"{rid}_max_abs_pitch_deg", f"{rid}_max_abs_roll_deg",
                       f"{rid}_min_altitude_m", f"{rid}_mean_altitude_m",
                       f"{rid}_min_speed_mps", f"{rid}_mean_speed_mps",
                       f"{rid}_action_pitch_mean", f"{rid}_action_pitch_min",
                       f"{rid}_action_pitch_max",
                       f"{rid}_action_heading_mean", f"{rid}_action_heading_min",
                       f"{rid}_action_heading_max",
                       f"{rid}_action_speed_mean", f"{rid}_action_speed_min",
                       f"{rid}_action_speed_max"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_records:
            r_out = dict(r)
            r_out["blue_first_target"] = r.get("blue_first_launch", {}).get("target", "") if r.get("blue_first_launch") else ""
            r_out["blue_first_step"] = r.get("blue_first_launch", {}).get("step", "") if r.get("blue_first_launch") else ""
            r_out["blue_first_range"] = r.get("blue_first_launch", {}).get("range", "") if r.get("blue_first_launch") else ""
            pa = r.get("per_agent", {})
            for rid in ["red_0", "red_1", "red_2"]:
                a = pa.get(rid, {})
                for k in ["max_abs_pitch_deg","max_abs_roll_deg","min_altitude_m","mean_altitude_m",
                          "min_speed_mps","mean_speed_mps",
                          "action_pitch_mean","action_pitch_min","action_pitch_max",
                          "action_heading_mean","action_heading_min","action_heading_max",
                          "action_speed_mean","action_speed_min","action_speed_max"]:
                    r_out[f"{rid}_{k}"] = a.get(k, 0)
            w.writerow(r_out)
    print(f"Saved: {csv_path}")

    # Death events CSV
    de_path = os.path.join(args.output_dir, "death_events.csv")
    de_fieldnames = ["episode", "mav_mode", "opp_mode", "agent_id", "death_reason",
                     "shooter_id", "step"]
    with open(de_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=de_fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_records:
            for de in r.get("death_events", []):
                de_out = {"episode": r["episode"], "mav_mode": r["mav_mode"],
                          "opp_mode": r["opp_mode"]}
                de_out.update({k: de.get(k, "") for k in de_fieldnames if k not in de_out})
                w.writerow(de_out)
    print(f"Saved: {de_path}")

    # Summary JSON
    summary = {
        "checkpoint": str(ckpt),
        "config": args.config,
        "episodes": args.episodes,
        "modes": sorted(set((r["mav_mode"], r["opp_mode"]) for r in all_records)),
        "records": all_records,
    }
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    _generate_report(all_records, args.output_dir)


if __name__ == "__main__":
    from collections import Counter
    main()
