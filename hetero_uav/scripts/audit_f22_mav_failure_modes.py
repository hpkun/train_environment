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
    with open(config_path) as f:
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
        red0_alt, red0_spd, red0_roll, red0_pitch = [], [], [], []
        red0_act_mean, red0_act_sat = [], []
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

            # Track MAV state
            mav_sim = env.red_planes.get("red_0")
            if mav_sim is not None and mav_sim.is_alive:
                rpy = mav_sim.get_rpy(); vel = mav_sim.get_velocity()
                red0_alt.append(mav_sim.get_geodetic()[2])
                red0_spd.append(float(np.linalg.norm(vel)))
                red0_roll.append(float(np.rad2deg(rpy[0])))
                red0_pitch.append(float(np.rad2deg(rpy[1])))
                red0_act_mean.append(float(np.mean(np.abs(raw_act[0]))))
                red0_act_sat.append(float(np.mean(np.abs(raw_act[0]) >= 0.999)))

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

        ep_data.update({
            "ep_len": step,
            "winner": winner,
            "red_alive": red_alive, "blue_alive": blue_alive,
            "mav_alive": mav_death_step < 0,
            "mav_death_step": mav_death_step,
            "mav_death_reason": mav_death_reason,
            "first_death": first_death,
            "red0_alt_min": float(np.min(red0_alt)) if red0_alt else 0,
            "red0_alt_max": float(np.max(red0_alt)) if red0_alt else 0,
            "red0_alt_mean": float(np.mean(red0_alt)) if red0_alt else 0,
            "red0_spd_min": float(np.min(red0_spd)) if red0_spd else 0,
            "red0_spd_max": float(np.max(red0_spd)) if red0_spd else 0,
            "red0_spd_mean": float(np.mean(red0_spd)) if red0_spd else 0,
            "red0_roll_max": float(np.max(np.abs(red0_roll))) if red0_roll else 0,
            "red0_pitch_max": float(np.max(np.abs(red0_pitch))) if red0_pitch else 0,
            "red0_act_mean": float(np.mean(red0_act_mean)) if red0_act_mean else 0,
            "red0_act_sat": float(np.mean(red0_act_sat)) if red0_act_sat else 0,
            "red_fired": red_fired, "blue_fired": blue_fired,
            "red_hits": red_hits, "blue_hits": blue_hits,
            "blue_launch_targets": dict(blue_launch_targets),
            "blue_hit_targets": dict(blue_hit_targets),
            "blue_first_launch": blue_first,
            "death_events": death_events,
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
        alt_m = np.mean([r["red0_alt_mean"] for r in recs])
        pitch_m = np.mean([r["red0_pitch_max"] for r in recs])
        rh = sum(r["red_hits"] for r in recs)
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
    lines.append("## Root cause assessment")
    lines.append(f"- **Control/flight-dynamics**: MAV pitch range {np.mean([r['red0_pitch_max'] for r in records]):.0f} deg, altitude {np.mean([r['red0_alt_min'] for r in records]):.0f}-{np.mean([r['red0_alt_max'] for r in records]):.0f} m")
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
                  "red0_alt_min", "red0_alt_max", "red0_alt_mean",
                  "red0_spd_min", "red0_spd_max", "red0_spd_mean",
                  "red0_roll_max", "red0_pitch_max",
                  "red0_act_mean", "red0_act_sat",
                  "red_fired", "blue_fired", "red_hits", "blue_hits",
                  "blue_first_target", "blue_first_step", "blue_first_range"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_records:
            r_out = dict(r)
            r_out["blue_first_target"] = r.get("blue_first_launch", {}).get("target", "") if r.get("blue_first_launch") else ""
            r_out["blue_first_step"] = r.get("blue_first_launch", {}).get("step", "") if r.get("blue_first_launch") else ""
            r_out["blue_first_range"] = r.get("blue_first_launch", {}).get("range", "") if r.get("blue_first_launch") else ""
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
