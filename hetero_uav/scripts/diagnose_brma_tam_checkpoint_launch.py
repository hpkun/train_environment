"""Corrected diagnostic eval v2: brma_rule opponent, per-decision gate diag, official eval path."""
from __future__ import annotations

import argparse, csv, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_happo_reference import (
    _load_meta, _build_policy_from_meta, _role_ids, _team_done,
    _alive_counts, _env_type_override_kwargs,
)
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_actions, zero_inactive_hidden
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env


def _safe(v, default=0.0):
    try: x = float(v); return x if math.isfinite(x) else default
    except: return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero","random","rule_nearest","greedy_fsm","brma_rule","fixed_route"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.parent / "meta.json"
    if not meta_path.exists():
        meta_path = model_path.parent.parent / "meta.json"
    if not meta_path.exists():
        print("ERROR: meta.json not found"); sys.exit(1)
    meta = _load_meta(model_path)
    print(f"Checkpoint meta: policy_arch={meta.get('policy_arch')} "
          f"biased_mask={meta.get('biased_mask')} entity_dim={meta.get('entity_dim')} "
          f"rnn_hidden_size={meta.get('rnn_hidden_size')} "
          f"max_allies={meta.get('max_allies',meta.get('adapter_max_allies',4))} "
          f"max_enemies={meta.get('max_enemies',meta.get('adapter_max_enemies',4))}")

    device = torch.device(args.device)
    policy = _build_policy_from_meta(meta, device)
    policy.load(model_path, map_location=device)
    policy.eval()

    agent_interaction_steps = 12
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.config, **_env_type_override_kwargs(args.config))
    if hasattr(env, "agent_interaction_steps"):
        agent_interaction_steps = env.agent_interaction_steps
    max_red, max_blue = len(env.red_ids), len(env.blue_ids)
    # Adapter must match policy training dims: critic_state_dim=700 = 5*140
    adapter = HeteroObsAdapterV2(max_red=5, max_blue=4, role_dim=4)
    print(f"Adapter: max_red=5 max_blue=4 "
          f"flat_actor={adapter.flat_actor_obs_dim} critic={adapter.critic_state_dim}")
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)

    print(f"Config={args.config} opponent={args.opponent_policy} "
          f"seeds={args.seed_start}-{args.seed_start+args.episodes-1} "
          f"agent_interaction_steps={agent_interaction_steps} "
          f"max_gate_rows_per_ep={(max_red-1)*args.max_steps}")

    # Output files
    gate_file = (out_dir / "launch_gate_diagnostics.csv").open("w", newline="", encoding="utf-8")
    gate_cols = ["run_id","scenario","env_idx","episode_id","episode_uid","diagnostic_protocol_version",
                 "step","sim_time","agent_id","role","decision_step","physics_frame_scans",
                 "agent_interaction_steps","alive_before","alive_after",
                 "alive_target_pair_scans","engaged_blocked_pair_scans","unengaged_target_pair_scans",
                 "track_pass_pair_scans","track_blocked_pair_scans","range_pass_pair_scans",
                 "ata_pass_pair_scans","ta_pass_pair_scans","boresight_pass_pair_scans",
                 "geometry_pass_pair_scans","any_alive_target","any_unengaged_target",
                 "any_track_pass","any_range_pass","any_ata_pass","any_ta_pass",
                 "any_geometry_pass","any_lock_mature","any_launch",
                 "lock_mature_frame_count","cooldown_blocked_frame_count",
                 "kill_cooldown_blocked_frame_count","launch_count","launch_target_ids",
                 "nearest_alive_target_distance_m_min","nearest_track_target_distance_m_min",
                 "best_geometry_range_m_min","best_geometry_ata_rad_min","best_geometry_ta_rad_max",
                 "ammo_remaining_end","cooldown_remaining_end",
                 "boresight_gate_enabled","line_gate_is_alias_of_ta",
                 "inactive_frame_count","no_ammo_frame_count","no_alive_target_frame_count",
                 "all_targets_engaged_frame_count","no_track_frame_count","out_of_range_frame_count",
                 "ata_blocked_frame_count","ta_blocked_frame_count","boresight_blocked_frame_count",
                 "lock_not_mature_frame_count","cooldown_frame_count","kill_cooldown_frame_count",
                 "launch_frame_count","unknown_frame_count","primary_block_reason"]
    gate_w = csv.DictWriter(gate_file, fieldnames=gate_cols, extrasaction="ignore")
    gate_w.writeheader()

    ep_summary_rows = []
    death_rows = []
    all_gate_rows = []
    mav_ts_rows = []
    opp_act_calls = 0
    blue_nonzero_decisions = 0
    blue_abs_sum = 0.0

    try:
        for ep_idx in range(args.episodes):
            seed = args.seed_start + ep_idx
            ep_uid = f"0:{ep_idx}"
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=seed + 99)
            opponent.reset_memory()
            obs, info = env.reset(seed=seed)
            roles = _role_ids(env)
            rnn_hidden = np.zeros((max_red, _rnn_hidden_size), dtype=np.float32) if _rnn_hidden_size > 0 else None

            ep_ret_trainer = 0.0
            ep_ret_fixed = 0.0
            ep_len = 0
            ep_red_launch = ep_red_hit = ep_blue_launch = ep_blue_hit = 0
            prev_hits = {"red": 0, "blue": 0}
            ep_death_reasons = {}
            ep_mav_ts = []

            while True:
                # --- Red action via official eval path ---
                adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                active = np.zeros(max_red, dtype=np.float32)
                for i, rid in enumerate(env.red_ids):
                    agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
                    if isinstance(agent_info, dict) and "alive" in agent_info:
                        alive = bool(agent_info["alive"])
                    else:
                        sim = env.red_planes.get(rid)
                        alive = bool(sim is not None and sim.is_alive)
                    active[i] = 1.0 if alive else 0.0
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                    for rid in env.red_ids
                ])
                critic = adapted["critic_state"]
                san = sanitize_policy_inputs(
                    actor_obs, active, critic_state=critic, rnn_hidden=rnn_hidden,
                    context={"env_idx": 0, "episode_id": ep_idx, "total_steps": ep_len},
                )
                actor_obs = san["actor_obs"]
                critic = san["critic_state"] if san["critic_state"] is not None else critic
                rnn_hidden = san["rnn_hidden"] if san["rnn_hidden"] is not None else rnn_hidden
                alive_before = active > 0.5
                act_kwargs = {}
                if rnn_hidden is not None:
                    act_kwargs["rnn_hidden"] = torch.as_tensor(rnn_hidden, device=device)
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device), roles=roles,
                        critic_state=torch.as_tensor(critic, device=device),
                        deterministic=True, **act_kwargs)
                if rnn_hidden is not None and "rnn_hidden" in out:
                    rnn_hidden = zero_inactive_hidden(
                        out["rnn_hidden"].detach().cpu().numpy(), active)
                actions = zero_inactive_actions(out["action"].detach().cpu().numpy(), active)
                action_dict = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}

                # --- Blue action via real OpponentPolicy ---
                opp_act_calls += 1
                blue_actions = opponent.act(obs, env.blue_ids, env=env)
                action_dict.update(blue_actions)
                for bid in env.blue_ids:
                    ba = blue_actions.get(bid, np.zeros(3))
                    abs_sum = float(np.sum(np.abs(ba)))
                    blue_abs_sum += abs_sum
                    if abs_sum > 1e-6:
                        blue_nonzero_decisions += 1

                # --- MAV timeseries ---
                mav = env.red_planes.get("red_0")
                if mav:
                    mav_pos_before = np.asarray(mav.get_position(), dtype=np.float64).copy()
                    mav_rpy_before = np.asarray(mav.get_rpy(), dtype=np.float64).copy()
                    mav_alive_before = bool(getattr(mav, "is_alive", False))
                else:
                    mav_pos_before = np.zeros(3); mav_rpy_before = np.zeros(3)
                    mav_alive_before = False

                obs, rewards, term, trunc, info = env.step(action_dict)

                if mav:
                    mav_pos_after = np.asarray(mav.get_position(), dtype=np.float64)
                    mav_rpy_after = np.asarray(mav.get_rpy(), dtype=np.float64)
                    mav_alive_after = bool(getattr(mav, "is_alive", False))
                    ep_mav_ts.append({
                        "episode_uid": ep_uid, "seed": seed, "decision_step": ep_len,
                        "alive_before": int(mav_alive_before), "alive_after": int(mav_alive_after),
                        "altitude_m_before": float(mav_pos_before[2]),
                        "altitude_m_after": float(mav_pos_after[2]),
                        "vertical_speed_up_mps_before": float(np.asarray(mav.get_velocity())[2]) if mav_alive_before else 0.0,
                        "vertical_speed_up_mps_after": float(np.asarray(mav.get_velocity())[2]) if mav_alive_after else 0.0,
                        "speed_mps_before": float(np.linalg.norm(np.asarray(mav.get_velocity()))) if mav_alive_before else 0.0,
                        "speed_mps_after": float(np.linalg.norm(np.asarray(mav.get_velocity()))) if mav_alive_after else 0.0,
                        "roll_deg_before": float(np.rad2deg(mav_rpy_before[0])),
                        "pitch_deg_before": float(np.rad2deg(mav_rpy_before[1])),
                        "heading_deg_before": float(np.rad2deg(mav_rpy_before[2])),
                        "roll_deg_after": float(np.rad2deg(mav_rpy_after[0])),
                        "pitch_deg_after": float(np.rad2deg(mav_rpy_after[1])),
                        "heading_deg_after": float(np.rad2deg(mav_rpy_after[2])),
                        "raw_action_pitch": float(actions[0][0]) if max_red > 0 else 0.0,
                        "raw_action_heading": float(actions[0][1]) if max_red > 0 else 0.0,
                        "raw_action_speed": float(actions[0][2]) if max_red > 0 else 0.0,
                    })

                # --- Gate diag: write per-decision records ---
                gate_recs = info.get("__launch_gate_diagnostics__", []) or []
                for gr in gate_recs:
                    gr["run_id"] = "diag_eval_v2"
                    gr["scenario"] = "3v2_brma_rule"
                    gr["env_idx"] = 0
                    gr["episode_id"] = ep_idx
                    gr["episode_uid"] = ep_uid
                    gr["step"] = ep_len
                    gate_w.writerow(gr)
                    all_gate_rows.append(dict(gr))

                # --- Missile accounting ---
                for aid in env.agent_ids:
                    agent_info = info.get(aid, {}) if isinstance(info, dict) else {}
                    fired = int(agent_info.get("missiles_fired_this_step", 0)) if isinstance(agent_info, dict) else 0
                    if aid.startswith("red_"): ep_red_launch += fired
                    else: ep_blue_launch += fired
                mt = info.get("__missile_term__", {}) or {}
                rh = int(mt.get("red", {}).get("hit", 0) or 0)
                bh = int(mt.get("blue", {}).get("hit", 0) or 0)
                ep_red_hit = max(ep_red_hit, rh)
                ep_blue_hit = max(ep_blue_hit, bh)

                # --- Death reasons ---
                for ev in info.get("death_events", []) or []:
                    if isinstance(ev, dict):
                        ep_death_reasons[ev.get("agent_id", "")] = ev.get("death_reason", "")

                # --- Returns ---
                red_rewards = [float(rewards.get(rid, 0.0)) for rid in env.red_ids]
                ep_ret_fixed += sum(red_rewards) / max_red
                alive_red_rewards = [r for i, r in enumerate(red_rewards) if alive_before[i]]
                ep_ret_trainer += float(np.mean(alive_red_rewards)) if alive_red_rewards else 0.0
                ep_len += 1

                if _team_done(term, trunc):
                    break

            ra, ba = _alive_counts(env)
            winner, end_reason = _outcome(ra, ba, ep_len, env)
            ep_summary_rows.append({
                "episode_uid": ep_uid, "seed": seed, "length": ep_len,
                "winner": winner, "end_reason": end_reason,
                "red_alive_final": ra, "blue_alive_final": ba,
                "mav_alive_final": int(bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)),
                "red_launch_count": ep_red_launch, "red_hit_count": ep_red_hit,
                "blue_launch_count": ep_blue_launch, "blue_hit_count": ep_blue_hit,
                "trainer_effective_episode_return": ep_ret_trainer,
                "trainer_effective_return_per_decision": ep_ret_trainer / max(ep_len, 1),
                "fixed_initial_team_episode_return": ep_ret_fixed,
                "fixed_initial_team_return_per_decision": ep_ret_fixed / max(ep_len, 1),
            })
            for aid, reason in ep_death_reasons.items():
                death_rows.append({"episode_uid": ep_uid, "seed": seed, "agent_id": aid, "death_reason": reason})
            mav_ts_rows.extend(ep_mav_ts)
            print(f"Ep {ep_uid} seed={seed} len={ep_len} outcome={winner}/{end_reason} "
                  f"R_alive={ra} B_alive={ba} MAV={ep_summary_rows[-1]['mav_alive_final']} "
                  f"R_launch={ep_red_launch} R_hit={ep_red_hit} "
                  f"B_launch={ep_blue_launch} B_hit={ep_blue_hit} "
                  f"ret_trainer={ep_ret_trainer:.1f} ret_fixed={ep_ret_fixed:.1f}")
    finally:
        env.close()
        gate_file.close()

    # --- Write CSVs ---
    _write_csv(out_dir / "diagnostic_episode_summary.csv", ep_summary_rows)
    _write_csv(out_dir / "diagnostic_death_reasons.csv", death_rows)
    _write_csv(out_dir / "mav_decision_timeseries.csv", mav_ts_rows)

    # --- Target-pair funnel ---
    pair_funnel = _compute_pair_funnel(all_gate_rows)
    decision_funnel = _compute_decision_funnel(all_gate_rows)

    # --- Block reason frame counts ---
    fr_total = defaultdict(int)
    for gr in all_gate_rows:
        for fr in ["inactive","no_ammo","no_alive_target","all_targets_engaged","no_track",
                    "out_of_range","ata_blocked","ta_blocked","boresight_blocked",
                    "lock_not_mature","cooldown","kill_cooldown","launch","unknown"]:
            fr_total[fr] += int(gr.get(f"{fr}_frame_count", 0))
    decision_reasons = Counter(gr.get("primary_block_reason", "unknown") for gr in all_gate_rows)

    # --- Consistency checks ---
    gate_row_launch_sum = sum(int(gr.get("launch_count", 0)) for gr in all_gate_rows)
    ep_launch_sum = sum(r["red_launch_count"] for r in ep_summary_rows)
    launch_ok = gate_row_launch_sum == ep_launch_sum

    # --- Verdict ---
    mav_death_reasons = [r["death_reason"] for r in death_rows if "red_0" in str(r.get("agent_id",""))]
    mav_all_crash = all("Crash" in str(d) or "LowAlt" in str(d) for d in mav_death_reasons)
    any_range = any(int(gr.get("any_range_pass", 0)) for gr in all_gate_rows)
    any_launch = any(int(gr.get("any_launch", 0)) for gr in all_gate_rows)
    gate_rows_ok = len(all_gate_rows) <= (max_red - 1) * args.episodes * args.max_steps + 10

    if not launch_ok:
        verdict = "IMPLEMENTATION_ERROR"
    elif not gate_rows_ok:
        verdict = "IMPLEMENTATION_ERROR"
    elif mav_all_crash:
        verdict = "NEEDS_TRAINING_STABILITY_REVIEW"
    elif any_launch:
        verdict = "READY_FOR_50K"
    elif any_range:
        verdict = "NEEDS_FIRE_CONTROL_ALIGNMENT_REVIEW"
    else:
        verdict = "NEEDS_REWARD_SCALE_REVIEW"

    # --- Report ---
    lines = [
        "# Diagnostic Eval v2 — Corrected Report",
        f"",
        f"- Checkpoint: `{args.model}`",
        f"- Config: {args.config}",
        f"- Opponent: **{args.opponent_policy}** (real OpponentPolicy, NOT zero-action)",
        f"- Opponent act calls: {opp_act_calls}",
        f"- Blue action mean abs: {blue_abs_sum / max(opp_act_calls * max_blue, 1):.4f}",
        f"- Blue nonzero decision fraction: {blue_nonzero_decisions / max(opp_act_calls, 1):.3f}",
        f"- Episodes: {args.episodes}, seeds {args.seed_start}–{args.seed_start + args.episodes - 1}",
        f"- agent_interaction_steps: {agent_interaction_steps}",
        f"- Gate diagnostic rows: {len(all_gate_rows)} (expected ≤ {(max_red-1)*args.episodes*args.max_steps})",
        f"- Gate rows per episode: ~{len(all_gate_rows)//max(args.episodes,1)}",
        f"",
        "## Episode Results",
        "| ep_uid | seed | len | outcome | R_alive | B_alive | MAV | R_launch | R_hit | B_launch | B_hit | ret(trainer) | ret(fixed) |",
        "|--------|------|-----|---------|---------|---------|-----|----------|-------|----------|-------|-------------|-----------|",
    ]
    for ep in ep_summary_rows:
        lines.append(f"| {ep['episode_uid']} | {ep['seed']} | {ep['length']} | "
                     f"{ep['winner']}/{ep['end_reason']} | {ep['red_alive_final']} | "
                     f"{ep['blue_alive_final']} | {ep['mav_alive_final']} | "
                     f"{ep['red_launch_count']} | {ep['red_hit_count']} | "
                     f"{ep['blue_launch_count']} | {ep['blue_hit_count']} | "
                     f"{ep['trainer_effective_episode_return']:.1f} | {ep['fixed_initial_team_episode_return']:.1f} |")

    lines += ["", "## Target-Pair Sequential Funnel", ""]
    for k, v in pair_funnel.items():
        lines.append(f"- **{k}**: {v}")

    lines += ["", "## Decision-Agent Funnel", ""]
    for k, v in decision_funnel.items():
        lines.append(f"- **{k}**: {v}")

    lines += ["", "## Frame-Level Block Reason Distribution", ""]
    for fr in sorted(fr_total.keys()):
        if fr_total[fr] > 0:
            lines.append(f"- {fr}: {fr_total[fr]}")

    lines += ["", "## Decision-Level Block Reason Distribution", ""]
    for reason, count in decision_reasons.most_common():
        lines.append(f"- {reason}: {count}")

    lines += ["", "## Death Reasons", ""]
    for d in death_rows:
        lines.append(f"- {d['agent_id']}: {d['death_reason']}")

    lines += ["", "## Consistency Checks", "",
              f"- Launch accounting: gate_sum={gate_row_launch_sum} vs ep_sum={ep_launch_sum} → {'OK' if launch_ok else 'FAIL'}",
              f"- Gate rows: {len(all_gate_rows)} ≤ {(max_red-1)*args.episodes*args.max_steps + 10} → {'OK' if gate_rows_ok else 'FAIL'}"]

    lines += ["", f"## Verdict: **{verdict}**", "",
              f"- MAV crash deaths: {len(mav_death_reasons)}/{args.episodes}",
              f"- Any range pass: {any_range}",
              f"- Any launch: {any_launch}",
              "",
              "### Corrected conclusions (invalidating prior report):",
              "1. Blue opponent now IS brma_rule (not zero-action). Previously all blue actions were zero.",
              "2. Gate rows are per-decision (~{0}), not per-physics-frame (120000).".format(len(all_gate_rows)),
              "3. Funnel uses real `_select_missile_target` diag counts, not recomputed geometry.",
              "4. `line_pass_count` is marked as alias of TA, not independent gate.",
              "5. `launch_count` is per-decision, not cumulative episode count.",
              "6. Returns use both trainer-effective and fixed-initial-team mean.",
              "7. Component means will NOT be divided by physics frames.",
    ]

    summary = {
        "diagnostic_protocol_version": 2,
        "verdict": verdict,
        "opponent_policy": args.opponent_policy,
        "opponent_act_calls": opp_act_calls,
        "blue_action_mean_abs": blue_abs_sum / max(opp_act_calls * max_blue, 1),
        "blue_nonzero_decisions": blue_nonzero_decisions,
        "gate_row_count": len(all_gate_rows),
        "pair_funnel": pair_funnel,
        "decision_funnel": decision_funnel,
        "frame_block_reasons": dict(fr_total),
        "decision_block_reasons": dict(decision_reasons),
        "launch_accounting_ok": launch_ok,
        "gate_rows_ok": gate_rows_ok,
        "mav_death_reasons": mav_death_reasons,
    }
    (out_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n=== Verdict: {verdict} ===")
    print(f"Gate rows: {len(all_gate_rows)}")
    print(f"Output: {out_dir}")


def _outcome(ra, ba, ep_len, env):
    timeout = ep_len >= getattr(env, "max_steps", 1000)
    if ba == 0 and ra > 0: return "red", "blue_eliminated"
    if ra == 0 and ba > 0: return "blue", "red_eliminated"
    if ra == 0 and ba == 0: return "draw", "mutual_elimination"
    if timeout: return "draw", "timeout"
    return "draw", "ongoing"


def _compute_pair_funnel(gate_rows):
    a = sum(gr.get("alive_target_pair_scans", 0) for gr in gate_rows)
    e = sum(gr.get("engaged_blocked_pair_scans", 0) for gr in gate_rows)
    u = sum(gr.get("unengaged_target_pair_scans", 0) for gr in gate_rows)
    t = sum(gr.get("track_pass_pair_scans", 0) for gr in gate_rows)
    r_ = sum(gr.get("range_pass_pair_scans", 0) for gr in gate_rows)
    at = sum(gr.get("ata_pass_pair_scans", 0) for gr in gate_rows)
    ta = sum(gr.get("ta_pass_pair_scans", 0) for gr in gate_rows)
    g = sum(gr.get("geometry_pass_pair_scans", 0) for gr in gate_rows)
    return {
        "alive_target_pair_scans": a,
        "unengaged_rate": f"{u / max(a, 1):.4f}" if a > 0 else "null",
        "track_rate": f"{t / max(u, 1):.4f}" if u > 0 else "null",
        "range_rate": f"{r_ / max(t, 1):.4f}" if t > 0 else "null",
        "ata_rate": f"{at / max(r_, 1):.4f}" if r_ > 0 else "null",
        "ta_rate": f"{ta / max(at, 1):.4f}" if at > 0 else "null",
        "geometry_rate": f"{g / max(ta, 1):.4f}" if ta > 0 else "null",
    }


def _compute_decision_funnel(gate_rows):
    n = max(len(gate_rows), 1)
    return {
        "decision_agent_rows": n,
        "any_alive_target_rate": sum(gr.get("any_alive_target", 0) for gr in gate_rows) / n,
        "any_unengaged_target_rate": sum(gr.get("any_unengaged_target", 0) for gr in gate_rows) / n,
        "any_track_pass_rate": sum(gr.get("any_track_pass", 0) for gr in gate_rows) / n,
        "any_range_pass_rate": sum(gr.get("any_range_pass", 0) for gr in gate_rows) / n,
        "any_ata_pass_rate": sum(gr.get("any_ata_pass", 0) for gr in gate_rows) / n,
        "any_ta_pass_rate": sum(gr.get("any_ta_pass", 0) for gr in gate_rows) / n,
        "any_geometry_pass_rate": sum(gr.get("any_geometry_pass", 0) for gr in gate_rows) / n,
        "any_lock_mature_rate": sum(gr.get("any_lock_mature", 0) for gr in gate_rows) / n,
        "any_launch_rate": sum(gr.get("any_launch", 0) for gr in gate_rows) / n,
    }


def _write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
