"""Deterministic diagnostic eval: 3V2, 5 episodes, brma_rule opponent.

Loads a checkpoint, runs deterministic rollout, records launch gate diagnostics
and death reasons. Does NOT modify policy, env, reward, or fire-control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import (
    BRMARecurrentMaskedHAPPOReferencePolicy,
    BRMARecurrentHAPPOReferencePolicy,
)
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTanhPolicy
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_actions, zero_inactive_hidden
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env import make_env
from scripts.experiment_logging_schema import (
    LAUNCH_GATE_DIAGNOSTICS_COLUMNS,
    ensure_csv,
)
from scripts.rich_logging import RichExperimentLogger


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _load_policy(model_path: Path, meta: dict, device):
    arch = meta.get("policy_arch", "")
    entity_dim = int(meta.get("entity_dim", 30))
    critic_state_dim = int(meta.get("critic_state_dim", 480))
    action_dim = int(meta.get("action_dim", 3))
    hidden_dim = int(meta.get("hidden_dim", 128))
    num_heads = int(meta.get("num_attention_heads", 4))
    rnn_hidden = int(meta.get("rnn_hidden_size", 128))
    max_allies = int(meta.get("max_allies", 4))
    max_enemies = int(meta.get("max_enemies", 4))
    num_agents = int(meta.get("num_agents", 3))
    biased_mask = bool(meta.get("biased_mask", False))

    if arch in ("pure_happo",):
        policy = PureHAPPOPolicy(
            actor_obs_dim=meta.get("actor_obs_dim", 96),
            critic_state_dim=critic_state_dim,
            action_dim=action_dim, num_agents=num_agents,
        )
    elif arch == "brma_recurrent_masked":
        policy = BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=entity_dim, critic_state_dim=critic_state_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            num_attention_heads=num_heads, rnn_hidden_size=rnn_hidden,
            max_allies=max_allies, max_enemies=max_enemies,
            biased_mask=biased_mask,
        )
    elif arch == "brma_recurrent":
        policy = BRMARecurrentHAPPOReferencePolicy(
            entity_dim=entity_dim, critic_state_dim=critic_state_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            num_attention_heads=num_heads, rnn_hidden_size=rnn_hidden,
            max_allies=max_allies, max_enemies=max_enemies,
        )
    else:
        raise ValueError(f"unsupported arch: {arch}")
    state = torch.load(model_path, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


def _run_episode(env, policy, adapter, device, seed: int, rnn_hidden=None):
    obs, _info = env.reset(seed=seed)
    red_ids = env.red_ids
    episode_data = {
        "steps": [], "reward_total": {rid: 0.0 for rid in red_ids},
        "death_reasons": {},
        "launch_gate_rows": [],
        "red_launches": 0, "red_hits": 0,
        "blue_launches": 0, "blue_hits": 0,
    }
    prev_hits = {"red": 0, "blue": 0}
    for step in range(env.max_steps):
        adapted = adapter.adapt_all(obs, red_ids=red_ids, blue_ids=env.blue_ids)
        actor_obs_np = np.stack([adapted["actor_obs"][rid] for rid in red_ids], axis=0)
        actor_obs_t = torch.as_tensor(actor_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        critic_t = torch.as_tensor(adapted["critic_state"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            result = policy.act(actor_obs_t, critic_state=critic_t, deterministic=True,
                               rnn_hidden=rnn_hidden)
        actions_np = result["action"].squeeze(0).cpu().numpy()
        if rnn_hidden is not None:
            rnn_hidden = result.get("rnn_hidden")
        action_dict = {rid: actions_np[i] for i, rid in enumerate(red_ids)}
        for bid in env.blue_ids:
            action_dict[bid] = np.zeros(3, dtype=np.float32)
        obs, rewards, term, trunc, info = env.step(action_dict)

        # Collect reward component sums
        comps = info.get("reward_components", {})
        for rid in red_ids:
            episode_data["reward_total"][rid] += float(rewards.get(rid, 0.0))
            c = comps.get(rid, {})
            episode_data["steps"].append({
                "step": step, "agent_id": rid,
                "role": env.agent_roles.get(rid, ""),
                "total": float(rewards.get(rid, 0.0)),
                **{k: _safe(c.get(k, 0)) for k in (
                    "brma_pitch", "brma_roll", "brma_vel",
                    "tam_speed_weighted", "tam_angle_weighted", "tam_distance_weighted",
                    "uav_event_total", "uav_event_kill", "uav_event_loss",
                    "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
                    "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
                    "mav_total", "uav_total", "total",
                )},
            })

        # Launch gate diagnostics
        gate_recs = info.get("__launch_gate_diagnostics__", []) or []
        for gr in gate_recs:
            gr["step"] = step
            gr["episode_id"] = 0
            gr["env_idx"] = 0
            gr["episode_uid"] = "0:0"
            episode_data["launch_gate_rows"].append(gr)

        # Missile events
        for rec in info.get("__launch_quality_step__", []) or []:
            if "red_" in str(rec.get("shooter_id", "")):
                episode_data["red_launches"] += 1
            else:
                episode_data["blue_launches"] += 1
        mt = info.get("__missile_term__", {}) or {}
        rh = int(mt.get("red", {}).get("hit", 0) or 0)
        bh = int(mt.get("blue", {}).get("hit", 0) or 0)
        episode_data["red_hits"] = rh
        episode_data["blue_hits"] = bh

        # Death reasons
        for ev in info.get("death_events", []) or []:
            if isinstance(ev, dict):
                episode_data["death_reasons"][ev.get("agent_id", "")] = ev.get("death_reason", "")

        if all(term.values()) or all(trunc.values()):
            break

    red_alive = sum(1 for rid in red_ids if env.red_planes.get(rid) and env.red_planes[rid].is_alive)
    blue_alive = sum(1 for bid in env.blue_ids if env.blue_planes.get(bid) and env.blue_planes[bid].is_alive)
    timed_out = bool(all(trunc.values()) if step >= env.max_steps - 1 else False)
    if blue_alive == 0 and red_alive > 0:
        winner, end_reason = "red", "blue_eliminated"
    elif red_alive == 0 and blue_alive > 0:
        winner, end_reason = "blue", "red_eliminated"
    elif red_alive == 0 and blue_alive == 0:
        winner, end_reason = "draw", "mutual_elimination"
    elif timed_out:
        winner, end_reason = "draw", "timeout"
    else:
        winner, end_reason = "draw", "ongoing"

    return {
        "episode_length": step + 1,
        "winner": winner, "end_reason": end_reason,
        "red_alive_final": red_alive, "blue_alive_final": blue_alive,
        "mav_alive_final": int(bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)),
        "red_launches": episode_data["red_launches"], "red_hits": episode_data["red_hits"],
        "blue_launches": episode_data["blue_launches"], "blue_hits": episode_data["blue_hits"],
        "reward_total": episode_data["reward_total"],
        "episode_return": sum(episode_data["reward_total"].values()) / max(len(red_ids), 1),
        "episode_return_per_step": sum(episode_data["reward_total"].values()) / max(step + 1, 1) / max(len(red_ids), 1),
        "steps": episode_data["steps"],
        "death_reasons": episode_data["death_reasons"],
        "launch_gate_rows": episode_data["launch_gate_rows"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)
    meta_path = model_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    print(f"Checkpoint meta: {json.dumps({k: meta.get(k) for k in ('policy_arch','biased_mask','entity_dim','num_agents')}, default=str)}")

    device = torch.device(args.device)
    policy = _load_policy(model_path, meta, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_csv(out_dir / "launch_gate_diagnostics.csv", LAUNCH_GATE_DIAGNOSTICS_COLUMNS)
    gate_file = (out_dir / "launch_gate_diagnostics.csv").open("a", newline="", encoding="utf-8")
    gate_writer = csv.DictWriter(gate_file, fieldnames=LAUNCH_GATE_DIAGNOSTICS_COLUMNS)
    gate_writer.writeheader()

    episodes = []
    obs_adapter = HeteroObsAdapterV2(max_red=5, max_blue=4, role_dim=4)
    env = make_env(args.config, max_steps=args.max_steps)

    try:
        for ep_idx in range(args.episodes):
            seed = args.seed_start + ep_idx
            ep_uid = f"0:{ep_idx}"
            rnn_hidden = None
            if hasattr(policy, "init_hidden"):
                rnn_hidden = policy.init_hidden(3, device)  # 3 red agents for 3V2
            ep = _run_episode(env, policy, obs_adapter, device, seed, rnn_hidden)
            ep["seed"] = seed
            ep["episode_uid"] = ep_uid
            episodes.append(ep)
            print(f"Ep {ep_uid} seed={seed} len={ep['episode_length']} "
                  f"outcome={ep['winner']}/{ep['end_reason']} ret={ep['episode_return']:.2f} "
                  f"R_launch={ep['red_launches']} R_hit={ep['red_hits']} "
                  f"B_launch={ep['blue_launches']} B_hit={ep['blue_hits']} "
                  f"MAV_alive={ep['mav_alive_final']} R_alive={ep['red_alive_final']}")

            # Write gate rows
            for gr in ep["launch_gate_rows"]:
                gr["episode_id"] = ep_idx
                gr["episode_uid"] = ep_uid
                gr["env_idx"] = 0
                gate_writer.writerow(gr)
    finally:
        env.close()
        gate_file.close()

    # ---- Summaries ----
    ep_summary_rows = []
    death_rows = []
    for ep in episodes:
        ep_summary_rows.append({
            "episode_uid": ep["episode_uid"], "seed": ep["seed"],
            "length": ep["episode_length"], "winner": ep["winner"],
            "end_reason": ep["end_reason"],
            "red_alive_final": ep["red_alive_final"],
            "blue_alive_final": ep["blue_alive_final"],
            "mav_alive_final": ep["mav_alive_final"],
            "red_launch_count": ep["red_launches"], "red_hit_count": ep["red_hits"],
            "blue_launch_count": ep["blue_launches"], "blue_hit_count": ep["blue_hits"],
            "episode_return": ep["episode_return"],
            "episode_return_per_step": ep["episode_return_per_step"],
        })
        for aid, reason in ep["death_reasons"].items():
            death_rows.append({
                "episode_uid": ep["episode_uid"], "seed": ep["seed"],
                "agent_id": aid, "death_reason": reason,
            })

    # Write CSVs
    with (out_dir / "diagnostic_episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_summary_rows[0].keys()) if ep_summary_rows else [])
        w.writeheader(); w.writerows(ep_summary_rows)
    with (out_dir / "diagnostic_death_reasons.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode_uid", "seed", "agent_id", "death_reason"])
        w.writeheader(); w.writerows(death_rows)

    # ---- Gate funnel stats ----
    all_gate_rows = []
    for ep in episodes:
        all_gate_rows.extend(ep["launch_gate_rows"])
    gate_funnel: dict = {"block_counts": defaultdict(int), "total_steps": 0,
                          "alive_target_steps": 0, "track_pass_total": 0,
                          "range_pass_total": 0, "ata_pass_total": 0,
                          "ta_pass_total": 0, "geometry_pass_total": 0,
                          "deconfliction_pass_total": 0, "lock_mature_total": 0,
                          "launch_total": 0}
    for gr in all_gate_rows:
        reason = str(gr.get("primary_block_reason", "unknown"))
        gate_funnel["block_counts"][reason] += 1
        gate_funnel["total_steps"] += 1
        gate_funnel["alive_target_steps"] += 1 if int(gr.get("alive_target_count", 0)) > 0 else 0
        gate_funnel["track_pass_total"] += int(gr.get("track_pass_count", 0))
        gate_funnel["range_pass_total"] += int(gr.get("range_pass_count", 0))
        gate_funnel["ata_pass_total"] += int(gr.get("ata_pass_count", 0))
        gate_funnel["ta_pass_total"] += int(gr.get("ta_pass_count", 0))
        gate_funnel["geometry_pass_total"] += int(gr.get("geometry_pass_count", 0))
        gate_funnel["deconfliction_pass_total"] += int(gr.get("deconfliction_pass_count", 0))
        gate_funnel["lock_mature_total"] += int(gr.get("lock_mature", 0))
        gate_funnel["launch_total"] += int(gr.get("launch_executed", 0))

    # ---- Reward component summary ----
    reward_summary = defaultdict(float)
    for ep in episodes:
        for s in ep["steps"]:
            role = s.get("role", "")
            for k, v in s.items():
                if k in ("step", "agent_id", "role"):
                    continue
                reward_summary[f"{role}_{k}"] += _safe(v)
                reward_summary[f"all_{k}"] += _safe(v)

    # ---- Diagnostic report ----
    lines = [
        "# BRMA/TAM Scripted Composite v1 — Diagnostic Eval Report",
        "",
        f"- Checkpoint: `{args.model}`",
        f"- Episodes: {args.episodes}, seeds {args.seed_start}–{args.seed_start + args.episodes - 1}",
        f"- Config: {args.config}",
        f"- Opponent: brma_rule, deterministic policy",
        "",
        "## Episode Results",
        "",
        "| ep_uid | seed | len | outcome | R_alive | B_alive | MAV | R_launch | R_hit | B_launch | B_hit | ret | ret/step |",
        "|--------|------|-----|---------|---------|---------|-----|----------|-------|----------|-------|-----|----------|",
    ]
    for ep in episodes:
        lines.append(
            f"| {ep['episode_uid']} | {ep['seed']} | {ep['episode_length']} | "
            f"{ep['winner']}/{ep['end_reason']} | {ep['red_alive_final']} | {ep['blue_alive_final']} | "
            f"{ep['mav_alive_final']} | {ep['red_launches']} | {ep['red_hits']} | "
            f"{ep['blue_launches']} | {ep['blue_hits']} | {ep['episode_return']:.1f} | {ep['episode_return_per_step']:.3f} |"
        )

    lines += [
        "",
        "## Fire-Control Funnel",
        "",
        f"- Total agent-steps recorded: {gate_funnel['total_steps']}",
        f"- Steps with alive targets: {gate_funnel['alive_target_steps']}",
        f"- Track Pass: {gate_funnel['track_pass_total']}",
        f"- Range Pass: {gate_funnel['range_pass_total']}",
        f"- ATA Pass: {gate_funnel['ata_pass_total']}",
        f"- TA Pass: {gate_funnel['ta_pass_total']}",
        f"- Geometry Pass: {gate_funnel['geometry_pass_total']}",
        f"- Deconfliction Pass: {gate_funnel['deconfliction_pass_total']}",
        f"- Lock Mature: {gate_funnel['lock_mature_total']}",
        f"- Launches: {gate_funnel['launch_total']}",
        "",
        "### Primary Block Reason Distribution",
        "",
        "| Reason | Count | Pct |",
        "|--------|-------|-----|",
    ]
    total = max(gate_funnel["total_steps"], 1)
    for reason in ["inactive", "no_alive_target", "no_track", "out_of_range",
                    "ata_blocked", "ta_blocked", "deconfliction_blocked",
                    "no_ammo", "cooldown", "lock_not_mature", "launch", "unknown"]:
        c = gate_funnel["block_counts"].get(reason, 0)
        if c > 0:
            lines.append(f"| {reason} | {c} | {100.0 * c / total:.1f}% |")

    # Death reasons
    mav_deaths = [r for r in death_rows if "red_0" in str(r.get("agent_id", ""))]
    uav_deaths = [r for r in death_rows if "red_0" not in str(r.get("agent_id", ""))]
    lines += [
        "",
        "## Death Reasons",
        "",
        "### MAV deaths",
    ]
    for d in mav_deaths:
        lines.append(f"- {d['agent_id']}: {d['death_reason']}")
    lines.append("")
    lines.append("### UAV deaths")
    for d in uav_deaths:
        lines.append(f"- {d['agent_id']}: {d['death_reason']}")

    # Reward summary
    lines += [
        "",
        "## Reward Component Summary (sum over all agent-steps)",
    ]
    for k in sorted(reward_summary.keys()):
        lines.append(f"- {k}: {reward_summary[k]:.1f}")

    # Verdict
    red_launch_total = sum(ep["red_launches"] for ep in episodes)
    most_common_block = max(gate_funnel["block_counts"], key=gate_funnel["block_counts"].get) if gate_funnel["block_counts"] else "unknown"

    if not all_gate_rows:
        verdict = "IMPLEMENTATION_ERROR"
    elif red_launch_total > 0:
        verdict = "READY_FOR_50K"
    elif most_common_block in ("no_track", "out_of_range"):
        verdict = "NEEDS_REWARD_SCALE_REVIEW"
    elif most_common_block in ("ata_blocked", "ta_blocked", "deconfliction_blocked", "lock_not_mature"):
        verdict = "NEEDS_FIRE_CONTROL_ALIGNMENT_REVIEW"
    else:
        verdict = "NEEDS_TRAINING_STABILITY_REVIEW"

    lines += [
        "",
        f"## Verdict: **{verdict}**",
        f"",
        f"- Red launches: {red_launch_total}",
        f"- Most common block: **{most_common_block}** ({gate_funnel['block_counts'].get(most_common_block, 0)} steps)",
    ]

    summary = {
        "verdict": verdict,
        "episodes": len(episodes),
        "total_red_launches": red_launch_total,
        "gate_funnel": dict(gate_funnel),
        "most_common_block": most_common_block,
    }
    (out_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== Verdict: {verdict} ===")
    print(f"Red launches: {red_launch_total}")
    print(f"Most common block: {most_common_block}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
