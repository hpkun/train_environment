"""Audit whether blue rule opponent preferentially targets red_0 MAV."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from happo_mav_audit_common import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_HAPPO_CONFIG,
    checkpoint_path,
    load_policy,
    make_hetero_env,
    rel,
    role_ids,
    team_done,
    write_json,
    write_md,
)


def _policy_actions(policy, device, adapter, obs, info, env, deterministic: bool):
    import torch

    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
        for rid in env.red_ids
    ])
    with torch.no_grad():
        out = policy.act(
            torch.as_tensor(actor_obs, device=device),
            roles=role_ids(env),
            critic_state=torch.as_tensor(adapted["critic_state"], device=device),
            deterministic=deterministic,
        )
    return out["action"].detach().cpu().numpy()


def _run_episode(policy, device, adapter, args, seed: int) -> dict:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from algorithms.mappo.opponent_policy import OpponentPolicy

    env = make_hetero_env(args.config, max_steps_override=args.max_steps_override)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=seed + 81)
    blue_launch_targets = []
    blue_hit_targets = []
    first_missile_target = None
    death_events = []
    try:
        obs, info = env.reset(seed=seed)
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        while True:
            acts = _policy_actions(policy, device, adapter, obs, info, env, not args.stochastic)
            actions = {rid: acts[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            actions.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            for record in info.get("__launch_quality_step__", []):
                if str(record.get("shooter_id", "")).startswith("blue_"):
                    target = record.get("target_id")
                    blue_launch_targets.append(target)
                    if first_missile_target is None:
                        first_missile_target = target
            for record in info.get("__launch_quality_done__", []):
                if str(record.get("shooter_id", "")).startswith("blue_") and record.get("termination_reason") == "hit":
                    blue_hit_targets.append(record.get("target_id"))
            death_events.extend(info.get("death_events", []))
            if team_done(terminated, truncated):
                break
        return {
            "blue_launch_targets": blue_launch_targets,
            "blue_hit_targets": blue_hit_targets,
            "first_missile_target": first_missile_target,
            "death_events": death_events,
        }
    finally:
        env.close()


def build_audit(args) -> dict:
    model = rel(args.model) if args.model else checkpoint_path(rel(args.experiment_dir), args.checkpoint)
    if not model.exists():
        raise FileNotFoundError(f"checkpoint not found: {model}")
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    policy, device = load_policy(model, args.device)
    adapter = HeteroObsAdapterV2()
    episodes = [_run_episode(policy, device, adapter, args, seed=3000 + ep) for ep in range(args.episodes)]
    launch_counts = Counter()
    hit_counts = Counter()
    first_counts = Counter()
    for ep in episodes:
        launch_counts.update(t for t in ep["blue_launch_targets"] if t)
        hit_counts.update(t for t in ep["blue_hit_targets"] if t)
        if ep["first_missile_target"]:
            first_counts[ep["first_missile_target"]] += 1
    total_targets = sum(launch_counts.values())
    total_first = sum(first_counts.values())
    total_hits = sum(hit_counts.values())
    mav_launches = launch_counts.get("red_0", 0)
    uav_launches = sum(v for k, v in launch_counts.items() if str(k).startswith("red_") and k != "red_0")
    data = {
        "model": str(model),
        "config": args.config,
        "episodes": args.episodes,
        "blue_missile_launches_by_target": dict(launch_counts),
        "blue_hits_by_target": dict(hit_counts),
        "first_missile_target_counts": dict(first_counts),
        "blue_lock_target_counts": {},
        "blue_selected_target_counts": {},
        "mav_target_fraction": mav_launches / max(total_targets, 1),
        "mav_first_target_fraction": first_counts.get("red_0", 0) / max(total_first, 1),
        "mav_missile_target_fraction": mav_launches / max(total_targets, 1),
        "uav_target_fraction": uav_launches / max(total_targets, 1),
        "hits_on_red_0_mav": hit_counts.get("red_0", 0),
        "hits_on_red_uavs": sum(v for k, v in hit_counts.items() if str(k).startswith("red_") and k != "red_0"),
        "unavailable_fields": ["blue_lock_target_counts", "blue_selected_target_counts"],
    }
    if total_targets == 0:
        data["conclusion"] = "blue missile target preference unavailable because no blue launches were recorded"
    elif data["mav_missile_target_fraction"] > 0.5:
        data["conclusion"] = "blue missile launches preferentially target the MAV"
    else:
        data["conclusion"] = "blue missile launches do not show MAV preference in this audit"
    return data


def write_report(data: dict, output_md: str) -> None:
    lines = [
        "# Blue Target Preference Against MAV",
        "",
        f"- mav_target_fraction: {data['mav_target_fraction']}",
        f"- mav_first_target_fraction: {data['mav_first_target_fraction']}",
        f"- mav_missile_target_fraction: {data['mav_missile_target_fraction']}",
        f"- uav_target_fraction: {data['uav_target_fraction']}",
        f"- unavailable_fields: {data['unavailable_fields']}",
        f"- conclusion: {data['conclusion']}",
    ]
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit blue target preference against MAV")
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=DEFAULT_HAPPO_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--output-json", default="outputs/happo_3v2_reference_200k/blue_target_audit/blue_target_preference_against_mav.json")
    parser.add_argument("--output-md", default="outputs/happo_3v2_reference_200k/blue_target_audit/blue_target_preference_against_mav.md")
    args = parser.parse_args()
    try:
        data = build_audit(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_json = write_json(args.output_json, data)
    write_report(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"mav_missile_target_fraction: {data['mav_missile_target_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
