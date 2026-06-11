"""Audit MAV survival ablations for HAPPO reference checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from happo_mav_audit_common import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_F16_SURROGATE_CONFIG,
    DEFAULT_HAPPO_CONFIG,
    SAFE_MAV_ACTION,
    aggregate_records,
    checkpoint_path,
    load_policy,
    make_hetero_env,
    rel,
    role_ids,
    sim_metrics,
    summarize_episode,
    team_done,
    update_missile_stats,
    write_json,
    write_md,
)


CASE_CONFIG = {
    "learned_all": {"config": DEFAULT_HAPPO_CONFIG, "mav_mode": "learned", "scale": 1.0},
    "mav_zero_action": {"config": DEFAULT_HAPPO_CONFIG, "mav_mode": "fixed", "action": SAFE_MAV_ACTION},
    "mav_safe_loiter_candidates": {"config": DEFAULT_HAPPO_CONFIG, "mav_mode": "fixed", "action": SAFE_MAV_ACTION},
    "mav_action_scale_0_3": {"config": DEFAULT_HAPPO_CONFIG, "mav_mode": "scaled", "scale": 0.3},
    "mav_action_scale_0_1": {"config": DEFAULT_HAPPO_CONFIG, "mav_mode": "scaled", "scale": 0.1},
    "f16_mav_surrogate_learned_all": {"config": DEFAULT_F16_SURROGATE_CONFIG, "mav_mode": "learned", "scale": 1.0},
    "f16_mav_surrogate_safe_mav": {"config": DEFAULT_F16_SURROGATE_CONFIG, "mav_mode": "fixed", "action": SAFE_MAV_ACTION},
}


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


def _run_episode(policy, device, adapter, checkpoint: str, case_name: str, spec: dict, args, seed: int) -> dict:
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from algorithms.mappo.opponent_policy import OpponentPolicy

    env = make_hetero_env(spec["config"], max_steps_override=args.max_steps_override)
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=seed + 99)
    try:
        obs, info = env.reset(seed=seed)
        missile_stats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
        prev_hits = {"red": 0, "blue": 0}
        red0_series, death_events, red0_actions = [], [], []
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        steps = 0
        while True:
            red0_series.append(sim_metrics(env.red_planes.get("red_0")))
            actions_np = _policy_actions(policy, device, adapter, obs, info, env, not args.stochastic)
            if spec["mav_mode"] == "fixed":
                actions_np[0] = np.asarray(spec["action"], dtype=np.float32)
            elif spec["mav_mode"] == "scaled":
                actions_np[0] = actions_np[0] * float(spec["scale"])
            actions = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            actions.update(opponent.act(obs, env.blue_ids, env=env))
            red0_actions.append(actions_np[0].astype(np.float32))
            obs, rewards, terminated, truncated, info = env.step(actions)
            death_events.extend(info.get("death_events", []))
            update_missile_stats(missile_stats, info, env, prev_hits)
            steps += 1
            if team_done(terminated, truncated):
                break
        red0_series.append(sim_metrics(env.red_planes.get("red_0")))
        out = summarize_episode(env, steps, truncated, missile_stats, red0_series, death_events, red0_actions)
        out.update({"checkpoint": checkpoint, "case": case_name, "config": spec["config"]})
        return out
    finally:
        env.close()


def _run_case(model: Path, checkpoint: str, case_name: str, spec: dict, args) -> dict:
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    policy, device = load_policy(model, args.device)
    adapter = HeteroObsAdapterV2()
    records = [
        _run_episode(policy, device, adapter, checkpoint, case_name, spec, args, seed=2000 + ep)
        for ep in range(args.episodes)
    ]
    summary = aggregate_records(records)
    return {
        "checkpoint": checkpoint,
        "case": case_name,
        "config": spec["config"],
        **summary,
        "episodes_detail": records,
    }


def build_audit(args) -> dict:
    exp_dir = rel(args.experiment_dir)
    try:
        models = [(name, checkpoint_path(exp_dir, name)) for name in args.checkpoints]
    except FileNotFoundError as exc:
        raise exc
    records = []
    for checkpoint, model in models:
        for case in args.cases:
            records.append(_run_case(model, checkpoint, case, CASE_CONFIG[case], args))
            print(f"{checkpoint} {case}: mav_survival={records[-1]['mav_survival_rate']:.3f}", flush=True)
    best_realistic = max(
        (r["mav_survival_rate"] for r in records if not r["case"].startswith("f16_mav_surrogate")),
        default=0.0,
    )
    f16_best = max((r["mav_survival_rate"] for r in records if r["case"].startswith("f16_mav_surrogate")), default=0.0)
    safe_best = max((r["mav_survival_rate"] for r in records if "safe" in r["case"] or "zero_action" in r["case"]), default=0.0)
    return {
        "records": records,
        "summary": {
            "best_realistic_mav_survival_rate": float(best_realistic),
            "safe_mav_can_survive": bool(safe_best > 0.5),
            "f16_surrogate_improves": bool(f16_best > best_realistic),
            "f16_surrogate_best_mav_survival_rate": float(f16_best),
        },
        "conclusion": {
            "safe_fixed_action_improves": bool(safe_best > best_realistic),
            "action_scale_improves": bool(max((r["mav_survival_rate"] for r in records if "scale" in r["case"]), default=0.0) > best_realistic),
            "f16_surrogate_improves": bool(f16_best > best_realistic),
        },
    }


def write_report(data: dict, output_md: str) -> None:
    lines = ["# HAPPO MAV Survival Ablation", ""]
    for record in data["records"]:
        lines.append(
            f"- {record['checkpoint']} {record['case']}: mav_survival={record['mav_survival_rate']}, "
            f"mav_first_death={record['mav_first_death_rate']}, blue_dead={record['blue_dead_mean']}, "
            f"red_hits={record['red_missile_hits_mean']}"
        )
    lines.extend(["", "## Summary", f"- {data['summary']}", "", "## Conclusion", f"- {data['conclusion']}"])
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAV survival ablation diagnostics")
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--checkpoints", nargs="+", choices=["best", "latest"], default=["best", "latest"])
    parser.add_argument("--cases", nargs="+", choices=list(CASE_CONFIG), default=list(CASE_CONFIG))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--output-json", default="outputs/happo_3v2_reference_200k/mav_survival_ablation/happo_mav_survival_ablation.json")
    parser.add_argument("--output-md", default="outputs/happo_3v2_reference_200k/mav_survival_ablation/happo_mav_survival_ablation.md")
    args = parser.parse_args()
    try:
        data = build_audit(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out_json = write_json(args.output_json, data)
    write_report(data, args.output_md)
    print(f"output_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
