"""Audit F-22/F-16 MAV control stability under fixed actions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from happo_mav_audit_common import (
    DEFAULT_F16_SURROGATE_CONFIG,
    DEFAULT_HAPPO_CONFIG,
    aggregate_records,
    make_hetero_env,
    rel,
    sim_metrics,
    summarize_episode,
    team_done,
    update_missile_stats,
    write_json,
    write_md,
)


ACTION_CASES = {
    "zero": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    "speed_0_3": np.asarray([0.0, 0.0, 0.3], dtype=np.float32),
    "mild_descend": np.asarray([-0.1, 0.0, 0.3], dtype=np.float32),
    "mild_climb": np.asarray([0.1, 0.0, 0.3], dtype=np.float32),
    "turn_left": np.asarray([0.0, -0.2, 0.3], dtype=np.float32),
    "turn_right": np.asarray([0.0, 0.2, 0.3], dtype=np.float32),
    "slow": np.asarray([0.0, 0.0, -0.2], dtype=np.float32),
}


def _blue_actions(policy: str, obs: dict, env):
    if policy == "zero":
        return {bid: np.zeros(3, dtype=np.float32) for bid in env.blue_ids}
    from algorithms.mappo.opponent_policy import OpponentPolicy

    return OpponentPolicy(mode="brma_rule", seed=17).act(obs, env.blue_ids, env=env)


def _run_episode(config: str, action: np.ndarray, blue_policy: str, steps: int, seed: int) -> dict:
    env = make_hetero_env(config, max_steps_override=steps)
    try:
        obs, info = env.reset(seed=seed)
        missile_stats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
        prev_hits = {"red": 0, "blue": 0}
        red0_series, death_events, red0_actions = [], [], []
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        step = 0
        while step < steps:
            red0_series.append(sim_metrics(env.red_planes.get("red_0")))
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            actions["red_0"] = action.astype(np.float32)
            actions.update(_blue_actions(blue_policy, obs, env))
            red0_actions.append(action.astype(np.float32))
            obs, rewards, terminated, truncated, info = env.step(actions)
            death_events.extend(info.get("death_events", []))
            update_missile_stats(missile_stats, info, env, prev_hits)
            step += 1
            if team_done(terminated, truncated):
                break
        red0_series.append(sim_metrics(env.red_planes.get("red_0")))
        return summarize_episode(env, step, truncated, missile_stats, red0_series, death_events, red0_actions)
    finally:
        env.close()


def _case_records(config_name: str, config: str, action_name: str, action: np.ndarray,
                  blue_policy: str, episodes: int, steps: int) -> dict:
    records = [
        _run_episode(config, action, blue_policy, steps, seed=1000 + ep)
        for ep in range(episodes)
    ]
    summary = aggregate_records(records)
    stable = (
        summary["mav_death_rate"] == 0.0
        and summary["red0_max_abs_roll_deg"] < 120.0
        and (summary["red0_min_altitude"] is None or summary["red0_min_altitude"] > 1000.0)
    )
    return {
        "config_name": config_name,
        "config": config,
        "blue_policy": blue_policy,
        "action_name": action_name,
        "action": [float(x) for x in action],
        "stable": bool(stable),
        **summary,
        "episodes_detail": records,
    }


def build_audit(args) -> dict:
    configs = {
        "f22_mav": args.f22_config,
        "f16_mav_surrogate": args.f16_surrogate_config,
    }
    records = []
    for config_name, config in configs.items():
        if not rel(config).exists():
            records.append({
                "config_name": config_name,
                "config": config,
                "error": f"config not found: {rel(config)}",
                "stable": False,
            })
            continue
        for action_name, action in ACTION_CASES.items():
            for blue_policy in ("zero", "brma_rule"):
                records.append(_case_records(
                    config_name, config, action_name, action,
                    blue_policy, args.episodes, args.steps,
                ))
    f22_stable = any(r.get("config_name") == "f22_mav" and r.get("stable") for r in records)
    f16_stable = any(r.get("config_name") == "f16_mav_surrogate" and r.get("stable") for r in records)
    f22_death = np.mean([r.get("mav_death_rate", 1.0) for r in records if r.get("config_name") == "f22_mav"])
    f16_death = np.mean([r.get("mav_death_rate", 1.0) for r in records if r.get("config_name") == "f16_mav_surrogate"])
    return {
        "cases": records,
        "summary": {
            "f22_stable": bool(f22_stable),
            "f16_surrogate_stable": bool(f16_stable),
            "f16_surrogate_more_stable": bool(f16_death < f22_death),
            "f22_mean_death_rate": float(f22_death) if records else None,
            "f16_surrogate_mean_death_rate": float(f16_death) if records else None,
        },
        "recommendations": {
            "use_f16_surrogate_for_algorithm_validation": bool(f16_death < f22_death),
            "needs_f22_control_sign_pid_sweep": bool(not f22_stable),
        },
    }


def write_report(data: dict, output_md: str) -> None:
    lines = ["# MAV Control Stability Sweep", ""]
    for record in data["cases"]:
        if "error" in record:
            lines.append(f"- {record['config_name']}: {record['error']}")
            continue
        lines.append(
            f"- {record['config_name']} {record['blue_policy']} {record['action_name']}: "
            f"death_rate={record['mav_death_rate']}, max_roll={record['red0_max_abs_roll_deg']:.1f}, "
            f"min_alt={record['red0_min_altitude']}, stable={record['stable']}"
        )
    lines.extend(["", "## Summary", f"- {data['summary']}"])
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MAV control stability sweep diagnostics")
    parser.add_argument("--f22-config", default=DEFAULT_HAPPO_CONFIG)
    parser.add_argument("--f16-surrogate-config", default=DEFAULT_F16_SURROGATE_CONFIG)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-json", default="outputs/environment_audit/mav_control_stability_sweep.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/mav_control_stability_sweep.md")
    args = parser.parse_args()
    data = build_audit(args)
    out_json = write_json(args.output_json, data)
    write_report(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"f22_stable: {data['summary']['f22_stable']}")
    print(f"f16_surrogate_stable: {data['summary']['f16_surrogate_stable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
