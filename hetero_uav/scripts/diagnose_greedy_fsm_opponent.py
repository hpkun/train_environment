"""Diagnose rule_nearest versus greedy_fsm blue opponent behavior.

This script does not train and does not change the environment default
opponent. Red actions are zero; blue actions come from the selected scripted
policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env

DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
POLICIES = ["rule_nearest", "greedy_fsm"]


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        if not isinstance(agent_obs, dict):
            continue
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def _alive_counts(env) -> tuple[int, int, bool]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    mav_alive = False
    for aid in env.red_ids:
        if getattr(env, "agent_roles", {}).get(aid) == "mav":
            sim = env.red_planes.get(aid)
            mav_alive = bool(sim is not None and sim.is_alive)
            break
    return red_alive, blue_alive, mav_alive


def _terminated_or_truncated(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(
        bool(v) for v in truncated.values()
    )


def diagnose_one(config: str, policy_name: str, steps: int, seed: int) -> dict:
    env = make_env(config, env_type="jsbsim_hetero")
    policy = OpponentPolicy(mode=policy_name, seed=seed)
    actions_seen: list[np.ndarray] = []
    state_counts: Counter[str] = Counter()
    nan_detected = False
    steps_executed = 0

    try:
        obs, info = env.reset(seed=seed)
        nan_detected = nan_detected or _obs_has_nan(obs)
        for _ in range(steps):
            actions = {
                rid: np.zeros(3, dtype=np.float32)
                for rid in env.red_ids
            }
            blue_actions = policy.act(obs, env.blue_ids)
            for action in blue_actions.values():
                arr = np.asarray(action, dtype=np.float32)
                if np.isnan(arr).any():
                    nan_detected = True
                actions_seen.append(arr)
            if getattr(policy, "last_states", None):
                state_counts.update(policy.last_states.values())
            actions.update(blue_actions)

            obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            nan_detected = nan_detected or _obs_has_nan(obs)
            if _terminated_or_truncated(terminated, truncated):
                break

        red_alive, blue_alive, mav_alive = _alive_counts(env)
        if actions_seen:
            action_arr = np.stack(actions_seen).astype(np.float32)
            action_mean_abs = float(np.mean(np.abs(action_arr)))
            action_min = float(np.min(action_arr))
            action_max = float(np.max(action_arr))
        else:
            action_mean_abs = 0.0
            action_min = 0.0
            action_max = 0.0

        return {
            "config": config,
            "opponent_policy": policy_name,
            "steps_executed": steps_executed,
            "nan_detected": bool(nan_detected),
            "blue_action_mean_abs": action_mean_abs,
            "blue_action_min": action_min,
            "blue_action_max": action_max,
            "blue_state_counts": dict(sorted(state_counts.items())),
            "red_alive_final": red_alive,
            "blue_alive_final": blue_alive,
            "mav_alive_final": mav_alive,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/greedy_fsm_opponent_diagnostic.json",
    )
    args = parser.parse_args()

    records = []
    for config in args.configs:
        for policy_name in POLICIES:
            record = diagnose_one(
                config=config,
                policy_name=policy_name,
                steps=args.steps,
                seed=args.seed,
            )
            records.append(record)
            print(
                f"{Path(config).stem:38s} {policy_name:12s} "
                f"steps={record['steps_executed']:3d} "
                f"nan={record['nan_detected']} "
                f"action_abs={record['blue_action_mean_abs']:.3f} "
                f"states={record['blue_state_counts']}"
            )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "configs_checked": len(args.configs),
        "records": len(records),
        "nan_records": sum(1 for record in records if record["nan_detected"]),
    }
    out.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"output_json: {out}")
    if summary["nan_records"]:
        raise RuntimeError("NaN detected during greedy_fsm opponent diagnosis")


if __name__ == "__main__":
    main()
