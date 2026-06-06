"""Diagnose greedy_fsm blue opponent without training."""
from __future__ import annotations

import argparse
import json
import sys
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
CLOSE_RANGE_CONFIG = (
    "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml"
)


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _red_actions(env, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
    if mode == "bounded_random":
        return {
            rid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
            for rid in env.red_ids
        }
    raise ValueError(mode)


def _validate_blue_actions(actions: dict[str, np.ndarray], blue_ids: list[str]) -> None:
    for bid in blue_ids:
        action = np.asarray(actions.get(bid), dtype=np.float32)
        if action.shape != (3,):
            raise RuntimeError(f"{bid}: greedy_fsm action shape is {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError(f"{bid}: greedy_fsm action contains NaN/Inf")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise RuntimeError(f"{bid}: greedy_fsm action out of [-1, 1]")


def diagnose_config(config: str, red_policy: str, steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Keep this diagnostic script's stdout usable on Windows. The JSBSim
    # low-level output suppressor can corrupt the process stdout file
    # descriptor in repeated env construction/close cycles; disabling it here
    # affects only diagnostic logging, not the environment mechanics.
    env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    policy = OpponentPolicy("greedy_fsm", seed=seed + 17)
    actions_seen: list[np.ndarray] = []
    defensive_triggered = False
    target_assignment_used = False
    state_counts: dict[str, int] = {}
    assigned_target_counts: dict[str, int] = {}
    nan_detected = False
    steps_executed = 0

    try:
        obs, info = env.reset(seed=seed)
        nan_detected = nan_detected or _contains_nan(obs)
        for _step in range(steps):
            blue_actions = policy.act(obs, env.blue_ids, env=env)
            _validate_blue_actions(blue_actions, env.blue_ids)
            actions_seen.extend(np.asarray(a, dtype=np.float32) for a in blue_actions.values())
            defensive_triggered = defensive_triggered or any(
                state == "evade" for state in policy.last_states.values()
            )
            assigned = getattr(policy, "last_assigned_targets", {})
            target_assignment_used = target_assignment_used or len(set(assigned.values())) > 1
            for slot in assigned.values():
                key = str(slot)
                assigned_target_counts[key] = assigned_target_counts.get(key, 0) + 1
            for state in policy.last_states.values():
                state_counts[state] = state_counts.get(state, 0) + 1

            actions = _red_actions(env, red_policy, rng)
            actions.update(blue_actions)
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            nan_detected = (
                nan_detected
                or _contains_nan(obs)
                or _contains_nan(rewards)
            )
            if all(terminated.values()) or all(truncated.values()):
                break

        if actions_seen:
            action_arr = np.stack(actions_seen).astype(np.float32)
            action_min = float(np.min(action_arr))
            action_max = float(np.max(action_arr))
            action_mean = float(np.mean(action_arr))
        else:
            action_min = action_max = action_mean = 0.0

        return {
            "config": config,
            "red_policy": red_policy,
            "red_count": len(env.red_ids),
            "blue_count": len(env.blue_ids),
            "steps_executed": steps_executed,
            "blue_action_min": action_min,
            "blue_action_max": action_max,
            "blue_action_mean": action_mean,
            "defensive_action_triggered": bool(defensive_triggered),
            "target_assignment_used": bool(target_assignment_used),
            "assigned_target_counts": assigned_target_counts,
            "used_env_refresh_engaged_targets": bool(
                getattr(policy, "used_env_refresh_engaged_targets", False)
            ),
            "used_env_own_kinematics": bool(
                getattr(policy, "used_env_own_kinematics", False)
            ),
            "used_env_own_positions": bool(
                getattr(policy, "used_env_own_positions", False)
            ),
            "state_counts": state_counts,
            "nan_detected": bool(nan_detected),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--include-close-range", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="outputs/blue_greedy_fsm_diagnostic/summary.json",
    )
    args = parser.parse_args()

    configs = list(args.configs) if args.configs else list(DEFAULT_CONFIGS)
    if args.include_close_range:
        configs.append(CLOSE_RANGE_CONFIG)

    records = []
    warnings = []
    for config in configs:
        for red_policy in ("zero", "bounded_random"):
            record = diagnose_config(config, red_policy, args.steps, args.seed)
            records.append(record)
            is_close = Path(config).name == Path(CLOSE_RANGE_CONFIG).name
            has_attack = any(
                state in record["state_counts"]
                for state in ("attack_nearest", "attack_mav_priority")
            )
            if is_close and not has_attack:
                warnings.append(
                    f"{Path(config).name} red={red_policy}: close-range did not trigger attack state"
                )
            print(
                f"{Path(config).stem:38s} red={red_policy:14s} "
                f"steps={record['steps_executed']:3d} "
                f"action_min={record['blue_action_min']:.3f} "
                f"action_max={record['blue_action_max']:.3f} "
                f"action_mean={record['blue_action_mean']:.3f} "
                f"defensive={record['defensive_action_triggered']} "
                f"assignment={record['target_assignment_used']} "
                f"states={record['state_counts']} "
                f"nan={record['nan_detected']}",
                flush=True,
            )

    summary = {
        "records": len(records),
        "nan_records": sum(1 for r in records if r["nan_detected"]),
        "any_defensive_action_triggered": any(
            r["defensive_action_triggered"] for r in records
        ),
        "any_target_assignment_used": any(
            r["target_assignment_used"] for r in records
        ),
        "warnings": warnings,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"output_json: {out}", flush=True)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    if summary["nan_records"]:
        raise RuntimeError("NaN detected in greedy_fsm diagnostic")


if __name__ == "__main__":
    main()
