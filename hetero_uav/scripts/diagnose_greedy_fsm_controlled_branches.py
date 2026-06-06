"""Controlled branch diagnostics for the greedy_fsm opponent.

This script builds artificial observations and verifies each FSM branch without
waiting for the live environment geometry to trigger it naturally.
"""
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


def _obs_cases() -> list[dict]:
    return [
        {
            "case": "search_acquire_case",
            "expected_state": "search_acquire",
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
            },
            "checks": {
                "min_speed": 0.8,
                "max_abs_heading": 0.2,
            },
        },
        {
            "case": "nearest_attack_case",
            "expected_state": "attack_nearest",
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "enemy_states": np.array(
                    [[0.3, 0.2, 0.1], [0.8, -0.1, 0.0]],
                    dtype=np.float32,
                ),
                "enemy_observed_mask": np.array([1.0, 1.0], dtype=np.float32),
            },
        },
        {
            "case": "mav_priority_case",
            "expected_state": "attack_mav_priority",
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "enemy_states": np.array(
                    [[0.2, -0.2, 0.0], [0.5, 0.3, 0.1]],
                    dtype=np.float32,
                ),
                "enemy_observed_mask": np.array([1.0, 1.0], dtype=np.float32),
                "enemy_roles": np.array(
                    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            },
        },
        {
            "case": "evade_case",
            "expected_state": "evade",
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([1.0], dtype=np.float32),
                "enemy_states": np.array([[0.4, -0.3, 0.0]], dtype=np.float32),
                "enemy_observed_mask": np.array([1.0], dtype=np.float32),
            },
        },
        {
            "case": "recover_altitude_case",
            "expected_state": "recover_altitude",
            "obs": {
                "altitude": np.array([0.1], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
            },
        },
        {
            "case": "heading_wrap_positive_case",
            "expected_state": "turn_back",
            "agent_index": 0,
            "preload_lost_target": True,
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "ego_geo_state": np.array([0, 0, 1, 0, 0, 0.9], dtype=np.float32),
                "enemy_observed_mask": np.zeros(1, dtype=np.float32),
            },
            "checks": {
                "heading_less_than": 0.0,
                "heading_not_equal": 1.0,
            },
        },
        {
            "case": "heading_wrap_negative_case",
            "expected_state": "turn_back",
            "agent_index": 1,
            "preload_lost_target": True,
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "ego_geo_state": np.array([0, 0, 1, 0, 0, -0.9], dtype=np.float32),
                "enemy_observed_mask": np.zeros(1, dtype=np.float32),
            },
            "checks": {
                "heading_greater_than": 0.0,
                "heading_not_equal": -1.0,
            },
        },
        {
            "case": "search_acquire_wrap_case",
            "expected_state": "search_acquire",
            "agent_index": 0,
            "obs": {
                "altitude": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "ego_geo_state": np.array([0, 0, 1, 0, 0, 0.99], dtype=np.float32),
            },
            "checks": {
                "heading_less_than": 0.0,
                "heading_not_equal": 1.0,
            },
        },
    ]


def run_case(case: dict) -> dict:
    policy = OpponentPolicy("greedy_fsm", seed=0)
    agent_index = int(case.get("agent_index", 0))
    if case.get("preload_lost_target"):
        policy.last_targets[agent_index] = 0
        policy.lost_target_steps[agent_index] = 1
    action, actual_state = policy._greedy_fsm_action(
        case["obs"], agent_index=agent_index
    )
    action = np.asarray(action, dtype=np.float32)
    nan_detected = bool(np.isnan(action).any())
    action_in_bounds = bool(
        action.shape == (3,)
        and action.dtype == np.float32
        and np.all(action >= -1.0)
        and np.all(action <= 1.0)
    )
    passed = (
        actual_state == case["expected_state"]
        and action_in_bounds
        and not nan_detected
    )
    checks = case.get("checks", {})
    if "min_speed" in checks:
        passed = passed and bool(action[2] > float(checks["min_speed"]))
    if "max_abs_heading" in checks:
        passed = passed and bool(abs(float(action[1])) <= float(checks["max_abs_heading"]))
    if "heading_less_than" in checks:
        passed = passed and bool(float(action[1]) < float(checks["heading_less_than"]))
    if "heading_greater_than" in checks:
        passed = passed and bool(float(action[1]) > float(checks["heading_greater_than"]))
    if "heading_not_equal" in checks:
        passed = passed and bool(
            not np.isclose(float(action[1]), float(checks["heading_not_equal"]))
        )
    return {
        "case": case["case"],
        "expected_state": case["expected_state"],
        "actual_state": actual_state,
        "action": action.astype(float).tolist(),
        "action_in_bounds": action_in_bounds,
        "nan_detected": nan_detected,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/greedy_fsm_controlled_branches.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in _obs_cases()]
    summary = {
        "cases_checked": len(records),
        "passed_cases": sum(1 for record in records if record["passed"]),
        "all_passed": all(record["passed"] for record in records),
    }

    for record in records:
        print(
            f"{record['case']:28s} expected={record['expected_state']:20s} "
            f"actual={record['actual_state']:20s} passed={record['passed']}"
        )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"output_json: {out}")
    if not summary["all_passed"]:
        raise RuntimeError("greedy_fsm controlled branch diagnostics failed")


if __name__ == "__main__":
    main()
