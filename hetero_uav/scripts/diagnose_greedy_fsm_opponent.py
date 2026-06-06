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

from algorithms.mappo.opponent_policy import OpponentPolicy, _wrap_heading_norm
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


def _min_red_blue_distance(env) -> float:
    distances = []
    for red in env.red_planes.values():
        if red is None or not red.is_alive:
            continue
        red_pos = red.get_position()
        for blue in env.blue_planes.values():
            if blue is None or not blue.is_alive:
                continue
            distances.append(float(np.linalg.norm(red_pos - blue.get_position())))
    return float(np.min(distances)) if distances else float("inf")


def _terminated_or_truncated(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(
        bool(v) for v in truncated.values()
    )


def diagnose_one(config: str, policy_name: str, steps: int, seed: int) -> dict:
    env = make_env(config, env_type="jsbsim_hetero")
    policy = OpponentPolicy(mode=policy_name, seed=seed)
    actions_seen: list[np.ndarray] = []
    min_distance_series: list[float] = []
    turn_back_heading_deltas: list[float] = []
    turn_back_heading_values: list[float] = []
    state_counts: Counter[str] = Counter()
    nan_detected = False
    steps_executed = 0

    try:
        obs, info = env.reset(seed=seed)
        min_distance_series.append(_min_red_blue_distance(env))
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
                for bid, state in policy.last_states.items():
                    if state != "turn_back" or bid not in blue_actions:
                        continue
                    action = np.asarray(blue_actions[bid], dtype=np.float32)
                    current_heading = policy._get_current_heading_norm(
                        obs.get(bid, {})
                    )
                    heading_delta = _wrap_heading_norm(
                        float(action[1]) - current_heading
                    )
                    turn_back_heading_deltas.append(abs(float(heading_delta)))
                    turn_back_heading_values.append(float(action[1]))
            actions.update(blue_actions)

            obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            min_distance_series.append(_min_red_blue_distance(env))
            nan_detected = nan_detected or _obs_has_nan(obs)
            if _terminated_or_truncated(terminated, truncated):
                break

        red_alive, blue_alive, mav_alive = _alive_counts(env)
        if actions_seen:
            action_arr = np.stack(actions_seen).astype(np.float32)
            action_mean_abs = float(np.mean(np.abs(action_arr)))
            action_mean = float(np.mean(action_arr))
            action_std = float(np.std(action_arr))
            action_min = float(np.min(action_arr))
            action_max = float(np.max(action_arr))
            saturation_rate = float(np.mean(np.any(np.abs(action_arr) > 0.95, axis=1)))
        else:
            action_mean_abs = 0.0
            action_mean = 0.0
            action_std = 0.0
            action_min = 0.0
            action_max = 0.0
            saturation_rate = 0.0

        state_transition_count = int(sum(state_counts.values()))
        if state_counts:
            dominant_state, dominant_count = state_counts.most_common(1)[0]
            dominant_state_ratio = float(dominant_count / max(1, state_transition_count))
        else:
            dominant_state = ""
            dominant_state_ratio = 0.0
        closest_distance = (
            float(np.min(min_distance_series)) if min_distance_series else float("inf")
        )
        final_distance = (
            float(min_distance_series[-1]) if min_distance_series else float("inf")
        )
        post_pass_separation = final_distance
        turn_back_count = int(state_counts.get("turn_back", 0))
        turn_back_heading_delta_mean_abs = (
            float(np.mean(turn_back_heading_deltas))
            if turn_back_heading_deltas else 0.0
        )
        warnings = []
        if turn_back_count > 0 and post_pass_separation > 10000.0:
            warnings.append(
                "turn_back triggered but did not reduce post-pass separation"
            )

        return {
            "config": config,
            "opponent_policy": policy_name,
            "steps_executed": steps_executed,
            "nan_detected": bool(nan_detected),
            "blue_action_mean": action_mean,
            "blue_action_mean_abs": action_mean_abs,
            "blue_action_std": action_std,
            "blue_action_min": action_min,
            "blue_action_max": action_max,
            "blue_action_saturation_rate": saturation_rate,
            "blue_state_counts": dict(sorted(state_counts.items())),
            "state_transition_count": state_transition_count,
            "dominant_state": dominant_state,
            "dominant_state_ratio": dominant_state_ratio,
            "heading_wrap_used": True,
            "turn_back_count": turn_back_count,
            "turn_back_heading_delta_mean_abs": turn_back_heading_delta_mean_abs,
            "turn_back_heading_values_sample": turn_back_heading_values[:10],
            "closest_distance_m": closest_distance,
            "final_distance_m": final_distance,
            "post_pass_separation_m": post_pass_separation,
            "warnings": warnings,
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
                f"sat={record['blue_action_saturation_rate']:.3f} "
                f"dominant={record['dominant_state'] or 'none'} "
                f"dominant_ratio={record['dominant_state_ratio']:.3f} "
                f"turn_back={record['turn_back_count']} "
                f"post_pass={record['post_pass_separation_m']:.1f} "
                f"states={record['blue_state_counts']}"
            )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    greedy_records = [
        record for record in records
        if record["opponent_policy"] == "greedy_fsm"
    ]
    greedy_state_coverage = sorted({
        state
        for record in greedy_records
        for state in record["blue_state_counts"].keys()
    })
    saturation_values = [
        float(record["blue_action_saturation_rate"])
        for record in greedy_records
    ]
    summary = {
        "configs_checked": len(args.configs),
        "policies_checked": list(POLICIES),
        "records": len(records),
        "nan_records": sum(1 for record in records if record["nan_detected"]),
        "greedy_fsm_state_coverage": greedy_state_coverage,
        "greedy_fsm_has_non_patrol_state": any(
            state != "patrol" for state in greedy_state_coverage),
        "greedy_fsm_acquired_target": any(
            state in {"attack_nearest", "attack_mav_priority"}
            for state in greedy_state_coverage),
        "greedy_fsm_action_saturation_mean": float(
            np.mean(saturation_values)) if saturation_values else 0.0,
        "heading_wrap_used": True,
    }
    out.write_text(
        json.dumps({"records": records, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"output_json: {out}")
    if greedy_state_coverage == ["patrol"]:
        print("warning: greedy_fsm remained in patrol for all diagnosed steps")
    if greedy_state_coverage == ["search_acquire"]:
        print("warning: greedy_fsm only searched and never acquired target")
    if summary["greedy_fsm_acquired_target"]:
        print("greedy_fsm acquired target")
    if summary["nan_records"]:
        raise RuntimeError("NaN detected during greedy_fsm opponent diagnosis")


if __name__ == "__main__":
    main()
