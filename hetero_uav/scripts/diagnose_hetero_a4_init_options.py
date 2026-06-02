from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


FT_PER_M = 1.0 / 0.3048


SCENARIOS = [
    {"name": "default"},
    {"name": "red0_altitude_plus_1000m", "red0_altitude_delta_m": 1000.0},
    {"name": "red0_altitude_plus_2000m", "red0_altitude_delta_m": 2000.0},
    {"name": "red0_speed_plus_30mps", "red0_speed_delta_mps": 30.0},
    {"name": "red0_speed_plus_50mps", "red0_speed_delta_mps": 50.0},
    {"name": "red0_action_bound_0.5", "red0_action_bound": 0.5},
    {"name": "red0_action_bound_0.3", "red0_action_bound": 0.3},
]


class DiagnosticHeteroEnv(HeteroUavCombatEnv):
    def __init__(self, *args, red0_altitude_delta_m: float = 0.0,
                 red0_speed_delta_mps: float = 0.0, **kwargs):
        self._diag_red0_altitude_delta_m = red0_altitude_delta_m
        self._diag_red0_speed_delta_mps = red0_speed_delta_mps
        super().__init__(*args, **kwargs)

    def _make_init_state(self, color: str, index: int) -> dict:
        state = super()._make_init_state(color, index)
        if color == "Red" and index == 0:
            if self._diag_red0_altitude_delta_m:
                state["ic/h-sl-ft"] += self._diag_red0_altitude_delta_m * FT_PER_M
            if self._diag_red0_speed_delta_mps:
                state["ic/u-fps"] += self._diag_red0_speed_delta_mps * FT_PER_M
        return state


def _make_actions(env: HeteroUavCombatEnv, policy: str, rng: np.random.Generator,
                  red0_action_bound: float | None) -> dict:
    if policy == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    actions = {}
    for aid in env.agent_ids:
        if policy == "bounded_random":
            bound = red0_action_bound if aid == "red_0" and red0_action_bound else 1.0
            actions[aid] = rng.uniform(-bound, bound, size=3).astype(np.float32)
        elif policy == "random":
            actions[aid] = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        else:
            raise ValueError(f"unknown policy: {policy}")
    return actions


def _scan_red0(env: HeteroUavCombatEnv) -> dict:
    sim = env.red_planes["red_0"]
    values = np.concatenate([
        sim.get_geodetic().astype(np.float64),
        sim.get_position().astype(np.float64),
        sim.get_velocity().astype(np.float64),
        np.asarray(sim.get_rpy(), dtype=np.float64),
    ])
    return {
        "altitude": float(sim.get_geodetic()[2]),
        "crashed": bool(sim.is_crash),
        "alive": bool(sim.is_alive),
        "nan_detected": bool(np.isnan(values).any() or np.isinf(values).any()),
    }


def run_rollout(scenario: dict, policy: str, steps: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    env = DiagnosticHeteroEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=12,
        max_steps=steps,
        suppress_jsbsim_output=True,
        red0_altitude_delta_m=float(scenario.get("red0_altitude_delta_m", 0.0)),
        red0_speed_delta_mps=float(scenario.get("red0_speed_delta_mps", 0.0)),
    )
    min_altitude = float("inf")
    nan_detected = False
    terminated = {}
    truncated = {}
    info = {}
    steps_executed = 0
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            scan = _scan_red0(env)
            min_altitude = min(min_altitude, scan["altitude"])
            nan_detected = nan_detected or scan["nan_detected"]
            actions = _make_actions(env, policy, rng, scenario.get("red0_action_bound"))
            _obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            scan = _scan_red0(env)
            min_altitude = min(min_altitude, scan["altitude"])
            nan_detected = nan_detected or scan["nan_detected"]
            if all(terminated.values()) or all(truncated.values()):
                break
        scan = _scan_red0(env)
        improved = min_altitude > 3000.0 and not scan["crashed"] and not nan_detected
        reason = "episode_limit" if all(truncated.values()) else "terminated" if all(terminated.values()) else "running"
        return {
            "scenario": scenario["name"],
            "policy": policy,
            "steps_executed": steps_executed,
            "red0_final_altitude": scan["altitude"],
            "red0_min_altitude": min_altitude,
            "red0_crashed": scan["crashed"],
            "red0_alive": scan["alive"],
            "nan_detected": nan_detected,
            "terminated_truncated_reason": reason,
            "improved": improved,
        }
    finally:
        env.close()


def run_all(steps: int = 200, seed: int = 0) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for policy in ("zero", "bounded_random", "random"):
            rows.append(run_rollout(scenario, policy, steps, seed))
    return rows


def _print_row(row: dict) -> None:
    print(
        f"{row['scenario']} policy={row['policy']} steps_executed={row['steps_executed']} "
        f"red_0_final_altitude={row['red0_final_altitude']:.3f} "
        f"red_0_min_altitude={row['red0_min_altitude']:.3f} "
        f"red_0_crashed={row['red0_crashed']} red_0_alive={row['red0_alive']} "
        f"nan_detected={row['nan_detected']} "
        f"terminated_truncated_reason={row['terminated_truncated_reason']} "
        f"improved={row['improved']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for row in run_all(args.steps, args.seed):
        _print_row(row)


if __name__ == "__main__":
    main()
