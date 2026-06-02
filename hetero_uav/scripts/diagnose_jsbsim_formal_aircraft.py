from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.env import UavCombatEnv
from uav_env.JSBSim.pid_controller import PIDController
from uav_env.JSBSim.simulator import AircraftSimulator
from uav_env.JSBSim.utils import in_range_rad


SCENARIOS = {
    "level": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "climb": np.array([0.2, 0.0, 0.0], dtype=np.float32),
    "descend": np.array([-0.2, 0.0, 0.0], dtype=np.float32),
    "turn_left": np.array([0.0, -0.25, 0.0], dtype=np.float32),
    "turn_right": np.array([0.0, 0.25, 0.0], dtype=np.float32),
    "speed_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "slow_down": np.array([0.0, 0.0, -1.0], dtype=np.float32),
}


def action_to_targets(action: np.ndarray) -> tuple[float, float, float]:
    target_pitch = float(action[0]) * math.radians(UavCombatEnv.PITCH_DEG)
    target_heading = float(action[1]) * math.pi
    target_velocity = UavCombatEnv.VELOCITY_MIN + (float(action[2]) + 1.0) * 0.5 * (
        UavCombatEnv.VELOCITY_MAX - UavCombatEnv.VELOCITY_MIN
    )
    return target_pitch, target_heading, target_velocity


def run_scenario(model: str, scenario: str, duration: float, sim_freq: int = 60) -> dict:
    action = SCENARIOS[scenario]
    sim = AircraftSimulator(
        uid=f"diag_{model}_{scenario}",
        color="Red",
        model=model,
        num_missiles=0,
        sim_freq=sim_freq,
        suppress_jsbsim_output=True,
    )
    pid = PIDController(1.0 / sim_freq)
    steps = max(1, int(round(duration * sim_freq)))
    initial_altitude = float(sim.get_geodetic()[2])
    initial_speed = float(np.linalg.norm(sim.get_velocity()))
    initial_heading = float(sim.get_rpy()[2])
    max_abs_roll = abs(float(sim.get_rpy()[0]))
    max_abs_pitch = abs(float(sim.get_rpy()[1]))
    nan_detected = False

    try:
        targets = action_to_targets(action)
        for _ in range(steps):
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            current_speed = float(np.linalg.norm(vel))
            vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            aileron, elevator, rudder, throttle = pid.compute_control(
                rpy,
                current_speed,
                targets[0],
                targets[1],
                targets[2],
                ned_velocity=vel_ned,
            )
            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            sim.set_property_value("fcs/throttle-cmd-norm", throttle)
            sim.run()

            rpy = sim.get_rpy()
            values = np.concatenate([
                sim.get_geodetic().astype(np.float64),
                sim.get_position().astype(np.float64),
                sim.get_velocity().astype(np.float64),
                np.asarray(rpy, dtype=np.float64),
            ])
            nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
            max_abs_roll = max(max_abs_roll, abs(float(rpy[0])))
            max_abs_pitch = max(max_abs_pitch, abs(float(rpy[1])))
            if not sim.is_alive:
                break

        final_altitude = float(sim.get_geodetic()[2])
        final_speed = float(np.linalg.norm(sim.get_velocity()))
        final_heading = float(sim.get_rpy()[2])
        return {
            "model": model,
            "scenario": scenario,
            "initial_altitude": initial_altitude,
            "final_altitude": final_altitude,
            "altitude_delta": final_altitude - initial_altitude,
            "initial_speed": initial_speed,
            "final_speed": final_speed,
            "speed_delta": final_speed - initial_speed,
            "initial_heading": initial_heading,
            "final_heading": final_heading,
            "heading_delta": in_range_rad(final_heading - initial_heading),
            "max_abs_roll": max_abs_roll,
            "max_abs_pitch": max_abs_pitch,
            "crashed": bool(sim.is_crash or final_altitude <= 0.0),
            "shotdown": bool(sim.is_shotdown),
            "nan_detected": nan_detected,
        }
    finally:
        sim.close()


def annotate(rows: list[dict]) -> None:
    by_scenario = {row["scenario"]: row for row in rows}
    level = by_scenario.get("level")
    if not level:
        return
    level_alt = level["final_altitude"]
    level_speed = level["final_speed"]
    for row in rows:
        scenario = row["scenario"]
        if scenario == "climb":
            row["direction_ok"] = row["final_altitude"] > level_alt
        elif scenario == "descend":
            row["direction_ok"] = row["final_altitude"] < level_alt
        elif scenario == "turn_left":
            row["direction_ok"] = row["heading_delta"] < 0.0
        elif scenario == "turn_right":
            row["direction_ok"] = row["heading_delta"] > 0.0
        elif scenario == "speed_up":
            row["direction_ok"] = row["final_speed"] >= level_speed - 1e-3
        elif scenario == "slow_down":
            row["direction_ok"] = row["final_speed"] <= level_speed + 1e-3
        elif scenario == "level":
            row["direction_ok"] = (not row["crashed"]) and (not row["nan_detected"])
        else:
            row["direction_ok"] = ""


def print_row(row: dict) -> None:
    print(
        f"model={row['model']} scenario={row['scenario']} "
        f"initial_altitude={row['initial_altitude']:.3f} "
        f"final_altitude={row['final_altitude']:.3f} "
        f"altitude_delta={row['altitude_delta']:.3f} "
        f"initial_speed={row['initial_speed']:.3f} "
        f"final_speed={row['final_speed']:.3f} "
        f"speed_delta={row['speed_delta']:.3f} "
        f"initial_heading={row['initial_heading']:.6f} "
        f"final_heading={row['final_heading']:.6f} "
        f"heading_delta={row['heading_delta']:.6f} "
        f"max_abs_roll={row['max_abs_roll']:.6f} "
        f"max_abs_pitch={row['max_abs_pitch']:.6f} "
        f"crashed={row['crashed']} "
        f"shotdown={row['shotdown']} "
        f"nan_detected={row['nan_detected']} "
        f"direction_ok={row.get('direction_ok', '')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A-4", "f16"], required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="level",
    )
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    rows = [run_scenario(args.model, scenario, args.duration) for scenario in scenarios]
    annotate(rows)
    for row in rows:
        print_row(row)


if __name__ == "__main__":
    main()
