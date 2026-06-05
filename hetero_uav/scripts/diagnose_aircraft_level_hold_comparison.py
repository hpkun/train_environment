"""Compare A-4 and f16 level-hold behavior under the same high-level actions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _action_to_targets(action: np.ndarray) -> tuple[float, float, float]:
    from uav_env.JSBSim.env import UavCombatEnv

    target_pitch = float(action[0]) * math.radians(UavCombatEnv.PITCH_DEG)
    target_heading = float(action[1]) * math.pi
    target_velocity = UavCombatEnv.VELOCITY_MIN + (float(action[2]) + 1.0) * 0.5 * (
        UavCombatEnv.VELOCITY_MAX - UavCombatEnv.VELOCITY_MIN
    )
    return target_pitch, target_heading, target_velocity


def _case_specs() -> list[dict]:
    return [
        {"model": "A-4", "case": "a4_zero", "action": [0.0, 0.0, 0.0]},
        {"model": "f16", "case": "f16_zero", "action": [0.0, 0.0, 0.0]},
        {"model": "A-4", "case": "a4_pitch_bias_005", "action": [0.05, 0.0, 0.0]},
        {"model": "f16", "case": "f16_pitch_bias_005", "action": [0.05, 0.0, 0.0]},
    ]


def run_case(spec: dict, duration: float = 60.0) -> dict:
    from uav_env.JSBSim.pid_controller import PIDController
    from uav_env.JSBSim.simulator import AircraftSimulator

    sim = AircraftSimulator(
        uid=spec["case"],
        color="Red",
        model=spec["model"],
        num_missiles=0,
        sim_freq=60,
        suppress_jsbsim_output=True,
    )
    pid = PIDController(1.0 / 60.0)
    action = np.asarray(spec["action"], dtype=np.float32)
    steps = max(1, int(round(float(duration) * 60.0)))
    altitudes: list[float] = []
    speeds: list[float] = []
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    nan_detected = False
    initial_altitude = float(sim.get_geodetic()[2])
    initial_speed = float(np.linalg.norm(sim.get_velocity()))

    try:
        for _ in range(steps):
            rpy = sim.get_rpy()
            vel = sim.get_velocity()
            targets = _action_to_targets(action)
            vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            aileron, elevator, rudder, throttle = pid.compute_control(
                rpy,
                float(np.linalg.norm(vel)),
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
            vel = sim.get_velocity()
            values = np.concatenate([
                sim.get_geodetic().astype(np.float64),
                sim.get_position().astype(np.float64),
                vel.astype(np.float64),
                np.asarray(rpy, dtype=np.float64),
            ])
            nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
            altitudes.append(float(sim.get_geodetic()[2]))
            speeds.append(float(np.linalg.norm(vel)))
            max_abs_pitch = max(max_abs_pitch, abs(float(rpy[1])))
            max_abs_roll = max(max_abs_roll, abs(float(rpy[0])))
            if not sim.is_alive:
                break

        final_altitude = float(sim.get_geodetic()[2])
        final_speed = float(np.linalg.norm(sim.get_velocity()))
        return {
            "case": spec["case"],
            "model": spec["model"],
            "action": list(map(float, action)),
            "initial_altitude_m": initial_altitude,
            "final_altitude_m": final_altitude,
            "min_altitude_m": float(np.min(altitudes)),
            "max_altitude_m": float(np.max(altitudes)),
            "altitude_delta_m": final_altitude - initial_altitude,
            "initial_speed_mps": initial_speed,
            "final_speed_mps": final_speed,
            "min_speed_mps": float(np.min(speeds)),
            "max_speed_mps": float(np.max(speeds)),
            "final_pitch_deg": float(np.rad2deg(sim.get_rpy()[1])),
            "final_roll_deg": float(np.rad2deg(sim.get_rpy()[0])),
            "crashed": bool(sim.is_crash or final_altitude <= 0.0),
            "nan_detected": nan_detected,
            "max_abs_pitch_deg": float(np.rad2deg(max_abs_pitch)),
            "max_abs_roll_deg": float(np.rad2deg(max_abs_roll)),
        }
    finally:
        sim.close()


def _conclusion(records: list[dict]) -> str:
    by_case = {record["case"]: record for record in records}
    a4_drop = by_case["a4_zero"]["altitude_delta_m"]
    f16_drop = by_case["f16_zero"]["altitude_delta_m"]
    if a4_drop < f16_drop - 500.0:
        return "A-4 drops significantly more than f16 under zero action"
    return "A-4 and f16 zero-action altitude loss are not clearly separated in this short diagnostic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-json", default="outputs/environment_audit/aircraft_level_hold_comparison.json")
    args = parser.parse_args()

    records = [run_case(spec, args.duration) for spec in _case_specs()]
    data = {"records": records, "conclusion": _conclusion(records)}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"output_json: {output}")
    print(f"conclusion: {data['conclusion']}")
    for record in records:
        print(
            f"{record['case']}: model={record['model']} "
            f"altitude_delta_m={record['altitude_delta_m']:.3f} "
            f"final_altitude_m={record['final_altitude_m']:.3f} "
            f"crashed={record['crashed']} nan_detected={record['nan_detected']}"
        )


if __name__ == "__main__":
    main()
