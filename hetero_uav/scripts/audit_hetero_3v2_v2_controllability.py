"""Measure formal high-level command response for MAV and attack UAV."""
from __future__ import annotations

import argparse
import math

import numpy as np

from hetero_3v2_v2_audit_common import AUDIT_DIR, ROOT, V1_CONFIG, write_csv, write_json
from uav_env.make_env import make_env


COMMANDS = {
    "pitch_up": (math.radians(10), None, 250.0),
    "pitch_down": (math.radians(-10), None, 250.0),
    "turn_left": (0.0, math.radians(-45), 250.0),
    "turn_right": (0.0, math.radians(45), 250.0),
    "turn_180": (0.0, math.pi, 250.0),
    "accelerate": (0.0, None, 330.0),
    "decelerate": (0.0, None, 180.0),
    "combined": (math.radians(8), math.radians(60), 300.0),
}


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V1_CONFIG)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    rows, summaries = [], []
    for aid in ("red_0", "red_1"):
        for name, command in COMMANDS.items():
            env = make_env(str(ROOT / args.config))
            env.reset(seed=100)
            aircraft = env.aircraft[aid]
            controller = env.controllers[aid]
            initial_alt = float(aircraft.get_position()[2])
            initial_yaw = float(aircraft.get_rpy()[2])
            target_pitch = command[0]
            target_heading = initial_yaw if command[1] is None else wrap(
                initial_yaw + command[1])
            target_speed = command[2]
            errors = []
            failed = False
            peak_error = 0.0
            max_load = 0.0
            for step in range(args.steps):
                roll, pitch, yaw = aircraft.get_rpy()
                velocity = aircraft.get_velocity()
                speed = float(np.linalg.norm(velocity))
                controls = controller.compute_control(
                    (roll, pitch, yaw), speed, target_pitch, target_heading,
                    target_speed, ned_velocity=(velocity[0], velocity[1], -velocity[2]))
                for prop, value in zip(
                        ("fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
                         "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm"), controls):
                    aircraft.set_property_value(prop, float(value))
                for _ in range(env.agent_interaction_steps):
                    aircraft.run()
                roll, pitch, yaw = aircraft.get_rpy()
                speed = float(np.linalg.norm(aircraft.get_velocity()))
                pitch_error = target_pitch - pitch
                heading_error = wrap(target_heading - yaw)
                speed_error = target_speed - speed
                error = math.sqrt(
                    pitch_error ** 2 + heading_error ** 2 + (speed_error / 100.0) ** 2)
                errors.append(error)
                peak_error = max(peak_error, abs(error))
                try:
                    load = abs(float(aircraft.get_property_value("accelerations/Nz")))
                except Exception:
                    load = 0.0
                max_load = max(max_load, load)
                finite = bool(np.isfinite(np.r_[
                    aircraft.get_position(), aircraft.get_velocity(),
                    aircraft.get_rpy()]).all())
                failed = failed or not finite or aircraft.get_position()[2] < 100.0
                rows.append({
                    "agent_id": aid, "role": env.roles[aid], "command": name,
                    "step": step + 1, "command_pitch_rad": target_pitch,
                    "command_heading_rad": target_heading,
                    "command_speed_mps": target_speed, "actual_roll_rad": roll,
                    "actual_pitch_rad": pitch, "actual_heading_rad": yaw,
                    "actual_speed_mps": speed,
                    "altitude_m": float(aircraft.get_position()[2]),
                    "pitch_error_rad": pitch_error,
                    "heading_error_rad": heading_error,
                    "speed_error_mps": speed_error, "load_factor": load,
                    "finite": int(finite),
                })
            threshold = max(0.1 * errors[0], 0.05)
            settled = next((index + 1 for index in range(len(errors))
                            if max(errors[index:]) <= threshold), None)
            rise = next((index + 1 for index, value in enumerate(errors)
                         if value <= 0.1 * errors[0]), None)
            final = rows[-1]
            summaries.append({
                "agent_id": aid, "role": env.roles[aid], "command": name,
                "rise_time_steps": rise, "settling_time_steps": settled,
                "peak_composite_error": peak_error,
                "steady_pitch_error_rad": final["pitch_error_rad"],
                "steady_heading_error_rad": final["heading_error_rad"],
                "steady_speed_error_mps": final["speed_error_mps"],
                "altitude_loss_m": initial_alt - final["altitude_m"],
                "max_load_factor": max_load, "flight_failure": int(failed),
                "error_step_20": errors[min(19, len(errors)-1)],
                "error_step_50": errors[min(49, len(errors)-1)],
                "error_step_100": errors[min(99, len(errors)-1)],
            })
            env.close()
    output = AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "controllability.csv", rows)
    write_csv(output / "controllability_summary.csv", summaries)
    write_json(output / "controllability_summary.json", {
        "tests": len(summaries),
        "flight_failures": sum(row["flight_failure"] for row in summaries),
        "all_finite": all(row["finite"] for row in rows),
        "summary": summaries,
    })
    print(output / "controllability_summary.json")


if __name__ == "__main__":
    main()
