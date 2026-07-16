"""Validate eight command tasks over the required altitude/speed matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import load_config
from aircombat_env_v1.geometry import paper_direction_errors
from aircombat_env_v1.pid import PaperAutopilot


FIELDS = ["time", "target_pitch", "target_heading", "target_true_airspeed",
          "aileron", "elevator", "rudder", "throttle", "roll", "pitch",
          "heading", "p", "q", "r", "altitude", "true_airspeed", "alpha",
          "beta", "load_factor"]


def command(task, elapsed, base_speed):
    phase = int(elapsed / 0.2)
    if task == "level":
        return 0.0, 0.0, base_speed
    if task == "left_turn":
        return 0.0, np.deg2rad(-30.0), base_speed
    if task == "right_turn":
        return 0.0, np.deg2rad(30.0), base_speed
    if task == "climb":
        return np.deg2rad(10.0), 0.0, base_speed
    if task == "descent":
        return np.deg2rad(-10.0), 0.0, base_speed
    if task == "accelerate":
        return 0.0, 0.0, base_speed + 30.0
    if task == "decelerate":
        return 0.0, 0.0, max(120.0, base_speed - 30.0)
    if task == "combined":
        headings = (0.0, np.deg2rad(30.0), np.deg2rad(-30.0))
        pitches = (0.0, np.deg2rad(8.0), np.deg2rad(-8.0))
        return pitches[(phase // 15) % 3], headings[(phase // 20) % 3], (
            base_speed + (-20.0, 0.0, 20.0)[(phase // 25) % 3])
    raise ValueError(f"unknown task: {task}")


def settling_time(rows, tolerance=0.05):
    errors = []
    for row in rows:
        roll_error, pitch_error = paper_direction_errors(
            row["roll"], row["pitch"], row["heading"],
            row["target_pitch"], row["target_heading"])
        speed_error = (row["target_true_airspeed"] - row["true_airspeed"]) / 50.0
        errors.append(max(abs(roll_error), abs(pitch_error), abs(speed_error)))
    for index in range(len(errors)):
        if max(errors[index:], default=0.0) <= tolerance:
            return rows[index]["time"] - rows[0]["time"]
    return None


def metrics(rows, initial_altitude, crashed, nonfinite, dt):
    error_integral = max_overshoot = 0.0
    previous = None
    saturated = 0
    for row in rows:
        roll_error, pitch_error = paper_direction_errors(
            row["roll"], row["pitch"], row["heading"],
            row["target_pitch"], row["target_heading"])
        speed_error = (row["target_true_airspeed"] - row["true_airspeed"]) / 50.0
        errors = np.array([roll_error, pitch_error, speed_error])
        error_integral += float(np.abs(errors).sum()) * dt
        if previous is not None:
            max_overshoot = max(max_overshoot,
                                float(np.max(np.maximum(0.0, -(errors * previous)))))
        previous = errors
        saturated += int(abs(row["aileron"]) > 0.999
                         or abs(row["elevator"]) > 0.999
                         or row["throttle"] < 0.001 or row["throttle"] > 0.999)
    return {
        "tracking_error_integral": error_integral,
        "maximum_overshoot": max_overshoot,
        "settling_time_s": settling_time(rows),
        "saturation_ratio": saturated / max(1, len(rows)),
        "maximum_alpha_deg": max((abs(np.rad2deg(r["alpha"])) for r in rows), default=0.0),
        "maximum_beta_deg": max((abs(np.rad2deg(r["beta"])) for r in rows), default=0.0),
        "maximum_load_factor": max((abs(r["load_factor"]) for r in rows), default=0.0),
        "altitude_loss_m": max(0.0, initial_altitude
                               - min((r["altitude"] for r in rows), default=initial_altitude)),
        "crashed": crashed,
        "has_nan_or_inf": nonfinite,
    }


def run_case(simulator, config, task, altitude, speed, duration):
    controller = PaperAutopilot.from_config(config)
    simulator.reset(altitude_m=altitude, speed_mps=speed,
                    elevator_trim=config["trim"]["elevator_trim"],
                    throttle_base=config["trim"]["throttle_base"])
    rows, crashed, nonfinite = [], False, False
    for index in range(round(duration / simulator.dt)):
        state = simulator.state()
        target = command(task, index * simulator.dt, speed)
        controls = controller.step(
            state["roll"], state["pitch"], state["heading"],
            state["true_airspeed"], *target, simulator.dt)
        simulator.set_controls(*controls)
        try:
            state = simulator.run()
        except RuntimeError:
            crashed = True
            break
        values = tuple(state.values()) + tuple(controls)
        if not np.all(np.isfinite(values)):
            nonfinite = True
            break
        row = {field: state.get(field) for field in FIELDS}
        row.update(target_pitch=target[0], target_heading=target[1],
                   target_true_airspeed=target[2], aileron=controls[0],
                   elevator=controls[1], rudder=controls[2], throttle=controls[3])
        rows.append(row)
        if state["altitude"] < 1000.0 or state["true_airspeed"] < 120.0:
            crashed = True
            break
    return rows, metrics(rows, altitude, crashed, nonfinite, simulator.dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    config = load_config()
    output = args.output or (ROOT / "aircombat_env_v1" / "outputs" /
                             f"validation_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    tasks = ("level", "left_turn", "right_turn", "climb", "descent",
             "accelerate", "decelerate", "combined")
    summary = []
    for altitude in (5000, 6000, 7000):
        for speed in (220, 250, 300):
            for task in tasks:
                rows, result = run_case(
                    simulator, config, task, altitude, speed, args.duration)
                name = f"{task}_h{altitude}_v{speed}"
                with (output / f"{name}.csv").open(
                        "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                result.update(task=task, altitude_m=altitude, speed_mps=speed,
                              csv=f"{name}.csv")
                summary.append(result)
                print(name)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
