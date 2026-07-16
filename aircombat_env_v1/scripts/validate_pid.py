"""Run 72 short responses and eight representative long stability cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, save_config
from aircombat_env_v1.execution import run_command_hold
from aircombat_env_v1.metrics import step_response_metrics, trajectory_metrics
from aircombat_env_v1.pid import PaperAutopilot


FIELDS = ["time", "target_pitch", "target_heading", "target_true_airspeed",
          "aileron", "elevator", "rudder", "throttle", "roll", "pitch",
          "heading", "p", "q", "r", "altitude", "true_airspeed", "v_north",
          "v_east", "v_down", "longitude", "latitude", "alpha", "beta",
          "load_factor"]
TASKS = ("level", "left_turn", "right_turn", "climb", "descent",
         "accelerate", "decelerate", "combined")
LONG_CASES = (("level", 6000, 250), ("left_turn", 6000, 250),
              ("right_turn", 6000, 250), ("climb", 6000, 250),
              ("descent", 6000, 250), ("combined", 6000, 250),
              ("level", 5000, 220), ("level", 7000, 300))


def command_at_decision_step(task, decision_step, base_speed):
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
        speeds = (base_speed, base_speed + 20.0, base_speed - 20.0)
        return (pitches[(decision_step // 20) % 3],
                headings[(decision_step // 15) % 3],
                speeds[(decision_step // 25) % 3])
    raise ValueError(f"unknown task: {task}")


def failure_reasons(metrics, thresholds, long_run):
    checks = [
        (thresholds["no_nan_or_inf"] and metrics["has_nan_or_inf"], "nan_or_inf"),
        (thresholds["no_crash"] and metrics["crashed"], "crash_or_simulation_failure"),
        (metrics["heading_final_error_deg"] > thresholds["heading_final_error_deg"],
         "heading_final_error"),
        (metrics["pitch_final_error_deg"] > thresholds["pitch_final_error_deg"],
         "pitch_final_error"),
        (metrics["speed_final_error_mps"] > thresholds["speed_final_error_mps"],
         "speed_final_error"),
        (metrics["aileron_saturation_ratio"] > thresholds["aileron_saturation_ratio"],
         "aileron_saturation"),
        (metrics["elevator_saturation_ratio"] > thresholds["elevator_saturation_ratio"],
         "elevator_saturation"),
        (metrics["throttle_saturation_ratio"] > thresholds["throttle_saturation_ratio"],
         "throttle_saturation"),
        (metrics["maximum_alpha_deg"] > thresholds["maximum_alpha_deg"], "maximum_alpha"),
        (metrics["maximum_beta_deg"] > thresholds["maximum_beta_deg"], "maximum_beta"),
        (metrics["maximum_load_factor"] > thresholds["maximum_load_factor"],
         "maximum_load_factor"),
        (long_run and metrics["altitude_loss_m"] > thresholds["long_run_altitude_loss_m"],
         "long_run_altitude_loss"),
    ]
    return [reason for failed, reason in checks if failed]


def run_case(simulator, config, task, altitude, speed, duration, long_run):
    controller = PaperAutopilot.from_config(config)
    initial = simulator.reset(altitude_m=altitude, speed_mps=speed,
                              elevator_trim=config["trim"]["elevator_trim"],
                              throttle_base=config["trim"]["throttle_base"])
    decision_hz = config["timing"]["decision_frequency_hz"]
    hold = config["timing"]["physics_steps_per_decision"]
    rows, failure = [], None
    for decision in range(max(1, math.ceil(duration * decision_hz))):
        target = command_at_decision_step(task, decision, speed)
        held_rows, failure = run_command_hold(simulator, controller, target, hold)
        rows.extend(held_rows)
        if failure:
            break
    result = trajectory_metrics(rows, config, altitude, failure)
    result.update(step_response_metrics(rows, initial))
    reasons = failure_reasons(result, config["validation_thresholds"], long_run)
    result.update(passed=not reasons, failure_reasons=reasons)
    return rows, result


def write_case(output, name, rows):
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--short-duration", type=float, default=40.0)
    parser.add_argument("--long-duration", type=float, default=200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mark-validated", action="store_true")
    args = parser.parse_args()
    if args.short_duration <= 0 or args.long_duration <= 0:
        parser.error("durations must be positive")
    config = load_config(args.config)
    output = args.output or (ROOT / "aircombat_env_v1" / "outputs" /
                             f"validation_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    short_results, long_results = [], []
    for altitude in (5000, 6000, 7000):
        for speed in (220, 250, 300):
            for task in TASKS:
                rows, result = run_case(
                    simulator, config, task, altitude, speed, args.short_duration, False)
                name = f"short_{task}_h{altitude}_v{speed}"
                write_case(output, name, rows)
                result.update(kind="short", task=task, altitude_m=altitude,
                              speed_mps=speed, csv=f"{name}.csv")
                short_results.append(result)
                print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}")
    for task, altitude, speed in LONG_CASES:
        rows, result = run_case(
            simulator, config, task, altitude, speed, args.long_duration, True)
        name = f"long_{task}_h{altitude}_v{speed}"
        write_case(output, name, rows)
        result.update(kind="long", task=task, altitude_m=altitude,
                      speed_mps=speed, csv=f"{name}.csv")
        long_results.append(result)
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}")
    summary = {
        "short_cases_total": len(short_results),
        "short_cases_passed": sum(result["passed"] for result in short_results),
        "long_cases_total": len(long_results),
        "long_cases_passed": sum(result["passed"] for result in long_results),
        "all_short_passed": all(result["passed"] for result in short_results),
        "all_long_passed": all(result["passed"] for result in long_results),
        "short_cases": short_results, "long_cases": long_results,
    }
    summary["overall_passed"] = summary["all_short_passed"] and summary["all_long_passed"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.mark_validated:
        if not summary["overall_passed"]:
            raise RuntimeError("refusing --mark-validated because validation did not pass")
        config["parameter_status"] = "validated"
        config["gain_source"] = "offline_tuned_on_local_jsbsim_f16"
        save_config(config, args.config)
        print(f"marked validated: {args.config}")
    print(f"summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
