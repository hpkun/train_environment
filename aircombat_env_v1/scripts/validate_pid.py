"""Quick flight-control health checks and full final acceptance validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, save_config
from aircombat_env_v1.validation import (command_at_decision_step, failure_reasons,
                                         quick_cases, run_validation_case)


FIELDS = ["time", "target_pitch", "target_heading", "target_true_airspeed",
          "aileron", "elevator", "rudder", "throttle", "roll", "pitch",
          "heading", "p", "q", "r", "altitude", "true_airspeed", "v_north",
          "v_east", "v_down", "longitude", "latitude", "alpha", "beta",
          "load_factor"]
STATIC_TASKS = ("level", "left_turn", "right_turn", "climb", "descent",
                "accelerate", "decelerate", "bounded_combined")


def ensure_mark_validated_allowed(mode, mark_validated):
    if mark_validated and mode != "full":
        raise ValueError("--mark-validated is allowed only with --mode full")


def write_case(output, name, rows):
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def run_cases(simulator, config, output, cases, duration, kind):
    results = []
    for index, case in enumerate(cases):
        rows, metrics = run_validation_case(simulator, config, case, duration)
        name = f"{kind}_{index:02d}_{case['task']}_h{case['altitude']}_v{case['speed']}"
        write_case(output, name, rows)
        result = {"kind": kind, **case, "duration": duration,
                  "csv": f"{name}.csv", **metrics}
        results.append(result)
        print(f"{name}: {'PASS' if metrics['passed'] else 'FAIL'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quick-duration", type=float, default=200.0)
    parser.add_argument("--short-duration", type=float, default=40.0)
    parser.add_argument("--long-duration", type=float, default=200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mark-validated", action="store_true")
    args = parser.parse_args()
    try:
        ensure_mark_validated_allowed(args.mode, args.mark_validated)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.quick_duration, args.short_duration, args.long_duration) <= 0:
        parser.error("durations must be positive")
    config = load_config(args.config)
    output = args.output or (ROOT / "aircombat_env_v1" / "outputs" /
                             f"validation_{args.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    if args.mode == "quick":
        results = run_cases(simulator, config, output, quick_cases(),
                            args.quick_duration, "quick")
        summary = {"mode": "quick", "cases_total": 8,
                   "cases_passed": sum(result["passed"] for result in results),
                   "level_200s_passed": results[0]["passed"],
                   "overall_passed": all(result["passed"] for result in results),
                   "cases": results}
    else:
        matrix = [{"task": task, "altitude": altitude, "speed": speed,
                   "dynamic": task == "bounded_combined"}
                  for altitude in (5000, 6000, 7000)
                  for speed in (220, 250, 300) for task in STATIC_TASKS]
        short = run_cases(simulator, config, output, matrix, args.short_duration, "full_short")
        long = run_cases(simulator, config, output, quick_cases(), args.long_duration, "full_long")
        summary = {"mode": "full", "short_cases_total": len(short),
                   "short_cases_passed": sum(result["passed"] for result in short),
                   "long_cases_total": len(long),
                   "long_cases_passed": sum(result["passed"] for result in long),
                   "all_short_passed": all(result["passed"] for result in short),
                   "all_long_passed": all(result["passed"] for result in long),
                   "short_cases": short, "long_cases": long}
        summary["overall_passed"] = summary["all_short_passed"] and summary["all_long_passed"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.mark_validated:
        if not summary["overall_passed"]:
            raise RuntimeError("refusing --mark-validated because full validation failed")
        config["parameter_status"] = "validated"
        config["gain_source"] = "offline_tuned_on_local_jsbsim_f16"
        save_config(config, args.config)
    print(f"summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
