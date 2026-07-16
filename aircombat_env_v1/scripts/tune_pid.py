"""Strict sequential PID tuning with isolated stages and candidate outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, mark_candidate, save_config
from aircombat_env_v1.execution import run_command_hold
from aircombat_env_v1.metrics import step_response_metrics, trajectory_metrics
from aircombat_env_v1.pid import PaperAutopilot


SEED = 20260316
LARGE_PENALTY = 1e12


def reset_simulator(simulator, config):
    state = config["initial_state"]
    return simulator.reset(
        altitude_m=state["altitude_m"], speed_mps=state["speed_mps"],
        heading_deg=state["heading_deg"], roll_deg=state["roll_deg"],
        pitch_deg=state["pitch_deg"], elevator_trim=config["trim"]["elevator_trim"],
        throttle_base=config["trim"]["throttle_base"])


def stage_cases(stage):
    if stage == "roll_pd":
        return [(0.0, np.deg2rad(value), 250.0) for value in (-60, -30, -10, 10, 30, 60)]
    if stage == "pitch_pd":
        return [(np.deg2rad(value), 0.0, 250.0) for value in (-20, -10, -5, 5, 10, 20)]
    if stage == "speed_pi":
        return [(0.0, 0.0, float(value)) for value in (220, 280, 300, 330)]
    return stage_cases("roll_pd") + stage_cases("pitch_pd") + stage_cases("speed_pi")


def prepare_stage(config, stage):
    prepared = copy.deepcopy(config)
    if stage == "roll_pd":
        prepared["pid"]["roll"]["ki"] = 0.0
    elif stage == "pitch_pd":
        prepared["pid"]["pitch"]["ki"] = 0.0
    elif stage == "speed_pi":
        prepared["pid"]["speed"]["kd"] = 0.0
    return prepared


def apply_parameters(config, stage, values):
    tuned = copy.deepcopy(config)
    if stage == "roll_pd":
        tuned["pid"]["roll"].update(kp=float(values[0]), ki=0.0, kd=float(values[1]))
    elif stage == "pitch_pd":
        tuned["pid"]["pitch"].update(kp=float(values[0]), ki=0.0, kd=float(values[1]))
    elif stage == "speed_pi":
        tuned["pid"]["speed"].update(kp=float(values[0]), ki=float(values[1]), kd=0.0)
    elif stage == "roll_pitch_integral":
        tuned["pid"]["roll"]["ki"] = float(values[0])
        tuned["pid"]["pitch"]["ki"] = float(values[1])
    return tuned


def run_case(simulator, config, target, duration):
    controller = PaperAutopilot.from_config(config)
    initial = reset_simulator(simulator, config)
    hold = config["timing"]["physics_steps_per_decision"]
    decisions = max(1, math.ceil(duration * config["timing"]["decision_frequency_hz"]))
    rows, failure = [], None
    for _ in range(decisions):
        held_rows, failure = run_command_hold(simulator, controller, target, hold)
        rows.extend(held_rows)
        if failure:
            break
    metrics = trajectory_metrics(rows, config, initial["altitude"], failure)
    metrics.update(step_response_metrics(rows, initial))
    return metrics


def average_metrics(items):
    keys = items[0].keys()
    averaged = {}
    for key in keys:
        values = [item[key] for item in items
                  if isinstance(item[key], (int, float)) and not isinstance(item[key], bool)
                  and np.isfinite(item[key])]
        if values:
            averaged[key] = float(np.mean(values))
    averaged["crashed"] = any(item["crashed"] for item in items)
    averaged["has_nan_or_inf"] = any(item["has_nan_or_inf"] for item in items)
    return averaged


def objective_cost(metrics):
    if metrics["crashed"] or metrics["has_nan_or_inf"]:
        return LARGE_PENALTY
    overshoot = (metrics.get("heading_overshoot_deg", 0.0) / 10.0
                 + metrics.get("pitch_overshoot_deg", 0.0) / 5.0
                 + metrics.get("speed_overshoot_mps", 0.0) / 15.0)
    return float(metrics["roll_error_integral"] + metrics["pitch_error_integral"]
                 + metrics["speed_error_integral"]
                 + 0.02 * metrics["control_increment_energy"]
                 + 5.0 * (metrics["aileron_saturation_ratio"]
                          + metrics["elevator_saturation_ratio"]
                          + metrics["throttle_saturation_ratio"])
                 + overshoot + (metrics["altitude_loss_m"] / 100.0) ** 2)


def evaluate(simulator, config, cases, duration):
    case_metrics = [run_case(simulator, config, target, duration) for target in cases]
    metrics = average_metrics(case_metrics)
    return objective_cost(metrics), metrics


def write_stage(output, stage, trace, result, config, parameter_names):
    fieldnames = ["evaluation", "total_cost", *parameter_names,
                  "roll_error_integral", "pitch_error_integral", "speed_error_integral",
                  "control_increment_energy", "aileron_saturation_ratio",
                  "elevator_saturation_ratio", "throttle_saturation_ratio",
                  "altitude_loss_m", "maximum_alpha_deg", "maximum_beta_deg",
                  "maximum_load_factor", "crashed", "has_nan_or_inf"]
    with (output / f"{stage}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(trace)
    best_trace = min(trace, key=lambda item: item["total_cost"])
    payload = {"stage": stage, "seed": SEED, "best_cost": float(result.fun),
               "best_values": dict(zip(parameter_names, map(float, result.x))),
               "optimizer_success": bool(result.success), "message": str(result.message),
               "best_metrics": {key: value for key, value in best_trace.items()
                                if key not in {"evaluation", "total_cost", *parameter_names}},
               "pid": config["pid"]}
    (output / f"{stage}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stage_config = mark_candidate(copy.deepcopy(config))
    save_config(stage_config, output / f"{stage}_candidate.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--roll-duration", type=float, default=30.0)
    parser.add_argument("--pitch-duration", type=float, default=20.0)
    parser.add_argument("--speed-duration", type=float, default=60.0)
    parser.add_argument("--joint-duration", type=float, default=60.0)
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=6)
    parser.add_argument("--accept-candidate", action="store_true")
    args = parser.parse_args()
    durations = {"roll_pd": args.roll_duration, "pitch_pd": args.pitch_duration,
                 "speed_pi": args.speed_duration,
                 "roll_pitch_integral": args.joint_duration}
    if min(durations.values()) <= 0 or args.maxiter < 1 or args.popsize < 2:
        parser.error("durations/maxiter/popsize are out of range")
    output = (ROOT / "aircombat_env_v1" / "outputs" /
              f"tuning_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    stages = [
        ("roll_pd", ("roll_kp", "roll_kd"), [(0.05, 3.0), (0.0, 0.5)]),
        ("pitch_pd", ("pitch_kp", "pitch_kd"), [(0.05, 3.0), (0.0, 0.5)]),
        ("speed_pi", ("speed_kp", "speed_ki"), [(0.0001, 0.05), (0.0, 0.01)]),
        ("roll_pitch_integral", ("roll_ki", "pitch_ki"), [(0.0, 0.08), (0.0, 0.08)]),
    ]
    for stage, names, bounds in stages:
        config = prepare_stage(config, stage)
        trace = []

        def objective(values):
            candidate = apply_parameters(config, stage, values)
            cost, metrics = evaluate(simulator, candidate, stage_cases(stage), durations[stage])
            trace.append({"evaluation": len(trace), "total_cost": cost,
                          **dict(zip(names, map(float, values))), **metrics})
            return cost

        result = differential_evolution(objective, bounds, seed=SEED,
                                        maxiter=args.maxiter, popsize=args.popsize,
                                        polish=False, workers=1, updating="immediate")
        config = apply_parameters(config, stage, result.x)
        write_stage(output, stage, trace, result, config, names)
        print(f"{stage}: cost={result.fun:.6g}, values={result.x}")
    joint_cost, joint_metrics = evaluate(
        simulator, config, stage_cases("joint_validation"), args.joint_duration)
    joint = {"stage": "joint_validation", "total_cost": joint_cost, **joint_metrics,
             "pid": config["pid"]}
    (output / "joint_validation.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    candidate = mark_candidate(copy.deepcopy(config))
    with (output / "joint_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joint_metrics) + ["total_cost"],
                                extrasaction="ignore")
        writer.writeheader(); writer.writerow({**joint_metrics, "total_cost": joint_cost})
    save_config(candidate, output / "joint_validation_candidate.yaml")
    save_config(candidate, output / "candidate_config.yaml")
    if args.accept_candidate:
        save_config(candidate, args.config)
        print(f"accepted candidate into {args.config}")
    print(f"joint cost={joint_cost:.6g}; outputs={output}")


if __name__ == "__main__":
    main()
