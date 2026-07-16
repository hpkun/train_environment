"""Conservative sequential Roll-PD, Pitch-PD, and Speed-PI tuning."""

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
from aircombat_env_v1.execution import bounded_combined_command, run_command_hold
from aircombat_env_v1.metrics import (aggregate_case_metrics, stable_segment_errors,
                                      trajectory_metrics)
from aircombat_env_v1.pid import PaperAutopilot


SEED = 20260316
FAILURE_BASE = 1e6


def reset_simulator(simulator, config):
    state = config["initial_state"]
    return simulator.reset(
        altitude_m=state["altitude_m"], speed_mps=state["speed_mps"],
        heading_deg=state["heading_deg"], roll_deg=state["roll_deg"],
        pitch_deg=state["pitch_deg"], elevator_trim=config["trim"]["elevator_trim"],
        throttle_base=config["trim"]["throttle_base"])


def stage_cases(stage):
    if stage == "roll_pd":
        return [{"name": f"heading_{value:+d}",
                 "target": (0.0, np.deg2rad(value), 250.0)}
                for value in (-60, -30, -10, 10, 30, 60)]
    if stage == "pitch_pd":
        return [{"name": f"pitch_{value:+d}",
                 "target": (np.deg2rad(value), 0.0, 250.0)}
                for value in (-20, -10, -5, 5, 10, 20)]
    if stage == "speed_pi":
        return [{"name": f"speed_{value}", "target": (0.0, 0.0, float(value))}
                for value in (220, 280, 300, 330)]
    return (stage_cases("roll_pd") + stage_cases("pitch_pd")
            + stage_cases("speed_pi") + [{"name": "bounded_combined", "dynamic": True}])


def prepare_config(config):
    prepared = copy.deepcopy(config)
    prepared["pid"]["roll"]["ki"] = 0.0
    prepared["pid"]["pitch"]["ki"] = 0.0
    prepared["pid"]["speed"]["kd"] = 0.0
    return prepared


def apply_parameters(config, stage, values):
    tuned = prepare_config(config)
    if stage == "roll_pd":
        tuned["pid"]["roll"].update(kp=float(values[0]), kd=float(values[1]))
    elif stage == "pitch_pd":
        tuned["pid"]["pitch"].update(kp=float(values[0]), kd=float(values[1]))
    elif stage == "speed_pi":
        tuned["pid"]["speed"].update(kp=float(values[0]), ki=float(values[1]))
    return tuned


def case_cost(metrics):
    return float(metrics["roll_error_integral"] + metrics["pitch_error_integral"]
                 + metrics["speed_error_integral"]
                 + 0.05 * metrics["control_increment_energy"]
                 + 0.2 * metrics["control_rate_energy"]
                 + 10.0 * (metrics["aileron_saturation_ratio"]
                           + metrics["elevator_saturation_ratio"]
                           + metrics["throttle_saturation_ratio"])
                 + metrics["altitude_loss_m"] / 100.0)


def failure_penalty(metrics, completed_fraction):
    limits = {"maximum_alpha_deg": 25.0, "maximum_beta_deg": 15.0,
              "maximum_load_factor": 9.0}
    excess = sum(max(0.0, metrics[key] - limit) / limit
                 for key, limit in limits.items())
    return FAILURE_BASE + (1.0 - completed_fraction) * 1e5 + excess * 1e5


def joint_health_reasons(metrics, thresholds, dynamic):
    heading = "stable_heading_error_deg" if dynamic else "heading_final_error_deg"
    pitch = "stable_pitch_error_deg" if dynamic else "pitch_final_error_deg"
    speed = "stable_speed_error_mps" if dynamic else "speed_final_error_mps"
    checks = [
        (metrics["maximum_alpha_deg"] > thresholds["maximum_alpha_deg"], "maximum_alpha"),
        (metrics["maximum_beta_deg"] > thresholds["maximum_beta_deg"], "maximum_beta"),
        (metrics["maximum_load_factor"] > thresholds["maximum_load_factor"], "maximum_load"),
        (metrics["aileron_saturation_ratio"] > thresholds["aileron_saturation_ratio"],
         "aileron_saturation"),
        (metrics["elevator_saturation_ratio"] > thresholds["elevator_saturation_ratio"],
         "elevator_saturation"),
        (metrics["throttle_saturation_ratio"] > thresholds["throttle_saturation_ratio"],
         "throttle_saturation"),
        (metrics[heading] > thresholds["heading_final_error_deg"], "heading_final_error"),
        (metrics[pitch] > thresholds["pitch_final_error_deg"], "pitch_final_error"),
        (metrics[speed] > thresholds["speed_final_error_mps"], "speed_final_error"),
    ]
    return [reason for failed, reason in checks if failed]


def run_case(simulator, config, case, duration, strict_health=False):
    controller = PaperAutopilot.from_config(config)
    initial = reset_simulator(simulator, config)
    decision_hz = config["timing"]["decision_frequency_hz"]
    hold = config["timing"]["physics_steps_per_decision"]
    decisions = max(1, math.ceil(duration * decision_hz))
    rows, failure = [], None
    for decision in range(decisions):
        target = (bounded_combined_command(
            decision, 250.0, duration, decision_hz, stable_duration=10.0)
                  if case.get("dynamic") else case["target"])
        held, failure = run_command_hold(simulator, controller, target, hold)
        rows.extend(held)
        if failure:
            break
    expected_steps = decisions * hold
    completed_fraction = len(rows) / expected_steps
    metrics = trajectory_metrics(rows, config, initial["altitude"], failure)
    simulation_complete = failure is None and len(rows) == expected_steps
    health_reasons = []
    if strict_health and simulation_complete:
        dynamic = bool(case.get("dynamic"))
        if dynamic:
            metrics.update(stable_segment_errors(
                rows, min(10.0, duration), config["timing"]["sim_frequency_hz"]))
        health_reasons = joint_health_reasons(
            metrics, config["validation_thresholds"], dynamic)
    complete = simulation_complete and not health_reasons
    cost = case_cost(metrics)
    if not complete:
        cost += failure_penalty(metrics, completed_fraction)
    return {"name": case["name"], "complete": complete,
            "completed_fraction": completed_fraction, "cost": cost,
            "health_failure_reasons": health_reasons, "metrics": metrics}


def evaluate(simulator, config, cases, duration, strict_health=False):
    results = [run_case(simulator, config, case, duration, strict_health)
               for case in cases]
    metrics = aggregate_case_metrics(results)
    cost = float(np.mean([result["cost"] for result in results]))
    return cost, metrics, results


def rejection_reason(candidate_cost, baseline_cost, metrics):
    if not np.isfinite(candidate_cost):
        return "candidate_cost_not_finite"
    if metrics["failed_case_count"]:
        return "candidate_has_failed_cases"
    if candidate_cost >= baseline_cost:
        return "candidate_did_not_improve_baseline"
    return None


def select_stage_config(baseline, proposed, candidate_cost, baseline_cost, metrics):
    reason = rejection_reason(candidate_cost, baseline_cost, metrics)
    return (proposed if reason is None else baseline), reason is None, reason


def acceptance_eligible(stage_reports, joint_cost, joint_metrics):
    return bool(all(report["stage_accepted"] for report in stage_reports)
                and np.isfinite(joint_cost) and joint_cost < FAILURE_BASE
                and joint_metrics["failed_case_count"] == 0)


def require_acceptance_eligible(stage_reports, joint_cost, joint_metrics):
    if not acceptance_eligible(stage_reports, joint_cost, joint_metrics):
        raise RuntimeError(
            "refusing --accept-candidate: stages or joint validation invalid")


def stage_row(evaluation, cost, values, names, metrics):
    return {"evaluation": evaluation, "total_cost": cost,
            **dict(zip(names, map(float, values))), **metrics}


def write_stage(output, stage, trace, report, config):
    fields = list(dict.fromkeys(key for row in trace for key in row))
    with (output / f"{stage}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(trace)
    (output / f"{stage}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_config(mark_candidate(copy.deepcopy(config)), output / f"{stage}_candidate.yaml")


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
                 "speed_pi": args.speed_duration}
    if min([*durations.values(), args.joint_duration]) <= 0 or args.maxiter < 1 or args.popsize < 2:
        parser.error("durations/maxiter/popsize are out of range")
    output = (ROOT / "aircombat_env_v1" / "outputs" /
              f"tuning_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    config = prepare_config(load_config(args.config))
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    stages = [
        ("roll_pd", ("roll_kp", "roll_kd"), [(0.1, 1.5), (0.0, 0.15)]),
        ("pitch_pd", ("pitch_kp", "pitch_kd"), [(0.1, 1.5), (0.0, 0.15)]),
        ("speed_pi", ("speed_kp", "speed_ki"), [(0.001, 0.03), (0.0, 0.005)]),
    ]
    stage_reports = []
    for stage, names, bounds in stages:
        cases, duration = stage_cases(stage), durations[stage]
        baseline_cost, baseline_metrics, _ = evaluate(simulator, config, cases, duration)
        trace = []

        def objective(values):
            candidate = apply_parameters(config, stage, values)
            cost, metrics, _ = evaluate(simulator, candidate, cases, duration)
            trace.append(stage_row(len(trace), cost, values, names, metrics))
            return cost

        result = differential_evolution(objective, bounds, seed=SEED,
                                        maxiter=args.maxiter, popsize=args.popsize,
                                        polish=False, workers=1, updating="immediate")
        proposed = apply_parameters(config, stage, result.x)
        candidate_cost, candidate_metrics, _ = evaluate(
            simulator, proposed, cases, duration)
        config, accepted, reason = select_stage_config(
            config, proposed, candidate_cost, baseline_cost, candidate_metrics)
        boundary_hit = any(np.isclose(value, bound).any()
                           for value, bound in zip(result.x, bounds))
        report = {"stage": stage, "baseline_cost": baseline_cost,
                  "candidate_cost": candidate_cost, "stage_accepted": accepted,
                  "rejection_reason": reason,
                  "failed_case_count": candidate_metrics["failed_case_count"],
                  "failed_cases": candidate_metrics["failed_cases"],
                  "accepted_parameters": {name: float(value) for name, value in
                                          zip(names, result.x)} if accepted else None,
                  "boundary_hit": bool(boundary_hit), "optimizer_message": str(result.message)}
        stage_reports.append(report)
        write_stage(output, stage, trace, report, config)
        print(f"{stage}: baseline={baseline_cost:.6g} candidate={candidate_cost:.6g} "
              f"accepted={accepted}")
    joint_cost, joint_metrics, joint_cases = evaluate(
        simulator, config, stage_cases("joint_validation"), args.joint_duration,
        strict_health=True)
    joint_valid = (np.isfinite(joint_cost) and joint_cost < FAILURE_BASE
                   and joint_metrics["failed_case_count"] == 0)
    joint = {"stage": "joint_validation", "joint_cost": joint_cost,
             "joint_valid": bool(joint_valid), **joint_metrics,
             "cases": [{"name": case["name"], "complete": case["complete"],
                        "cost": case["cost"],
                        "health_failure_reasons": case["health_failure_reasons"]}
                       for case in joint_cases]}
    (output / "joint_validation.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
    candidate = mark_candidate(prepare_config(config))
    save_config(candidate, output / "candidate_config.yaml")
    all_stages_accepted = all(report["stage_accepted"] for report in stage_reports)
    eligible = acceptance_eligible(stage_reports, joint_cost, joint_metrics)
    manifest = {"all_stages_accepted": all_stages_accepted,
                "joint_valid": bool(joint_valid), "acceptance_eligible": bool(eligible),
                "stage_reports": stage_reports}
    (output / "acceptance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.accept_candidate:
        require_acceptance_eligible(stage_reports, joint_cost, joint_metrics)
        save_config(candidate, args.config)
        print(f"accepted candidate into {args.config}")
    print(f"joint cost={joint_cost:.6g}, valid={joint_valid}; outputs={output}")


if __name__ == "__main__":
    main()
