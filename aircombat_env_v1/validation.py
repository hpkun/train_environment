"""Shared quick-validation cases and single-case execution."""

from __future__ import annotations

import math

import numpy as np

from .execution import bounded_combined_command, run_command_hold, then_level_command
from .metrics import stable_segment_errors, trajectory_metrics
from .pid import PaperAutopilot


def quick_cases():
    return [
        {"task": "level", "altitude": 6000, "speed": 250},
        {"task": "left_turn", "altitude": 6000, "speed": 250},
        {"task": "right_turn", "altitude": 6000, "speed": 250},
        {"task": "climb_then_level", "altitude": 6000, "speed": 250, "dynamic": True},
        {"task": "descent_then_level", "altitude": 6000, "speed": 250, "dynamic": True},
        {"task": "bounded_combined", "altitude": 6000, "speed": 250, "dynamic": True},
        {"task": "level", "altitude": 5000, "speed": 220},
        {"task": "level", "altitude": 7000, "speed": 300},
    ]


def command_at_decision_step(task, decision, base_speed, duration, decision_hz=5):
    if task == "level":
        return 0.0, 0.0, base_speed
    if task == "left_turn":
        return 0.0, np.deg2rad(-30.0), base_speed
    if task == "right_turn":
        return 0.0, np.deg2rad(30.0), base_speed
    if task == "climb":
        return np.deg2rad(5.0), 0.0, base_speed
    if task == "descent":
        return np.deg2rad(-5.0), 0.0, base_speed
    if task == "accelerate":
        return 0.0, 0.0, base_speed + 30.0
    if task == "decelerate":
        return 0.0, 0.0, max(120.0, base_speed - 30.0)
    if task == "climb_then_level":
        return then_level_command(decision, base_speed, 5.0, decision_hz)
    if task == "descent_then_level":
        return then_level_command(decision, base_speed, -5.0, decision_hz)
    if task == "bounded_combined":
        return bounded_combined_command(
            decision, base_speed, duration, decision_hz, stable_duration=20.0)
    raise ValueError(f"unknown task: {task}")


def failure_reasons(metrics, thresholds, dynamic=False):
    heading = "stable_heading_error_deg" if dynamic else "heading_final_error_deg"
    pitch = "stable_pitch_error_deg" if dynamic else "pitch_final_error_deg"
    speed = "stable_speed_error_mps" if dynamic else "speed_final_error_mps"
    checks = [
        (metrics["has_nan_or_inf"], "nan_or_inf"),
        (metrics["crashed"], "crash_or_simulation_failure"),
        (metrics["maximum_alpha_deg"] > thresholds["maximum_alpha_deg"], "maximum_alpha"),
        (metrics["maximum_beta_deg"] > thresholds["maximum_beta_deg"], "maximum_beta"),
        (metrics["maximum_load_factor"] > thresholds["maximum_load_factor"],
         "maximum_load_factor"),
        (metrics["aileron_saturation_ratio"] > thresholds["aileron_saturation_ratio"],
         "aileron_saturation"),
        (metrics["elevator_saturation_ratio"] > thresholds["elevator_saturation_ratio"],
         "elevator_saturation"),
        (metrics["throttle_saturation_ratio"] > thresholds["throttle_saturation_ratio"],
         "throttle_saturation"),
        (metrics[heading] > thresholds["heading_final_error_deg"],
         "heading_final_error"),
        (metrics[pitch] > thresholds["pitch_final_error_deg"], "pitch_final_error"),
        (metrics[speed] > thresholds["speed_final_error_mps"], "speed_final_error"),
    ]
    return [reason for failed, reason in checks if failed]


def run_validation_case(simulator, config, case, duration):
    controller = PaperAutopilot.from_config(config)
    altitude, speed, task = case["altitude"], case["speed"], case["task"]
    simulator.reset(altitude_m=altitude, speed_mps=speed,
                    elevator_trim=config["trim"]["elevator_trim"],
                    throttle_base=config["trim"]["throttle_base"])
    decision_hz = config["timing"]["decision_frequency_hz"]
    hold = config["timing"]["physics_steps_per_decision"]
    rows, failure = [], None
    for decision in range(max(1, math.ceil(duration * decision_hz))):
        target = command_at_decision_step(task, decision, speed, duration, decision_hz)
        held, failure = run_command_hold(simulator, controller, target, hold)
        rows.extend(held)
        if failure:
            break
    metrics = trajectory_metrics(rows, config, altitude, failure)
    dynamic = bool(case.get("dynamic"))
    if dynamic:
        metrics.update(stable_segment_errors(
            rows, min(10.0, duration), config["timing"]["sim_frequency_hz"]))
    reasons = failure_reasons(metrics, config["validation_thresholds"], dynamic)
    metrics.update(passed=not reasons, failure_reasons=reasons,
                   completed=failure is None)
    return rows, metrics
