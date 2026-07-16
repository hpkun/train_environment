"""Common trajectory and step-response metrics for tuning and validation."""

from __future__ import annotations

import numpy as np

from .geometry import in_range_rad, paper_direction_errors


ROLL_SCALE = np.deg2rad(30.0)
PITCH_SCALE = np.deg2rad(10.0)
SPEED_SCALE = 50.0


def trajectory_metrics(rows, config, initial_altitude, failure_reason=None):
    dt = 1.0 / config["timing"]["sim_frequency_hz"]
    trim = config["trim"]
    totals = {"roll_error_integral": 0.0, "pitch_error_integral": 0.0,
              "speed_error_integral": 0.0, "control_increment_energy": 0.0,
              "control_rate_energy": 0.0}
    saturation = {"aileron": 0, "elevator": 0, "throttle": 0}
    previous_controls = None
    heading_errors, pitch_errors, speed_errors = [], [], []
    for row in rows:
        e_roll, e_pitch = paper_direction_errors(
            row["roll"], row["pitch"], row["heading"],
            row["target_pitch"], row["target_heading"])
        e_speed = row["target_true_airspeed"] - row["true_airspeed"]
        heading_errors.append(abs(np.rad2deg(in_range_rad(
            row["target_heading"] - row["heading"]))))
        pitch_errors.append(abs(np.rad2deg(row["target_pitch"] - row["pitch"])))
        speed_errors.append(abs(e_speed))
        totals["roll_error_integral"] += abs(e_roll) / ROLL_SCALE * dt
        totals["pitch_error_integral"] += abs(e_pitch) / PITCH_SCALE * dt
        totals["speed_error_integral"] += abs(e_speed) / SPEED_SCALE * dt
        increments = (row["aileron"], row["elevator"] - trim["elevator_trim"],
                      row["rudder"], row["throttle"] - trim["throttle_base"])
        totals["control_increment_energy"] += float(np.dot(increments, increments)) * dt
        controls = np.array((row["aileron"], row["elevator"], row["throttle"]))
        if previous_controls is not None:
            delta = controls - previous_controls
            totals["control_rate_energy"] += float(delta @ delta)
        previous_controls = controls
        saturation["aileron"] += abs(row["aileron"]) >= 0.999
        saturation["elevator"] += abs(row["elevator"]) >= 0.999
        saturation["throttle"] += row["throttle"] <= 0.001 or row["throttle"] >= 0.999
    count = max(1, len(rows))
    totals.update({
        "aileron_saturation_ratio": saturation["aileron"] / count,
        "elevator_saturation_ratio": saturation["elevator"] / count,
        "throttle_saturation_ratio": saturation["throttle"] / count,
        "altitude_loss_m": max(0.0, initial_altitude
                               - min((r["altitude"] for r in rows), default=initial_altitude)),
        "maximum_alpha_deg": max((abs(np.rad2deg(r["alpha"])) for r in rows), default=0.0),
        "maximum_beta_deg": max((abs(np.rad2deg(r["beta"])) for r in rows), default=0.0),
        "maximum_load_factor": max((abs(r["load_factor"]) for r in rows), default=0.0),
        "crashed": failure_reason not in (None, "nan_or_inf"),
        "has_nan_or_inf": failure_reason == "nan_or_inf",
        "failure_reason": failure_reason,
        "heading_error_rms_deg": float(np.sqrt(np.mean(np.square(heading_errors))))
        if heading_errors else float("inf"),
        "pitch_error_rms_deg": float(np.sqrt(np.mean(np.square(pitch_errors))))
        if pitch_errors else float("inf"),
        "speed_error_rms_mps": float(np.sqrt(np.mean(np.square(speed_errors))))
        if speed_errors else float("inf"),
        "heading_error_p95_deg": float(np.percentile(heading_errors, 95))
        if heading_errors else float("inf"),
        "pitch_error_p95_deg": float(np.percentile(pitch_errors, 95))
        if pitch_errors else float("inf"),
        "speed_error_p95_mps": float(np.percentile(speed_errors, 95))
        if speed_errors else float("inf"),
    })
    if rows:
        last = rows[-1]
        totals["heading_final_error_deg"] = abs(np.rad2deg(in_range_rad(
            last["target_heading"] - last["heading"])))
        totals["pitch_final_error_deg"] = abs(np.rad2deg(
            last["target_pitch"] - last["pitch"]))
        totals["speed_final_error_mps"] = abs(
            last["target_true_airspeed"] - last["true_airspeed"])
    else:
        totals.update(heading_final_error_deg=float("inf"),
                      pitch_final_error_deg=float("inf"),
                      speed_final_error_mps=float("inf"))
    return totals


MEAN_METRICS = ("roll_error_integral", "pitch_error_integral",
                "speed_error_integral", "control_increment_energy",
                "control_rate_energy")
MAX_METRICS = ("aileron_saturation_ratio", "elevator_saturation_ratio",
               "throttle_saturation_ratio", "altitude_loss_m",
               "maximum_alpha_deg", "maximum_beta_deg", "maximum_load_factor",
               "heading_final_error_deg", "pitch_final_error_deg",
               "speed_final_error_mps", "heading_error_rms_deg",
               "pitch_error_rms_deg", "speed_error_rms_mps",
               "heading_error_p95_deg", "pitch_error_p95_deg",
               "speed_error_p95_mps")


def aggregate_case_metrics(case_results):
    """Average performance metrics and keep worst-case safety metrics."""
    aggregate = {}
    for key in MEAN_METRICS:
        aggregate[key] = float(np.mean([case["metrics"][key] for case in case_results]))
    for key in MAX_METRICS:
        aggregate[key] = float(np.max([case["metrics"][key] for case in case_results]))
    failed = [case["name"] for case in case_results if not case["complete"]]
    aggregate.update(failed_case_count=len(failed), failed_cases=failed,
                     crashed=any(case["metrics"]["crashed"] for case in case_results),
                     has_nan_or_inf=any(case["metrics"]["has_nan_or_inf"]
                                        for case in case_results))
    return aggregate


def stable_segment_errors(rows, duration, sim_frequency_hz):
    """Return worst errors in the final commanded-stable segment."""
    tail = rows[-max(1, round(duration * sim_frequency_hz)):]
    if not tail:
        return {"stable_heading_error_deg": float("inf"),
                "stable_pitch_error_deg": float("inf"),
                "stable_speed_error_mps": float("inf")}
    return {
        "stable_heading_error_deg": max(abs(np.rad2deg(in_range_rad(
            row["target_heading"] - row["heading"]))) for row in tail),
        "stable_pitch_error_deg": max(abs(np.rad2deg(
            row["target_pitch"] - row["pitch"])) for row in tail),
        "stable_speed_error_mps": max(abs(
            row["target_true_airspeed"] - row["true_airspeed"]) for row in tail),
    }


def _first_step(rows, target_key, actual_key, initial_value, circular=False):
    def difference(a, b):
        return in_range_rad(a - b) if circular else a - b
    start = next((i for i, row in enumerate(rows)
                  if abs(difference(row[target_key], initial_value)) > 1e-9), None)
    if start is None:
        return None
    target = rows[start][target_key]
    end = next((i for i in range(start + 1, len(rows))
                if abs(difference(rows[i][target_key], target)) > 1e-9), len(rows))
    return start, end, target, difference(target, initial_value)


def _step_metric(rows, target_key, actual_key, initial_value, tolerance, circular=False):
    step = _first_step(rows, target_key, actual_key, initial_value, circular)
    if step is None:
        return None
    start, end, target, command_change = step
    direction = 1.0 if command_change > 0 else -1.0
    segment = rows[start:end]
    progress = [direction * ((in_range_rad(r[actual_key] - initial_value))
                             if circular else r[actual_key] - initial_value)
                for r in segment]
    target_distance = abs(command_change)
    reached_index = next((i for i, value in enumerate(progress)
                          if value >= target_distance), None)
    overshoot = None
    peak = None
    if reached_index is not None:
        peak_progress = max(progress[reached_index:])
        peak = initial_value + direction * peak_progress
        overshoot = max(0.0, peak_progress - target_distance)
    errors = [abs((in_range_rad(target - r[actual_key])) if circular
                  else target - r[actual_key]) for r in segment]
    settling = None
    for index in range(len(errors)):
        if max(errors[index:], default=float("inf")) <= tolerance:
            settling = segment[index]["time"] - rows[start]["time"]
            break
    return {"command_step_time_s": rows[start]["time"], "initial_value": initial_value,
            "target_value": target, "reached_target": reached_index is not None,
            "peak_after_reaching": peak, "overshoot": overshoot,
            "settling_time_s": settling}


def step_response_metrics(rows, initial_state):
    specs = {
        "heading": ("target_heading", "heading", initial_state["heading"],
                    np.deg2rad(5.0), True, 180.0 / np.pi),
        "pitch": ("target_pitch", "pitch", initial_state["pitch"],
                  np.deg2rad(3.0), False, 180.0 / np.pi),
        "speed": ("target_true_airspeed", "true_airspeed",
                  initial_state["true_airspeed"], 10.0, False, 1.0),
    }
    result = {}
    for name, (target, actual, initial, tolerance, circular, scale) in specs.items():
        metric = _step_metric(rows, target, actual, initial, tolerance, circular)
        result[f"{name}_command_step_time_s"] = None if metric is None else metric["command_step_time_s"]
        result[f"{name}_reached_target"] = None if metric is None else metric["reached_target"]
        for source, suffix in (("initial_value", "initial_value"),
                               ("target_value", "target_value"),
                               ("peak_after_reaching", "peak_after_reaching"),
                               ("overshoot", "overshoot")):
            value = None if metric is None else metric[source]
            result[f"{name}_{suffix}_{'deg' if name != 'speed' else 'mps'}"] = (
                None if value is None else value * scale)
        result[f"{name}_settling_time_s"] = None if metric is None else metric["settling_time_s"]
    return result
