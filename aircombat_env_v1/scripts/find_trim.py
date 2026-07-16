"""Deterministic elevator/throttle trim search with auditable outputs."""

from __future__ import annotations

import argparse
import copy
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
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, mark_candidate, save_config


LARGE_PENALTY = 1e12
CANDIDATE_FIELDS = ["refinement_round", "elevator_trim", "throttle_base",
                    "total_cost", "altitude_drift_m", "speed_drift_mps",
                    "pitch_drift_deg", "pitch_rms_deg", "pitch_rate_rms_deg_s",
                    "maximum_alpha_deg", "maximum_load_factor", "valid",
                    "failure_reason", "boundary_hit"]
TRAJECTORY_FIELDS = ["time", "elevator", "throttle", "altitude", "true_airspeed",
                     "roll", "pitch", "heading", "p", "q", "r", "alpha",
                     "beta", "load_factor"]


def reset_kwargs(config, elevator, throttle):
    state = config["initial_state"]
    return {"altitude_m": state["altitude_m"], "speed_mps": state["speed_mps"],
            "heading_deg": state["heading_deg"], "roll_deg": state["roll_deg"],
            "pitch_deg": state["pitch_deg"], "elevator_trim": elevator,
            "throttle_base": throttle}


def evaluate(simulator, config, elevator, throttle, duration):
    rows, failure = [], None
    try:
        initial = simulator.reset(**reset_kwargs(config, elevator, throttle))
        controls = (0.0, elevator, 0.0, throttle)
        simulator.set_controls(*controls)
        for _ in range(round(duration / simulator.dt)):
            state = simulator.run()
            if not np.all(np.isfinite(tuple(state.values()))):
                failure = "nan_or_inf"
                break
            row = dict(state, elevator=elevator, throttle=throttle)
            rows.append(row)
            if state["altitude"] < 1000.0:
                failure = "altitude_below_1000_m"
                break
            if state["true_airspeed"] < 120.0:
                failure = "airspeed_below_120_mps"
                break
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        initial, failure = {"altitude": config["initial_state"]["altitude_m"],
                            "true_airspeed": config["initial_state"]["speed_mps"],
                            "pitch": 0.0}, f"simulation_error:{type(exc).__name__}"
    if not rows or failure:
        metrics = {key: 0.0 for key in CANDIDATE_FIELDS if key not in
                   ("refinement_round", "elevator_trim", "throttle_base",
                    "valid", "failure_reason", "boundary_hit")}
        metrics.update(total_cost=LARGE_PENALTY, valid=False, failure_reason=failure)
        return metrics, rows
    final = rows[-1]
    pitch = np.array([row["pitch"] for row in rows])
    pitch_rate = np.array([row["q"] for row in rows])
    altitude_drift = final["altitude"] - initial["altitude"]
    speed_drift = final["true_airspeed"] - initial["true_airspeed"]
    pitch_drift = np.rad2deg(final["pitch"] - initial["pitch"])
    # Neutral references are elevator=0 and mid-throttle=0.5. This small term
    # selects among dynamically equivalent trim candidates; it is not a paper term.
    neutral_control_cost = 0.2 * elevator ** 2 + 0.02 * (throttle - 0.5) ** 2
    total_cost = ((altitude_drift / 100.0) ** 2 + (speed_drift / 10.0) ** 2
                  + pitch_drift ** 2 + np.mean(np.rad2deg(pitch) ** 2)
                  + 10.0 * np.mean(np.rad2deg(pitch_rate) ** 2)
                  + neutral_control_cost)
    metrics = {
        "total_cost": float(total_cost), "altitude_drift_m": altitude_drift,
        "speed_drift_mps": speed_drift, "pitch_drift_deg": pitch_drift,
        "pitch_rms_deg": float(np.sqrt(np.mean(np.rad2deg(pitch) ** 2))),
        "pitch_rate_rms_deg_s": float(np.sqrt(np.mean(np.rad2deg(pitch_rate) ** 2))),
        "maximum_alpha_deg": float(max(abs(np.rad2deg(r["alpha"])) for r in rows)),
        "maximum_load_factor": float(max(abs(r["load_factor"]) for r in rows)),
        "valid": True, "failure_reason": None}
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.grid_size < 3:
        parser.error("duration must be positive and grid-size at least 3")
    output = (ROOT / "aircombat_env_v1" / "outputs" /
              f"trim_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    e_bounds, t_bounds = (-0.25, 0.25), (0.05, 0.8)
    candidates, best, best_rows = [], None, []
    for refinement in range(1, 4):
        elevators, throttles = np.linspace(*e_bounds, args.grid_size), np.linspace(*t_bounds, args.grid_size)
        for elevator in elevators:
            for throttle in throttles:
                metrics, rows = evaluate(simulator, config, float(elevator), float(throttle), args.duration)
                boundary = bool(np.isclose(elevator, e_bounds).any()
                                or np.isclose(throttle, t_bounds).any())
                item = {"refinement_round": refinement, "elevator_trim": float(elevator),
                        "throttle_base": float(throttle), **metrics, "boundary_hit": boundary}
                candidates.append(item)
                if best is None or item["total_cost"] < best["total_cost"]:
                    best, best_rows = item, rows
                    print(f"cost={best['total_cost']:.6g} elevator={elevator:+.6f} throttle={throttle:.6f}")
        e_step = (e_bounds[1] - e_bounds[0]) / (args.grid_size - 1)
        t_step = (t_bounds[1] - t_bounds[0]) / (args.grid_size - 1)
        e_bounds = (max(-1.0, best["elevator_trim"] - e_step),
                    min(1.0, best["elevator_trim"] + e_step))
        t_bounds = (max(0.0, best["throttle_base"] - t_step),
                    min(1.0, best["throttle_base"] + t_step))
    if best is None or not best["valid"]:
        raise RuntimeError("no valid trim candidate found")
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader(); writer.writerows(candidates)
    (output / "best_candidate.json").write_text(
        json.dumps({**best, "control_penalty_reference":
                    "elevator=0, throttle=0.5 engineering neutral"}, indent=2), encoding="utf-8")
    with (output / "best_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(best_rows)
    candidate_config = mark_candidate(copy.deepcopy(config))
    candidate_config["trim"].update(elevator_trim=best["elevator_trim"],
                                    throttle_base=best["throttle_base"])
    save_config(candidate_config, output / "candidate_config.yaml")
    if best["boundary_hit"]:
        print("WARNING: best candidate touches a search boundary")
    if args.accept:
        save_config(candidate_config, args.config)
        print(f"accepted candidate into {args.config}")
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
