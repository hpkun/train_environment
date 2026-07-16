"""Sequential offline tuning: roll PD, pitch PD, speed PI, then small I terms."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, save_config
from aircombat_env_v1.geometry import paper_direction_errors
from aircombat_env_v1.pid import PaperAutopilot


SEED = 20260316
LARGE_PENALTY = 1e12


def reset_simulator(simulator, config):
    state = config["initial_state"]
    return simulator.reset(
        altitude_m=state["altitude_m"], speed_mps=state["speed_mps"],
        heading_deg=state["heading_deg"], roll_deg=state["roll_deg"],
        pitch_deg=state["pitch_deg"],
        elevator_trim=config["trim"]["elevator_trim"],
        throttle_base=config["trim"]["throttle_base"])


def stage_cases(stage):
    if stage == "roll_pd":
        return [(0.0, np.deg2rad(value), 250.0)
                for value in (-60, -30, -10, 10, 30, 60)]
    if stage == "pitch_pd":
        return [(np.deg2rad(value), 0.0, 250.0)
                for value in (-20, -10, -5, 5, 10, 20)]
    if stage == "speed_pi":
        return [(0.0, 0.0, float(value)) for value in (220, 280, 300, 330)]
    return (stage_cases("roll_pd") + stage_cases("pitch_pd")
            + stage_cases("speed_pi"))


def apply_parameters(config, stage, values):
    tuned = copy.deepcopy(config)
    if stage == "roll_pd":
        tuned["pid"]["roll"].update(kp=float(values[0]), kd=float(values[1]))
    elif stage == "pitch_pd":
        tuned["pid"]["pitch"].update(kp=float(values[0]), kd=float(values[1]))
    elif stage == "speed_pi":
        tuned["pid"]["speed"].update(kp=float(values[0]), ki=float(values[1]))
    elif stage == "small_integral":
        tuned["pid"]["roll"]["ki"] = float(values[0])
        tuned["pid"]["pitch"]["ki"] = float(values[1])
    return tuned


def evaluate(simulator, config, cases, duration):
    total_cost = 0.0
    for target_pitch, target_heading, target_speed in cases:
        controller = PaperAutopilot.from_config(config)
        initial = reset_simulator(simulator, config)
        previous_error = None
        overshoot = saturation = error_integral = control_energy = 0.0
        try:
            for _ in range(round(duration / simulator.dt)):
                state = simulator.state()
                controls = controller.step(
                    state["roll"], state["pitch"], state["heading"],
                    state["true_airspeed"], target_pitch, target_heading,
                    target_speed, simulator.dt)
                simulator.set_controls(*controls)
                state = simulator.run()
                numeric = tuple(state.values()) + tuple(controls)
                if (not np.all(np.isfinite(numeric)) or state["altitude"] < 1000.0
                        or state["true_airspeed"] < 120.0):
                    return LARGE_PENALTY
                roll_error, pitch_error = paper_direction_errors(
                    state["roll"], state["pitch"], state["heading"],
                    target_pitch, target_heading)
                speed_error = (target_speed - state["true_airspeed"]) / 50.0
                errors = np.array([roll_error, pitch_error, speed_error])
                error_integral += float(errors @ errors) * simulator.dt
                control_energy += float(np.square(controls).sum()) * simulator.dt
                saturation += float(abs(controls[0]) > 0.999
                                    or abs(controls[1]) > 0.999
                                    or controls[3] < 0.001 or controls[3] > 0.999)
                if previous_error is not None:
                    overshoot += float(np.sum(np.maximum(
                        0.0, -(errors * previous_error))))
                previous_error = errors
        except (RuntimeError, ValueError, FloatingPointError):
            return LARGE_PENALTY
        altitude_loss = max(0.0, initial["altitude"] - state["altitude"]) / 100.0
        saturation_ratio = saturation / max(1, round(duration / simulator.dt))
        total_cost += (error_integral + 0.02 * control_energy
                       + 5.0 * saturation_ratio + 2.0 * overshoot
                       + altitude_loss ** 2)
    return float(total_cost / len(cases))


def write_stage(output_dir, stage, trace, result, config):
    csv_path = output_dir / f"{stage}.csv"
    names = [f"x{index}" for index in range(len(trace[0]["values"]))]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evaluation", "cost", *names])
        writer.writeheader()
        for index, item in enumerate(trace):
            row = {"evaluation": index, "cost": item["cost"]}
            row.update({name: value for name, value in zip(names, item["values"])})
            writer.writerow(row)
    payload = {
        "stage": stage, "seed": SEED, "best_cost": float(result.fun),
        "best_values": [float(value) for value in result.x],
        "success": bool(result.success), "message": str(result.message),
        "pid": config["pid"],
    }
    (output_dir / f"{stage}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--popsize", type=int, default=6)
    args = parser.parse_args()
    if args.duration <= 0 or args.maxiter < 1 or args.popsize < 2:
        parser.error("duration/maxiter/popsize are out of range")

    output_dir = (ROOT / "aircombat_env_v1" / "outputs" /
                  f"tuning_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    stages = [
        ("roll_pd", [(0.05, 3.0), (0.0, 0.5)]),
        ("pitch_pd", [(0.05, 3.0), (0.0, 0.5)]),
        ("speed_pi", [(0.0001, 0.05), (0.0, 0.01)]),
        ("small_integral", [(0.0, 0.08), (0.0, 0.08)]),
    ]
    for stage, bounds in stages:
        trace = []

        def objective(values):
            candidate = apply_parameters(config, stage, values)
            cost = evaluate(simulator, candidate, stage_cases(stage), args.duration)
            trace.append({"values": [float(value) for value in values], "cost": cost})
            return cost

        result = differential_evolution(
            objective, bounds, seed=SEED, maxiter=args.maxiter,
            popsize=args.popsize, polish=False, workers=1, updating="immediate")
        config = apply_parameters(config, stage, result.x)
        write_stage(output_dir, stage, trace, result, config)
        print(f"{stage}: cost={result.fun:.6g}, values={result.x}")

    joint_cost = evaluate(simulator, config, stage_cases("joint"), args.duration)
    joint = {"stage": "joint_validation", "seed": SEED,
             "cost": joint_cost, "pid": config["pid"]}
    (output_dir / "joint_validation.json").write_text(
        json.dumps(joint, indent=2), encoding="utf-8")
    with (output_dir / "joint_validation.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cost"])
        writer.writeheader()
        writer.writerow({"cost": joint_cost})
    config["initial_guess_only"] = False
    config["paper_reported"] = False
    save_config(config, args.config)
    print(f"joint cost={joint_cost:.6g}; outputs={output_dir}")


if __name__ == "__main__":
    main()
