"""Deterministic two-dimensional elevator/throttle trim search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import DEFAULT_CONFIG, load_config, save_config


LARGE_PENALTY = 1e12


def reset_kwargs(config, elevator, throttle):
    state = config["initial_state"]
    return {
        "altitude_m": state["altitude_m"], "speed_mps": state["speed_mps"],
        "heading_deg": state["heading_deg"], "roll_deg": state["roll_deg"],
        "pitch_deg": state["pitch_deg"], "elevator_trim": elevator,
        "throttle_base": throttle,
    }


def trim_cost(simulator, config, elevator, throttle, duration):
    try:
        initial = simulator.reset(**reset_kwargs(config, elevator, throttle))
        state = initial
        pitch_sq = q_sq = 0.0
        for _ in range(round(duration / simulator.dt)):
            simulator.set_controls(0.0, elevator, 0.0, throttle)
            state = simulator.run()
            values = tuple(state.values())
            if not np.all(np.isfinite(values)):
                return LARGE_PENALTY
            if state["altitude"] < 1000.0 or state["true_airspeed"] < 120.0:
                return LARGE_PENALTY
            pitch_sq += state["pitch"] ** 2 * simulator.dt
            q_sq += state["q"] ** 2 * simulator.dt
        altitude_drift = (state["altitude"] - initial["altitude"]) / 100.0
        speed_drift = (state["true_airspeed"] - initial["true_airspeed"]) / 10.0
        pitch_drift = np.rad2deg(state["pitch"] - initial["pitch"])
        control_cost = 0.2 * elevator ** 2 + 0.02 * throttle ** 2
        return float(altitude_drift ** 2 + speed_drift ** 2
                     + pitch_drift ** 2 + pitch_sq + 10.0 * q_sq + control_cost)
    except (RuntimeError, ValueError, FloatingPointError):
        return LARGE_PENALTY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.duration <= 0 or args.grid_size < 3:
        parser.error("duration must be positive and grid-size at least 3")

    config = load_config(args.config)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    e_bounds, t_bounds = (-0.25, 0.25), (0.05, 0.8)
    best = (LARGE_PENALTY, 0.0, config["trim"]["throttle_base"])
    for _ in range(3):
        elevators = np.linspace(*e_bounds, args.grid_size)
        throttles = np.linspace(*t_bounds, args.grid_size)
        for elevator in elevators:
            for throttle in throttles:
                cost = trim_cost(simulator, config, elevator, throttle, args.duration)
                if cost < best[0]:
                    best = (cost, float(elevator), float(throttle))
                    print(f"cost={best[0]:.6g} elevator={best[1]:+.6f} "
                          f"throttle={best[2]:.6f}")
        e_step = (e_bounds[1] - e_bounds[0]) / (args.grid_size - 1)
        t_step = (t_bounds[1] - t_bounds[0]) / (args.grid_size - 1)
        e_bounds = (max(-1.0, best[1] - e_step), min(1.0, best[1] + e_step))
        t_bounds = (max(0.0, best[2] - t_step), min(1.0, best[2] + t_step))

    if best[0] >= LARGE_PENALTY:
        raise RuntimeError("no finite trim candidate found")
    config["trim"].update(elevator_trim=best[1], throttle_base=best[2])
    save_config(config, args.config)
    print(f"updated {args.config}")


if __name__ == "__main__":
    main()
