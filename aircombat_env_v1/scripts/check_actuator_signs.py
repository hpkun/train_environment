"""Measure small open-loop F-16 actuator responses without changing config."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.config import load_config


FIELDS = ["case", "time", "aileron", "elevator", "rudder", "throttle",
          "roll", "pitch", "heading", "p", "q", "r", "altitude",
          "true_airspeed", "alpha", "beta", "load_factor"]


def initial_kwargs(config):
    state = config["initial_state"]
    return {
        "altitude_m": state["altitude_m"],
        "speed_mps": state["speed_mps"],
        "heading_deg": state["heading_deg"],
        "roll_deg": state["roll_deg"],
        "pitch_deg": state["pitch_deg"],
        "elevator_trim": config["trim"]["elevator_trim"],
        "throttle_base": config["trim"]["throttle_base"],
    }


def run_pulse(simulator, config, name, controls, duration):
    start = simulator.reset(**initial_kwargs(config))
    simulator.set_controls(*controls)
    rows = []
    steps = round(duration / simulator.dt)
    for _ in range(steps):
        state = simulator.run()
        row = {field: state.get(field) for field in FIELDS}
        row.update({"case": name, "aileron": controls[0],
                    "elevator": controls[1], "rudder": controls[2],
                    "throttle": controls[3]})
        rows.append(row)
    return rows, start, rows[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    base_throttle = config["trim"]["throttle_base"]
    tests = [
        ("aileron_plus", (0.05, 0.0, 0.0, base_throttle), 0.5, "roll"),
        ("aileron_minus", (-0.05, 0.0, 0.0, base_throttle), 0.5, "roll"),
        ("elevator_plus", (0.0, 0.02, 0.0, base_throttle), 0.5, "pitch"),
        ("elevator_minus", (0.0, -0.02, 0.0, base_throttle), 0.5, "pitch"),
        ("throttle_plus", (0.0, 0.0, 0.0, min(1.0, base_throttle + 0.05)),
         5.0, "true_airspeed"),
        ("throttle_minus", (0.0, 0.0, 0.0, max(0.0, base_throttle - 0.05)),
         5.0, "true_airspeed"),
    ]
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    all_rows = []
    deltas = {}
    for name, controls, duration, response in tests:
        rows, start, end = run_pulse(simulator, config, name, controls, duration)
        all_rows.extend(rows)
        deltas[name] = end[response] - start[response]

    output = args.output or (ROOT / "aircombat_env_v1" / "outputs" /
                             f"actuator_signs_{datetime.now():%Y%m%d_%H%M%S}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    for actuator in ("aileron", "elevator", "throttle"):
        plus, minus = deltas[f"{actuator}_plus"], deltas[f"{actuator}_minus"]
        paired_effect = plus - minus
        print(f"{actuator}: +pulse delta={plus:+.6g}, -pulse delta={minus:+.6g}; "
              f"paired +command direction="
              f"{'positive' if paired_effect > 0 else 'negative'}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
