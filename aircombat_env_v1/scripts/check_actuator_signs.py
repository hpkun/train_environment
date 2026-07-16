"""Measure paired open-loop actuator responses without changing configuration."""

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
from aircombat_env_v1.config import load_config


FIELDS = ["case", "phase", "time", "aileron", "elevator", "rudder", "throttle",
          "roll", "pitch", "heading", "p", "q", "r", "altitude",
          "true_airspeed", "alpha", "beta", "load_factor"]


def reset_kwargs(config):
    state = config["initial_state"]
    return {"altitude_m": state["altitude_m"], "speed_mps": state["speed_mps"],
            "heading_deg": state["heading_deg"], "roll_deg": state["roll_deg"],
            "pitch_deg": state["pitch_deg"],
            "elevator_trim": config["trim"]["elevator_trim"],
            "throttle_base": config["trim"]["throttle_base"]}


def record_row(name, phase, state, controls):
    row = {field: state.get(field) for field in FIELDS}
    row.update(case=name, phase=phase, aileron=controls[0], elevator=controls[1],
               rudder=controls[2], throttle=controls[3])
    return row


def run_case(simulator, config, name, pulse_controls, stabilization_duration,
             pulse_duration, response_key):
    simulator.reset(**reset_kwargs(config))
    trim_controls = (0.0, config["trim"]["elevator_trim"], 0.0,
                     config["trim"]["throttle_base"])
    rows = []
    simulator.set_controls(*trim_controls)
    for _ in range(round(stabilization_duration / simulator.dt)):
        rows.append(record_row(name, "stabilization", simulator.run(), trim_controls))
    start = simulator.state()
    simulator.set_controls(*pulse_controls)
    early_state = start
    early_steps = max(1, round(0.2 / simulator.dt))
    for index in range(round(pulse_duration / simulator.dt)):
        state = simulator.run()
        rows.append(record_row(name, "pulse", state, pulse_controls))
        if index + 1 == early_steps:
            early_state = state
    end = simulator.state()
    response_end = early_state if response_key in ("p", "q") else end
    return rows, start, end, response_end[response_key] - start[response_key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stabilization-duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stabilization_duration < 0.0:
        parser.error("stabilization-duration must be non-negative")
    config = load_config()
    elevator, throttle = config["trim"]["elevator_trim"], config["trim"]["throttle_base"]
    definitions = {
        "aileron": ((0.05, elevator, 0.0, throttle),
                    (-0.05, elevator, 0.0, throttle), 0.5, "p"),
        "elevator": ((0.0, elevator + 0.02, 0.0, throttle),
                     (0.0, elevator - 0.02, 0.0, throttle), 0.5, "q"),
        "throttle": ((0.0, elevator, 0.0, min(1.0, throttle + 0.05)),
                     (0.0, elevator, 0.0, max(0.0, throttle - 0.05)),
                     5.0, "true_airspeed"),
    }
    output = args.output or (ROOT / "aircombat_env_v1" / "outputs" /
                             f"actuator_signs_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    simulator = AircraftSimulator(config["timing"]["sim_frequency_hz"])
    all_rows, summary = [], []
    for actuator, (plus_controls, minus_controls, duration, response) in definitions.items():
        cases = {}
        for sign, controls in (("plus", plus_controls), ("minus", minus_controls)):
            rows, start, end, delta = run_case(
                simulator, config, f"{actuator}_{sign}", controls,
                args.stabilization_duration, duration, response)
            all_rows.extend(rows)
            cases[sign] = {"start": start, "end": end, "delta": delta}
        paired = cases["plus"]["delta"] - cases["minus"]["delta"]
        item = {"actuator": actuator, "response": response,
                "plus_delta": cases["plus"]["delta"],
                "minus_delta": cases["minus"]["delta"],
                "paired_effect": paired,
                "detected_direction": "positive" if paired > 0.0 else "negative",
                "start_state": {key: cases[key]["start"] for key in cases},
                "end_state": {key: cases[key]["end"] for key in cases},
                "stabilization_duration": args.stabilization_duration,
                "pulse_duration": duration}
        summary.append(item)
        print(f"{actuator}: {item['detected_direction']} ({response}, paired={paired:+.6g})")
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
