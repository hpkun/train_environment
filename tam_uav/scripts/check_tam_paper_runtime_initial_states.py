"""Check nominal Table 5-7 states against the reset JSBSim runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import SCENARIOS, make_paper_env
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION, PAPER_NOMINAL_PROTOCOL)


TOLERANCE = {"longitude_deg": 1e-5, "latitude_deg": 1e-5,
             "altitude_m": 2.0, "speed_mps": 2.0, "heading_deg": 1.0}


def wrapped_heading_error_deg(actual, expected):
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def check_scenario(scenario, seed=2026):
    config_path = ROOT / "uav_env" / "JSBSim" / "configs" / SCENARIOS[scenario]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured = {
        item["id"]: (item, side)
        for side, entries in (("red", config["red_agents"]),
                              ("blue", config["blue_agents"]))
        for item in entries
    }
    env = make_paper_env(
        ROOT, scenario, initial_perturbation=NOMINAL_PERTURBATION,
        dynamics_backend="jsbsim", experiment_protocol=PAPER_NOMINAL_PROTOCOL)
    try:
        env.reset(seed=seed)
        rows = []
        for agent in env.task.agents:
            expected, expected_side = configured[agent.agent_id]
            expected_type = str(expected["type"])
            expected_model = str(config["aircraft_type_params"][expected_type]["aircraft_model"])
            runtime = {
                "longitude_deg": agent._get_property("position/long-gc-deg"),
                "latitude_deg": agent._get_property("position/lat-geod-deg"),
                "altitude_m": agent._get_property("position/h-sl-ft") * agent.FT2M,
                "speed_mps": agent.speed,
                "heading_deg": float(np.rad2deg(agent.heading) % 360.0),
            }
            expected_state = {
                "longitude_deg": float(expected["lon_deg"]),
                "latitude_deg": float(expected["lat_deg"]),
                "altitude_m": float(expected["altitude_m"]),
                "speed_mps": float(expected["speed_mps"]),
                "heading_deg": float(expected["heading_deg"]),
            }
            error = {key: abs(runtime[key] - expected_state[key])
                     for key in runtime if key != "heading_deg"}
            error["heading_deg"] = wrapped_heading_error_deg(
                runtime["heading_deg"], expected_state["heading_deg"])
            rows.append({
                "agent_id": agent.agent_id,
                "configured_side": expected_side, "runtime_side": agent.side,
                "configured_role": expected["role"],
                "runtime_role": agent.aircraft_type.role,
                "configured_aircraft_type": expected_type,
                "runtime_aircraft_type": agent.aircraft_type.name,
                "configured_aircraft_model": expected_model,
                "runtime_aircraft_model": agent.aircraft_type.aircraft_model,
                "alive": agent.alive,
                "missile_count": agent.missile_left,
                "configured_initial_state": expected_state,
                "runtime_initial_state": runtime,
                "absolute_error": error,
                "within_tolerance": (
                    all(error[key] <= TOLERANCE[key] for key in error)
                    and agent.side == expected_side
                    and agent.aircraft_type.role == expected["role"]
                    and agent.aircraft_type.name == expected_type
                    and agent.aircraft_type.aircraft_model == expected_model
                    and agent.alive
                    and agent.missile_left == int(
                        config["aircraft_type_params"][expected_type]["missile_num"])),
            })
        max_error = {key: max(row["absolute_error"][key] for row in rows)
                     for key in TOLERANCE}
        return {
            "scenario": scenario, "dynamics_backend": "jsbsim",
            "initial_perturbation": NOMINAL_PERTURBATION,
            "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
            "tolerance": TOLERANCE, "agents": rows, "maximum_absolute_error": max_error,
            "within_tolerance": all(row["within_tolerance"] for row in rows),
        }
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("2v2", "3v2", "5v4", "all"), default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="outputs/tam_paper_runtime_initial_states.json")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    result = {scenario: check_scenario(scenario, args.seed) for scenario in scenarios}
    output = resolve_tam_output(ROOT, args.output, create_parent=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
