from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.core.aircraft import JSBSimAircraftPlatform
from uav_env.JSBSim.core.aircraft_types import AircraftType
from uav_env.JSBSim.core.utils import load_yaml, wrap_pi

SCENARIOS = {
    "level": np.array([0.0, 0.0, 0.5], dtype=np.float32),
    "climb": np.array([0.2, 0.0, 0.5], dtype=np.float32),
    "descend": np.array([-0.2, 0.0, 0.5], dtype=np.float32),
    "turn_left": np.array([0.0, -0.5, 0.5], dtype=np.float32),
    "turn_right": np.array([0.0, 0.5, 0.5], dtype=np.float32),
    "speed_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "slow_down": np.array([0.0, 0.0, -0.2], dtype=np.float32),
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _aircraft_type(config: dict, model: str) -> AircraftType:
    for name, raw in config.get("aircraft_type_params", {}).items():
        if raw.get("aircraft_model") == model:
            raw_control = raw.get("control", {})
            return AircraftType(
                name=name,
                aircraft_model=str(raw.get("aircraft_model", model)),
                model_path=str(raw.get("model_path", "")),
                role=str(raw.get("role", name)),
                radar_range=float(raw.get("radar_range", 90000.0)),
                missile_num=int(raw.get("missile_num", 2)),
                max_speed_scale=float(raw.get("max_speed_scale", 1.0)),
                max_g=float(raw.get("max_g", 9.0)),
                reward_role=str(raw.get("reward_role", raw.get("role", name))),
                control={
                    "elevator_sign": float(raw_control.get("elevator_sign", 1.0)),
                    "aileron_sign": float(raw_control.get("aileron_sign", 1.0)),
                    "rudder_sign": float(raw_control.get("rudder_sign", 1.0)),
                    "throttle_sign": float(raw_control.get("throttle_sign", 1.0)),
                    "heading_sign": float(raw_control.get("heading_sign", 1.0)),
                },
            )
    raise ValueError(f"model {model!r} is not declared in aircraft_type_params")


def run_diagnostic(model: str, scenario: str, action: np.ndarray, duration: float,
                   config_path: str = "uav_env/configs/hetero_train_2v2_mav_attack.yaml") -> dict:
    config = load_yaml(str(_resolve(config_path)))
    model_root = _resolve(config.get("jsbsim_model_root", "uav_env/JSBSim/models"))
    platform = JSBSimAircraftPlatform(
        "diag_0",
        "red",
        _aircraft_type(config, model),
        np.array([0.0, 0.0, 6000.0], dtype=np.float32),
        np.array([250.0, 0.0, 0.0], dtype=np.float32),
        0.0,
        model_root=str(model_root),
        model_name=model,
        reference_lat=float(config.get("reference_lat", 60.0)),
        reference_lon=float(config.get("reference_lon", 120.0)),
        reference_alt=float(config.get("reference_alt", 0.0)),
        simulation_frequency=int(config.get("simulation_frequency", 60)),
    )
    dt = 1.0 / float(config.get("decision_frequency", 5))
    steps = max(1, int(round(duration / dt)))
    speed_range = tuple(config.get("speed_range", [102.0, 408.0]))

    initial_altitude = float(platform.position[2])
    initial_speed = float(platform.speed)
    initial_heading = float(platform.heading)
    min_altitude = initial_altitude
    max_altitude = initial_altitude
    max_abs_pitch = abs(float(platform.pitch))
    max_abs_roll = abs(float(platform.roll))
    nan_detected = False

    for _ in range(steps):
        platform.step(action, dt, speed_range)
        values = np.concatenate([
            platform.position.astype(np.float64),
            platform.velocity.astype(np.float64),
            np.array([platform.pitch, platform.roll, platform.heading, platform.speed], dtype=np.float64),
        ])
        nan_detected = nan_detected or bool(np.isnan(values).any() or np.isinf(values).any())
        min_altitude = min(min_altitude, float(platform.position[2]))
        max_altitude = max(max_altitude, float(platform.position[2]))
        max_abs_pitch = max(max_abs_pitch, abs(float(platform.pitch)))
        max_abs_roll = max(max_abs_roll, abs(float(platform.roll)))
        if not platform.alive:
            break

    final_altitude = float(platform.position[2])
    final_speed = float(platform.speed)
    final_heading = float(platform.heading)
    result = {
        "model": model,
        "scenario": scenario,
        "action": " ".join(f"{x:.3f}" for x in action.tolist()),
        "duration": float(duration),
        "initial_altitude": initial_altitude,
        "final_altitude": final_altitude,
        "altitude_delta": final_altitude - initial_altitude,
        "initial_speed": initial_speed,
        "final_speed": final_speed,
        "speed_delta": final_speed - initial_speed,
        "initial_heading": initial_heading,
        "final_heading": final_heading,
        "heading_delta": wrap_pi(final_heading - initial_heading),
        "max_abs_pitch": max_abs_pitch,
        "max_abs_roll": max_abs_roll,
        "min_altitude": min_altitude,
        "max_altitude": max_altitude,
        "crashed": bool(platform.crashed),
        "alive": bool(platform.alive),
        "nan_detected": nan_detected,
        "expected_heading_sign": "",
        "actual_heading_sign": "",
        "heading_direction_ok": "",
        "pitch_direction_ok": "",
        "speed_direction_ok": "",
    }
    platform.close()
    return result


def _sign(value: float, tol: float = 1e-3) -> int:
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def annotate_direction_checks(rows: list[dict]) -> None:
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["scenario"]] = row

    for scenarios in by_model.values():
        level = scenarios.get("level")
        if level is None:
            continue
        level_alt = level["final_altitude"]
        level_speed = level["final_speed"]
        for name, row in scenarios.items():
            if name == "turn_left":
                row["expected_heading_sign"] = -1
                row["actual_heading_sign"] = _sign(row["heading_delta"])
                row["heading_direction_ok"] = row["actual_heading_sign"] < 0
            elif name == "turn_right":
                row["expected_heading_sign"] = 1
                row["actual_heading_sign"] = _sign(row["heading_delta"])
                row["heading_direction_ok"] = row["actual_heading_sign"] > 0
            if name == "climb":
                row["pitch_direction_ok"] = row["final_altitude"] > level_alt
            elif name == "descend":
                row["pitch_direction_ok"] = row["final_altitude"] < level_alt
            if name == "speed_up":
                row["speed_direction_ok"] = row["final_speed"] >= level_speed - 1e-3
            elif name == "slow_down":
                row["speed_direction_ok"] = row["final_speed"] < level_speed


def _print_result(row: dict) -> None:
    print(
        f"{row['model']} {row['scenario']}: "
        f"alt {row['initial_altitude']:.1f}->{row['final_altitude']:.1f} "
        f"delta={row['altitude_delta']:.1f}, "
        f"speed {row['initial_speed']:.1f}->{row['final_speed']:.1f} "
        f"delta={row['speed_delta']:.1f}, "
        f"heading_delta={row['heading_delta']:.3f}, "
        f"max_abs_pitch={row['max_abs_pitch']:.3f}, "
        f"max_abs_roll={row['max_abs_roll']:.3f}, "
        f"min_alt={row['min_altitude']:.1f}, max_alt={row['max_altitude']:.1f}, "
        f"crashed={row['crashed']}, alive={row['alive']}, nan={row['nan_detected']}, "
        f"expected_heading_sign={row['expected_heading_sign']}, "
        f"actual_heading_sign={row['actual_heading_sign']}, "
        f"heading_direction_ok={row['heading_direction_ok']}, "
        f"pitch_direction_ok={row['pitch_direction_ok']}, "
        f"speed_direction_ok={row['speed_direction_ok']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A-4", "F-16"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--action", nargs=3, type=float)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="level")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    models = ["A-4", "F-16"] if args.all else [args.model or "A-4"]
    if args.all:
        scenario_items = list(SCENARIOS.items())
    else:
        action = np.array(args.action, dtype=np.float32) if args.action else SCENARIOS[args.scenario]
        scenario_items = [(args.scenario, action)]

    rows = []
    for model in models:
        for scenario, action in scenario_items:
            row = run_diagnostic(model, scenario, action, args.duration)
            rows.append(row)
    annotate_direction_checks(rows)
    for row in rows:
        _print_result(row)

    if args.output_csv:
        path = Path(args.output_csv)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote_csv: {path}")


if __name__ == "__main__":
    main()
