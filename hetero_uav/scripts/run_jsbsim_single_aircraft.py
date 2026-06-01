from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.core.aircraft import JSBSimAircraftPlatform
from uav_env.JSBSim.core.aircraft_types import AircraftType
from uav_env.JSBSim.core.utils import load_yaml


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _type_for_model(config: dict, model: str) -> AircraftType:
    for name, raw in config.get("aircraft_type_params", {}).items():
        if raw.get("aircraft_model") == model:
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
            )
    raise ValueError(f"model {model!r} is not declared in aircraft_type_params")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A-4", "F-16"], required=True)
    parser.add_argument("--duration", "--seconds", dest="duration", type=float, default=10.0)
    parser.add_argument("--action", nargs=3, type=float, default=[0.0, 0.0, 0.5],
                        metavar=("PITCH", "HEADING", "SPEED"))
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--config", default="uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    args = parser.parse_args()

    if importlib.util.find_spec("jsbsim") is None:
        print("JSBSim single-aircraft run skipped: Python package 'jsbsim' is not installed.")
        print("install hint: pip install -r requirements.txt")
        print("install hint: pip install jsbsim==1.1.6")
        return

    config = load_yaml(str(_resolve(args.config)))
    aircraft_type = _type_for_model(config, args.model)
    model_root = _resolve(config.get("jsbsim_model_root", "uav_env/JSBSim/models"))
    sim_hz = int(config.get("simulation_frequency", 60))

    aircraft = JSBSimAircraftPlatform(
        "test_0",
        "red",
        aircraft_type,
        np.array([0.0, 0.0, 6000.0], dtype=np.float32),
        np.array([250.0, 0.0, 0.0], dtype=np.float32),
        0.0,
        model_root=str(model_root),
        model_name=args.model,
        reference_lat=float(config.get("reference_lat", 60.0)),
        reference_lon=float(config.get("reference_lon", 120.0)),
        reference_alt=float(config.get("reference_alt", 0.0)),
        simulation_frequency=sim_hz,
    )

    dt = 1.0 / 5.0
    steps = int(round(args.duration / dt))
    print_every = max(1, int(round(args.print_interval / dt)))
    action = np.array(args.action, dtype=np.float32)
    print(
        "time, local_position_m, altitude_m, speed_mps, heading_rad, "
        "pitch_rad, roll_rad, alive, crashed"
    )
    for step in range(steps + 1):
        sim_time = step * dt
        if step % print_every == 0 or step == steps:
            print(
                f"{sim_time:.1f}, {aircraft.position.tolist()}, {aircraft.position[2]:.3f}, "
                f"{aircraft.speed:.3f}, {aircraft.heading:.6f}, {aircraft.pitch:.6f}, "
                f"{aircraft.roll:.6f}, {aircraft.alive}, {aircraft.crashed}"
            )
        if step == steps:
            break
        aircraft.step(action, dt, tuple(config.get("speed_range", [102.0, 408.0])))

    print("summary:")
    print(f"model: {args.model}")
    print(f"duration: {args.duration}")
    print(f"action: {action.tolist()}")
    print(f"alive: {aircraft.alive}")
    print(f"crashed: {aircraft.crashed}")
    print(f"position_m: {aircraft.position.tolist()}")
    print(f"velocity_mps: {aircraft.velocity.tolist()}")
    print(f"speed_mps: {aircraft.speed:.3f}")
    print(f"heading_rad: {aircraft.heading:.6f}")
    print(f"pitch_rad: {aircraft.pitch:.6f}")
    print(f"roll_rad: {aircraft.roll:.6f}")
    aircraft.close()


if __name__ == "__main__":
    main()
