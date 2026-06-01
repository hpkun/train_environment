from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.core.aircraft import SimpleKinematicAircraftPlatform
from uav_env.JSBSim.core.aircraft_types import AircraftType
from uav_env.JSBSim.core.original_missile import MissileSimulator


def _type(name: str) -> AircraftType:
    return AircraftType(
        name=name,
        aircraft_model="F-16",
        model_path="",
        role=name,
        radar_range=90000.0,
        missile_num=2,
        max_speed_scale=1.0,
        max_g=9.0,
        reward_role="attack",
    )


def main() -> None:
    shooter = SimpleKinematicAircraftPlatform(
        "shooter", "red", _type("attack_uav"),
        np.array([0.0, 0.0, 6000.0], dtype=np.float32),
        np.array([250.0, 0.0, 0.0], dtype=np.float32),
        0.0,
    )
    target = SimpleKinematicAircraftPlatform(
        "target", "blue", _type("attack_uav"),
        np.array([2500.0, 0.0, 6000.0], dtype=np.float32),
        np.array([180.0, 0.0, 0.0], dtype=np.float32),
        0.0,
    )
    shooter.reset_runtime()
    target.reset_runtime()
    missile = MissileSimulator.create(shooter, target, "M0001", dt=1 / 60)
    distances = []
    print("step,status,distance,speed,target_alive")
    for step in range(1, 3601):
        missile.run()
        distance = missile.target_distance
        speed = float(np.linalg.norm(missile.get_velocity()))
        distances.append(distance)
        print(f"{step},{missile.status_name},{distance:.3f},{speed:.3f},{target.alive}")
        if missile.is_done:
            break
    print(f"final_status: {missile.status_name}")
    print(f"target_alive: {target.alive}")
    print(f"initial_distance: {distances[0]:.3f}")
    print(f"min_distance: {min(distances):.3f}")
    print(f"final_distance: {distances[-1]:.3f}")
    print(f"distance_generally_decreased: {min(distances) < distances[0]}")


if __name__ == "__main__":
    main()
