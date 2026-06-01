"""Aircraft platform proxy.

This debug implementation uses simple 3D kinematics. The class boundary is kept
small so it can later be backed by JSBSim aircraft models without changing the
environment API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft_types import AircraftType
from .utils import clamp, heading_to_unit, wrap_pi


@dataclass
class AircraftPlatform:
    agent_id: str
    side: str
    aircraft_type: AircraftType
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    pitch: float = 0.0
    roll: float = 0.0
    alive: bool = True
    crashed: bool = False
    out_of_boundary: bool = False
    missile_left: int = 0
    missile_cooldown: int = 0

    def reset_runtime(self) -> None:
        self.alive = True
        self.crashed = False
        self.out_of_boundary = False
        self.missile_left = self.aircraft_type.missile_num
        self.missile_cooldown = 0

    @property
    def type_id(self) -> int:
        return self.aircraft_type.type_id

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def step(self, action: np.ndarray, dt: float, speed_range: tuple[float, float]) -> None:
        if not self.alive:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        target_pitch = float(action[0]) * (np.pi / 2.0)
        target_heading = float(action[1]) * np.pi
        speed_min, speed_max = speed_range
        speed_hi = speed_max * self.aircraft_type.max_speed_scale
        target_speed = speed_min + (float(action[2]) + 1.0) * 0.5 * (speed_hi - speed_min)

        turn_rate = np.deg2rad(18.0) * max(0.5, self.aircraft_type.max_g / 9.0)
        pitch_rate = np.deg2rad(12.0) * max(0.5, self.aircraft_type.max_g / 9.0)
        speed_rate = 45.0 * max(0.5, self.aircraft_type.max_speed_scale)

        heading_error = wrap_pi(target_heading - self.heading)
        self.heading = wrap_pi(self.heading + clamp(heading_error, -turn_rate * dt, turn_rate * dt))
        pitch_error = target_pitch - self.pitch
        self.pitch += clamp(pitch_error, -pitch_rate * dt, pitch_rate * dt)
        self.pitch = clamp(self.pitch, -np.deg2rad(35.0), np.deg2rad(35.0))
        self.roll = clamp(heading_error * 0.5, -np.deg2rad(65.0), np.deg2rad(65.0))

        current_speed = self.speed
        next_speed = current_speed + clamp(target_speed - current_speed,
                                           -speed_rate * dt, speed_rate * dt)
        next_speed = clamp(next_speed, speed_min, speed_hi)
        self.velocity = heading_to_unit(self.heading, self.pitch) * next_speed
        self.position = self.position + self.velocity * dt
        self.missile_cooldown = max(0, self.missile_cooldown - 1)

    def kill(self, reason: str = "killed") -> None:
        self.alive = False
        if reason == "crash":
            self.crashed = True
        if reason == "boundary":
            self.out_of_boundary = True
