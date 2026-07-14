"""Three-dimensional point-mass missile with paper PN guidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G = 9.80665


def _wrap_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class PaperMissile:
    missile_id: str
    shooter_id: str
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    config: dict

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).copy()
        initial = np.asarray(self.velocity, dtype=np.float64)
        initial_norm = max(float(np.linalg.norm(initial)), 1e-8)
        configured_speed = float(self.config["missile_initial_speed_mps"])
        self.speed_mps = configured_speed
        direction = initial / initial_norm
        self.yaw_rad = float(np.arctan2(direction[1], direction[0]))
        self.pitch_rad = float(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
        self.velocity = self._direction() * self.speed_mps
        self.previous_speed_mps = self.speed_mps
        self.flight_time_s = 0.0
        self.previous_los_yaw_rad = None
        self.previous_los_pitch_rad = None
        self.los_yaw_rad = 0.0
        self.los_pitch_rad = 0.0
        self.los_yaw_rate_rps = 0.0
        self.los_pitch_rate_rps = 0.0
        self.commanded_overload_g = 0.0
        self.status = "flying"
        self.termination_reason = None
        self.navigation_gain_y = float(self.config["navigation_gain_y"])
        self.navigation_gain_z = float(self.config["navigation_gain_z"])

    @property
    def alive(self) -> bool:
        return self.status == "flying"

    def _direction(self) -> np.ndarray:
        cp = np.cos(self.pitch_rad)
        return np.array([cp * np.cos(self.yaw_rad), cp * np.sin(self.yaw_rad),
                         np.sin(self.pitch_rad)], dtype=np.float64)

    def step(self, target_position: np.ndarray, target_velocity: np.ndarray, dt: float) -> str | None:
        del target_velocity  # LOS rates are intentionally finite-differenced per physics step.
        if not self.alive:
            return self.termination_reason
        target_position = np.asarray(target_position, dtype=np.float64)
        rel = target_position - self.position
        distance = float(np.linalg.norm(rel))
        if distance <= float(self.config["hit_radius_m"]):
            self.status, self.termination_reason = "hit", "hit"
            return self.termination_reason
        self.los_yaw_rad = float(np.arctan2(rel[1], rel[0]))
        self.los_pitch_rad = float(np.arctan2(rel[2], max(np.linalg.norm(rel[:2]), 1e-8)))
        if self.previous_los_yaw_rad is not None:
            self.los_yaw_rate_rps = _wrap_pi(
                self.los_yaw_rad - self.previous_los_yaw_rad) / dt
            self.los_pitch_rate_rps = _wrap_pi(
                self.los_pitch_rad - self.previous_los_pitch_rad) / dt
        self.previous_los_yaw_rad = self.los_yaw_rad
        self.previous_los_pitch_rad = self.los_pitch_rad

        cos_pitch = max(abs(float(np.cos(self.pitch_rad))), 1e-3)
        ny = self.navigation_gain_y * self.speed_mps * self.los_yaw_rate_rps / (G * cos_pitch)
        nz = self.navigation_gain_z * self.speed_mps * self.los_pitch_rate_rps / G + np.cos(self.pitch_rad)
        magnitude = float(np.hypot(ny, nz))
        limit = float(self.config["maximum_overload_g"])
        if magnitude > limit:
            ny, nz = ny * limit / magnitude, nz * limit / magnitude
            magnitude = limit
        self.commanded_overload_g = magnitude
        self.yaw_rad = _wrap_pi(self.yaw_rad + G / max(self.speed_mps, 1.0) * ny
                                * np.cos(self.pitch_rad) * dt)
        self.pitch_rad = float(np.clip(
            self.pitch_rad + G / max(self.speed_mps, 1.0)
            * (nz - np.cos(self.pitch_rad)) * dt,
            -np.pi / 2.0 + 1e-4, np.pi / 2.0 - 1e-4))

        self.previous_speed_mps = self.speed_mps
        drag = float(self.config["drag_coefficient"]) * self.speed_mps ** 2
        thrust = (float(self.config["powered_acceleration_mps2"])
                  if self.flight_time_s < float(self.config["powered_duration_s"]) else 0.0)
        self.speed_mps = float(np.clip(
            self.speed_mps + (thrust - drag - G * np.sin(self.pitch_rad)) * dt,
            1.0, float(self.config["maximum_missile_speed_mps"])))
        self.velocity = self._direction() * self.speed_mps
        self.position += self.velocity * dt
        self.flight_time_s += dt
        if not np.isfinite(np.concatenate([self.position, self.velocity])).all():
            self.status, self.termination_reason = "miss", "nonfinite"
        elif self.flight_time_s >= float(self.config["missile_max_flight_time_s"]):
            self.status, self.termination_reason = "miss", "timeout"
        return self.termination_reason
