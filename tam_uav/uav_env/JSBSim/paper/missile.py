"""Paper Eq. (7)-(9) point-mass missile with analytic LOS rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G = 9.80665


def analytic_los_rates(relative_position: np.ndarray,
                       relative_velocity: np.ndarray) -> tuple[float, float, float, float]:
    """Return LOS yaw, pitch and their analytic time derivatives."""
    x, y, z = np.asarray(relative_position, dtype=np.float64)
    xd, yd, zd = np.asarray(relative_velocity, dtype=np.float64)
    rho2 = x * x + y * y
    rho = float(np.sqrt(max(rho2, 1e-12)))
    range2 = max(rho2 + z * z, 1e-12)
    yaw = float(np.arctan2(y, x))
    pitch = float(np.arctan2(z, rho))
    yaw_rate = float((x * yd - y * xd) / max(rho2, 1e-12))
    rho_rate = float((x * xd + y * yd) / rho)
    pitch_rate = float((rho * zd - z * rho_rate) / range2)
    return yaw, pitch, yaw_rate, pitch_rate


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
        direction = initial / max(float(np.linalg.norm(initial)), 1e-8)
        self.speed_mps = float(self.config["missile_initial_speed_mps"])
        self.yaw_rad = float(np.arctan2(direction[1], direction[0]))
        self.pitch_rad = float(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
        self.velocity = self._direction() * self.speed_mps
        self.previous_speed_mps = self.speed_mps
        self.decision_start_speed_mps = self.speed_mps
        self.flight_time_s = 0.0
        self.los_yaw_rad = self.los_pitch_rad = 0.0
        self.los_yaw_rate_rps = self.los_pitch_rate_rps = 0.0
        self.longitudinal_overload_g = 0.0
        self.lateral_overload_y_g = 0.0
        self.lateral_overload_z_g = 0.0
        self.commanded_overload_g = 0.0
        self.distance_m = float("inf")
        self.status = "flying"
        self.termination_reason = None
        self.navigation_gain_y = float(self.config["navigation_gain_y"])
        self.navigation_gain_z = float(self.config["navigation_gain_z"])

    @property
    def alive(self) -> bool:
        return self.status == "flying"

    def mark_decision_start(self) -> None:
        if self.alive:
            self.decision_start_speed_mps = self.speed_mps

    def _direction(self) -> np.ndarray:
        cp = np.cos(self.pitch_rad)
        return np.array([cp * np.cos(self.yaw_rad), cp * np.sin(self.yaw_rad),
                         np.sin(self.pitch_rad)], dtype=np.float64)

    def step(self, target_position: np.ndarray, target_velocity: np.ndarray, dt: float) -> str | None:
        if not self.alive:
            return self.termination_reason
        target_position = np.asarray(target_position, dtype=np.float64)
        target_velocity = np.asarray(target_velocity, dtype=np.float64)
        relative_position = target_position - self.position
        relative_velocity = target_velocity - self.velocity
        self.distance_m = float(np.linalg.norm(relative_position))
        if self.distance_m <= float(self.config["hit_radius_m"]):
            self.status, self.termination_reason = "hit", "hit"
            return self.termination_reason

        (self.los_yaw_rad, self.los_pitch_rad, self.los_yaw_rate_rps,
         self.los_pitch_rate_rps) = analytic_los_rates(relative_position, relative_velocity)
        cos_pitch = float(np.cos(self.pitch_rad))
        self.lateral_overload_y_g = (
            self.navigation_gain_y * self.speed_mps / G * cos_pitch
            * self.los_yaw_rate_rps)
        self.lateral_overload_z_g = (
            self.navigation_gain_z * self.speed_mps / G * self.los_pitch_rate_rps
            + cos_pitch)
        lateral_norm = float(np.hypot(self.lateral_overload_y_g,
                                      self.lateral_overload_z_g))
        limit = float(self.config["maximum_overload_g"])
        if lateral_norm > limit:
            scale = limit / lateral_norm
            self.lateral_overload_y_g *= scale
            self.lateral_overload_z_g *= scale
            lateral_norm = limit
        self.commanded_overload_g = lateral_norm

        safe_cos = cos_pitch if abs(cos_pitch) >= 1e-4 else np.copysign(1e-4, cos_pitch or 1.0)
        yaw_dot = G * self.lateral_overload_y_g / (max(self.speed_mps, 1.0) * safe_cos)
        pitch_dot = G / max(self.speed_mps, 1.0) * (
            self.lateral_overload_z_g - cos_pitch)
        self.yaw_rad = _wrap_pi(self.yaw_rad + yaw_dot * dt)
        self.pitch_rad = float(np.clip(self.pitch_rad + pitch_dot * dt,
                                       -np.pi / 2.0 + 1e-4,
                                       np.pi / 2.0 - 1e-4))

        drag_accel = float(self.config["drag_coefficient"]) * self.speed_mps ** 2
        thrust_accel = (float(self.config["powered_acceleration_mps2"])
                        if self.flight_time_s < float(self.config["powered_duration_s"])
                        else 0.0)
        self.longitudinal_overload_g = (thrust_accel - drag_accel) / G
        speed_dot = G * (self.longitudinal_overload_g - np.sin(self.pitch_rad))
        self.previous_speed_mps = self.speed_mps
        self.speed_mps = float(np.clip(
            self.speed_mps + speed_dot * dt, 1.0,
            float(self.config["maximum_missile_speed_mps"])))
        self.velocity = self._direction() * self.speed_mps
        self.position += self.velocity * dt
        self.flight_time_s += dt
        if not np.isfinite(np.concatenate([self.position, self.velocity,
                                           [self.speed_mps]])).all():
            self.status, self.termination_reason = "miss", "nonfinite"
        elif self.flight_time_s >= float(self.config["missile_max_flight_time_s"]):
            self.status, self.termination_reason = "miss", "timeout"
        return self.termination_reason

    def telemetry(self) -> dict:
        return {
            "missile_id": self.missile_id,
            "target_id": self.target_id,
            "los_yaw_rate_rps": self.los_yaw_rate_rps,
            "los_pitch_rate_rps": self.los_pitch_rate_rps,
            "n_y_g": self.lateral_overload_y_g,
            "n_z_g": self.lateral_overload_z_g,
            "total_overload_g": self.commanded_overload_g,
            "longitudinal_overload_g": self.longitudinal_overload_g,
            "speed_mps": self.speed_mps,
            "decision_start_speed_mps": self.decision_start_speed_mps,
            "distance_m": self.distance_m,
            "termination_reason": self.termination_reason,
        }
