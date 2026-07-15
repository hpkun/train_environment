"""Single deterministic point-mass proportional-navigation missile model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import unit


@dataclass
class FormalMissile:
    missile_id: str
    shooter_id: str
    target_id: str
    position: np.ndarray
    velocity: np.ndarray
    navigation_gain: float = 3.0
    max_overload_g: float = 30.0
    speed_mps: float = 600.0
    hit_radius_m: float = 300.0
    max_flight_time_sec: float = 60.0
    arming_time_sec: float = 0.2
    state: str = "launched"
    termination_reason: str = ""
    flight_time_sec: float = 0.0

    @property
    def is_launched(self) -> bool:
        return self.state == "launched"

    def step(self, dt: float, target) -> dict | None:
        if not self.is_launched:
            return None
        if not target.is_alive:
            return self._finish("miss", "target_dead")
        old_position = self.position.copy()
        rel = target.get_position() - self.position
        distance = float(np.linalg.norm(rel))
        rel_velocity = target.get_velocity() - self.velocity
        closing = max(0.0, -float(np.dot(rel_velocity, unit(rel))))
        omega = np.cross(rel, rel_velocity) / max(distance * distance, 1e-6)
        acceleration = self.navigation_gain * closing * np.cross(omega, unit(self.velocity))
        magnitude = float(np.linalg.norm(acceleration))
        limit = self.max_overload_g * 9.81
        if magnitude > limit:
            acceleration *= limit / magnitude
        direction = unit(self.velocity + acceleration * dt)
        if not direction.any():
            direction = unit(rel)
        self.velocity = direction * self.speed_mps
        self.position = self.position + self.velocity * dt
        self.flight_time_sec += dt
        segment = self.position - old_position
        t = np.clip(np.dot(target.get_position() - old_position, segment) /
                    max(np.dot(segment, segment), 1e-6), 0.0, 1.0)
        miss_distance = float(np.linalg.norm(old_position + t * segment - target.get_position()))
        if self.flight_time_sec >= self.arming_time_sec and miss_distance <= self.hit_radius_m:
            target.shotdown()
            return self._finish("hit", "hit")
        if self.flight_time_sec >= self.max_flight_time_sec:
            return self._finish("miss", "timeout")
        if np.linalg.norm(self.position[:2]) > 100_000 or abs(self.position[2]) > 30_000:
            return self._finish("miss", "out_of_bounds")
        if not np.isfinite(np.r_[self.position, self.velocity]).all():
            return self._finish("miss", "out_of_bounds")
        return None

    def _finish(self, state: str, reason: str) -> dict:
        self.state = state
        self.termination_reason = reason
        return {"event": state, "reason": reason, "missile_id": self.missile_id,
                "shooter_id": self.shooter_id, "target_id": self.target_id,
                "flight_time_sec": self.flight_time_sec}
