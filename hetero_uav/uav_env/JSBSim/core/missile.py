"""Simplified missile launch and hit logic."""

from __future__ import annotations

from dataclasses import dataclass

from .aircraft import AircraftPlatform
from .sensor import SensorSuite
from .utils import heading_to_unit, los_angle, safe_norm


@dataclass
class MissileEvent:
    shooter_id: str
    target_id: str
    shooter_side: str
    hit: bool


class MissileManager:
    def __init__(self, config: dict):
        params = config.get("missile_params", {})
        self.attack_range = float(params.get("attack_range", 14000.0))
        self.hit_range = float(params.get("hit_range", 6000.0))
        self.cooldown_steps = int(params.get("cooldown_steps", 25))
        self.max_los_angle = float(params.get("max_los_angle_rad", 1.57079632679))

    def try_launch(self, shooter: AircraftPlatform, target: AircraftPlatform,
                   sensor: SensorSuite) -> MissileEvent | None:
        if not shooter.alive or not target.alive:
            return None
        if shooter.missile_left <= 0 or shooter.missile_cooldown > 0:
            return None
        rel = target.position - shooter.position
        distance = safe_norm(rel)
        if distance > self.attack_range or not sensor.can_detect(shooter, target):
            return None
        forward = heading_to_unit(shooter.heading, shooter.pitch)
        if los_angle(forward, rel) > self.max_los_angle:
            return None
        shooter.missile_left -= 1
        shooter.missile_cooldown = self.cooldown_steps
        hit = distance <= self.hit_range
        if hit:
            target.kill("killed")
        return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, hit)
