"""Simplified missile launch and hit logic."""

from __future__ import annotations

from dataclasses import dataclass

from .aircraft import AircraftPlatform
from .sensor import SensorSuite
from .utils import heading_to_unit, los_angle, safe_norm


@dataclass
class MissileEvent:
    shooter_id: str
    target_id: str | None
    shooter_side: str
    fired: bool
    hit: bool
    reason: str
    distance: float | None
    missile_left_after: int


class MissileManager:
    def __init__(self, config: dict):
        params = config.get("missile_params", {})
        self.attack_range = float(params.get("attack_range", 14000.0))
        self.launch_range = float(params.get("launch_range", params.get("hit_range", 6500.0)))
        self.hit_range = float(params.get("hit_range", 6500.0))
        self.cooldown_steps = int(params.get("cooldown_steps", 25))
        self.max_los_angle = float(params.get("max_los_angle_rad", 1.57079632679))
        self.fire_only_in_kill_zone = bool(params.get("fire_only_in_kill_zone", True))

    def try_launch(self, shooter: AircraftPlatform, target: AircraftPlatform | None,
                   sensor: SensorSuite) -> MissileEvent | None:
        if not shooter.alive:
            return MissileEvent(shooter.agent_id, None, shooter.side, False, False,
                                "shooter_dead", None, shooter.missile_left)
        if target is None or not target.alive:
            return MissileEvent(shooter.agent_id, None, shooter.side, False, False,
                                "no_target", None, shooter.missile_left)
        if shooter.missile_left <= 0 or shooter.missile_cooldown > 0:
            reason = "no_missile" if shooter.missile_left <= 0 else "cooldown"
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                reason, safe_norm(target.position - shooter.position),
                                shooter.missile_left)
        rel = target.position - shooter.position
        distance = safe_norm(rel)
        if not sensor.can_detect(shooter, target):
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "not_visible", distance, shooter.missile_left)
        if distance > self.attack_range:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "out_of_range", distance, shooter.missile_left)
        if self.fire_only_in_kill_zone and distance > self.launch_range:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "out_of_launch_range", distance, shooter.missile_left)
        forward = heading_to_unit(shooter.heading, shooter.pitch)
        if los_angle(forward, rel) > self.max_los_angle:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "los_blocked", distance, shooter.missile_left)
        shooter.missile_left -= 1
        shooter.missile_cooldown = self.cooldown_steps
        hit = distance <= self.hit_range
        return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, True, hit,
                            "hit" if hit else "miss", distance, shooter.missile_left)
