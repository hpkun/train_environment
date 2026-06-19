"""Missile fire-control manager around the migrated MissileSimulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import AircraftPlatform
from .original_missile import MissileSimulator
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
    missile_id: str | None = None


class MissileManager:
    def __init__(self, config: dict):
        params = config.get("missile_params", {})
        self.attack_range = float(params.get("attack_range", 14000.0))
        self.launch_range = float(params.get("launch_range", params.get("hit_range", 6500.0)))
        self.hit_range = float(params.get("hit_range", 6500.0))
        self.cooldown_steps = int(params.get("cooldown_steps", 25))
        self.max_los_angle = float(params.get("max_los_angle_rad", 1.57079632679))
        self.fire_only_in_kill_zone = bool(params.get("fire_only_in_kill_zone", True))
        self.missile_model = str(params.get("missile_model", "AIM-9L"))
        self.missile_substeps = int(params.get("missile_substeps", 12))
        self.active_missiles: list[MissileSimulator] = []
        self._uid_counter = 0

    def reset(self) -> None:
        self.active_missiles.clear()
        self._uid_counter = 0

    def evaluate_launch(self, shooter: AircraftPlatform, target: AircraftPlatform | None,
                        sensor: SensorSuite) -> MissileEvent | None:
        if not shooter.alive:
            return MissileEvent(shooter.agent_id, None, shooter.side, False, False,
                                "shooter_dead", None, shooter.missile_left)
        if target is None or not target.alive:
            return MissileEvent(shooter.agent_id, None, shooter.side, False, False,
                                "no_target", None, shooter.missile_left)
        distance = safe_norm(target.position - shooter.position)
        if shooter.missile_left <= 0:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "no_missile", distance, shooter.missile_left)
        if shooter.missile_cooldown > 0:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "cooldown", distance, shooter.missile_left)
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
        if los_angle(forward, target.position - shooter.position) > self.max_los_angle:
            return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, False, False,
                                "los_blocked", distance, shooter.missile_left)
        return MissileEvent(shooter.agent_id, target.agent_id, shooter.side, True, False,
                            "launched", distance, shooter.missile_left - 1)

    def create_missile(self, shooter: AircraftPlatform, target: AircraftPlatform,
                       event: MissileEvent, dt: float) -> MissileEvent:
        shooter.missile_left -= 1
        shooter.missile_cooldown = self.cooldown_steps
        self._uid_counter += 1
        uid = f"{shooter.agent_id}_M{self._uid_counter:04d}"
        missile = MissileSimulator.create(
            parent=shooter, target=target, uid=uid, missile_model=self.missile_model,
            dt=dt / max(1, self.missile_substeps))
        self.active_missiles.append(missile)
        event.missile_left_after = shooter.missile_left
        event.missile_id = uid
        return event

    def step_active(self, dt: float) -> list[MissileEvent]:
        events = []
        substeps = max(1, self.missile_substeps)
        for missile in list(self.active_missiles):
            previous_status = missile.status
            for _ in range(substeps):
                if missile.is_alive:
                    missile.run()
            if missile.is_done and previous_status != missile.status:
                target = missile.target_aircraft
                shooter = missile.parent_aircraft
                reason = "hit" if missile.is_success else self._miss_reason(missile)
                events.append(MissileEvent(
                    shooter.agent_id,
                    target.agent_id if target is not None else None,
                    shooter.side,
                    False,
                    missile.is_success,
                    reason,
                    missile.target_distance if target is not None else None,
                    shooter.missile_left,
                    missile.uid,
                ))
        self.active_missiles = [m for m in self.active_missiles if m.is_alive]
        return events

    def incoming_missiles_for(self, agent: AircraftPlatform) -> list[MissileSimulator]:
        return [m for m in agent.under_missiles if m.is_alive]

    @property
    def active_missile_count(self) -> int:
        return len(self.active_missiles)

    def _miss_reason(self, missile: MissileSimulator) -> str:
        if missile.target_aircraft is not None and not missile.target_aircraft.is_alive and not missile.is_success:
            return "target_dead"
        if getattr(missile, "_t", 0.0) > getattr(missile, "_t_max", 60.0):
            return "timeout"
        return "miss"
