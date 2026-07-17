"""Paper automatic weapon management and point-mass missile lifecycle."""

from __future__ import annotations

import numpy as np

from .missile import PaperMissile


COINCIDENT_POSITION_EPSILON_M = 1.0e-6


class PaperWeaponManager:
    def __init__(self, published: dict, inferred: dict):
        self.published = published
        self.inferred = inferred
        self.missiles: list[PaperMissile] = []
        self.last_launch_time_s: dict[str, float] = {}
        self.counter = 0
        self.total_fired = 0
        self.total_hits = 0
        self._reported_terminations: set[str] = set()

    def reset(self):
        self.missiles.clear()
        self.last_launch_time_s.clear()
        self.counter = self.total_fired = self.total_hits = 0
        self._reported_terminations.clear()

    def begin_decision_step(self) -> None:
        for missile in self.missiles:
            missile.mark_decision_start()

    def try_launch(self, shooter, target, simulation_time_s: float) -> dict | None:
        if (not shooter.alive or shooter.aircraft_type.role == "mav"
                or shooter.missile_left <= 0 or target is None or not target.alive):
            return None
        distance = float(np.linalg.norm(target.position - shooter.position))
        if distance <= COINCIDENT_POSITION_EPSILON_M:
            return None
        if distance > float(self.published["maximum_attack_range_m"]):
            return None
        elapsed = simulation_time_s - self.last_launch_time_s.get(shooter.agent_id, -1e30)
        if elapsed + 1e-9 < float(self.published["launch_interval_s"]):
            return None
        direction = target.position - shooter.position
        self.counter += 1
        missile_id = f"{shooter.agent_id}_M{self.counter:04d}"
        config = {**self.published, **self.inferred}
        missile = PaperMissile(missile_id, shooter.agent_id, target.agent_id,
                               shooter.position.copy(), direction, config)
        self.missiles.append(missile)
        shooter.missile_left -= 1
        self.last_launch_time_s[shooter.agent_id] = simulation_time_s
        self.total_fired += 1
        return {"missile_id": missile_id, "shooter_id": shooter.agent_id,
                "target_id": target.agent_id, "reason": "launched", "distance_m": distance,
                "missiles_left": shooter.missile_left}

    def step_physics_once(self, by_id: dict, physics_dt: float) -> list[dict]:
        events = []
        for missile in self.missiles:
            if not missile.alive:
                continue
            target = by_id.get(missile.target_id)
            if target is None or not target.alive:
                missile.status, missile.termination_reason = "miss", "target_dead"
            else:
                missile.step(target.position, target.velocity, physics_dt)
            if (missile.termination_reason
                    and missile.missile_id not in self._reported_terminations):
                hit = missile.termination_reason == "hit" and target is not None and target.alive
                if hit:
                    target.kill("shotdown")
                    self.total_hits += 1
                self._reported_terminations.add(missile.missile_id)
                events.append({"missile_id": missile.missile_id,
                               "shooter_id": missile.shooter_id,
                               "target_id": missile.target_id,
                               "event_type": "missile_termination",
                               "reason": missile.termination_reason,
                               "hit": hit})
                if hit:
                    events.append({
                        "event_type": "aircraft_death",
                        "agent_id": target.agent_id,
                        "side": target.side,
                        "reason": target.death_reason,
                    })
        return events
