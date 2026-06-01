"""Episode termination helpers."""

from __future__ import annotations

from .aircraft import AircraftPlatform


class TerminationChecker:
    def __init__(self, config: dict):
        self.episode_limit = int(config.get("episode_limit", 1000))
        self.terminate_on_mav_loss = bool(config.get("terminate_on_mav_loss", True))

    def check(self, agents: list[AircraftPlatform], step_count: int) -> tuple[bool, bool, str | None]:
        red_alive = [a for a in agents if a.side == "red" and a.alive]
        blue_alive = [a for a in agents if a.side == "blue" and a.alive]
        mav_alive = any(a.alive and a.aircraft_type.role == "mav" for a in agents if a.side == "red")
        if not blue_alive:
            return True, False, "red"
        if not red_alive:
            return True, False, "blue"
        if self.terminate_on_mav_loss and not mav_alive:
            return True, False, "blue"
        if step_count >= self.episode_limit:
            return False, True, "draw"
        return False, False, None
