"""Episode termination helpers."""

from __future__ import annotations

from .aircraft import AircraftPlatform


class TerminationChecker:
    def __init__(self, config: dict):
        self.episode_limit = int(config.get("episode_limit", 1000))
        self.terminate_on_mav_loss = bool(config.get("terminate_on_mav_loss", True))

    def check(self, agents: list[AircraftPlatform], step_count: int) -> tuple[bool, bool, str | None, str | None]:
        red_alive = [a for a in agents if a.side == "red" and a.alive]
        blue_alive = [a for a in agents if a.side == "blue" and a.alive]
        mav_alive = any(a.alive and a.aircraft_type.role == "mav" for a in agents if a.side == "red")
        if not blue_alive:
            return True, False, "red_win", "blue_all_destroyed"
        if not red_alive:
            return True, False, "blue_win", "red_all_destroyed"
        if self.terminate_on_mav_loss and not mav_alive:
            return True, False, "blue_win", "mav_loss"
        if step_count >= self.episode_limit:
            red_kills = sum(1 for a in agents if a.side == "blue" and not a.alive)
            blue_kills = sum(1 for a in agents if a.side == "red" and not a.alive)
            if red_kills > blue_kills or len(red_alive) > len(blue_alive):
                return False, True, "red_win", "episode_limit_alive_advantage"
            if blue_kills > red_kills or len(blue_alive) > len(red_alive):
                return False, True, "blue_win", "episode_limit_alive_advantage"
            return False, True, "draw", "episode_limit_draw"
        return False, False, None, None
