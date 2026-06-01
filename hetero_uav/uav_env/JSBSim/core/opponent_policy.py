"""Rule-based policies for non-controlled agents."""

from __future__ import annotations

import numpy as np

from .aircraft import AircraftPlatform


class RuleNearestOpponentPolicy:
    """Aim each agent at the nearest alive opponent."""

    def __init__(self, speed_norm: float = 0.6):
        self.speed_norm = float(speed_norm)

    def act(self, agent: AircraftPlatform, agents: list[AircraftPlatform]) -> np.ndarray:
        targets = [a for a in agents if a.side != agent.side and a.alive]
        if not agent.alive or not targets:
            return np.zeros(3, dtype=np.float32)

        target = min(targets, key=lambda other: np.linalg.norm(other.position - agent.position))
        rel = target.position - agent.position
        heading = float(np.arctan2(rel[1], rel[0]))
        horizontal_range = float(np.linalg.norm(rel[:2]) + 1e-8)
        pitch = float(np.arctan2(rel[2], horizontal_range))
        return np.array([
            np.clip(pitch / (np.pi / 2.0), -1.0, 1.0),
            np.clip(heading / np.pi, -1.0, 1.0),
            np.clip(self.speed_norm, -1.0, 1.0),
        ], dtype=np.float32)


def make_opponent_policy(name: str):
    if name == "rule_nearest":
        return RuleNearestOpponentPolicy()
    raise KeyError(f"unknown opponent_policy {name!r}")
