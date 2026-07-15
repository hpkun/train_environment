"""Single paper-style immediate-reward greedy blue opponent."""
from __future__ import annotations

import numpy as np

from .geometry import combat_geometry


class PaperGreedyOpponent:
    """Evaluate a fixed basic-maneuver set with TAM Table 1 weight ratios."""

    def actions(self, env, team: str = "blue") -> dict[str, np.ndarray]:
        ids = env.blue_ids if team == "blue" else env.red_ids
        return {aid: self._action(env, aid) for aid in ids}

    def _action(self, env, aid: str) -> np.ndarray:
        aircraft = env.aircraft[aid]
        if not aircraft.is_alive:
            return np.zeros(3, np.float32)
        target_ids = env.red_ids if aid.startswith("blue") else env.blue_ids
        targets = [env.aircraft[x] for x in target_ids if env.aircraft[x].is_alive]
        if not targets:
            return np.asarray([0.0, 0.0, 0.0], np.float32)
        target = min(targets, key=lambda x: np.linalg.norm(x.get_position() - aircraft.get_position()))
        yaw = float(aircraft.get_rpy()[2])
        speed = float(np.linalg.norm(aircraft.get_velocity()))
        speed_action = np.clip(2 * (speed - 102.0) / 306.0 - 1.0, -1.0, 1.0)
        candidates = []
        for pitch in (-0.2, 0.0, 0.2):
            for delta in (-np.pi / 6, 0.0, np.pi / 6):
                for dv in (-0.2, 0.2):
                    candidates.append(np.asarray([pitch, _wrap(yaw + delta) / np.pi,
                                                  np.clip(speed_action + dv, -1, 1)], np.float32))
        warning = any(m.is_launched and m.target_id == aid for m in env.missiles)
        return max(candidates, key=lambda action: self._score(aircraft, target, action, warning))

    @staticmethod
    def _score(aircraft, target, action, warning: bool) -> float:
        position = aircraft.get_position().copy()
        speed = 102.0 + (float(action[2]) + 1.0) * 153.0
        heading = float(action[1]) * np.pi
        pitch = float(action[0]) * np.pi / 2
        velocity = speed * np.asarray([np.cos(pitch) * np.cos(heading),
                                       np.cos(pitch) * np.sin(heading), np.sin(pitch)])
        predicted = position + velocity * 0.2
        altitude = predicted[2]
        height = np.clip(1.0 - abs(altitude - 6000.0) / 5250.0, -1.0, 1.0)
        target_speed = np.linalg.norm(target.get_velocity())
        speed_score = np.clip((speed - target_speed) / 150.0, -1.0, 1.0)
        los = target.get_position() - predicted
        ata = np.arccos(np.clip(np.dot(velocity, los) /
                                max(np.linalg.norm(velocity) * np.linalg.norm(los), 1e-6), -1, 1))
        angle_score = 1.0 - ata / np.pi
        distance = np.linalg.norm(los)
        distance_score = np.clip((20_000.0 - distance) / 20_000.0, -1.0, 1.0)
        avoidance = abs(float(action[1])) if warning else 0.0
        return float(10 * height + 10 * speed_score + 15 * angle_score +
                     10 * distance_score + 30 * avoidance)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
