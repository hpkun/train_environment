"""Single immediate-greedy blue opponent using the formal UAV score."""
from __future__ import annotations

import numpy as np

from .reward import missile_risk, uav_dense_components


class PaperGreedyOpponent:
    """Exhaustively score every visible target x fixed basic maneuver pair."""

    def actions(self, env, team: str = "blue") -> dict[str, np.ndarray]:
        ids = env.blue_ids if team == "blue" else env.red_ids
        return {aid: self._action(env, aid) for aid in ids}

    def scored_candidates(self, env, aid: str) -> list[dict]:
        aircraft = env.aircraft[aid]
        target_ids = env.red_ids if aid.startswith("blue") else env.blue_ids
        targets = [env.aircraft[x] for x in target_ids if env.aircraft[x].is_alive]
        if not aircraft.is_alive or not targets:
            return []
        incoming = [m for m in env.missiles if m.is_launched and m.target_id == aid]
        current_risk = missile_risk(aircraft.get_position(), aircraft.get_velocity(), incoming)
        rows = []
        for target in targets:
            for action in self._candidate_actions(aircraft):
                position, velocity = self._predict_state(aircraft, action)
                components = uav_dense_components(
                    position, velocity, target.get_position(), target.get_velocity(),
                    incoming, current_risk)
                rows.append({"target_id": target.uid, "action": action,
                             "score": components["dense"], "components": components})
        return rows

    def _action(self, env, aid: str) -> np.ndarray:
        aircraft = env.aircraft[aid]
        if not aircraft.is_alive:
            return np.zeros(3, np.float32)
        rows = self.scored_candidates(env, aid)
        if not rows:
            yaw = _wrap(float(aircraft.get_rpy()[2]))
            return np.asarray([0.0, yaw / np.pi, 0.0], np.float32)
        best = max(rows, key=lambda row: row["score"])
        return np.clip(best["action"], -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _candidate_actions(aircraft) -> list[np.ndarray]:
        yaw = float(aircraft.get_rpy()[2])
        speed = float(np.linalg.norm(aircraft.get_velocity()))
        speed_action = np.clip(2 * (speed - 102.0) / 306.0 - 1.0, -1.0, 1.0)
        return [
            np.asarray([pitch, _wrap(yaw + delta) / np.pi,
                        np.clip(speed_action + dv, -1.0, 1.0)], np.float32)
            for pitch in (-0.20, 0.0, 0.20)
            for delta in (-np.pi / 6, 0.0, np.pi / 6)
            for dv in (-0.20, 0.20)
        ]

    @staticmethod
    def _predict_state(aircraft, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = 102.0 + (float(action[2]) + 1.0) * 153.0
        heading = float(action[1]) * np.pi
        pitch = float(action[0]) * np.pi / 2
        velocity = speed * np.asarray([
            np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading), np.sin(pitch)])
        return aircraft.get_position() + velocity * 0.2, velocity


def _wrap(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)
