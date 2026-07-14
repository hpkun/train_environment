"""Observation-only greedy candidate-manoeuvre blue opponent."""

from __future__ import annotations

import numpy as np


MANOEUVRES = {
    "level": (24, 20, 20, 20),
    "accelerate_level": (39, 20, 20, 20),
    "left_turn": (28, 8, 20, 12),
    "right_turn": (28, 31, 20, 27),
    "climb": (30, 20, 8, 20),
    "dive": (25, 20, 31, 20),
    "left_climb": (30, 8, 8, 12),
    "right_climb": (30, 31, 8, 27),
    "left_dive": (27, 8, 31, 12),
    "right_dive": (27, 31, 31, 27),
}


class GreedyPaperOpponent:
    """Scores ten fixed 4D commands using only the formal local observation."""

    def act(self, agent_id: str, observation: dict) -> tuple[np.ndarray, dict]:
        visible = np.flatnonzero(observation["enemy_mask"] > 0.5)
        if not len(visible):
            action = np.asarray(MANOEUVRES["level"], dtype=np.int64)
            return action, {"target_slot": None, "scores": {}, "manoeuvre": "level",
                            "action_indices": action.tolist()}
        # Formal enemy slots are stable IDs. The highest geometric proxy wins.
        rows = observation["enemy_states"][visible]
        target_offset = int(np.argmax(1.0 - rows[:, 2] - 0.35 * rows[:, 3] - 0.15 * rows[:, 4]))
        slot = int(visible[target_offset])
        row = observation["enemy_states"][slot]
        relative_altitude = float(row[1])
        ata = float(row[3])
        scores = {}
        for name, action in MANOEUVRES.items():
            _, aileron, elevator, rudder = action
            turn_strength = (abs(aileron - 19.5) + abs(rudder - 19.5)) / 39.0
            climb_direction = (elevator - 19.5) / 19.5
            alignment_need = float(np.clip(ata, 0.0, 1.0))
            scores[name] = float(-ata + (0.6 * alignment_need - 0.4 * (1.0 - alignment_need)) * turn_strength
                                 + 0.2 * np.sign(relative_altitude) * climb_direction
                                 + 0.25 * (1.0 - alignment_need) * action[0] / 39.0)
        if ata < 0.5 and abs(relative_altitude) < 0.2:
            scores["accelerate_level"] += 2.0
        manoeuvre = max(sorted(scores), key=scores.get)
        action = np.asarray(MANOEUVRES[manoeuvre], dtype=np.float64)
        # Minimal self-state stabilization permitted by the paper environment:
        # it uses only formal ego pitch/roll and does not alter target selection.
        pitch_norm = float(observation["ego_state"][4])
        roll_norm = float(observation["ego_state"][6])
        action[1] -= 14.0 * roll_norm
        action[2] += 14.0 * pitch_norm
        action = np.clip(np.rint(action), 0, 39).astype(np.int64)
        return action, {"agent_id": agent_id, "target_slot": slot, "scores": scores,
                        "manoeuvre": manoeuvre, "action_indices": action.tolist()}
