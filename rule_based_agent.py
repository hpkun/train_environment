"""Minimal rule-action helpers for the formal paper-derived 3V3 profile."""
from __future__ import annotations

import numpy as np


VELOCITY_MIN_MPS = 102.0
VELOCITY_MAX_MPS = 408.0


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _paper_absolute_action(pitch_rad: float, heading_rad: float,
                           speed_mps: float = 300.0) -> np.ndarray:
    """Map absolute paper setpoints to the normalized three-action vector."""
    pitch = float(np.clip(pitch_rad / (np.pi / 2.0), -1.0, 1.0))
    heading = float(np.clip(_wrap_pi(heading_rad) / np.pi, -1.0, 1.0))
    speed = float(np.clip(
        2.0 * (speed_mps - VELOCITY_MIN_MPS)
        / (VELOCITY_MAX_MPS - VELOCITY_MIN_MPS) - 1.0,
        -1.0, 1.0))
    return np.asarray([pitch, heading, speed], dtype=np.float32)


def _strict_target_action(obs: dict, target_index: int,
                          own_heading: float = 0.0) -> np.ndarray:
    enemies = np.asarray(obs.get("enemy_states", ()), dtype=np.float64)
    if target_index < 0 or target_index >= enemies.shape[0]:
        return _paper_absolute_action(0.0, own_heading, 300.0)
    state = enemies[target_index]
    if state.size != 10 or not np.all(np.isfinite(state)) or state[9] <= 0.0:
        return _paper_absolute_action(0.0, own_heading, 300.0)
    pitch = float(state[6])
    relative_bearing = float(state[7])
    return _paper_absolute_action(
        pitch, _wrap_pi(own_heading + relative_bearing), 300.0)


def blue_coordinated_actions(
        blue_obs: dict[str, dict], num_blue: int, num_red: int,
        engaged_targets=None, own_positions=None, own_headings=None,
        **_unused) -> dict[str, np.ndarray]:
    """Strict-observation fallback for dynamic nearest unique-first pursuit.

    ``UavCombatEnv.blue_policy_actions`` is the formal path and supplies true
    internal kinematics. This fallback exists only for callers that expose the
    strict entity observation but not the environment controller.
    """
    del engaged_targets, own_positions
    own_headings = own_headings or {}
    assignments = {}
    taken = set()
    for blue_index in range(num_blue):
        blue_id = f"blue_{blue_index}"
        obs = blue_obs.get(blue_id, {})
        enemies = np.asarray(obs.get("enemy_states", ()), dtype=np.float64)
        mask = np.asarray(obs.get("entity_mask", ()), dtype=np.int64)
        rows = []
        for target_index in range(min(num_red, enemies.shape[0])):
            mask_index = num_blue + target_index
            valid = mask_index < mask.size and mask[mask_index] == 0
            state = enemies[target_index]
            if (valid and state.size == 10 and np.all(np.isfinite(state))
                    and state[9] > 0.0):
                rows.append((float(state[9]), target_index))
        rows.sort(key=lambda row: (row[0], row[1]))
        available = [row for row in rows if row[1] not in taken]
        selected = available or rows
        assignments[blue_id] = selected[0][1] if selected else None
        if selected:
            taken.add(selected[0][1])

    return {
        blue_id: (_paper_absolute_action(
            0.0, own_headings.get(blue_id, 0.0), 300.0)
            if target_index is None else _strict_target_action(
                blue_obs.get(blue_id, {}), target_index,
                own_headings.get(blue_id, 0.0)))
        for blue_id, target_index in assignments.items()
    }
