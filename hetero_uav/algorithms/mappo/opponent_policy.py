"""Scripted opponent policies for Stage 1 MAPPO baseline runners.

These policies live at the training/evaluation script layer. They do not modify
the environment, reward, missile, PID, termination, action mapping, or aircraft
models.
"""
from __future__ import annotations

import numpy as np


class OpponentPolicy:
    """Generate blue-side high-level actions for baseline evaluation.

    Supported modes:
    - ``zero``: all blue actions are zero, for smoke/debug only.
    - ``random``: sample uniformly from [-1, 1].
    - ``rule_nearest``: steer toward the nearest non-zero red entity in each
      blue agent's own ``enemy_states`` observation.
    """

    MODES = {"zero", "random", "rule_nearest"}

    def __init__(self, mode: str = "zero", seed: int | None = None):
        if mode not in self.MODES:
            raise ValueError(f"unknown opponent policy mode: {mode}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def act(self, obs_dict: dict, blue_ids: list[str],
            deterministic: bool = True) -> dict[str, np.ndarray]:
        del deterministic
        if self.mode == "zero":
            return {
                bid: np.zeros(3, dtype=np.float32)
                for bid in blue_ids
            }
        if self.mode == "random":
            return {
                bid: self.rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
                for bid in blue_ids
            }
        return {
            bid: self._rule_nearest_action(obs_dict.get(bid, {}))
            for bid in blue_ids
        }

    @staticmethod
    def _rule_nearest_action(obs: dict) -> np.ndarray:
        enemy_states = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
        if enemy_states.ndim != 2 or enemy_states.shape[0] == 0:
            return np.array([0.0, 0.0, 0.3], dtype=np.float32)

        best_state = None
        best_dist = float("inf")
        for state in enemy_states:
            if state.size < 3 or np.allclose(state, 0.0):
                continue
            rel = state[:3].astype(np.float32)
            dist = float(np.linalg.norm(rel))
            if dist < best_dist:
                best_dist = dist
                best_state = state

        if best_state is None:
            return np.array([0.0, 0.0, 0.3], dtype=np.float32)

        # Existing BRMA observations place enemy states in ego body-relative
        # order. y > 0 means target is to the right; z > 0 means target is above.
        pitch = float(best_state[2]) * 2.0
        heading = float(best_state[1]) * 2.0
        speed = 0.8

        action = np.array([pitch, heading, speed], dtype=np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
