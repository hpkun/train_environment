"""Small potential-based reward for the minimal 1v1 environment."""

from __future__ import annotations

import numpy as np


def potential(geometry):
    alignment = np.cos(geometry["boresight_angle"])
    range_score = np.exp(-((geometry["distance_m"] - 1500.0) / 2500.0) ** 2)
    closure_score = np.clip(geometry["closure_mps"] / 300.0, -1.0, 1.0)
    return float(0.6 * alignment + 0.25 * range_score + 0.15 * closure_score)


def terminal_reward(event):
    if event in ("red_hit", "blue_crash"):
        return 10.0
    if event in ("blue_hit", "red_crash"):
        return -10.0
    return 0.0


def step_reward(previous_potential, next_potential, red_dwell_delta,
                blue_dwell_delta, event=None):
    shaped = (0.99 * next_potential - previous_potential - 0.001
              + 0.02 * (red_dwell_delta - blue_dwell_delta))
    return float(shaped + terminal_reward(event))
