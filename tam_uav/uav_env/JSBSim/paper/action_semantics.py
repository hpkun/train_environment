"""Published 40-level action discretization and its exact center semantics."""

from __future__ import annotations

import numpy as np


ACTION_LEVELS = 40
LOWER_CENTER_INDEX = 19
UPPER_CENTER_INDEX = 20
LOWER_CENTER_VALUE = -1.0 / 39.0
UPPER_CENTER_VALUE = 1.0 / 39.0
NEAREST_POSITIVE_CENTER = "nearest_positive_center"
NEUTRAL_ACTION_SEMANTICS = "nearest_positive_center_not_exact_zero"
INACTIVE_ACTION_PLACEHOLDER = (20, 20, 20, 20)


def map_action_indices(indices) -> np.ndarray:
    """Map four discrete bins to throttle and three direct-control commands."""
    values = np.clip(np.asarray(indices, dtype=np.int64).reshape(4),
                     0, ACTION_LEVELS - 1)
    denominator = float(ACTION_LEVELS - 1)
    return np.array([
        0.4 + values[0] / denominator * 0.5,
        -1.0 + values[1] / denominator * 2.0,
        -1.0 + values[2] / denominator * 2.0,
        -1.0 + values[3] / denominator * 2.0,
    ], dtype=np.float64)
