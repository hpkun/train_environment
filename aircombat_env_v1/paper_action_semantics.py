"""Published TAM 40-level direct flight-control action mapping."""
from __future__ import annotations
import numpy as np

ACTION_LEVELS = 40
INACTIVE_ACTION_PLACEHOLDER = (20, 20, 20, 20)


def map_action_indices(indices):
    """Map indices to [throttle, aileron, elevator, rudder]."""
    values = np.clip(np.asarray(indices, dtype=np.int64).reshape(4), 0, 39)
    return np.array([0.4 + values[0] / 39.0 * 0.5,
                     -1.0 + values[1] / 39.0 * 2.0,
                     -1.0 + values[2] / 39.0 * 2.0,
                     -1.0 + values[3] / 39.0 * 2.0], dtype=np.float64)
