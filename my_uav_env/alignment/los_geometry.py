"""Pure 3D line-of-sight geometry functions for situation reward and diagnostics.

This module does not import the environment or JSBSim.  It provides the
canonical body-x q_LOS computation used by the updated ``_situation_reward()``
and by the diagnostic / candidate modules.
"""
from __future__ import annotations

import numpy as np

from my_uav_env.alignment.state_extractor import (
    _rotation_inertial_to_body,
    compute_q_los_placeholder,
)


def angle_between_vectors_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Return the 3D angle in [0, pi] between two vectors.

    Returns 0.0 if either vector has near-zero norm.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 0.0
    cos_angle = float(np.dot(a, b)) / (a_norm * b_norm)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def compute_3d_range(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Return the 3D Euclidean distance between two NEU positions."""
    return float(np.linalg.norm(
        np.asarray(pos_b, dtype=np.float64) - np.asarray(pos_a, dtype=np.float64)))


def compute_body_x_q_los(
    observer_pos: np.ndarray,
    observer_rpy: np.ndarray,
    target_pos: np.ndarray,
) -> float:
    """3D angle between the observer body x-axis and the LOS to target.

    Rotates ``target_pos - observer_pos`` into the observer's body frame
    using the same rotation-matrix convention as ``state_extractor``, then
    returns ``compute_q_los_placeholder`` — the angle between the LOS
    vector and the body x-axis.  Range: [0, pi].
    """
    roll, pitch, heading = (float(observer_rpy[0]),
                            float(observer_rpy[1]),
                            float(observer_rpy[2]))
    r_bi = _rotation_inertial_to_body(roll, pitch, heading)
    rel_pos_body = r_bi @ (np.asarray(target_pos, dtype=np.float64)
                           - np.asarray(observer_pos, dtype=np.float64))
    return compute_q_los_placeholder(rel_pos_body)


def compute_velocity_q_los(
    observer_pos: np.ndarray,
    observer_vel: np.ndarray,
    target_pos: np.ndarray,
) -> float:
    """3D angle between the observer velocity vector and the LOS to target.

    Range: [0, pi].  Returns 0.0 if distance or velocity is near-zero.
    """
    rel_pos = (np.asarray(target_pos, dtype=np.float64)
               - np.asarray(observer_pos, dtype=np.float64))
    return angle_between_vectors_rad(
        np.asarray(observer_vel, dtype=np.float64), rel_pos)
