"""Candidate 3D situation-reward functions for paper-alignment experiments.

These functions are NOT wired into ``UavCombatEnv._situation_reward()``.
They exist so geometry-ablation experiments can compare the current 2D-AO/TA
reward against 3D q_LOS formulations without changing the training environment.

The core question: does paper eq.20 use a 2D horizontal angle or a 3D
body-frame / velocity-frame line-of-sight angle?

Three formulations:

1. **current** — 2D AO / TA via get2d_AO_TA_R (horizontal plane only).
2. **body-x** — 3D angle between LOS and observer body x-axis.
3. **velocity** — 3D angle between LOS and observer velocity vector.

All use the same Ta/Td decomposition and weights (1.0 / 0.8) as the
current environment.
"""
from __future__ import annotations

import numpy as np

from .geometry_diagnostics import (
    compute_current_ao_ta_r,
)
from .los_geometry import (
    compute_3d_range,
    compute_body_x_q_los,
    compute_velocity_q_los,
)
from .reward_utils import (
    ta_angle_advantage_fixed,
    td_distance_advantage,
)


def situation_reward_current_formula(
    AO_rad: float,
    TA_rad: float,
    distance_m: float,
) -> float:
    """Current situation-reward formula using given 2D AO/TA and distance.

    Matches ``UavCombatEnv._situation_reward()`` under ``fixed_ta_v1``:
      r_adv_ij = 1.0 * Ta(AO) * Td(R) - 0.8 * Ta(TA) * Td(R)
    """
    Ta_ij = ta_angle_advantage_fixed(np.rad2deg(AO_rad))
    Ta_ji = ta_angle_advantage_fixed(np.rad2deg(TA_rad))
    Td = td_distance_advantage(distance_m)
    return 1.0 * Ta_ij * Td - 0.8 * Ta_ji * Td


def situation_reward_3d_body_x_candidate(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_rpy: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_rpy: np.ndarray,
) -> float:
    """Candidate situation reward using body-x q_LOS (3D) for both sides.

    q_ij = angle between ego body x-axis and LOS to target (3D).
    q_ji = angle between target body x-axis and LOS to ego (3D).

    Uses the same Ta/Td decomposition and weights as the current formula.
    """
    q_ij = compute_body_x_q_los(ego_pos, ego_rpy, target_pos)
    q_ji = compute_body_x_q_los(target_pos, target_rpy, ego_pos)
    distance = compute_3d_range(ego_pos, target_pos)
    Ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
    Ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))
    Td = td_distance_advantage(distance)
    return 1.0 * Ta_ij * Td - 0.8 * Ta_ji * Td


def situation_reward_3d_velocity_candidate(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
) -> float:
    """Candidate situation reward using velocity-LOS q (3D) for both sides.

    q_ij = angle between ego velocity vector and LOS to target (3D).
    q_ji = angle between target velocity vector and LOS to ego (3D).

    Uses the same Ta/Td decomposition and weights as the current formula.
    """
    q_ij = compute_velocity_q_los(ego_pos, ego_vel, target_pos)
    q_ji = compute_velocity_q_los(target_pos, target_vel, ego_pos)
    distance = compute_3d_range(ego_pos, target_pos)
    Ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
    Ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))
    Td = td_distance_advantage(distance)
    return 1.0 * Ta_ij * Td - 0.8 * Ta_ji * Td


def compare_situation_reward_candidates(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_rpy: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_rpy: np.ndarray,
) -> dict:
    """Return all three candidate rewards plus the 2D AO/TA for one pair."""
    ao_ta = compute_current_ao_ta_r(ego_pos, ego_vel, target_pos, target_vel)
    body_x = situation_reward_3d_body_x_candidate(
        ego_pos, ego_vel, ego_rpy, target_pos, target_vel, target_rpy,
    )
    vel_q = situation_reward_3d_velocity_candidate(
        ego_pos, ego_vel, target_pos, target_vel,
    )
    current_val = situation_reward_current_formula(
        ao_ta["AO_rad"], ao_ta["TA_rad"], ao_ta["R_m"],
    )
    return {
        "current_2d_ao_ta": float(current_val),
        "candidate_3d_body_x": float(body_x),
        "candidate_3d_velocity": float(vel_q),
        "ao_deg": ao_ta["AO_deg"],
        "ta_deg": ao_ta["TA_deg"],
        "distance_m": ao_ta["R_m"],
    }
