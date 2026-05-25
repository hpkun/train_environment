"""Pure geometry diagnostics for AO / TA / q_LOS comparison.

This module does not import the environment or JSBSim.  It uses the same
geometry utilities that the environment calls internally, so the outputs
reflect what the current reward and observation code would see.
"""
from __future__ import annotations

import numpy as np

from my_uav_env.alignment.los_geometry import (
    angle_between_vectors_rad,
    compute_body_x_q_los,
    compute_velocity_q_los,
)
from my_uav_env.alignment.reward_utils import ta_angle_advantage_fixed, td_distance_advantage
from my_uav_env.alignment.state_extractor import (
    _rotation_inertial_to_body,
    compute_q_los_placeholder,
)
from my_uav_env.utils import get2d_AO_TA_R


def compute_pairwise_3d_q_los(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_rpy: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    target_rpy: np.ndarray,
) -> dict:
    """Compute all three q_LOS definitions for both ego and target.

    Returns a dict with keys ego_body_x_q_{rad,deg}, target_body_x_q_{rad,deg},
    ego_velocity_q_{rad,deg}, target_velocity_q_{rad,deg}.
    """
    ego_body_x = compute_body_x_q_los(ego_pos, ego_rpy, target_pos)
    tgt_body_x = compute_body_x_q_los(target_pos, target_rpy, ego_pos)
    ego_vel_q = compute_velocity_q_los(ego_pos, ego_vel, target_pos)
    tgt_vel_q = compute_velocity_q_los(target_pos, target_vel, ego_pos)
    return {
        "ego_body_x_q_rad": ego_body_x,
        "target_body_x_q_rad": tgt_body_x,
        "ego_velocity_q_rad": ego_vel_q,
        "target_velocity_q_rad": tgt_vel_q,
        "ego_body_x_q_deg": float(np.rad2deg(ego_body_x)),
        "target_body_x_q_deg": float(np.rad2deg(tgt_body_x)),
        "ego_velocity_q_deg": float(np.rad2deg(ego_vel_q)),
        "target_velocity_q_deg": float(np.rad2deg(tgt_vel_q)),
    }


# ---------------------------------------------------------------------------
#  2D AO/TA (current environment) and body LOS
# ---------------------------------------------------------------------------

def make_feat_2d(pos_neu: np.ndarray, vel_neu: np.ndarray) -> np.ndarray:
    """Convert NEU (z-up) position / velocity to the 6-dim feature used by get2d_AO_TA_R.

    Args:
        pos_neu: (3,)  [north, east, up]
        vel_neu: (3,)  [vn, ve, vu]

    Returns:
        feat: (6,)  [x, y, down, vx, vy, -vz]
    """
    pos_neu = np.asarray(pos_neu, dtype=np.float64)
    vel_neu = np.asarray(vel_neu, dtype=np.float64)
    return np.array([
        pos_neu[0], pos_neu[1], -pos_neu[2],
        vel_neu[0], vel_neu[1], -vel_neu[2],
    ], dtype=np.float64)


def compute_current_ao_ta_r(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_vel: np.ndarray,
) -> dict:
    """Compute the 2D AO / TA / R values that the current situation reward uses.

    ``get2d_AO_TA_R`` only uses the horizontal (north/east) components.
    Altitude differences are ignored for AO/TA, though the down component
    feeds into the signed side-flag computation on the cross-product.

    Returns keys: AO_rad, TA_rad, R_m, AO_deg, TA_deg.
    """
    ego_feat = make_feat_2d(ego_pos, ego_vel)
    tgt_feat = make_feat_2d(tgt_pos, tgt_vel)
    AO_unsigned, TA, R, side_flag = get2d_AO_TA_R(ego_feat, tgt_feat,
                                                    return_side=True)
    AO_signed = float(AO_unsigned * side_flag)
    return {
        "AO_rad": AO_signed,
        "TA_rad": float(TA),
        "R_m": float(R),
        "AO_deg": float(np.rad2deg(AO_signed)),
        "TA_deg": float(np.rad2deg(TA)),
    }


def compute_body_los_angles(
    ego_pos: np.ndarray,
    ego_rpy: np.ndarray,
    tgt_pos: np.ndarray,
) -> dict:
    """Compute body-frame line-of-sight angles (paper Table 2 geometry).

    Rotates the relative position vector into ego's body frame using the
    same rotation matrix that ``state_extractor`` uses, then computes:

    - theta_los: elevation of LOS in body frame (arctan2(z_body, horizontal))
    - psi_los:   azimuth of LOS in body frame (arctan2(y_body, x_body))
    - q_los_body_x: placeholder LOS angle = arccos(x_body / d)

    These are *3D* angles that account for altitude differences, unlike
    ``get2d_AO_TA_R`` which operates purely in the horizontal plane.
    """
    ego_pos = np.asarray(ego_pos, dtype=np.float64)
    tgt_pos = np.asarray(tgt_pos, dtype=np.float64)
    roll, pitch, heading = (float(ego_rpy[0]), float(ego_rpy[1]), float(ego_rpy[2]))
    r_bi = _rotation_inertial_to_body(roll, pitch, heading)
    rel_pos_body = r_bi @ (tgt_pos - ego_pos)
    x_body, y_body, z_body = (float(rel_pos_body[0]),
                              float(rel_pos_body[1]),
                              float(rel_pos_body[2]))
    d = float(np.linalg.norm(rel_pos_body))
    horizontal = float(np.linalg.norm(rel_pos_body[:2]))
    theta_los = float(np.arctan2(z_body, horizontal))
    psi_los = float(np.arctan2(y_body, x_body))
    q_los = compute_q_los_placeholder(rel_pos_body)
    return {
        "x_body": x_body,
        "y_body": y_body,
        "z_body": z_body,
        "theta_los_rad": theta_los,
        "psi_los_rad": psi_los,
        "q_los_body_x_rad": q_los,
        "theta_los_deg": float(np.rad2deg(theta_los)),
        "psi_los_deg": float(np.rad2deg(psi_los)),
        "q_los_body_x_deg": float(np.rad2deg(q_los)),
    }


# ---------------------------------------------------------------------------
#  Diagnostic print helper
# ---------------------------------------------------------------------------

def describe_geometry_case(
    name: str,
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_rpy: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_vel: np.ndarray,
    tgt_rpy: np.ndarray,
) -> str:
    """Return a human-readable diagnostic block for one geometry case."""
    ao_ta = compute_current_ao_ta_r(ego_pos, ego_vel, tgt_pos, tgt_vel)
    los = compute_body_los_angles(ego_pos, ego_rpy, tgt_pos)
    q3d = compute_pairwise_3d_q_los(
        ego_pos, ego_vel, ego_rpy, tgt_pos, tgt_vel, tgt_rpy,
    )
    lines = [
        f"=== {name} ===",
        f"  Ego pos (NEU):    [{ego_pos[0]:.0f}, {ego_pos[1]:.0f}, {ego_pos[2]:.0f}]",
        f"  Ego vel (NEU):    [{ego_vel[0]:.0f}, {ego_vel[1]:.0f}, {ego_vel[2]:.0f}]",
        f"  Ego rpy (deg):    [{np.rad2deg(ego_rpy[0]):.1f}, "
        f"{np.rad2deg(ego_rpy[1]):.1f}, {np.rad2deg(ego_rpy[2]):.1f}]",
        f"  Tgt pos (NEU):    [{tgt_pos[0]:.0f}, {tgt_pos[1]:.0f}, {tgt_pos[2]:.0f}]",
        f"  Tgt vel (NEU):    [{tgt_vel[0]:.0f}, {tgt_vel[1]:.0f}, {tgt_vel[2]:.0f}]",
        f"  Tgt rpy (deg):    [{np.rad2deg(tgt_rpy[0]):.1f}, "
        f"{np.rad2deg(tgt_rpy[1]):.1f}, {np.rad2deg(tgt_rpy[2]):.1f}]",
        f"  --- current 2D AO/TA (horizontal plane only) ---",
        f"  AO = {ao_ta['AO_deg']:+.2f} deg",
        f"  TA = {ao_ta['TA_deg']:+.2f} deg",
        f"  R  = {ao_ta['R_m']:.1f} m",
        f"  --- body-frame 3D LOS angles (ego → target) ---",
        f"  x_body = {los['x_body']:.4f}, y_body = {los['y_body']:.4f}, "
        f"z_body = {los['z_body']:.4f}",
        f"  theta_los (elevation) = {los['theta_los_deg']:+.3f} deg",
        f"  psi_los   (azimuth)    = {los['psi_los_deg']:+.3f} deg",
        f"  q_los_body_x           = {los['q_los_body_x_deg']:+.3f} deg",
        f"  --- 3D q_LOS candidates ---",
        f"  ego body-x q      = {q3d['ego_body_x_q_deg']:+.3f} deg",
        f"  target body-x q   = {q3d['target_body_x_q_deg']:+.3f} deg",
        f"  ego velocity q    = {q3d['ego_velocity_q_deg']:+.3f} deg",
        f"  target velocity q = {q3d['target_velocity_q_deg']:+.3f} deg",
        f"  --- note ---",
        "  Current AO/TA = 2D horizontal geometry (get2d_AO_TA_R).",
        "  body-x q      = 3D angle between LOS and body x-axis.",
        "  velocity q    = 3D angle between LOS and velocity vector.",
        "  Which of these corresponds to the paper's q_LOS (eq.20) needs",
        "  confirmation against the paper's Table 2 variable definitions.",
        "  They are NOT equivalent when there is a vertical offset.",
    ]
    return "\n".join(lines)
