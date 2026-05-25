"""Pure smoke test for the canonical los_geometry functions.

No JSBSim, no env import.  Uses static NEU positions and attitude to verify
body-x q_LOS, velocity q_LOS, and 3D range behave correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.los_geometry import (
    angle_between_vectors_rad,
    compute_3d_range,
    compute_body_x_q_los,
    compute_velocity_q_los,
)


def main() -> None:
    # ---- angle_between_vectors_rad ----
    assert angle_between_vectors_rad(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0])) == 0.0
    assert abs(angle_between_vectors_rad(
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])) - np.pi) < 1e-9
    assert angle_between_vectors_rad(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0])) == 0.0  # near-zero norm

    # ---- compute_3d_range ----
    assert compute_3d_range(
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 4.0, 12.0])) == 13.0

    # ---- compute_body_x_q_los: same-alt cases (rpy=[0,0,0], level flight) ----
    rpy_zero = np.array([0.0, 0.0, 0.0])
    origin = np.array([0.0, 0.0, 6000.0])

    # ahead: LOS along +x (body x-axis) → q ≈ 0
    q_ahead = compute_body_x_q_los(origin, rpy_zero,
                                   np.array([10000.0, 0.0, 6000.0]))
    assert q_ahead < 0.01

    # behind: LOS along -x → q ≈ π
    q_behind = compute_body_x_q_los(origin, rpy_zero,
                                    np.array([-10000.0, 0.0, 6000.0]))
    assert q_behind > np.pi - 0.01

    # right: LOS along +y → q ≈ π/2
    q_right = compute_body_x_q_los(origin, rpy_zero,
                                   np.array([0.0, 10000.0, 6000.0]))
    assert abs(q_right - np.pi / 2) < 0.01

    # ---- compute_velocity_q_los: same-alt, velocity aligned +x ----
    vel_fwd = np.array([300.0, 0.0, 0.0])

    q_vel_ahead = compute_velocity_q_los(origin, vel_fwd,
                                         np.array([10000.0, 0.0, 6000.0]))
    assert q_vel_ahead < 0.01

    q_vel_behind = compute_velocity_q_los(origin, vel_fwd,
                                          np.array([-10000.0, 0.0, 6000.0]))
    assert q_vel_behind > np.pi - 0.01

    # zero velocity → 0
    q_vel_zero = compute_velocity_q_los(origin, np.array([0.0, 0.0, 0.0]),
                                        np.array([10000.0, 0.0, 6000.0]))
    assert q_vel_zero == 0.0

    # zero distance → 0
    q_vel_zero_dist = compute_velocity_q_los(origin, vel_fwd, origin)
    assert q_vel_zero_dist == 0.0

    # All outputs finite for a set of cases
    cases = [
        ("ahead", origin, rpy_zero, np.array([10000.0, 0.0, 6000.0])),
        ("behind", origin, rpy_zero, np.array([-10000.0, 0.0, 6000.0])),
        ("right", origin, rpy_zero, np.array([0.0, 10000.0, 6000.0])),
        ("above_ahead", origin, rpy_zero, np.array([10000.0, 0.0, 8000.0])),
    ]
    for _name, obs_pos, obs_rpy, tgt_pos in cases:
        q = compute_body_x_q_los(obs_pos, obs_rpy, tgt_pos)
        assert np.isfinite(q), f"{_name}: body_x_q not finite: {q}"

    print("los geometry smoke test passed")


if __name__ == "__main__":
    main()
