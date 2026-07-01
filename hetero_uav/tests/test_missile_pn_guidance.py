"""Tests for BRMA-style PN missile guidance contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.simulator import MissileSimulator


def test_pn_acceleration_is_perpendicular_to_missile_velocity():
    missile_pos = np.array([0.0, 0.0, 0.0])
    missile_vel = np.array([600.0, 0.0, 0.0])
    target_pos = np.array([5000.0, 1000.0, 0.0])
    target_vel = np.array([250.0, -80.0, 0.0])

    a_cmd, diag = MissileSimulator.compute_pn_lateral_acceleration(
        missile_pos,
        missile_vel,
        target_pos,
        target_vel,
        navigation_gain=3.0,
        max_overload_g=30.0,
    )

    missile_dir = missile_vel / np.linalg.norm(missile_vel)
    assert abs(float(np.dot(a_cmd, missile_dir))) < 1e-6
    assert diag["closing_speed_mps"] > 0.0


def test_pn_acceleration_respects_max_overload():
    a_cmd, _diag = MissileSimulator.compute_pn_lateral_acceleration(
        missile_pos=np.array([0.0, 0.0, 0.0]),
        missile_vel=np.array([600.0, 0.0, 0.0]),
        target_pos=np.array([500.0, 5000.0, 0.0]),
        target_vel=np.array([-250.0, -500.0, 0.0]),
        navigation_gain=8.0,
        max_overload_g=30.0,
    )

    assert np.linalg.norm(a_cmd) <= 30.0 * 9.81 + 1e-6


def test_pn_guidance_responds_to_los_rate_not_pure_los_direction():
    missile_pos = np.array([0.0, 0.0, 0.0])
    missile_vel = np.array([600.0, 0.0, 0.0])
    target_pos = np.array([5000.0, 1000.0, 0.0])
    target_vel = np.array([250.0, 200.0, 0.0])

    a_cmd, _diag = MissileSimulator.compute_pn_lateral_acceleration(
        missile_pos,
        missile_vel,
        target_pos,
        target_vel,
        navigation_gain=3.0,
        max_overload_g=30.0,
    )

    los_dir = target_pos - missile_pos
    los_dir = los_dir / np.linalg.norm(los_dir)
    a_dir = a_cmd / max(np.linalg.norm(a_cmd), 1e-8)

    assert np.linalg.norm(a_cmd) > 1e-6
    assert abs(float(np.dot(a_dir, los_dir))) < 0.95
