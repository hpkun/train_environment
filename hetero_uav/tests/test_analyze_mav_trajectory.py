from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_mav_trajectory.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--fixed-episodes" in result.stdout
    assert "--workers" in result.stdout


def test_fixed_mav_actions_are_finite_and_clipped():
    from scripts.analyze_mav_trajectory import fixed_mav_action

    blue = {"blue_0": np.array([1000.0, 0.0, 6000.0])}
    for behavior in ("level_flight", "rear_retreat", "gentle_loiter", "climb_safe"):
        action = fixed_mav_action(
            behavior,
            current_heading=0.0,
            initial_heading=0.0,
            sim_time=4.0,
            mav_position=np.array([0.0, 0.0, 6000.0]),
            blue_positions=blue,
        )
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all(np.abs(action) <= 1.0)


def test_rear_retreat_points_away_from_blue():
    from scripts.analyze_mav_trajectory import fixed_mav_action

    action = fixed_mav_action(
        "rear_retreat",
        current_heading=0.0,
        initial_heading=0.0,
        sim_time=0.0,
        mav_position=np.array([0.0, 0.0, 6000.0]),
        blue_positions={"blue_0": np.array([1000.0, 0.0, 6000.0])},
    )
    decoded_heading = float(action[1]) * math.pi
    assert abs(abs(decoded_heading) - math.pi) < 1e-5


def test_predeath_window_stats_uses_last_ten_seconds():
    from scripts.analyze_mav_trajectory import predeath_window_stats

    rows = [
        {
            "sim_time": float(t),
            "mav_altitude_m": 6000.0 + t,
            "mav_speed_mps": 200.0 + t,
            "mav_roll_deg": -float(t),
            "mav_pitch_deg": float(t) / 2.0,
            "mav_yaw_deg": float(t) * 2.0,
        }
        for t in range(21)
    ]
    stats = predeath_window_stats(rows, death_time_sec=20.0, window_sec=10.0)
    assert stats["predeath_sample_count"] == 11
    assert stats["predeath_altitude_min_m"] == 6010.0
    assert stats["predeath_altitude_max_m"] == 6020.0
    assert stats["predeath_speed_mean_mps"] == 215.0
