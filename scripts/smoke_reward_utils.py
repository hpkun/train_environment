"""Pure Python smoke test for situation-reward helper functions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.reward_utils import (
    sample_ta_table,
    ta_angle_advantage_candidate_continuous,
    ta_angle_advantage_current,
    ta_angle_advantage_fixed,
    td_distance_advantage,
    td_distance_advantage_current,
)


def _assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def main() -> None:
    _assert_close(ta_angle_advantage_current(0.0), 1.0)
    _assert_close(ta_angle_advantage_current(4.0), 1.0)
    assert ta_angle_advantage_current(15.0) < 0.0
    assert ta_angle_advantage_current(35.0) > 0.0
    _assert_close(ta_angle_advantage_current(40.0), 0.0)

    _assert_close(td_distance_advantage_current(15000.0), 1.0)
    assert td_distance_advantage_current(30000.0) < 1.0
    _assert_close(td_distance_advantage(15000.0), 1.0)
    assert td_distance_advantage(30000.0) < 1.0

    _assert_close(ta_angle_advantage_fixed(0.0), 1.0)
    _assert_close(ta_angle_advantage_fixed(4.0), 1.0)
    _assert_close(ta_angle_advantage_fixed(15.0), 0.5)
    _assert_close(ta_angle_advantage_fixed(35.0), 0.0)
    _assert_close(ta_angle_advantage_fixed(40.0), 0.0)

    for _angle, value in sample_ta_table(ta_angle_advantage_fixed):
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)

    for boundary in (4.0, 15.0, 35.0):
        left = ta_angle_advantage_fixed(boundary - 1e-6)
        right = ta_angle_advantage_fixed(boundary + 1e-6)
        assert abs(left - right) < 1e-4

    for _angle, value in sample_ta_table(ta_angle_advantage_candidate_continuous):
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)

    print("current Ta table:")
    for angle, value in sample_ta_table(ta_angle_advantage_current):
        print(f"  q={angle:5.1f} deg -> {value: .6f}")

    print("fixed Ta table:")
    for angle, value in sample_ta_table(ta_angle_advantage_fixed):
        print(f"  q={angle:5.1f} deg -> {value: .6f}")

    print("reward utils smoke test passed")


if __name__ == "__main__":
    main()
