"""Pure Python smoke test for pitch and speed reward helpers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.reward_utils import (
    pitch_penalty_current,
    pitch_penalty_paper_candidate,
    sample_pitch_table,
    sample_speed_table,
    speed_penalty_current,
    speed_penalty_paper_candidate,
)


def _assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def main() -> None:
    _assert_close(pitch_penalty_current(0.0), 0.0)
    _assert_close(pitch_penalty_current(math.pi / 4.0), 0.0)
    _assert_close(pitch_penalty_current(math.pi / 3.0 + 1e-6), -1.0)

    _assert_close(speed_penalty_current(0.1), -1.0)
    assert np.isclose(speed_penalty_current(0.2), -1.0)
    _assert_close(speed_penalty_current(0.3), 0.0)
    _assert_close(speed_penalty_current(0.5), 0.0)

    for _deg, value in sample_pitch_table(pitch_penalty_paper_candidate):
        assert math.isfinite(value)
    for _mach, value in sample_speed_table(speed_penalty_paper_candidate):
        assert math.isfinite(value)

    print("current pitch table:")
    for deg, value in sample_pitch_table(pitch_penalty_current):
        print(f"  theta={deg:6.1f} deg -> {value: .6f}")

    print("candidate pitch table:")
    for deg, value in sample_pitch_table(pitch_penalty_paper_candidate):
        print(f"  theta={deg:6.1f} deg -> {value: .6f}")

    print("current speed table:")
    for mach, value in sample_speed_table(speed_penalty_current):
        print(f"  mach={mach:4.2f} -> {value: .6f}")

    print("candidate speed table:")
    for mach, value in sample_speed_table(speed_penalty_paper_candidate):
        print(f"  mach={mach:4.2f} -> {value: .6f}")

    print("pitch/speed reward utils smoke test passed")


if __name__ == "__main__":
    main()
