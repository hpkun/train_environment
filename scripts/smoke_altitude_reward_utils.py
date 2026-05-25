"""Pure Python smoke test for altitude reward helper functions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.reward_utils import (
    altitude_reward_current,
    altitude_reward_pairwise_mean_candidate,
    altitude_reward_paper_candidate,
    sample_altitude_table,
)


def _assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def main() -> None:
    _assert_close(altitude_reward_current(0.0), 0.0)
    _assert_close(altitude_reward_current(2000.0), 1.0)
    _assert_close(altitude_reward_current(5000.0), 1.0)
    _assert_close(altitude_reward_current(10000.0), 0.0)

    _assert_close(altitude_reward_paper_candidate(10000.0), 0.1)
    _assert_close(altitude_reward_paper_candidate(12000.0), 0.1)

    for _dz, value in sample_altitude_table(altitude_reward_paper_candidate):
        assert 0.0 <= value <= 1.0

    expected_pairwise = (
        altitude_reward_paper_candidate(6000.0 - 4000.0)
        + altitude_reward_paper_candidate(6000.0 - 1000.0)
    ) / 2.0
    _assert_close(
        altitude_reward_pairwise_mean_candidate(6000.0, [4000.0, 1000.0]),
        expected_pairwise,
    )

    print("current altitude table:")
    for dz, value in sample_altitude_table(altitude_reward_current):
        print(f"  dz={dz:8.1f} m -> {value: .6f}")

    print("paper candidate altitude table:")
    for dz, value in sample_altitude_table(altitude_reward_paper_candidate):
        print(f"  dz={dz:8.1f} m -> {value: .6f}")

    print("altitude reward utils smoke test passed")


if __name__ == "__main__":
    main()
