"""Pure formula smoke test for situation reward Ta/Td composition."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_utils import ta_angle_advantage_fixed, td_distance_advantage


def _assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def _r_adv(ao_deg: float, ta_deg: float, distance_m: float) -> float:
    ta_ij = ta_angle_advantage_fixed(ao_deg)
    td = td_distance_advantage(distance_m)
    ta_ji = ta_angle_advantage_fixed(ta_deg)
    return 1.0 * ta_ij * td - 0.8 * ta_ji * td


def main() -> None:
    _assert_close(_r_adv(0.0, 180.0, 10000.0), 1.0)
    _assert_close(_r_adv(180.0, 0.0, 10000.0), -0.8)
    _assert_close(_r_adv(0.0, 0.0, 10000.0), 0.2)

    td_far = td_distance_advantage(30000.0)
    assert 0.0 < td_far < 1.0
    assert _r_adv(0.0, 180.0, 30000.0) < 1.0

    print("situation reward formula smoke test passed")


if __name__ == "__main__":
    main()
