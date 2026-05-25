"""Pure smoke test for blue no-target cruise boundary helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_agent import (
    _boundary_patrol_heading_command,
    blue_pursuit_action,
)


def _fake_no_target_obs() -> dict:
    return {
        "ego_state": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "enemy_states": np.zeros((1, 11), dtype=np.float32),
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "death_mask": np.array([1.0, 0.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
    }


def main() -> None:
    # center: no patrol
    assert _boundary_patrol_heading_command(
        np.array([0.0, 0.0, 6000.0]), 0.0) == 0.0

    # mid area (25km): should not trigger patrol (inner_limit=28km)
    assert _boundary_patrol_heading_command(
        np.array([0.0, 25000.0, 6000.0]), np.pi / 2) == 0.0

    # near edge (39km): strong patrol
    east_cmd = _boundary_patrol_heading_command(
        np.array([0.0, 39000.0, 6000.0]), np.pi / 2)
    assert abs(east_cmd) > 0.1

    north_cmd = _boundary_patrol_heading_command(
        np.array([39000.0, 0.0, 6000.0]), 0.0)
    assert abs(north_cmd) > 0.1

    corner_cmd = _boundary_patrol_heading_command(
        np.array([39000.0, 39000.0, 6000.0]), 0.0)
    assert np.isfinite(corner_cmd)
    assert -1.0 <= corner_cmd <= 1.0

    # heading gain is pressure-scaled: further out → stronger
    near_30 = _boundary_patrol_heading_command(
        np.array([0.0, 30000.0, 6000.0]), np.pi / 2)
    near_39 = _boundary_patrol_heading_command(
        np.array([0.0, 39000.0, 6000.0]), np.pi / 2)
    assert abs(near_30) <= abs(near_39), \
        f"near_30={near_30:.4f}, near_39={near_39:.4f} — further out should be stronger"

    action = blue_pursuit_action(_fake_no_target_obs(), 1, 1, 0)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    print("blue cruise boundary smoke test passed")


if __name__ == "__main__":
    main()
