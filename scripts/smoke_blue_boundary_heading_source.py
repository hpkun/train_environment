"""Pure smoke test for blue boundary heading-source correction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_agent import (
    _boundary_outward_heading_component,
    _should_override_for_boundary_safety,
    blue_pursuit_action,
)


def _fake_combat_obs() -> dict:
    enemy = np.zeros((1, 11), dtype=np.float32)
    enemy[0, 0] = 0.05
    enemy[0, 5] = 0.1
    enemy[0, 6] = 0.5
    return {
        "ego_state": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "enemy_states": enemy,
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "death_mask": np.array([1.0, 1.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        # Velocity track points north, while own_heading can point east.
        "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
    }


def main() -> None:
    obs = _fake_combat_obs()
    own_position = np.array([0.0, 39000.0, 6000.0], dtype=np.float32)
    own_heading = np.pi / 2

    assert _boundary_outward_heading_component(own_position, own_heading) > 0.9
    assert _boundary_outward_heading_component(own_position, -np.pi / 2) < -0.9
    assert _should_override_for_boundary_safety(own_position, own_heading)

    yaw_action = blue_pursuit_action(
        obs, 1, 1, 0, own_position=own_position, own_heading=own_heading)
    fallback_action = blue_pursuit_action(obs, 1, 1, 0, own_position=own_position)

    assert yaw_action.shape == (3,)
    assert fallback_action.shape == (3,)
    assert np.all(np.isfinite(yaw_action))
    assert np.all(np.isfinite(fallback_action))

    # With own yaw=pi/2, target heading should remain near east while turning
    # westward by the rule agent's limited 10 degree authority. Fallback uses
    # velocity-track north as the absolute-heading base and lands near zero.
    assert yaw_action[1] > 0.3, yaw_action
    assert fallback_action[1] < 0.1, fallback_action
    assert not np.isclose(yaw_action[1], fallback_action[1])
    assert yaw_action[2] < 1.0

    no_own_position = blue_pursuit_action(obs, 1, 1, 0, own_heading=own_heading)
    assert no_own_position.shape == (3,)
    assert np.all(np.isfinite(no_own_position))

    print("blue boundary heading source smoke test passed")


if __name__ == "__main__":
    main()
