"""Static smoke test for blue own-position patrol integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_agent import blue_coordinated_actions


def _fake_no_target_blue_obs() -> dict:
    return {
        "ego_state": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "enemy_states": np.zeros((1, 11), dtype=np.float32),
        "death_mask": np.array([1.0, 0.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        # heading east: atan2(east, north) = pi/2
        "velocity": np.array([0.0, 300.0, 0.0], dtype=np.float32),
    }


def main() -> None:
    blue_obs = {"blue_0": _fake_no_target_blue_obs()}

    old_actions = blue_coordinated_actions(
        blue_obs, num_blue=1, num_red=1, engaged_targets=set())
    new_actions = blue_coordinated_actions(
        blue_obs,
        num_blue=1,
        num_red=1,
        engaged_targets=set(),
        own_positions={"blue_0": np.array([0.0, 39000.0, 6000.0], dtype=np.float32)},
    )

    old_action = old_actions["blue_0"]
    new_action = new_actions["blue_0"]

    assert old_action.shape == (3,)
    assert new_action.shape == (3,)
    assert np.all(np.isfinite(old_action))
    assert np.all(np.isfinite(new_action))
    assert not np.isclose(old_action[1], new_action[1])

    print("blue own positions integration static smoke test passed")


if __name__ == "__main__":
    main()
