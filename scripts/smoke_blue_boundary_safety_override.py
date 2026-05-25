"""Pure smoke test for blue boundary safety override in combat pursuit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_agent import (
    _should_override_for_boundary_safety,
    blue_pursuit_action,
)


def _fake_combat_obs() -> dict:
    enemy = np.zeros((1, 11), dtype=np.float32)
    enemy[0, 0] = 0.05       # body dx
    enemy[0, 5] = 0.1        # range normalized -> 8km
    enemy[0, 6] = 0.5        # target speed
    return {
        "ego_state": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "enemy_states": enemy,
        "ally_states": np.zeros((0, 11), dtype=np.float32),
        "death_mask": np.array([1.0, 1.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        # heading east/outward for position [0, 39000]
        "velocity": np.array([0.0, 300.0, 0.0], dtype=np.float32),
    }


def main() -> None:
    heading_east = np.pi / 2
    safe = np.array([0.0, 30000.0, 6000.0], dtype=np.float32)
    near_out = np.array([0.0, 39000.0, 6000.0], dtype=np.float32)
    near_in = np.array([0.0, 39000.0, 6000.0], dtype=np.float32)

    assert not _should_override_for_boundary_safety(safe, heading_east)
    assert _should_override_for_boundary_safety(near_out, heading_east)
    assert not _should_override_for_boundary_safety(near_in, -np.pi / 2)
    assert not _should_override_for_boundary_safety(None, heading_east)

    obs = _fake_combat_obs()
    no_own = blue_pursuit_action(obs, 1, 1, 0)
    safe_action = blue_pursuit_action(obs, 1, 1, 0, own_position=safe)
    override = blue_pursuit_action(obs, 1, 1, 0, own_position=near_out)

    assert no_own.shape == (3,)
    assert safe_action.shape == (3,)
    assert override.shape == (3,)
    assert np.all(np.isfinite(override))

    # Override should steer more toward center (west from east heading), and slow down.
    assert override[1] < no_own[1], (override, no_own)
    assert override[2] < no_own[2], (override, no_own)
    assert np.allclose(no_own, safe_action), (no_own, safe_action)

    print("blue boundary safety override smoke test passed")


if __name__ == "__main__":
    main()
