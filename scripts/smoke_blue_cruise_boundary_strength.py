"""Pure smoke test for strengthened blue no-target boundary patrol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_agent import (
    _blue_cruise_heading_command,
    _blue_cruise_speed_command,
    _boundary_outward_heading_component,
    _boundary_patrol_heading_command,
    _boundary_patrol_pressure,
    blue_coordinated_actions,
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
        "velocity": np.array([0.0, 300.0, 0.0], dtype=np.float32),
    }


def _assert_cmd(value: float) -> None:
    assert np.isfinite(value)
    assert -1.0 <= value <= 1.0


def main() -> None:
    obs = _fake_no_target_obs()

    center = np.array([0.0, 0.0, 6000.0], dtype=np.float32)
    assert _boundary_patrol_pressure(center) == 0.0
    assert _boundary_patrol_heading_command(center, 0.0) == 0.0
    assert _blue_cruise_speed_command(center) == 1.0

    mid = np.array([0.0, 25000.0, 6000.0], dtype=np.float32)
    assert _boundary_patrol_pressure(mid) == 0.0
    assert _boundary_patrol_heading_command(mid, np.pi / 2) == 0.0
    assert _blue_cruise_speed_command(mid) == 1.0

    near_patrol = np.array([0.0, 30000.0, 6000.0], dtype=np.float32)
    p_near = _boundary_patrol_pressure(near_patrol)
    assert p_near > 0.0, f"expected pressure > 0 at 30km, got {p_near}"
    assert _blue_cruise_speed_command(near_patrol) < 1.0
    _assert_cmd(_boundary_patrol_heading_command(near_patrol, np.pi / 2))

    near_edge = np.array([0.0, 39000.0, 6000.0], dtype=np.float32)
    p_edge = _boundary_patrol_pressure(near_edge)
    assert 0.85 <= p_edge <= 1.05, f"pressure near edge: {p_edge}"
    assert _blue_cruise_speed_command(near_edge) <= 0.3
    _assert_cmd(_boundary_patrol_heading_command(near_edge, np.pi / 2))

    outside = np.array([0.0, 45000.0, 6000.0], dtype=np.float32)
    p_out = _boundary_patrol_pressure(outside)
    assert p_out > 1.0, f"pressure outside: {p_out}"
    assert _blue_cruise_speed_command(outside) <= 0.25
    _assert_cmd(_boundary_patrol_heading_command(outside, np.pi / 2))

    assert _boundary_outward_heading_component(near_edge, np.pi / 2) > 0.9
    assert _boundary_outward_heading_component(near_edge, -np.pi / 2) < -0.9
    assert abs(_boundary_outward_heading_component(near_edge, 0.0)) < 0.1

    compare_pos = np.array([30000.0, 30000.0, 6000.0], dtype=np.float32)
    cmd_outward = _boundary_patrol_heading_command(compare_pos, np.pi / 4)
    cmd_tangent = _boundary_patrol_heading_command(compare_pos, 3.0 * np.pi / 4)
    cmd_inward = _boundary_patrol_heading_command(compare_pos, -3.0 * np.pi / 4)
    if abs(cmd_outward) < 0.999 and abs(cmd_tangent) < 0.999:
        assert abs(cmd_outward) > abs(cmd_tangent) > abs(cmd_inward), (
            cmd_outward, cmd_tangent, cmd_inward)
    else:
        assert abs(cmd_outward) >= 0.999, (cmd_outward, cmd_tangent, cmd_inward)
        assert abs(cmd_inward) < abs(cmd_outward), (
            cmd_outward, cmd_tangent, cmd_inward)

    speed_outward = _blue_cruise_speed_command(compare_pos, np.pi / 4)
    speed_tangent = _blue_cruise_speed_command(compare_pos, 3.0 * np.pi / 4)
    speed_inward = _blue_cruise_speed_command(compare_pos, -3.0 * np.pi / 4)
    assert speed_outward < speed_tangent <= speed_inward, (
        speed_outward, speed_tangent, speed_inward)

    assert _blue_cruise_speed_command(None) == 1.0
    assert _blue_cruise_heading_command(obs, 0, own_position=None) == 0.0

    blue_obs = {"blue_0": obs}
    old_action = blue_coordinated_actions(
        blue_obs, 1, 1, engaged_targets=set())["blue_0"]
    new_action = blue_coordinated_actions(
        blue_obs,
        1,
        1,
        engaged_targets=set(),
        own_positions={"blue_0": near_edge},
    )["blue_0"]
    assert old_action.shape == (3,)
    assert new_action.shape == (3,)
    assert (
        not np.isclose(old_action[1], new_action[1])
        or not np.isclose(old_action[2], new_action[2])
    )

    print("blue cruise boundary strength smoke test passed")


if __name__ == "__main__":
    main()
