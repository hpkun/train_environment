from __future__ import annotations

import numpy as np
import pytest

from uav_env.JSBSim.env import UavCombatEnv


class _FakeSim:
    is_alive = True

    def __init__(self) -> None:
        self.properties: dict[str, float] = {}

    def get_rpy(self):
        return np.zeros(3, dtype=np.float64)

    def set_property_value(self, name: str, value: float) -> None:
        self.properties[name] = float(value)


class _ExplodingPid:
    def compute_control(self, *args, **kwargs):
        raise AssertionError("TAM direct control must bypass PID")


def _make_direct_env(**overrides) -> UavCombatEnv:
    options = {
        "max_num_blue": 1,
        "max_num_red": 2,
        "action_interface": "tam_direct_fcs_4d",
        "tam_action_levels": 40,
        "tam_throttle_min": 0.4,
        "tam_throttle_max": 0.9,
        "scripted_evasion_red": False,
        "scripted_evasion_blue": False,
    }
    options.update(overrides)
    return UavCombatEnv(**options)


def test_tam_interface_exposes_four_actions_for_every_agent():
    env = _make_direct_env()

    assert all(space.shape == (4,) for space in env.action_space.spaces.values())


def test_tam_action_quantizes_all_axes_and_maps_throttle():
    env = _make_direct_env()
    raw = np.array([0.0, -0.4, 0.3, 2.0], dtype=np.float32)

    mapped = env._map_tam_direct_action(raw)

    expected_quantized = np.round((np.clip(raw, -1.0, 1.0) + 1.0) / 2.0 * 39.0) / 39.0 * 2.0 - 1.0
    expected_throttle = 0.4 + (expected_quantized[0] + 1.0) / 2.0 * 0.5
    np.testing.assert_allclose(mapped["quantized_action"], expected_quantized, atol=1e-7)
    assert mapped["throttle_cmd_norm"] == pytest.approx(expected_throttle)
    assert 0.4 <= mapped["throttle_cmd_norm"] <= 0.9
    assert mapped["rudder_cmd_norm"] == pytest.approx(1.0)


def test_tam_direct_action_writes_all_fcs_commands_without_pid():
    env = _make_direct_env()
    sim = _FakeSim()
    env.red_planes = {"red_0": sim, "red_1": None}
    env.blue_planes = {"blue_0": None}
    env.pid_controllers = {"red_0": _ExplodingPid()}

    targets = env._parse_actions({"red_0": np.array([1.0, -0.5, 0.25, -0.75])})
    env._apply_pid_controls(targets)

    assert sim.properties == pytest.approx(
        {
            "fcs/throttle-cmd-norm": 0.9,
            "fcs/aileron-cmd-norm": -0.4871794871794872,
            "fcs/elevator-cmd-norm": 0.23076923076923084,
            "fcs/rudder-cmd-norm": -0.7435897435897436,
        }
    )
    assert env._last_tam_action_commands["red_0"]["raw_action"] == [1.0, -0.5, 0.25, -0.75]


def test_legacy_interface_keeps_three_dimensional_action_space():
    env = UavCombatEnv(max_num_blue=1, max_num_red=1)

    assert env.action_space["red_0"].shape == (3,)
