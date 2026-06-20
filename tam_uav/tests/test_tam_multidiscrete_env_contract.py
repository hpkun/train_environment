from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.env import UavCombatEnv


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def test_formal_tam_env_uses_four_axis_multidiscrete_space():
    env = make_env(CONFIG)
    for aid in env.agent_ids:
        space = env.action_space[aid]
        assert isinstance(space, gymnasium.spaces.MultiDiscrete)
        np.testing.assert_array_equal(space.nvec, [40, 40, 40, 40])
        assert space.shape == (4,)


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([0, 0, 0, 0], [0.4, -1.0, -1.0, -1.0]),
        ([39, 39, 39, 39], [0.9, 1.0, 1.0, 1.0]),
        ([20, 20, 20, 20], [0.4 + 20 / 39 * 0.5, -1 + 40 / 39, -1 + 40 / 39, -1 + 40 / 39]),
    ],
)
def test_discrete_indices_map_exactly_to_fcs_commands(indices, expected):
    env = make_env(CONFIG)
    command = env._map_tam_direct_discrete_action(np.asarray(indices))
    actual = [
        command["throttle_cmd_norm"], command["aileron_cmd_norm"],
        command["elevator_cmd_norm"], command["rudder_cmd_norm"],
    ]
    np.testing.assert_allclose(actual, expected)
    assert command["action_distribution"] == "multidiscrete_categorical"
    assert command["action_indices"] == indices
    assert "raw_action" not in command
    assert "quantized_action" not in command


@pytest.mark.parametrize("indices", [[-1, 0, 0, 0], [40, 0, 0, 0], [1.5, 0, 0, 0], [0, 0, 0]])
def test_discrete_mapper_rejects_invalid_indices(indices):
    env = make_env(CONFIG)
    with pytest.raises(ValueError):
        env._map_tam_direct_discrete_action(indices)


def test_legacy_continuous_quantized_path_remains_box_diagnostic():
    env = UavCombatEnv(
        max_num_blue=1, max_num_red=1,
        action_interface="tam_direct_fcs_4d",
        tam_action_distribution="continuous_quantized",
    )
    assert isinstance(env.action_space["red_0"], gymnasium.spaces.Box)
    command = env._map_tam_direct_continuous_action([0.0, 0.0, 0.0, 0.0])
    assert command["action_distribution"] == "continuous_quantized"
    assert "quantized_action" in command


def test_formal_parse_executes_the_same_action_indices():
    env = make_env(CONFIG)
    env.reset(seed=0)
    indices = np.array([3, 7, 11, 15], dtype=np.int64)
    targets = env._parse_actions({"red_0": indices})
    assert targets["red_0"]["action_indices"] == indices.tolist()
    assert env._last_tam_action_commands["red_0"]["action_indices"] == indices.tolist()
    env.close()
