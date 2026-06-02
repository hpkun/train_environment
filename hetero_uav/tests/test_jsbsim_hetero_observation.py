from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


HETERO_FIELDS = {
    "ego_type",
    "ego_role",
    "ally_types",
    "ally_roles",
    "enemy_types",
    "enemy_roles",
}


def test_hetero_observation_has_type_role_metadata():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv, TYPE_VOCAB

    env = HeteroUavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _info = env.reset(seed=0)
        mav = np.eye(len(TYPE_VOCAB), dtype=np.float32)[TYPE_VOCAB.index("mav")]
        attack = np.eye(len(TYPE_VOCAB), dtype=np.float32)[TYPE_VOCAB.index("attack_uav")]

        for aid in env.agent_ids:
            assert HETERO_FIELDS.issubset(env.observation_space.spaces[aid].spaces)
            assert HETERO_FIELDS.issubset(obs[aid])

        np.testing.assert_array_equal(obs["red_0"]["ego_type"], mav)
        np.testing.assert_array_equal(obs["red_1"]["ego_type"], attack)
        np.testing.assert_array_equal(obs["blue_0"]["ego_type"], attack)
        np.testing.assert_array_equal(obs["blue_1"]["ego_type"], attack)
        np.testing.assert_array_equal(obs["red_0"]["ally_types"][0], attack)
        np.testing.assert_array_equal(obs["red_0"]["enemy_types"][0], attack)
        np.testing.assert_array_equal(obs["red_0"]["enemy_types"][1], attack)

        assert obs["red_0"]["ally_types"].shape == (env.max_num_red - 1, 4)
        assert obs["red_0"]["ally_roles"].shape == (env.max_num_red - 1, 4)
        assert obs["red_0"]["enemy_types"].shape == (env.max_num_blue, 4)
        assert obs["red_0"]["enemy_roles"].shape == (env.max_num_blue, 4)

        for _ in range(3):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, _rewards, _terminated, _truncated, _info = env.step(actions)
            for aid in env.agent_ids:
                assert HETERO_FIELDS.issubset(obs[aid])
                for key in HETERO_FIELDS:
                    assert not np.isnan(obs[aid][key]).any()
    finally:
        env.close()


def test_brma_observation_space_is_not_extended():
    from uav_env.JSBSim.envs.uav_combat_env import UavCombatEnv

    env = UavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _info = env.reset(seed=0)
        for aid in env.agent_ids:
            assert HETERO_FIELDS.isdisjoint(env.observation_space.spaces[aid].spaces)
            assert HETERO_FIELDS.isdisjoint(obs[aid])
    finally:
        env.close()
