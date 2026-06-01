from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _assert_no_backup_or_parent_imports():
    forbidden = [
        name
        for name in sys.modules
        if name == "my_uav_env"
        or name.startswith("my_uav_env.")
        or name == "uav_env.brma_env"
        or name.startswith("uav_env.brma_env.")
    ]
    assert forbidden == []


def test_hetero_env_reset_models_info_and_step():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv

    env = HeteroUavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.agent_ids)
        assert env.agent_models["red_0"] == "A-4"
        assert env.agent_models["red_1"] == "f16"
        assert env.agent_models["blue_0"] == "f16"
        assert env.agent_models["blue_1"] == "f16"
        assert info["agent_types"] == env.agent_types
        assert info["agent_roles"] == env.agent_roles
        assert info["agent_models"] == env.agent_models

        for _ in range(3):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert set(obs.keys()) == set(env.agent_ids)
            assert "agent_types" in info
            assert "agent_roles" in info
            assert "agent_models" in info
        _assert_no_backup_or_parent_imports()
    finally:
        env.close()
