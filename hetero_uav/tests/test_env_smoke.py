from __future__ import annotations

import numpy as np

from uav_env import make_env
from uav_env.wrappers import MAPPOEnvWrapper


def test_env_smoke():
    env = make_env("uav_env/configs/hetero_2v2_debug.yaml")
    obs, info = env.reset(seed=123)
    assert env.controlled_side == "red"
    assert env.num_agents == 2
    assert set(env.agent_ids) == {"red_0", "red_1"}
    assert isinstance(obs, dict)
    assert set(obs) == {"red_0", "red_1"}
    assert obs["red_0"]["flat"].shape == (env.obs_shape,)
    assert env.get_state().shape == (env.state_shape,)
    assert info["blue_alive"] == 2

    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert set(rewards) == set(env.agent_ids)
    assert set(terminated) == set(env.agent_ids)
    assert set(truncated) == set(env.agent_ids)
    assert info["blue_alive"] <= 2
    assert "blue_0" in info["agent_alive"]
    assert "mav_alive" in info
    assert "red_alive" in info
    assert "blue_alive" in info
    assert "agent_types" in info
    assert info["agent_types"]["red_0"] == "mav"
    env.close()


def test_3v3_exposes_only_red_agents():
    env = make_env("uav_env/configs/hetero_3v3_debug.yaml")
    obs, info = env.reset(seed=123)
    assert env.num_agents == 3
    assert set(env.agent_ids) == {"red_0", "red_1", "red_2"}
    assert set(obs) == set(env.agent_ids)
    assert info["blue_alive"] == 3


def test_all_controlled_side_mode():
    env = make_env("uav_env/configs/hetero_2v2_debug.yaml", controlled_side="all")
    obs, _info = env.reset(seed=123)
    assert env.num_agents == 4
    assert set(env.agent_ids) == {"red_0", "red_1", "blue_0", "blue_1"}
    assert set(obs) == set(env.agent_ids)


def test_mappo_wrapper_red_only_arrays():
    env = MAPPOEnvWrapper(make_env("uav_env/configs/hetero_2v2_debug.yaml"))
    obs, state, info = env.reset()
    assert obs.shape == (2, env.obs_shape)
    assert state.shape == (env.state_shape,)
    assert info["blue_alive"] == 2
    actions = np.zeros((2, env.action_shape), dtype=np.float32)
    obs, state, rewards, dones, info = env.step(actions)
    assert obs.shape == (2, env.obs_shape)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    assert "blue_alive" in info
