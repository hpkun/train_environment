from __future__ import annotations

import numpy as np

from uav_env import make_env


def test_env_smoke():
    env = make_env("uav_env/configs/hetero_2v2_debug.yaml")
    obs, info = env.reset(seed=123)
    assert env.num_agents == 4
    assert set(env.agent_ids) == {"red_0", "red_1", "blue_0", "blue_1"}
    assert isinstance(obs, dict)
    assert obs["red_0"]["flat"].shape == (env.obs_shape,)
    assert env.get_state().shape == (env.state_shape,)

    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert set(rewards) == set(env.agent_ids)
    assert set(terminated) == set(env.agent_ids)
    assert set(truncated) == set(env.agent_ids)
    assert "mav_alive" in info
    assert "red_alive" in info
    assert "blue_alive" in info
    assert "agent_types" in info
    assert info["agent_types"]["red_0"] == "mav"
    env.close()
