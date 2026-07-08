from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from gymnasium import spaces


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _make_tam_env(max_steps: int = 20):
    from uav_env import make_env

    return make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml",
        env_type="tam",
        max_steps=max_steps,
        suppress_jsbsim_output=True,
    )


def test_tam_env_entrypoints_construct():
    from uav_env import TamCombatEnv, make_tam_env

    env = make_tam_env(max_num_red=1, max_num_blue=1, max_steps=5, suppress_jsbsim_output=True)
    try:
        assert isinstance(env, TamCombatEnv)
    finally:
        env.close()


def test_tam_env_action_space_is_direct_fcs_box4():
    env = _make_tam_env(max_steps=5)
    try:
        assert set(env.action_space.spaces) == set(env.agent_ids)
        for aid in env.agent_ids:
            space = env.action_space.spaces[aid]
            assert isinstance(space, spaces.Box)
            assert space.shape == (4,)
            assert space.dtype == np.float32
            np.testing.assert_allclose(space.low, np.full(4, -1.0, dtype=np.float32))
            np.testing.assert_allclose(space.high, np.full(4, 1.0, dtype=np.float32))
    finally:
        env.close()


def test_tam_env_reset_and_neutral_steps():
    env = _make_tam_env(max_steps=8)
    try:
        obs, info = env.reset(seed=0)
        assert set(obs) == set(env.agent_ids)
        assert isinstance(info, dict)
        neutral = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(3):
            obs, rewards, terminated, truncated, info = env.step(neutral)
            assert set(obs) == set(env.agent_ids)
            assert set(rewards) == set(env.agent_ids)
            assert set(terminated) == set(env.agent_ids)
            assert set(truncated) == set(env.agent_ids)
            assert "reward_components" in info
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.1, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.0, 0.1, 0.0], dtype=np.float32),
        np.array([0.5, 0.0, 0.0, 0.1], dtype=np.float32),
    ],
)
def test_tam_env_direct_fcs_axis_smoke(action: np.ndarray):
    env = _make_tam_env(max_steps=8)
    try:
        env.reset(seed=1)
        actions = {aid: action.copy() for aid in env.agent_ids}
        for _ in range(3):
            _obs, _rewards, terminated, truncated, _info = env.step(actions)
            assert not all(terminated.values())
            assert not all(truncated.values())
    finally:
        env.close()
