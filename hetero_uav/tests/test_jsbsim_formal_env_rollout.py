from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from uav_env import make_env


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _zero_actions(env):
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _assert_no_nan(env):
    for sim in list(env.blue_planes.values()) + list(env.red_planes.values()):
        values = np.concatenate([
            sim.get_geodetic().astype(np.float64),
            sim.get_position().astype(np.float64),
            sim.get_velocity().astype(np.float64),
            np.asarray(sim.get_rpy(), dtype=np.float64),
        ])
        assert not np.isnan(values).any()
        assert not np.isinf(values).any()


def test_jsbsim_brma_zero_rollout_20_steps():
    env = make_env(
        "uav_env/JSBSim/configs/brma_2v2_debug.yaml",
        max_steps=40,
        agent_interaction_steps=2,
    )
    try:
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.agent_ids)
        for _ in range(20):
            obs, rewards, terminated, truncated, info = env.step(_zero_actions(env))
            _assert_no_nan(env)
    finally:
        env.close()


def test_jsbsim_hetero_zero_rollout_20_steps():
    env = make_env(
        "uav_env/JSBSim/configs/hetero_2v2_mav_attack.yaml",
        max_steps=40,
        agent_interaction_steps=2,
    )
    try:
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.agent_ids)
        assert info["agent_models"]["red_0"] == "A-4"
        assert info["agent_models"]["red_1"] == "f16"
        assert info["agent_models"]["blue_0"] == "f16"
        assert info["agent_models"]["blue_1"] == "f16"
        assert "agent_types" in info
        assert "agent_roles" in info
        assert "agent_models" in info
        for _ in range(20):
            obs, rewards, terminated, truncated, info = env.step(_zero_actions(env))
            _assert_no_nan(env)
    finally:
        env.close()
