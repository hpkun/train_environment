"""Smoke tests for balanced hetero config family."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorithms.mappo.adapter_utils import make_obs_adapter
from uav_env import make_env

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "v2_3v3": "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "v2_4v4": "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
    "v1_3v3": "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_3v3.yaml",
    "v1_4v4": "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_4v4.yaml",
}


def _has_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nan(v) for v in value)
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _adapter_for(obs_mode: str):
    return make_obs_adapter("v2" if obs_mode == "mav_shared_geo" else "v1")


def _zero_actions(env):
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _bounded_random_actions(env, rng):
    return {
        aid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        for aid in env.agent_ids
    }


def _reset_config(config: str):
    env = make_env(config, env_type="jsbsim_hetero")
    obs, info = env.reset(seed=0)
    return env, obs, info


def test_balanced_config_files_exist():
    for config in CONFIGS.values():
        assert (ROOT / config).exists(), config


@pytest.mark.parametrize("name,config", CONFIGS.items())
def test_balanced_config_reset_metadata_and_dims(name, config):
    env, obs, info = _reset_config(config)
    try:
        obs_mode = "mav_shared_geo" if name.startswith("v2") else "brma_sensor"
        adapter = _adapter_for(obs_mode)
        adapted = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

        assert len(env.red_ids) == len(env.blue_ids)
        assert info["observation_mode"] == obs_mode
        assert info["agent_types"]["red_0"] == "mav"
        assert info["agent_models"]["red_0"] == "A-4"
        assert env.red_planes["red_0"].num_left_missiles == 0

        for rid in env.red_ids[1:]:
            assert info["agent_types"][rid] == "attack_uav"
            assert env.red_planes[rid].num_left_missiles == 2
        for bid in env.blue_ids:
            assert info["agent_types"][bid] == "attack_uav"
            assert env.blue_planes[bid].num_left_missiles == 2

        if name.startswith("v2"):
            assert adapter.flat_actor_obs_dim == 96
            assert adapter.critic_state_dim == 480
        else:
            assert adapter.flat_actor_obs_dim == 140
            assert adapter.critic_state_dim == 700

        expected_mask = (
            [1, 1, 1, 0, 0] if "3v3" in name else [1, 1, 1, 1, 0]
        )
        assert adapted["red_valid_mask"].astype(int).tolist() == expected_mask
    finally:
        env.close()


@pytest.mark.parametrize("name,config", CONFIGS.items())
def test_balanced_config_zero_and_random_steps_no_nan(name, config):
    env, obs, info = _reset_config(config)
    rng = np.random.default_rng(7)
    try:
        adapter = _adapter_for(info["observation_mode"])
        for action_builder in (
            lambda: _zero_actions(env),
            lambda: _bounded_random_actions(env, rng),
        ):
            for _ in range(3):
                obs, rewards, terminated, truncated, info = env.step(action_builder())
                adapted = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                assert not _has_nan(obs)
                assert not _has_nan(adapted)
                assert not _has_nan(rewards)
    finally:
        env.close()
