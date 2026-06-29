from __future__ import annotations

import numpy as np
import pytest
import yaml

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


CFG = "uav_env/JSBSim/configs/diagnostic_mav_shared_geo_3v2.yaml"
FULL_GEO_KEYS = [
    "enemy_relative_pos_xyz",
    "enemy_relative_vel_xyz",
    "enemy_bearing_elevation",
    "enemy_speed_heading",
    "enemy_full_geo_valid_mask",
]


def test_canonical_config_uses_mav_shared_geo_not_versioned_mode():
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["observation_mode"] == "mav_shared_geo"


def test_mav_shared_geo_adds_full_enemy_geometry_fields():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        red_obs = obs["red_1"]
        assert red_obs["enemy_geo_states"].shape == (2, 5)
        assert red_obs["enemy_relative_pos_xyz"].shape == (2, 3)
        assert red_obs["enemy_relative_vel_xyz"].shape == (2, 3)
        assert red_obs["enemy_bearing_elevation"].shape == (2, 2)
        assert red_obs["enemy_speed_heading"].shape == (2, 2)
        assert red_obs["enemy_full_geo_valid_mask"].shape == (2,)
    finally:
        env.close()


def test_red_uav_receives_mav_shared_full_geometry_when_direct_unobserved():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        red_obs = obs["red_1"]
        shared = red_obs["enemy_track_source"][:, 1] > 0.5
        assert np.any(shared)
        assert np.all(red_obs["enemy_full_geo_valid_mask"][shared] == 1.0)
        assert np.any(np.linalg.norm(red_obs["enemy_relative_pos_xyz"][shared], axis=1) > 1e-6)
        assert np.any(np.linalg.norm(red_obs["enemy_relative_vel_xyz"][shared], axis=1) > 1e-6)
    finally:
        env.close()


def test_blue_does_not_receive_mav_shared_track():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        blue_obs = obs["blue_0"]
        assert blue_obs["enemy_track_source"].shape == (3, 2)
        assert np.all(blue_obs["enemy_track_source"][:, 1] == 0.0)
    finally:
        env.close()


def test_red_mav_launch_block_and_mav_shared_track_gate_unchanged():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        env._last_step_obs = obs
        has_track, reason = env._has_launch_track("red_0", "blue_0")
        assert has_track is False
        assert reason == "role_blocked_mav"
        has_track, reason = env._has_launch_track("red_1", "blue_0")
        assert has_track is True
        assert reason in {"direct", "mav_shared"}
    finally:
        env.close()


def test_boresight_gate_is_not_enabled_by_default():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        assert getattr(env, "use_boresight_launch_gate", False) is False
    finally:
        env.close()


def test_adapter_v2_consumes_canonical_full_geo_schema():
    env = make_env(CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, info = env.reset(seed=0)
        adapter = HeteroObsAdapterV2()
        out = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

        assert adapter.enemy_entity_dim > 7
        assert adapter.flat_actor_obs_dim > 96
        assert adapter.critic_state_dim == adapter.flat_actor_obs_dim * adapter.max_red
        assert out["actor_obs"]["red_1"].shape[0] == adapter.flat_actor_obs_dim
        assert out["critic_state"].shape[0] == adapter.critic_state_dim
        assert out["structured_actor_obs"]["red_1"]["enemy_entities"].shape[-1] == adapter.enemy_entity_dim
    finally:
        env.close()


def test_versioned_mav_shared_geo_mode_is_rejected(tmp_path):
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["observation_mode"] = "mav_shared_geo_v2"
    path = tmp_path / "legacy_v2.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown observation_mode"):
        make_env(str(path), env_type="jsbsim_hetero", max_steps=2)
