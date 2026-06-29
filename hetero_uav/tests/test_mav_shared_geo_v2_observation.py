from __future__ import annotations

import numpy as np

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.adapters.hetero_obs_adapter_v3 import HeteroObsAdapterV3


OLD_CFG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v7_role_aligned.yaml"
)
V2_CFG = "uav_env/JSBSim/configs/diagnostic_mav_shared_geo_v2_3v2.yaml"


def test_mav_shared_geo_legacy_fields_do_not_gain_v2_fields():
    env = make_env(OLD_CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        red_obs = obs["red_1"]
        assert "enemy_geo_states" in red_obs
        assert red_obs["enemy_geo_states"].shape == (2, 5)
        for key in [
            "enemy_relative_pos_xyz",
            "enemy_relative_vel_xyz",
            "enemy_bearing_elevation",
            "enemy_speed_heading",
            "enemy_full_geo_valid_mask",
        ]:
            assert key not in red_obs
    finally:
        env.close()


def test_mav_shared_geo_v2_adds_full_enemy_geometry_fields():
    env = make_env(V2_CFG, env_type="jsbsim_hetero", max_steps=2)
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
    env = make_env(V2_CFG, env_type="jsbsim_hetero", max_steps=2)
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


def test_blue_does_not_receive_mav_shared_track_in_v2():
    env = make_env(V2_CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, _info = env.reset(seed=0)
        blue_obs = obs["blue_0"]
        assert blue_obs["enemy_track_source"].shape == (3, 2)
        assert np.all(blue_obs["enemy_track_source"][:, 1] == 0.0)
    finally:
        env.close()


def test_red_mav_launch_block_and_mav_shared_track_gate_unchanged():
    env = make_env(V2_CFG, env_type="jsbsim_hetero", max_steps=2)
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


def test_adapter_v3_consumes_full_geo_without_changing_v2_adapter():
    env = make_env(V2_CFG, env_type="jsbsim_hetero", max_steps=2)
    try:
        obs, info = env.reset(seed=0)
        v2 = HeteroObsAdapterV2()
        v3 = HeteroObsAdapterV3()
        out2 = v2.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        out3 = v3.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

        assert out2["actor_obs"]["red_1"].shape[0] == v2.flat_actor_obs_dim
        assert out3["actor_obs"]["red_1"].shape[0] == v3.flat_actor_obs_dim
        assert v3.flat_actor_obs_dim > v2.flat_actor_obs_dim
        assert v3.schema_version == "hetero_obs_adapter_v3_mav_shared_full_geo"
    finally:
        env.close()
