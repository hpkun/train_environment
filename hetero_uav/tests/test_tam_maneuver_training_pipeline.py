from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


CFG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_maneuver_fcs_train_probe.yaml"
)


def test_tam_maneuver_training_smoke_config_constructs():
    from uav_env import make_env

    env = make_env(CFG, max_steps=5, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        assert env.tam_control_mode == "maneuver_fcs"
        assert env.action_space[env.red_ids[0]].shape == (4,)
    finally:
        env.close()


def test_train_pipeline_reads_tam_box4_action_dim():
    from uav_env import make_env
    from scripts.train_happo_reference import _build_policy, _infer_env_action_dim

    env = make_env(CFG, max_steps=5, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        action_dim = _infer_env_action_dim(env)
        policy = _build_policy(
            "brma_recurrent_masked",
            actor_dim=96,
            critic_dim=480,
            action_dim=action_dim,
            device="cpu",
            num_agents=len(env.red_ids),
        )
        assert action_dim == 4
        assert policy.action_dim == 4
    finally:
        env.close()


def test_train_pipeline_reads_box4_action_dim_from_mp_proxy_meta():
    from scripts.train_happo_reference import _MpEnvProxy, _infer_env_action_dim

    class _DummyRemote:
        pass

    proxy = _MpEnvProxy(
        {
            "red_ids": ["red_0", "red_1", "red_2"],
            "blue_ids": ["blue_0", "blue_1"],
            "agent_ids": ["red_0", "red_1", "red_2", "blue_0", "blue_1"],
            "max_steps": 1000,
            "agent_roles": {},
            "agent_types": {},
            "agent_models": {},
            "actual_reward_mode": "happo_ref_v0",
            "action_dim": 4,
        },
        _DummyRemote(),
    )

    assert proxy.action_dim == 4
    assert _infer_env_action_dim(proxy) == 4


def test_tam_maneuver_rejects_red_3d_action():
    from uav_env import make_env

    env = make_env(CFG, max_steps=5, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        env.reset(seed=3)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions[env.red_ids[0]] = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError, match="shape=\\(4,\\)"):
            env.step(actions)
    finally:
        env.close()


def test_blue_rule_legacy_action_adapter_explicit_if_used():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env

    env = make_env(CFG, max_steps=5, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        obs, _info = env.reset(seed=4)
        red_actions = {rid: np.zeros(4, dtype=np.float32) for rid in env.red_ids}
        blue_actions = OpponentPolicy(mode="brma_rule", seed=4).act(obs, env.blue_ids, env=env)
        assert all(np.asarray(a).shape == (3,) for a in blue_actions.values())
        obs, rewards, terminated, truncated, info = env.step({**red_actions, **blue_actions})
        assert info["tam_control_mode"] == "maneuver_fcs"
        for bid in env.blue_ids:
            assert info["tam_blue_legacy_pid_rule_adapter"][bid] is True
            assert info["tam_blue_legacy_pid_applied"][bid] is True
            assert "blue_legacy_pid_rule_adapter" in info["tam_action_warnings"][bid]
        for rid in env.red_ids:
            assert len(info["tam_sanitized_actions"][rid]) == 4
    finally:
        env.close()


def test_blue_dead_legacy_3d_action_is_diagnostic_ignored_without_padding():
    from uav_env import make_env

    env = make_env(CFG, max_steps=5, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        env.reset(seed=40)
        dead_blue = env.blue_ids[0]
        alive_blue = env.blue_ids[1]
        env.blue_planes[dead_blue].crash()
        actions = {rid: np.zeros(4, dtype=np.float32) for rid in env.red_ids}
        actions[dead_blue] = np.zeros(3, dtype=np.float32)
        actions[alive_blue] = np.zeros(3, dtype=np.float32)

        _obs, _rewards, _terminated, _truncated, info = env.step(actions)

        assert info["tam_blue_legacy_pid_rule_adapter"][dead_blue] == "inactive_ignored"
        assert info["tam_blue_legacy_pid_applied"][dead_blue] is False
        assert info["tam_blue_legacy_pid_targets"][dead_blue] is None
        assert "blue_legacy_pid_rule_adapter_inactive_ignored" in info["tam_action_warnings"][dead_blue]
        assert info["tam_blue_legacy_pid_rule_adapter"][alive_blue] is True
        assert info["tam_blue_legacy_pid_applied"][alive_blue] is True
        assert len(info["tam_raw_actions"][dead_blue]) == 3
    finally:
        env.close()


def test_tam_maneuver_smoke_rollout_random_actions():
    from uav_env import make_env

    rng = np.random.default_rng(5)
    env = make_env(CFG, max_steps=8, suppress_jsbsim_output=True, num_missiles_per_plane=0)
    try:
        env.reset(seed=5)
        for _ in range(3):
            actions = {
                aid: rng.uniform(-0.25, 0.25, size=4).astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert set(obs) == set(env.agent_ids)
            assert set(rewards) == set(env.agent_ids)
            assert info["tam_control_mode"] == "maneuver_fcs"
            assert info["tam_action_order"] == [
                "throttle", "roll_cmd", "pitch_or_load_cmd", "yaw_cmd"
            ]
    finally:
        env.close()
