from __future__ import annotations

import numpy as np

from uav_env import make_env
from uav_env.JSBSim.core.missile import MissileEvent
from uav_env.wrappers import MAPPOEnvWrapper

ZERO_SHOT_CONFIGS = [
    "uav_env/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]


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
    assert "termination_reason" in info
    assert "missile_summary" in info
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


def test_zero_shot_configs_smoke():
    for config_path in ZERO_SHOT_CONFIGS:
        env = make_env(config_path)
        obs, info = env.reset(seed=123)
        assert env.controlled_side == "red"
        assert env.config["opponent_policy"] == "rule_nearest"
        assert env.action_shape == 3
        assert set(obs) == set(env.agent_ids)
        assert info["blue_alive"] >= 2
        assert "mav" in info["agent_types"].values()
        for _ in range(3):
            actions = {
                aid: np.random.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert set(obs) == set(env.agent_ids)
            assert set(rewards) == set(env.agent_ids)
            assert "blue_alive" in info
            assert "agent_types" in info
        env.close()


def test_radar_range_changes_visible_mask():
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    _obs, _info = env.reset(seed=123)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["red_0"].position = np.array([0.0, 0.0, 6000.0], dtype=np.float32)
    by_id["red_1"].position = np.array([0.0, 1000.0, 6000.0], dtype=np.float32)
    by_id["blue_0"].position = np.array([100000.0, 0.0, 6000.0], dtype=np.float32)
    obs_all = env.task.observation.build_obs(env.task.agents)
    assert obs_all["red_0"]["visible_mask"][2] == 1.0
    assert obs_all["red_1"]["visible_mask"][2] == 0.0
    assert obs_all["red_1"]["enemy_states"][0][12] == 0.0
    env.close()


def test_mav_loss_triggers_termination():
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    _obs, _info = env.reset(seed=123)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["red_0"].kill("killed")
    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    _obs, rewards, terminated, _truncated, info = env.step(actions)
    assert all(terminated.values())
    assert info["termination_reason"] == "mav_loss"
    assert info["win_flag"] == "blue_win"
    assert rewards["red_0"] <= -70.0
    env.close()


def test_missile_left_decreases_on_launch():
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    _obs, info = env.reset(seed=123)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["red_1"].position = np.array([0.0, 0.0, 6000.0], dtype=np.float32)
    by_id["red_1"].heading = 0.0
    by_id["blue_0"].position = np.array([4000.0, 0.0, 6000.0], dtype=np.float32)
    before = by_id["red_1"].missile_left
    actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
    _obs, _rewards, _terminated, _truncated, info = env.step(actions)
    assert by_id["red_1"].missile_left < before
    assert info["missile_summary"]["launches"] >= 1
    env.close()


def test_mav_death_penalty_reward_component():
    env = make_env("uav_env/configs/hetero_train_2v2_mav_attack.yaml")
    _obs, _info = env.reset(seed=123)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["red_0"].kill("killed")
    event = MissileEvent("blue_0", "red_0", "blue", True, True, "hit", 1000.0,
                         by_id["blue_0"].missile_left)
    rewards = env.task.reward.compute(
        env.task.agents, [event], True, "blue_win", env.task.sensor,
        env.task.missiles.attack_range, env.task.missiles.max_los_angle)
    assert rewards["red_0"] <= -90.0
    env.close()


def test_debug_configs_complete_episode():
    for config_path in [
        "uav_env/configs/hetero_train_2v2_mav_attack.yaml",
        "uav_env/configs/hetero_test_3v3_mav_attack_scout.yaml",
    ]:
        env = make_env(config_path)
        _obs, _info = env.reset(seed=123)
        done = False
        steps = 0
        while not done:
            actions = {
                aid: np.random.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
                for aid in env.agent_ids
            }
            _obs, _rewards, terminated, truncated, info = env.step(actions)
            done = all(terminated.get(aid, False) or truncated.get(aid, False)
                       for aid in env.agent_ids)
            steps += 1
            assert steps <= env.config["episode_limit"]
        assert info["termination_reason"] is not None
        env.close()
