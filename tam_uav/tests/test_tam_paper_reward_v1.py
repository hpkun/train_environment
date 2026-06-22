from __future__ import annotations

import numpy as np
import pytest
import yaml

from uav_env import make_env
from uav_env.JSBSim.simulator import MissileSimulator


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"
MAV_KEYS = {
    "tam_mav_safety", "tam_mav_dist", "tam_mav_threat", "tam_mav_aspect",
    "tam_mav_support", "tam_mav_pos", "tam_mav_aware", "tam_mav_event",
    "tam_mav_death", "tam_mav_team_bonus", "tam_paper_reward_v1_total",
}
UAV_KEYS = {
    "tam_uav_height", "tam_uav_speed", "tam_uav_angle", "tam_uav_distance",
    "tam_uav_dodge", "tam_uav_dodge_angle", "tam_uav_dodge_speed",
    "tam_uav_event", "tam_uav_kill", "tam_uav_death",
    "tam_uav_out_of_zone", "tam_paper_reward_v1_total",
}


def _env(mode="tam_paper_reward_v1"):
    return make_env(CONFIG, env_type="jsbsim_hetero", hetero_reward_mode=mode,
                    max_steps=4)


def _neutral_actions(env):
    return {
        aid: np.asarray([39, 20, 20 if aid == "red_0" else 4, 20], np.int64)
        for aid in env.agent_ids
    }


def test_mode_accepts_and_one_step_rewards_and_components_are_finite():
    env = _env()
    try:
        env.reset(seed=3)
        _obs, rewards, _terminated, _truncated, info = env.step(_neutral_actions(env))
        assert env.hetero_reward_mode == "tam_paper_reward_v1"
        assert all(np.isfinite(value) for value in rewards.values())
        assert MAV_KEYS <= info["reward_components"]["red_0"].keys()
        assert UAV_KEYS <= info["reward_components"]["red_1"].keys()
        for aid in env.red_ids:
            assert rewards[aid] == pytest.approx(
                info["reward_components"][aid]["tam_paper_reward_v1_total"]
            )
        assert info["reward_components"]["red_1"]["height_formula_source"] == (
            "paper_undefined_PV_PH_v1_approx"
        )
        for aid in env.red_ids:
            numeric = [
                value for value in info["reward_components"][aid].values()
                if isinstance(value, (int, float, np.number))
            ]
            assert all(np.isfinite(value) for value in numeric)
    finally:
        env.close()


def test_config_keeps_happo_ref_v0_as_default():
    with open(CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["hetero_reward_mode"] == "happo_ref_v0"
    assert "tam_paper_reward_v1" in config


def test_uav_distance_reward_prefers_five_km_over_ten_km():
    env = _env()
    try:
        assert env._tam_paper_uav_distance_reward(5000.0) > (
            env._tam_paper_uav_distance_reward(10000.0)
        )
        assert env._tam_paper_uav_distance_reward(5000.0) == pytest.approx(1.0)
        assert env._tam_paper_uav_distance_reward(10000.0) == pytest.approx(-1.0)
    finally:
        env.close()


def test_uav_kill_and_death_events_have_paper_scale_and_death_is_once():
    env = _env()
    try:
        env.reset(seed=4)
        env._step_kill_count["red_1"] = 1
        rewards, components = env._compute_rewards()
        assert components["red_1"]["tam_uav_kill"] == pytest.approx(200.0)
        assert rewards["red_1"] >= 200.0

        env.red_planes["red_1"].crash()
        _rewards, first = env._compute_rewards()
        _rewards, second = env._compute_rewards()
        assert first["red_1"]["tam_uav_death"] == pytest.approx(-200.0)
        assert second["red_1"]["tam_uav_death"] == pytest.approx(0.0)
    finally:
        env.close()


def test_mav_death_is_negative_and_only_once():
    env = _env()
    try:
        env.reset(seed=5)
        env.red_planes["red_0"].crash()
        _rewards, first = env._compute_rewards()
        _rewards, second = env._compute_rewards()
        assert first["red_0"]["tam_mav_death"] < 0.0
        assert second["red_0"]["tam_mav_death"] == pytest.approx(0.0)
    finally:
        env.close()


def test_reward_mode_preserves_spaces_and_missile_parameters():
    base = _env("happo_ref_v0")
    paper = _env()
    try:
        assert repr(base.action_space) == repr(paper.action_space)
        assert repr(base.observation_space) == repr(paper.observation_space)
        base.reset(seed=8)
        paper.reset(seed=8)
        base_missile = MissileSimulator.create(
            base.red_planes["red_1"], base.blue_planes["blue_0"], "base_test"
        )
        paper_missile = MissileSimulator.create(
            paper.red_planes["red_1"], paper.blue_planes["blue_0"], "paper_test"
        )
        for name in ("_t_max", "_t_thrust", "_Isp", "_cD", "_m0", "_K", "_Rc"):
            assert getattr(base_missile, name) == getattr(paper_missile, name)
    finally:
        base.close()
        paper.close()
