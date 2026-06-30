"""Tests for the BRMA role-adapted no-missile MAV reward v8."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_"
    "brma_role_no_missile_reward_v8.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_dynamics_f22_visual_mav_"
    "brma_role_no_missile_reward_v8.yaml"
)


def _bare_env():
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = ["blue_0", "blue_1"]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {
        "red_0": "mav",
        "red_1": "attack_uav",
        "red_2": "attack_uav",
    }
    env.hetero_reward_mode = "brma_role_no_missile_reward_v8"
    return env


def _base_rewards_and_components():
    rewards = {"red_0": 10.0, "red_1": 11.0, "red_2": 12.0}
    components = {}
    for idx, rid in enumerate(["red_0", "red_1", "red_2"]):
        components[rid] = {
            "r_pitch": 0.10 + idx,
            "r_roll": 0.20 + idx,
            "r_alt": 0.30 + idx,
            "r_bound": 0.40 + idx,
            "r_vel": 0.50 + idx,
            "r_adv": 6.0 + idx,
            "r_end": 1.0,
            "total": rewards[rid],
        }
    return rewards, components


def test_reward_mode_registered_and_configs_load():
    from uav_env import make_env

    for cfg, max_red, max_blue in [(CFG_3V2, 3, 2), (CFG_5V4, 5, 4)]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "brma_role_no_missile_reward_v8"
            assert env.observation_mode == "mav_shared_geo"
            assert env.agent_roles["red_0"] == "mav"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.aircraft_type_params["attack_uav"]["num_missiles"] == 2
            assert env.max_num_red == max_red
            assert env.max_num_blue == max_blue
        finally:
            env.close()


def test_config_contains_no_tam_v7_reward_block():
    for cfg in [CFG_3V2, CFG_5V4]:
        data = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        assert data["hetero_reward_mode"] == "brma_role_no_missile_reward_v8"
        assert "tam_paper_reward_v7_role_aligned" not in data
        assert data["red_agent_types"][0] == "mav"
        assert data["aircraft_type_params"]["mav"]["num_missiles"] == 0


def test_v8_uses_parent_brma_weighted_components_without_double_weighting():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()
    rewards, components = env._compute_brma_role_no_missile_reward_v8(rewards, components)

    assert rewards["red_0"] == pytest.approx(4.0)
    assert components["red_0"]["r_adv"] == pytest.approx(0.0)
    assert components["red_0"]["brma_role_removed_situation"] == pytest.approx(6.0)
    assert components["red_0"]["brma_role_removed_situation_is_weighted"] == pytest.approx(1.0)
    assert components["red_0"]["brma_role_situation_active"] == pytest.approx(0.0)
    assert components["red_0"]["brma_role_no_missile_total"] == pytest.approx(4.0)
    assert components["red_0"]["total"] == pytest.approx(4.0)

    assert rewards["red_1"] == pytest.approx(11.0)
    assert components["red_1"]["r_adv"] == pytest.approx(7.0)
    assert components["red_1"]["brma_role_removed_situation"] == pytest.approx(0.0)
    assert components["red_1"]["brma_role_situation_active"] == pytest.approx(1.0)
    assert components["red_1"]["brma_role_no_missile_total"] == pytest.approx(11.0)


def test_v8_keeps_attack_uav_brma_flight_situation_terminal_components():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()
    rewards, components = env._compute_brma_role_no_missile_reward_v8(rewards, components)

    for rid in ["red_1", "red_2"]:
        comp = components[rid]
        for key in ("r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv", "r_end"):
            assert key in comp
        assert comp["brma_role_no_missile_active"] == pytest.approx(1.0)
        assert comp["brma_role_active_brma_flight"] == pytest.approx(1.0)
        assert comp["brma_role_active_brma_situation"] == pytest.approx(1.0)
        assert comp["brma_role_active_brma_terminal"] == pytest.approx(1.0)


def test_v8_forbids_tam_and_missile_process_active_fields():
    env = _bare_env()
    rewards, components = _base_rewards_and_components()
    _, components = env._compute_brma_role_no_missile_reward_v8(rewards, components)

    forbidden_tokens = (
        "tam_v7",
        "mav_safety",
        "mav_support",
        "team_credit",
        "fire",
        "launch",
        "lock",
        "guided",
        "shared_track",
        "missile_warning_reward",
        "missile_threat_reward",
        "dodge",
        "near_hit",
        "role_weighted",
        "loss_fraction",
        "event_scale",
        "lambda_support",
        "low_speed_exploit",
    )
    for comp in components.values():
        for key in comp:
            assert not any(token in key for token in forbidden_tokens)


def test_v8_needs_last_step_obs_cache_and_populates_it():
    from uav_env import make_env

    env = make_env(CFG_3V2, max_steps=5)
    try:
        assert env._needs_last_step_obs_cache() is True
        obs, _info = env.reset()
        assert env._last_step_obs
        assert set(env.red_ids + env.blue_ids).issubset(env._last_step_obs)
        assert set(env.red_ids + env.blue_ids).issubset(obs)
        zero_actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, *_ = env.step(zero_actions)
        assert env._last_step_obs
        assert set(env.red_ids + env.blue_ids).issubset(env._last_step_obs)
        assert set(env.red_ids + env.blue_ids).issubset(obs)
    finally:
        env.close()
