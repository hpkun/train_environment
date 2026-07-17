from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.trainer import _fixed_three_agent_team_mean
from scripts.eval_hetero_3v2_pure_happo_v1 import _validate_checkpoint_meta
from scripts.train_hetero_3v2_pure_happo_v1 import _contract_meta
from uav_env.JSBSim.formal_v2.contract import (
    CREDIT_MODE, ENV_TYPE, OBSERVATION_CONTRACT, REWARD_CONTRACT_VERSION,
)
from uav_env.JSBSim.formal_v2.reward import (
    EVENT_NORMALIZER, EVENT_REWARDS, MAV_DENSE_NORMALIZER,
    UAV_DENSE_NORMALIZER, _scaled_reward_fields, compute_role_rewards,
    mav_safety_components,
)
from uav_env.make_env import make_env

V1_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2.yaml"


def _zero_actions(env):
    return {agent_id: np.zeros(3, np.float32) for agent_id in env.red_ids}


def _disable_fire(env):
    for aircraft in env.aircraft.values():
        aircraft.num_left_missiles = 0


def test_v2_combat_capability_termination_contract():
    env = make_env(V2_CONFIG)
    env.reset(seed=0)
    _disable_fire(env)
    env.aircraft["red_1"].shotdown()
    env.aircraft["red_2"].shotdown()
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert all(terminated.values()) and not any(truncated.values())
    assert info["outcome"] == "blue_win"
    assert info["end_reason"] == "red_combat_capability_eliminated"
    assert info["red_attack_alive"] == 0
    assert info["mav_alive"] is True
    env.close()

    env = make_env(V2_CONFIG)
    env.reset(seed=1)
    _disable_fire(env)
    env.aircraft["red_0"].shotdown()
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert not any(terminated.values()) and not any(truncated.values())
    assert info["red_attack_alive"] == 2
    assert info["mav_alive"] is False
    env.close()

    env = make_env(V2_CONFIG)
    env.reset(seed=2)
    _disable_fire(env)
    for agent_id in ("red_1", "red_2", "blue_0", "blue_1"):
        env.aircraft[agent_id].shotdown()
    _, _, terminated, _, info = env.step(_zero_actions(env))
    assert all(terminated.values())
    assert info["outcome"] == "mutual_elimination"
    assert info["end_reason"] == "mutual_combat_capability_eliminated"
    env.close()


def test_v2_timeout_requires_both_sides_to_retain_attack_capability():
    env = make_env(V2_CONFIG)
    env.reset(seed=3)
    _disable_fire(env)
    env.step_count = env.max_steps - 1
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert not any(terminated.values()) and all(truncated.values())
    assert info["outcome"] == "draw"
    assert info["end_reason"] == "timeout"
    assert info["red_attack_alive"] > 0 and info["blue_attack_alive"] > 0
    env.close()


def test_fixed_three_agent_credit_never_changes_denominator():
    rewards = torch.tensor([
        [0.9, -0.6, -0.3],
        [0.9, 0.0, 0.0],
    ])
    team = _fixed_three_agent_team_mean(rewards)
    assert team.tolist() == pytest.approx([0.0, 0.3], abs=1e-7)
    assert _fixed_three_agent_team_mean(torch.tensor([[0.6, 0.0, 0.0]])).item() == pytest.approx(0.2)
    with pytest.raises(ValueError, match="exactly three"):
        _fixed_three_agent_team_mean(torch.ones(1, 2))


def test_v2_policy_and_trainer_accept_fixed_three_agent_credit():
    policy = PureHAPPOPolicy(
        actor_obs_dim=73,
        critic_state_dim=219,
        action_dim=3,
        num_agents=3,
        credit_mode=CREDIT_MODE,
    )
    trainer = PureHAPPOTrainer(policy)
    assert policy.credit_mode == CREDIT_MODE
    assert trainer.policy.credit_mode == CREDIT_MODE


def test_v4_raw_events_and_fixed_reward_scaling():
    assert EVENT_REWARDS["red_kill"] == 200.0
    assert EVENT_REWARDS["uav_death"] == -200.0
    assert EVENT_REWARDS["out_of_zone"] == -100.0
    assert EVENT_REWARDS["mav_death"] == -200.0
    assert EVENT_REWARDS["mav_team_kill"] == 100.0
    assert EVENT_REWARDS["mav_team_kill_cap"] == 200.0
    assert UAV_DENSE_NORMALIZER == 75.0
    assert MAV_DENSE_NORMALIZER == pytest.approx(1.8)
    assert EVENT_NORMALIZER == 100.0

    favorable = _scaled_reward_fields(75.0, 200.0, UAV_DENSE_NORMALIZER)
    assert favorable["normalized_dense_reward"] == 1.0
    assert favorable["normalized_event_reward"] == 2.0
    assert favorable["normalized_role_reward"] == 3.0
    clipped = _scaled_reward_fields(150.0, 0.0, UAV_DENSE_NORMALIZER)
    assert clipped["normalized_dense_reward"] == 1.0
    mav = _scaled_reward_fields(-1.8, -200.0, MAV_DENSE_NORMALIZER)
    assert mav["normalized_dense_reward"] == -1.0
    assert mav["normalized_event_reward"] == -2.0
    assert all(np.isfinite(value) for value in mav.values())


def test_v4_mav_event_credit_is_bounded_and_death_is_once():
    env = make_env(V2_CONFIG)
    env.reset(seed=4)
    selected = {"red_1": "blue_0", "red_2": "blue_1"}
    hit_1 = [{"event": "hit", "shooter_id": "red_1", "target_id": "blue_0"}]
    _, first = compute_role_rewards(env, selected, hit_1)
    assert first["per_agent"]["red_1"]["raw_event_reward"] == 200.0
    assert first["per_agent"]["red_0"]["mav_team_kill_credit_delta"] == 100.0
    _, second = compute_role_rewards(env, selected, hit_1)
    assert second["per_agent"]["red_0"]["mav_team_kill_credit_used"] == 200.0
    _, capped = compute_role_rewards(env, selected, hit_1)
    assert capped["per_agent"]["red_0"]["mav_team_kill_credit_delta"] == 0.0

    env.aircraft["red_0"].shotdown()
    env.newly_dead = {"red_0"}
    reward, death = compute_role_rewards(env, selected, [])
    assert death["per_agent"]["red_0"]["raw_event_reward"] == -200.0
    assert reward["red_0"] == -2.0
    env.newly_dead = set()
    reward, later = compute_role_rewards(env, selected, [])
    assert reward["red_0"] == 0.0
    assert later["per_agent"]["red_0"]["raw_event_reward"] == 0.0
    env.close()


def test_v4_out_of_zone_event_uses_raw_minus_100_and_normalized_minus_1():
    env = make_env(V2_CONFIG)
    env.reset(seed=7)
    env.aircraft["red_1"].shotdown()
    env.newly_dead = {"red_1"}
    env.death_reasons["red_1"] = "out_of_zone"
    rewards, details = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_1"}, [])
    uav = details["per_agent"]["red_1"]
    assert uav["raw_dense_reward"] == 0.0
    assert uav["raw_event_reward"] == -100.0
    assert uav["normalized_event_reward"] == -1.0
    assert rewards["red_1"] == -1.0
    env.close()


def test_v4_mav_threat_is_only_a_real_incoming_missile():
    env = make_env(V2_CONFIG)
    env.reset(seed=5)
    assert mav_safety_components(env)["safety_threat"] == 0.0
    missile = type("Missile", (), {
        "is_launched": True,
        "target_id": "red_0",
    })()
    env.missiles.append(missile)
    assert mav_safety_components(env)["safety_threat"] == -1.0
    env.close()


def test_v4_checkpoint_credit_and_reward_contract_are_strict():
    env = make_env(V2_CONFIG)
    meta = {
        **_contract_meta(env),
        "credit_mode": CREDIT_MODE,
        "algorithm_contract": "pure_happo_sequential_v2",
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
    }
    _validate_checkpoint_meta(meta, ENV_TYPE)
    with pytest.raises(ValueError, match="reward_contract"):
        _validate_checkpoint_meta(
            {**meta, "reward_contract": "paper_aligned_role_reward_v3"},
            ENV_TYPE,
        )
    with pytest.raises(ValueError, match="credit_mode"):
        _validate_checkpoint_meta(
            {**meta, "credit_mode": "shared_alive_team_mean"}, ENV_TYPE)
    env.close()


def test_v1_contract_and_mav_only_termination_remain_unchanged():
    env = make_env(V1_CONFIG)
    obs, info = env.reset(seed=6)
    assert env.actor_obs_dim == 68 and env.critic_state_dim == 204
    assert env.credit_mode == "shared_alive_team_mean"
    assert obs["red_0"]["flat"].shape == (68,)
    assert info["critic_state"].shape == (204,)
    _disable_fire(env)
    env.aircraft["red_1"].shotdown()
    env.aircraft["red_2"].shotdown()
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert not any(terminated.values()) and not any(truncated.values())
    assert info["outcome"] == "ongoing"
    env.close()
