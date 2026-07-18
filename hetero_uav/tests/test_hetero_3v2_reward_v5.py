from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy
from scripts.audit_hetero_3v2_reward_v5_ranking import _load_actor_only
from scripts.eval_hetero_3v2_pure_happo_v1 import _validate_checkpoint_meta
from scripts.train_hetero_3v2_pure_happo_v1 import (
    _contract_meta,
    _validate_resume_state,
    _validate_training_reward_contract,
)
from uav_env.JSBSim.formal_v2.contract import (
    ENV_TYPE,
    REWARD_CONTRACT_VERSION as V4_REWARD_CONTRACT,
    V5_REWARD_CONTRACT_VERSION,
)
from uav_env.JSBSim.formal_v2.reward_v5 import (
    POTENTIAL_BETA,
    POTENTIAL_GAMMA,
    _terminal_components,
    compute_team_rewards,
    potential_shaping_reward,
    role_potential_components,
)
from uav_env.make_env import make_env

V4_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2.yaml"
V5_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2_reward_v5.yaml"


def _context(**updates):
    context = {
        "red_attack_alive": 2,
        "blue_attack_alive": 2,
        "mav_alive": True,
        "terminated": False,
        "truncated": False,
        "team_done": False,
        "outcome": "ongoing",
        "end_reason": "",
    }
    context.update(updates)
    return context


def _meta(env):
    return {
        **_contract_meta(env),
        "policy_arch": "pure_happo",
        "credit_mode": "fixed_three_agent_team_mean",
        "algorithm_contract": ALGORITHM_CONTRACT,
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
    }


def test_v4_and_v5_are_selected_explicitly_and_keep_dimensions():
    v4 = make_env(V4_CONFIG)
    v5 = make_env(V5_CONFIG)
    assert v4.reward_contract == V4_REWARD_CONTRACT
    assert v5.reward_contract == V5_REWARD_CONTRACT_VERSION
    for env in (v4, v5):
        obs, info = env.reset(seed=0)
        assert env.actor_obs_dim == 73
        assert env.critic_state_dim == 219
        assert env.action_dim == 3
        assert obs["red_0"]["flat"].shape == (73,)
        assert info["critic_state"].shape == (219,)
        env.close()


def test_v5_returns_the_exact_same_reward_to_all_red_agents():
    env = make_env(V5_CONFIG)
    env.reset(seed=1)
    rewards, components = compute_team_rewards(
        env, env.selected_targets, [], _context())
    assert list(rewards.values()) == pytest.approx(
        [components["team_reward"]] * 3)
    assert sum(rewards.values()) / 3.0 == pytest.approx(
        components["team_reward"])
    env.close()


def test_v5_shared_events_have_exact_scale_and_out_of_zone_is_not_double_counted():
    env = make_env(V5_CONFIG)
    env.reset(seed=2)
    env.step_count = 1
    hit = [{
        "event": "hit", "shooter_id": "red_1", "target_id": "blue_0",
        "missile_id": "red_1_m0",
    }]
    _, kill = compute_team_rewards(env, env.selected_targets, hit, _context())
    assert kill["shared_event_raw"] == 200.0
    assert kill["shared_event_reward"] == 1.0
    assert kill["red_kill_count"] == 1

    env.step_count = 2
    env.aircraft["red_1"].shotdown()
    env.newly_dead = {"red_1"}
    env.death_reasons["red_1"] = "missile_hit"
    _, death = compute_team_rewards(env, env.selected_targets, [], _context(
        red_attack_alive=1))
    assert death["shared_event_raw"] == -200.0
    assert death["shared_event_reward"] == -1.0

    env.step_count = 3
    env.aircraft["red_0"].shotdown()
    env.newly_dead = {"red_0"}
    env.death_reasons["red_0"] = "missile_hit"
    _, mav_death = compute_team_rewards(env, env.selected_targets, [], _context(
        mav_alive=False))
    assert mav_death["shared_event_reward"] == -1.0

    env.reset(seed=3)
    env.step_count = 1
    env.aircraft["red_2"].shotdown()
    env.newly_dead = {"red_2"}
    env.death_reasons["red_2"] = "out_of_zone"
    _, out = compute_team_rewards(env, env.selected_targets, [], _context(
        red_attack_alive=1))
    assert out["out_of_zone_count"] == 1
    assert out["red_attack_death_count"] == 0
    assert out["shared_event_raw"] == -100.0
    assert out["shared_event_reward"] == -0.5
    env.close()


def test_v5_event_and_terminal_are_not_settled_twice_for_one_step():
    env = make_env(V5_CONFIG)
    env.reset(seed=4)
    env.step_count = 1
    event = [{
        "event": "hit", "shooter_id": "red_1", "target_id": "blue_0",
        "missile_id": "red_1_m0",
    }]
    context = _context(
        blue_attack_alive=0, terminated=True, team_done=True,
        outcome="red_win", end_reason="blue_combat_capability_eliminated")
    _, first = compute_team_rewards(env, env.selected_targets, event, context)
    _, second = compute_team_rewards(env, env.selected_targets, event, context)
    assert first["shared_event_reward"] == 1.0
    assert first["terminal_reward"] != 0.0
    assert second["shared_event_reward"] == 0.0
    assert second["terminal_reward"] == 0.0
    assert second["potential_shaping_reward"] == 0.0
    env.close()


def test_v5_terminal_outcome_and_retention_formula():
    red_win = _terminal_components(_context(
        blue_attack_alive=0, terminated=True, team_done=True,
        outcome="red_win"))
    assert red_win["outcome_bonus"] == 1.0
    assert red_win["terminal_retention_raw"] == 60.0
    assert red_win["terminal_reward"] == pytest.approx(1.3)

    blue_win = _terminal_components(_context(
        red_attack_alive=0, mav_alive=False, terminated=True, team_done=True,
        outcome="blue_win"))
    assert blue_win["outcome_bonus"] == -1.0
    assert blue_win["terminal_retention_raw"] == -90.0
    assert blue_win["terminal_reward"] == pytest.approx(-1.45)

    for outcome in ("mutual_elimination", "draw", "invalid"):
        row = _terminal_components(_context(
            terminated=outcome != "draw", truncated=outcome == "draw",
            team_done=True, outcome=outcome))
        assert row["outcome_bonus"] == 0.0


def test_v5_potential_formula_direction_boundary_and_fixed_divisor():
    positive, effective = potential_shaping_reward(-0.5, 0.5, False)
    negative, _ = potential_shaping_reward(0.5, -0.5, False)
    boundary, boundary_effective = potential_shaping_reward(0.5, 0.9, True)
    assert positive > 0.0
    assert negative < 0.0
    assert boundary == pytest.approx(-POTENTIAL_BETA * 0.5)
    assert effective == 0.5 and boundary_effective == 0.0

    env = make_env(V5_CONFIG)
    env.reset(seed=5)
    details = role_potential_components(env, env.selected_targets)
    expected = sum(details[aid]["phi"] for aid in env.red_ids) / 3.0
    assert env.v5_phi_previous == pytest.approx(expected)
    env.aircraft["red_1"].shotdown()
    details = role_potential_components(env, env.selected_targets)
    assert details["red_1"]["phi"] == 0.0
    env.close()


def test_v5_reset_reinitializes_previous_potential():
    env = make_env(V5_CONFIG)
    env.reset(seed=6)
    env.v5_phi_previous = 123.0
    env.v5_terminal_applied = True
    env.v5_last_event_step = 99
    env.reset(seed=6)
    assert -1.0 <= env.v5_phi_previous <= 1.0
    assert env.v5_phi_previous != 123.0
    assert env.v5_terminal_applied is False
    assert env.v5_last_event_step == -1
    env.close()


def test_v5_gamma_mismatch_is_rejected():
    env = make_env(V5_CONFIG)
    _validate_training_reward_contract(env, POTENTIAL_GAMMA)
    with pytest.raises(ValueError, match="gamma must equal potential_gamma"):
        _validate_training_reward_contract(env, 0.95)
    env.close()


def test_v4_and_v5_checkpoint_contracts_are_strictly_isolated():
    v4 = make_env(V4_CONFIG)
    v5 = make_env(V5_CONFIG)
    v4_meta = _meta(v4)
    v5_meta = _meta(v5)
    _validate_checkpoint_meta(v4_meta, ENV_TYPE, V4_REWARD_CONTRACT)
    _validate_checkpoint_meta(v5_meta, ENV_TYPE, V5_REWARD_CONTRACT_VERSION)
    with pytest.raises(ValueError, match="reward_contract"):
        _validate_checkpoint_meta(v4_meta, ENV_TYPE, V5_REWARD_CONTRACT_VERSION)
    with pytest.raises(ValueError, match="reward_contract"):
        _validate_checkpoint_meta(v5_meta, ENV_TYPE, V4_REWARD_CONTRACT)
    for key, expected in {
        "potential_gamma": 0.99,
        "potential_beta": 0.25,
        "event_scale": 200.0,
        "shared_team_reward": True,
    }.items():
        assert v5_meta[key] == expected
    v4.close()
    v5.close()


def test_v4_training_state_cannot_resume_a_v5_run(tmp_path: Path):
    env = make_env(V5_CONFIG)
    output = tmp_path / "v5_run"
    meta = {
        **_meta(env),
        "seed": 0,
        "training_hyperparameters": {"gamma": 0.99},
        "run_identity": str(output.resolve()),
    }
    state = {
        "formal_contract": meta["formal_contract"],
        "reward_contract": V4_REWARD_CONTRACT,
        "algorithm_contract": meta["algorithm_contract"],
        "actor_obs_dim": meta["actor_obs_dim"],
        "critic_state_dim": meta["critic_state_dim"],
        "action_dim": meta["action_dim"],
        "num_agents": meta["num_agents"],
        "seed": meta["seed"],
        "training_hyperparameters": meta["training_hyperparameters"],
        "run_identity": meta["run_identity"],
        "observation_contract": meta["observation_contract"],
        "resume_semantics": "episode_boundary_exact_training_state",
    }
    with pytest.raises(ValueError, match="reward_contract"):
        _validate_resume_state(state, meta, output)
    env.close()


def test_cross_reward_audit_loads_only_compatible_actor_parameters(tmp_path: Path):
    policy = PureHAPPOPolicy(
        73, 219, 3, 3, credit_mode="fixed_three_agent_team_mean")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.pt"
    torch.save(policy.state_dict(), checkpoint)
    meta = {
        "formal_contract": ENV_TYPE,
        "reward_contract": V4_REWARD_CONTRACT,
        "policy_arch": "pure_happo",
        "algorithm_contract": ALGORITHM_CONTRACT,
        "actor_obs_dim": 73,
        "critic_state_dim": 219,
        "action_dim": 3,
        "num_agents": 3,
    }
    (checkpoint_dir / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    loaded, loaded_meta = _load_actor_only(checkpoint, torch.device("cpu"))
    assert loaded_meta["reward_contract"] == V4_REWARD_CONTRACT
    assert all(torch.equal(
        loaded.state_dict()[key], policy.state_dict()[key])
        for key in policy.state_dict() if key.startswith("actors."))

    meta["actor_obs_dim"] = 68
    (checkpoint_dir / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="actor contract mismatch"):
        _load_actor_only(checkpoint, torch.device("cpu"))
