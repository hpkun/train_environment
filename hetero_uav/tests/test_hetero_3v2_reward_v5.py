from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy
from scripts.audit_hetero_3v2_reward_v5_ranking import (
    _load_actor_only,
    _paired_scenarios,
)
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
    discounted_potential_return,
    potential_shaping_reward,
    potential_targets,
    role_potential_components,
    theoretical_discounted_potential_return,
    v5_uav_dodge_potential_components,
)
from uav_env.JSBSim.formal_v2.reward import uav_dodge_components
from uav_env.JSBSim.formal_v1.missile import FormalMissile
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


def _controller_state(controller):
    return {
        key: copy.deepcopy(
            value.__dict__ if hasattr(value, "__dict__") else value)
        for key, value in controller.__dict__.items()
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
    boundary, boundary_effective = potential_shaping_reward(
        0.5, 0.9, terminated=True)
    timeout, timeout_effective = potential_shaping_reward(
        0.5, 0.9, terminated=False, truncated=True)
    invalid, invalid_effective = potential_shaping_reward(
        0.5, 0.9, terminated=False, invalid=True)
    assert positive > 0.0
    assert negative < 0.0
    assert boundary == pytest.approx(-POTENTIAL_BETA * 0.5)
    assert effective == 0.5 and boundary_effective == 0.0
    assert timeout_effective == 0.9
    assert timeout == pytest.approx(POTENTIAL_BETA * (0.99 * 0.9 - 0.5))
    assert invalid_effective == 0.0
    assert invalid == pytest.approx(-POTENTIAL_BETA * 0.5)

    env = make_env(V5_CONFIG)
    env.reset(seed=5)
    details = role_potential_components(env)
    expected = sum(details[aid]["phi"] for aid in env.red_ids) / 3.0
    assert env.v5_phi_previous == pytest.approx(expected)
    env.aircraft["red_1"].shotdown()
    details = role_potential_components(env)
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


def test_v5_potential_targets_use_current_state_and_ignore_cached_targets():
    env = make_env(V5_CONFIG)
    env.reset(seed=7)
    first = potential_targets(env)
    assert first["red_0"] is None
    assert first["red_1"] in env.blue_ids
    assert first["red_2"] in env.blue_ids
    expected_phi = sum(
        row["phi"] for row in role_potential_components(env).values()) / 3.0
    assert env.v5_phi_previous == pytest.approx(expected_phi)

    env.selected_targets.update({
        "red_0": "blue_0", "red_1": None, "red_2": "blue_1"})
    second = potential_targets(env)
    assert second == first
    assert role_potential_components(env) == role_potential_components(env)

    env.aircraft[first["red_1"]].shotdown()
    switched = potential_targets(env)
    assert switched["red_1"] != first["red_1"]
    env.close()


def test_v5_potential_evaluation_has_no_environment_side_effects():
    env = make_env(V5_CONFIG)
    env.reset(seed=8)
    env.selected_targets["red_1"] = "blue_1"
    env.last_fire_gates = {"red_1": {"allowed": True}}
    env.last_launch_time["red_1"] = 12.5
    env.previous_missile_speed = {"existing": 321.0}
    before = {
        "selected_targets": dict(env.selected_targets),
        "last_fire_gates": {key: dict(value) for key, value in env.last_fire_gates.items()},
        "last_launch_time": dict(env.last_launch_time),
        "previous_missile_speed": dict(env.previous_missile_speed),
        "missiles": [(m.state, m.position.copy(), m.velocity.copy()) for m in env.missiles],
        "aircraft": {aid: (
            aircraft.get_position().copy(), aircraft.get_velocity().copy(),
            aircraft.get_rpy().copy(), aircraft.is_alive,
        ) for aid, aircraft in env.aircraft.items()},
        "controllers": {aid: _controller_state(controller)
                        for aid, controller in env.controllers.items()},
        "v5_phi_previous": env.v5_phi_previous,
        "v5_terminal_applied": env.v5_terminal_applied,
        "v5_last_event_step": env.v5_last_event_step,
    }
    first = role_potential_components(env)
    second = role_potential_components(env)
    assert first == second
    assert env.selected_targets == before["selected_targets"]
    assert env.last_fire_gates == before["last_fire_gates"]
    assert env.last_launch_time == before["last_launch_time"]
    assert env.previous_missile_speed == before["previous_missile_speed"]
    assert env.v5_phi_previous == before["v5_phi_previous"]
    assert env.v5_terminal_applied == before["v5_terminal_applied"]
    assert env.v5_last_event_step == before["v5_last_event_step"]
    assert len(env.missiles) == len(before["missiles"])
    for aid, aircraft in env.aircraft.items():
        position, velocity, rpy, alive = before["aircraft"][aid]
        assert np.array_equal(aircraft.get_position(), position)
        assert np.array_equal(aircraft.get_velocity(), velocity)
        assert np.array_equal(aircraft.get_rpy(), rpy)
        assert aircraft.is_alive == alive
        assert _controller_state(env.controllers[aid]) == before["controllers"][aid]
    env.close()


def test_v5_dodge_is_pure_and_v4_history_behavior_is_unchanged():
    env = make_env(V5_CONFIG)
    env.reset(seed=9)
    target = env.aircraft["red_1"]
    position = target.get_position() - np.asarray([1_000.0, 0.0, 0.0])
    missile = FormalMissile(
        "blue_0_m0", "blue_0", "red_1", position.copy(),
        np.asarray([600.0, 0.0, 0.0]),
    )
    env.missiles.append(missile)
    env.previous_missile_speed.clear()
    first = v5_uav_dodge_potential_components(env, "red_1")
    second = v5_uav_dodge_potential_components(env, "red_1")
    assert first == second
    assert first["dodge_speed"] == 0.0
    assert env.previous_missile_speed == {}

    v4_result = uav_dodge_components(env, "red_1")
    assert v4_result["dodge_speed"] == 0.0
    assert env.previous_missile_speed[missile.missile_id] == pytest.approx(600.0)
    env.close()


@pytest.mark.parametrize(
    ("phis", "terminated", "truncated", "expected_final"),
    [
        ([0.2, 0.4, -0.1], False, False, -0.1),
        ([0.2, 0.4, -0.1], True, False, 0.0),
        ([0.2, 0.4, -0.1], False, True, -0.1),
    ],
)
def test_v5_discounted_potential_telescoping_identity(
        phis, terminated, truncated, expected_final):
    rewards = []
    for index in range(len(phis) - 1):
        is_last = index == len(phis) - 2
        reward, effective = potential_shaping_reward(
            phis[index], phis[index + 1],
            terminated=terminated and is_last,
            truncated=truncated and is_last,
        )
        rewards.append(reward)
    actual = discounted_potential_return(rewards)
    theoretical = theoretical_discounted_potential_return(
        phis[0], expected_final, len(rewards))
    assert effective == pytest.approx(expected_final)
    assert actual == pytest.approx(theoretical, abs=1e-12)


def test_v5_audit_scenarios_are_distinct_and_reusable_across_policies():
    first_policy = _paired_scenarios(2_000, 20)
    second_policy = _paired_scenarios(2_000, 20)
    assert first_policy == second_policy
    signatures = {
        json.dumps(row["perturbation"], sort_keys=True)
        for row in first_policy
    }
    assert len(signatures) == 20


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
