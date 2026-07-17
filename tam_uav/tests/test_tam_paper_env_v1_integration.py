from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.happo.happo_policy import MAV_ROLE_ID, UAV_ROLE_ID
from algorithms.happo.happo_trainer import HAPPOReferenceTrainer
from algorithms.happo.plain_categorical_policy import PlainCategoricalHAPPOPolicy
from uav_env.make_env import make_env


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"


def _env(name="tam_paper_env_v1_3v2.yaml", **kwargs):
    return make_env(str(CONFIG_DIR / name), dynamics_backend="simple", **kwargs)


def test_mav_is_unarmed_and_launch_interval_is_25_seconds():
    env = _env()
    env.reset(seed=1)
    by_id = {a.agent_id: a for a in env.task.agents}
    shooter, target, mav = by_id["red_1"], by_id["blue_0"], by_id["red_0"]
    target.position = shooter.position + np.array([5000.0, 0.0, 0.0])
    assert env.task.weapon.try_launch(mav, target, 0.0) is None
    first = env.task.weapon.try_launch(shooter, target, 0.0)
    assert first and first["reason"] == "launched"
    shooter.missile_left += 1
    assert env.task.weapon.try_launch(shooter, target, 24.99) is None
    assert env.task.weapon.try_launch(shooter, target, 25.0) is not None
    env.close()


def test_close_range_launch_is_allowed_without_tactical_minimum_range():
    env = _env()
    env.reset(seed=11)
    by_id = {a.agent_id: a for a in env.task.agents}
    shooter, target = by_id["red_1"], by_id["blue_0"]
    target.position = shooter.position + np.array([100.0, 0.0, 0.0])
    assert env.task.weapon.try_launch(shooter, target, 0.0) is not None
    env.close()


def test_coincident_position_does_not_launch():
    env = _env()
    env.reset(seed=111)
    by_id = {a.agent_id: a for a in env.task.agents}
    shooter, target = by_id["red_1"], by_id["blue_0"]
    target.position = shooter.position.copy()
    assert env.task.weapon.try_launch(shooter, target, 0.0) is None
    env.close()


def test_maximum_launch_range_and_interval_remain_enforced():
    env = _env()
    env.reset(seed=12)
    by_id = {a.agent_id: a for a in env.task.agents}
    shooter, target = by_id["red_1"], by_id["blue_0"]
    target.position = shooter.position + np.array([14000.0, 0.0, 0.0])
    assert env.task.weapon.try_launch(shooter, target, 0.0) is not None
    shooter.missile_left += 1
    env.task.weapon.last_launch_time_s.clear()
    target.position = shooter.position + np.array([14001.0, 0.0, 0.0])
    assert env.task.weapon.try_launch(shooter, target, 0.0) is None
    target.position = shooter.position + np.array([5000.0, 0.0, 0.0])
    assert env.task.weapon.try_launch(shooter, target, 0.0) is not None
    shooter.missile_left += 1
    assert env.task.weapon.try_launch(shooter, target, 24.99) is None
    env.close()


def test_mav_track_sharing_stops_immediately_after_death():
    env = _env()
    env.reset(seed=2)
    by_id = {a.agent_id: a for a in env.task.agents}
    uav, mav, enemy = by_id["red_1"], by_id["red_0"], by_id["blue_0"]
    enemy.position = uav.position + np.array([30000.0, 0.0, 0.0])
    mav.position = enemy.position + np.array([1000.0, 0.0, 0.0])
    obs = env.task.observation.build(env.task.agents, [])
    assert obs[uav.agent_id]["enemy_mask"][0] == 1.0
    mav.kill("shotdown")
    obs = env.task.observation.build(env.task.agents, [])
    assert obs[uav.agent_id]["enemy_mask"][0] == 0.0
    env.close()


def test_dead_slots_zero_and_mav_loss_does_not_end_episode():
    env = _env()
    env.reset(seed=5)
    by_id = {a.agent_id: a for a in env.task.agents}
    by_id["blue_0"].kill("shotdown")
    by_id["red_0"].kill("shotdown")
    obs = env.task.observation.build(env.task.agents, [])
    assert obs["red_1"]["enemy_mask"][0] == 0.0
    assert np.count_nonzero(obs["red_1"]["enemy_states"][0]) == 0
    terminated, truncated, _, _ = env.task._termination()
    assert not terminated and not truncated
    env.close()


def test_out_of_zone_event_is_minus_100_without_death_double_count():
    env = _env()
    env.reset(seed=6)
    uav = next(a for a in env.task.agents if a.agent_id == "red_1")
    uav.kill("boundary")
    rewards, components = env.task.reward.compute(
        env.task.agents, env.task.current_targets, env.task.target_scores,
        [], [], {uav.agent_id}, {a.agent_id: True for a in env.task.agents})
    assert components[uav.agent_id]["r_event"] == pytest.approx(-100.0)
    assert np.isfinite(rewards[uav.agent_id])
    env.close()


def test_info_contains_per_agent_reward_and_missile_accounting():
    env = _env()
    obs, _ = env.reset(seed=8)
    actions = {aid: np.array([30, 20, 20, 20]) for aid in env.agent_ids}
    _, _, _, _, info = env.step(actions)
    for aid in env.agent_ids:
        assert "reward_components" in info[aid]
        assert info[aid]["missiles_left"] >= 0
        assert "current_target_id" in info[aid]
    assert set(info["missile_termination_reasons"]) == {
        "hit", "timeout", "target_dead", "nonfinite"
    }
    assert info["environment_fidelity_revision"] == "published_rules_simplified_v4"
    assert info["reference_8_exact_blue_fsm_reproduced"] is False
    assert info["termination_resolution"] == "decision_step_boundary"
    assert info["experiment_protocol"] == "paper_nominal"
    assert info["initial_perturbation"] == "none"
    assert info["paper_nominal_experiment"] is True
    assert info["paper_generalization_experiment"] is False
    for metrics in info["aircraft_metrics"].values():
        assert "speed_limit_exceedance_count" in metrics
        assert "overload_limit_exceedance_count" in metrics
        assert metrics["speed_violation_count"] == metrics["speed_limit_exceedance_count"]
        assert metrics["overload_violation_count"] == metrics["overload_limit_exceedance_count"]
    assert info["structural_failures"] == 0
    assert info["structural_failure_step"] == []
    env.close()


@pytest.mark.parametrize("name", ["tam_paper_env_v1_2v2.yaml",
                                   "tam_paper_env_v1_3v2.yaml",
                                   "tam_paper_env_v1_5v4.yaml"])
def test_random_rollout_is_finite_and_makes_progress(name):
    env = _env(name, episode_limit_steps=40)
    obs, _ = env.reset(seed=3)
    for _ in range(40):
        actions = {aid: np.array([30, 20, 20, 20]) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert all(np.isfinite(value) for value in rewards.values())
        assert np.isfinite(env.get_state()).all()
        if any(terminated.values()) or any(truncated.values()):
            break
    assert info["episode_step"] > 0
    assert info["missiles_fired"] >= 0
    env.close()


def test_plain_happo_rollout_and_one_update_without_tam_network():
    env = _env(episode_limit_steps=16)
    obs, _ = env.reset(seed=4)
    actor_dim = len(env.flatten_observation(obs[env.agent_ids[0]]))
    state_dim = len(env.get_state())
    roles = np.array([MAV_ROLE_ID, UAV_ROLE_ID, UAV_ROLE_ID])
    policy = PlainCategoricalHAPPOPolicy(actor_dim, state_dim)
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)
    buffer = HAPPORolloutBuffer(8, 3, actor_dim, state_dim, 4, roles,
                                action_dtype=np.int64)
    for _ in range(8):
        actor_obs = np.stack([env.flatten_observation(obs[aid]) for aid in env.agent_ids])
        state = env.get_state()
        with torch.no_grad():
            out = policy.act(actor_obs, roles, state)
        actions = out["action"].cpu().numpy()
        next_obs, rewards, term, trunc, _ = env.step(
            {aid: actions[i] for i, aid in enumerate(env.agent_ids)})
        active = np.array([1.0 if env.task.agents[i].alive else 0.0 for i in range(3)])
        done = np.array([term[aid] or trunc[aid] for aid in env.agent_ids], dtype=np.float32)
        with torch.no_grad():
            next_value = policy.value(torch.as_tensor(env.get_state()).unsqueeze(0)).item()
        buffer.store(actor_obs, state, actions, out["log_prob"].cpu().numpy(),
                     np.array([rewards[aid] for aid in env.agent_ids]), done,
                     out["value"].item(), active, next_value=next_value)
        obs = next_obs
    before = policy.uav_actor[-1].weight.detach().clone()
    metrics = trainer.update(buffer)
    assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(before, policy.uav_actor[-1].weight)
    env.close()
