from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.trainer import _compute_grouped_gae
from scripts.eval_hetero_3v2_pure_happo_v1 import _validate_checkpoint_meta
from uav_env.JSBSim.formal_v1.contract import ENV_TYPE
from uav_env.JSBSim.formal_v1.reward import REWARD_CONTRACT_VERSION


def make_buffer(policy, length=12, inactive_agent=None, terminated_at=(), truncated_at=()):
    buffer = HAPPORolloutBuffer(
        length, policy.num_agents, policy.actor_obs_dim, policy.critic_state_dim,
        policy.action_dim, list(range(policy.num_agents)))
    for step in range(length):
        obs = torch.randn(policy.num_agents, policy.actor_obs_dim)
        state = torch.randn(policy.critic_state_dim)
        with torch.no_grad():
            out = policy.act(obs, critic_state=state)
            next_value = policy.value(torch.randn(policy.critic_state_dim))
        active = np.ones(policy.num_agents, np.float32)
        if inactive_agent is not None:
            active[inactive_agent] = 0.0
        terminated = float(step in terminated_at)
        episode_done = float(terminated or step in truncated_at)
        buffer.store(
            obs.numpy(), state.numpy(), out["action"].numpy(), out["log_prob"].numpy(),
            np.linspace(-0.1, 0.1, policy.num_agents, dtype=np.float32),
            np.full(policy.num_agents, episode_done, np.float32), out["value"].numpy(),
            active, next_value=next_value.numpy(), raw_actions=out["raw_action"].numpy(),
            terminated=terminated, episode_done=episode_done)
    return buffer


def test_raw_mean_can_use_nearly_full_action_range():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    with torch.no_grad():
        policy.actors[0][-1].weight.zero_()
        policy.actors[0][-1].bias.fill_(5.0)
    out = policy.act(torch.zeros(3, 4), critic_state=torch.zeros(6), deterministic=True)
    assert torch.all(out["action"][0] > 0.95)
    assert torch.all(out["action"][0] > torch.tanh(torch.tensor(0.999)))


def test_actor_parameters_are_disjoint_and_critic_is_scalar():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    ids = [{id(parameter) for parameter in actor.parameters()} for actor in policy.actors]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert policy.value(torch.zeros(6)).shape == (1,)
    assert policy.value(torch.zeros(5, 6)).shape == (5,)


def test_raw_action_log_prob_replay_and_boundary_gradient_are_finite():
    torch.manual_seed(4)
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    obs = torch.randn(3, 4)
    out = policy.act(obs, critic_state=torch.randn(6))
    replay, _, _, _ = policy.evaluate_actions(
        obs.unsqueeze(0), torch.randn(1, 6), out["action"].unsqueeze(0),
        out["raw_action"].unsqueeze(0))
    assert torch.allclose(replay.squeeze(0), out["log_prob"], atol=1e-6)
    raw = torch.full((2, 3), 20.0, requires_grad=True)
    distribution, _ = policy._distribution(torch.zeros(2, 4), 0)
    loss = policy._squashed_log_prob(distribution, raw).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()


def test_buffer_roundtrips_raw_action_and_boundaries():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    buffer = make_buffer(policy, length=3, terminated_at=(1,), truncated_at=(2,))
    data = buffer.get(torch.device("cpu"))
    assert data["raw_actions"].shape == (3, 3, 3)
    assert torch.all(data["raw_action_valid"] == 1)
    assert data["terminated"].tolist() == [0.0, 1.0, 0.0]
    assert data["episode_dones"].tolist() == [0.0, 1.0, 1.0]


def test_sequential_order_is_single_contiguous_and_factor_progresses():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    metrics = PureHAPPOTrainer(policy, ppo_epochs=3, seed=9).update(make_buffer(policy))
    order = metrics["agent_update_order"]
    assert len(order) == 3 and sorted(order) == [0, 1, 2]
    assert metrics["actor_update_trace"] == [
        [agent, epoch] for agent in order for epoch in range(3)]
    first, second, third = order
    assert metrics["factor_before_per_agent"][first] == {
        "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
    for stat in ("mean", "std", "min", "max"):
        assert metrics["factor_before_per_agent"][second][stat] == pytest.approx(
            metrics["factor_after_per_agent"][first][stat])
        assert metrics["factor_before_per_agent"][third][stat] == pytest.approx(
            metrics["factor_after_per_agent"][second][stat])


def test_inactive_ratio_is_one_for_factor_and_zero_lr_keeps_factor_one():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    metrics = PureHAPPOTrainer(policy, actor_lr=0.0, ppo_epochs=2, seed=2).update(
        make_buffer(policy, inactive_agent=1))
    assert metrics["final_factor"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["final_factor"]["std"] < 1e-6
    assert metrics["final_factor"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["final_factor"]["max"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["valid_sample_count_per_agent"][1] == 0


def test_fixed_seed_order_is_reproducible_and_single_agent_is_ppo():
    p1 = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    p2 = copy.deepcopy(p1)
    m1 = PureHAPPOTrainer(p1, actor_lr=0.0, seed=17).update(make_buffer(p1))
    m2 = PureHAPPOTrainer(p2, actor_lr=0.0, seed=17).update(make_buffer(p2))
    assert m1["agent_update_order"] == m2["agent_update_order"]
    single = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=1)
    result = PureHAPPOTrainer(single, seed=1).update(make_buffer(single))
    assert result["agent_update_order"] == [0]
    assert result["factor_before_per_agent"][0]["mean"] == 1.0


def test_actor_and_critic_optimizer_ownership_isolated():
    actor_only = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    critic_before = [parameter.detach().clone() for parameter in actor_only.critic.parameters()]
    PureHAPPOTrainer(actor_only, critic_lr=0.0, seed=3).update(make_buffer(actor_only))
    assert all(torch.equal(a, b) for a, b in zip(critic_before, actor_only.critic.parameters()))
    critic_only = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    actor_before = [[parameter.detach().clone() for parameter in actor.parameters()]
                    for actor in critic_only.actors]
    log_std_before = [parameter.detach().clone()
                      for parameter in critic_only.action_log_stds]
    PureHAPPOTrainer(critic_only, actor_lr=0.0, seed=3).update(make_buffer(critic_only))
    assert all(torch.equal(before, after) for rows, actor in zip(actor_before, critic_only.actors)
               for before, after in zip(rows, actor.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(
        log_std_before, critic_only.action_log_stds))


def test_epoch_metric_summaries_distinguish_mean_max_and_final():
    policy = PureHAPPOPolicy(actor_obs_dim=4, critic_state_dim=6, num_agents=3)
    metrics = PureHAPPOTrainer(policy, ppo_epochs=3, seed=5).update(make_buffer(policy))
    for key, raw_key in (
        ("actor_loss_epoch_summary_per_agent", "actor_loss_epochs_per_agent"),
        ("entropy_epoch_summary_per_agent", "entropy_epochs_per_agent"),
        ("approx_kl_epoch_summary_per_agent", "approx_kl_epochs_per_agent"),
        ("approx_kl_abs_epoch_summary_per_agent", "approx_kl_abs_epochs_per_agent"),
        ("clip_fraction_epoch_summary_per_agent", "clip_fraction_epochs_per_agent"),
    ):
        for summary, values in zip(metrics[key], metrics[raw_key]):
            assert summary["mean"] == pytest.approx(float(np.mean(values)))
            assert summary["max"] == pytest.approx(float(np.max(values)))
            assert summary["final"] == pytest.approx(float(values[-1]))


def test_checkpoint_meta_requires_sequential_v2_contract():
    meta = {
        "formal_contract": ENV_TYPE,
        "credit_mode": "shared_alive_team_mean",
        "reward_contract": {"version": REWARD_CONTRACT_VERSION},
        "algorithm_contract": "pure_happo_sequential_v2",
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
        "actor_obs_dim": 68,
        "critic_state_dim": 204,
        "action_dim": 3,
        "num_agents": 3,
    }
    _validate_checkpoint_meta(meta)
    old = dict(meta, algorithm_contract="legacy_pure_happo")
    with pytest.raises(ValueError, match="algorithm_contract"):
        _validate_checkpoint_meta(old)


def gae(rewards, next_values, bootstrap, continuation):
    values = torch.zeros_like(rewards)
    env_ids = torch.zeros(len(rewards), dtype=torch.long)
    return _compute_grouped_gae(
        rewards, values, next_values, bootstrap, continuation, env_ids, 1.0, 1.0)


def test_gae_bootstrap_and_episode_boundary_contract():
    advantage, _ = gae(torch.tensor([0.0]), torch.tensor([5.0]),
                       torch.tensor([1.0]), torch.tensor([1.0]))
    assert advantage.item() == 5.0
    terminal, _ = gae(torch.tensor([0.0]), torch.tensor([5.0]),
                      torch.tensor([0.0]), torch.tensor([0.0]))
    assert terminal.item() == 0.0
    timeout, _ = gae(torch.tensor([0.0]), torch.tensor([5.0]),
                     torch.tensor([1.0]), torch.tensor([0.0]))
    assert timeout.item() == 5.0


def test_gae_timeout_does_not_leak_into_next_episode_and_tail_bootstraps():
    advantage, returns = gae(
        torch.tensor([1.0, 10.0, 0.0]), torch.tensor([2.0, 0.0, 4.0]),
        torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0]))
    assert advantage[0].item() == 3.0
    assert advantage[1].item() == 10.0
    assert advantage[2].item() == 4.0
    assert torch.allclose(advantage, returns)
