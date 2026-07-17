from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from algorithms.happo.vanilla_happo import (
    VanillaHAPPOPolicy, VanillaHAPPORolloutBuffer, VanillaHAPPOTrainer)
from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, save_vanilla_happo_checkpoint)
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, PAPER_NOMINAL_PROTOCOL)
from uav_env.make_env import make_env


def formal_checkpoint_config(scenario):
    return {
        "scenario": scenario,
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "experiment_protocol": PAPER_NOMINAL_PROTOCOL,
        "initial_perturbation": "none",
        "dynamics_backend": "jsbsim",
        "paper_silent_assumptions_present": True,
    }


def policy_for(agent_ids=("red_0", "red_1", "red_2"), obs_dim=12, state_dim=30):
    roles = {aid: "mav" if i == 0 else "attack_uav"
             for i, aid in enumerate(agent_ids)}
    return VanillaHAPPOPolicy(agent_ids, roles, obs_dim, state_dim, hidden_dim=16)


def test_four_categorical_heads_joint_log_probability_and_argmax():
    torch.manual_seed(1)
    policy = policy_for(agent_ids=("red_0",), obs_dim=7, state_dim=9)
    obs, state = torch.randn(1, 7), torch.randn(9)
    out = policy.act(obs, state, deterministic=True)
    assert out["logits"].shape == (1, 4, 40)
    assert out["actions"].shape == (1, 4)
    torch.testing.assert_close(out["actions"], out["logits"].argmax(-1))
    dist = torch.distributions.Categorical(logits=out["logits"][0])
    torch.testing.assert_close(out["log_probs"][0],
                               dist.log_prob(out["actions"][0]).sum())


def test_available_action_mask_forces_dead_agent_action_20():
    policy = policy_for(agent_ids=("red_0",), obs_dim=7, state_dim=9)
    available = np.zeros((1, 4, 40), np.float32)
    available[:, :, 20] = 1.0
    for deterministic in (False, True):
        out = policy.act(np.zeros((1, 7), np.float32), np.zeros(9, np.float32),
                         available, deterministic)
        assert out["actions"].tolist() == [[20, 20, 20, 20]]


@pytest.mark.parametrize(("name", "count"), [
    ("tam_paper_env_v1_2v2.yaml", 2),
    ("tam_paper_env_v1_3v2.yaml", 3),
    ("tam_paper_env_v1_5v4.yaml", 5),
])
def test_independent_actor_count_and_dynamic_dimensions(name, count):
    env = make_env(f"uav_env/JSBSim/configs/{name}", dynamics_backend="simple")
    obs, _ = env.reset(seed=2)
    obs_dim = len(env.flatten_observation(obs[env.agent_ids[0]]))
    state_dim = len(env.get_state())
    policy = VanillaHAPPOPolicy(env.agent_ids, env.agent_roles, obs_dim, state_dim)
    assert len(policy.actors) == count
    assert policy.critic_state_dim == state_dim
    assert all(policy.actor_key(aid) == aid for aid in env.agent_ids)
    assert len(policy.critic.heads) == count
    assert policy.value(torch.as_tensor(env.get_state())).shape == (count,)
    env.close()


def _one_step_buffer(terminated, truncated, active=(1.0,)):
    buffer = VanillaHAPPORolloutBuffer(1, 1, 2, 3)
    available = np.ones((1, 4, 40), np.float32)
    buffer.add(obs=np.zeros((1, 2)), state=np.zeros(3), actions=np.zeros((1, 4)),
               log_probs=np.zeros(1), rewards=np.ones(1),
               value=np.array([2.0]), next_value=np.array([10.0]),
               terminated=np.array([terminated]), truncated=np.array([truncated]),
               active_masks=np.array(active), available_actions=available,
               agent_alive=np.array(active), episode_id=0, decision_step=0,
               policy_version=0)
    return buffer


def test_gae_termination_no_bootstrap_and_truncation_bootstraps():
    terminated, _ = _one_step_buffer(1.0, 0.0).compute_gae(gamma=0.9, gae_lambda=1.0)
    truncated, _ = _one_step_buffer(0.0, 1.0).compute_gae(gamma=0.9, gae_lambda=1.0)
    assert terminated[0, 0] == pytest.approx(-1.0)
    assert truncated[0, 0] == pytest.approx(8.0)


def test_dead_agent_gae_is_zero():
    advantage, _ = _one_step_buffer(0.0, 0.0, active=(0.0,)).compute_gae()
    assert advantage[0, 0] == 0.0


def test_two_and_three_agent_multiplicative_factor_and_detach():
    factor = torch.ones(2, requires_grad=False)
    old = torch.zeros(2)
    ratios = [torch.log(torch.tensor([2.0, 3.0], requires_grad=True)),
              torch.log(torch.tensor([5.0, 7.0], requires_grad=True)),
              torch.log(torch.tensor([11.0, 13.0], requires_grad=True))]
    factor = VanillaHAPPOTrainer.update_factor(factor, ratios[0], old, torch.ones(2))
    torch.testing.assert_close(factor, torch.tensor([2.0, 3.0]))
    factor = VanillaHAPPOTrainer.update_factor(factor, ratios[1], old, torch.ones(2))
    torch.testing.assert_close(factor, torch.tensor([10.0, 21.0]))
    factor = VanillaHAPPOTrainer.update_factor(factor, ratios[2], old, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(factor, torch.tensor([110.0, 21.0]))
    assert factor.requires_grad is False and factor.grad_fn is None


def test_randomized_order_changes_factor_correspondence_and_updates_all_parameters():
    torch.manual_seed(3)
    policy = policy_for(obs_dim=5, state_dim=8)
    trainer = VanillaHAPPOTrainer(policy, ppo_epochs=3, seed=7)
    buffer = VanillaHAPPORolloutBuffer(4, 3, 5, 8)
    available = np.ones((3, 4, 40), np.float32)
    for step in range(4):
        obs = np.random.randn(3, 5).astype(np.float32)
        state = np.random.randn(8).astype(np.float32)
        with torch.no_grad():
            out = policy.act(obs, state, available)
        buffer.add(obs=obs, state=state, actions=out["actions"].numpy(),
                   log_probs=out["log_probs"].numpy(), rewards=np.ones(3),
                   value=out["value"].numpy(), next_value=out["value"].numpy(),
                   terminated=np.zeros(3), truncated=np.zeros(3),
                   active_masks=np.ones(3), available_actions=available,
                   agent_alive=np.ones(3), episode_id=0, decision_step=step,
                   policy_version=0)
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    result = trainer.update(buffer)
    assert result.update_orders and len(result.update_orders) == 1
    assert result.metrics["agent_order_count"] == 1
    assert result.metrics["factor_initialization_count"] == 1
    for aid in policy.agent_ids:
        assert any(not torch.equal(before[f"actors.{aid}.{name}"], value)
                   for name, value in policy.actors[aid].named_parameters())
    assert any(not torch.equal(before[f"critic.{name}"], value)
               for name, value in policy.critic.named_parameters())
    assert all(np.isfinite(value) for value in result.metrics.values()
               if isinstance(value, (int, float, bool)))


def test_role_shared_is_explicit_ablation_only():
    with pytest.warns(FutureWarning):
        policy = VanillaHAPPOPolicy(
            ["red_0", "red_1", "red_2"],
            {"red_0": "mav", "red_1": "uav", "red_2": "uav"}, 5, 8,
            actor_sharing="role_shared_ablation")
    assert policy.actor_sharing == "parameter_sharing_ppo_ablation"
    assert len(policy.actors) == 2
    with pytest.raises(ValueError, match="independent"):
        VanillaHAPPOTrainer(policy)


def test_checkpoint_roundtrip_optimizer_rng_and_resume_step(tmp_path):
    torch.manual_seed(9)
    policy = policy_for(agent_ids=("red_0",), obs_dim=7, state_dim=9)
    trainer = VanillaHAPPOTrainer(policy, seed=9)
    rng = np.random.default_rng(9)
    obs, state = torch.randn(1, 7), torch.randn(9)
    before = policy.act(obs, state, deterministic=True)
    path = tmp_path / "checkpoint.pt"
    saved = save_vanilla_happo_checkpoint(
        path, policy, trainer, environment_steps=123, episodes=4,
        config=formal_checkpoint_config("2v2"), numpy_rng=rng,
        checkpoint_type="resumable", at_episode_boundary=True)
    for parameter in policy.parameters():
        parameter.data.add_(1.0)
    loaded = load_vanilla_happo_checkpoint(
        path, policy, trainer, numpy_rng=rng, for_resume=True,
        expected_scenario="2v2")
    after = policy.act(obs, state, deterministic=True)
    torch.testing.assert_close(before["logits"], after["logits"])
    torch.testing.assert_close(before["actions"], after["actions"])
    assert loaded["environment_steps"] == 123 and loaded["episodes"] == 4
    assert loaded["actor_optimizers"].keys() == saved["actor_optimizers"].keys()


def test_checkpoint_rejects_wrong_scenario_dimensions(tmp_path):
    policy = policy_for(agent_ids=("red_0",), obs_dim=7, state_dim=9)
    trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "checkpoint.pt"
    save_vanilla_happo_checkpoint(path, policy, trainer, environment_steps=0,
                                  episodes=0, config=formal_checkpoint_config("2v2"),
                                  numpy_rng=np.random.default_rng())
    wrong = policy_for(agent_ids=("red_0",), obs_dim=8, state_dim=9)
    with pytest.raises(ValueError, match="actor_obs_dim mismatch"):
        load_vanilla_happo_checkpoint(path, wrong, restore_rng=False)


def test_formal_buffer_contains_traceability_fields_and_no_forbidden_networks():
    buffer = VanillaHAPPORolloutBuffer(2, 3, 5, 8)
    for name in ("terminated", "truncated", "active_masks", "available_actions",
                 "agent_alive", "episode_ids", "decision_steps", "policy_versions"):
        assert hasattr(buffer, name)
    source = (Path(__file__).parents[1] / "algorithms/happo/vanilla_happo.py").read_text()
    for forbidden in ("nn.GRU", "nn.LSTM", "Transformer", "MultiheadAttention"):
        assert forbidden not in source
