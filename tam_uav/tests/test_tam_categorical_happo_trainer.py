from __future__ import annotations

import numpy as np
import torch

import algorithms.happo as happo
from algorithms.happo.happo_buffer import HAPPORolloutBuffer


def _policy():
    torch.manual_seed(17)
    return happo.TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96, critic_state_dim=480, action_levels=40,
        hidden_dim=32, rnn_hidden_size=16, num_attention_heads=4,
    )


def _buffer(policy, steps=5):
    buffer = HAPPORolloutBuffer(
        steps, 2, 96, 480, 4, [0, 1], rnn_hidden_size=16,
        action_dtype="int64", num_envs=1,
    )
    hidden = policy.init_hidden(2)
    buffer.set_rnn_hidden_initial(0, hidden.numpy())
    for step in range(steps):
        obs = torch.randn(2, 96)
        critic = torch.randn(480)
        with torch.no_grad():
            out = policy.act(obs, [0, 1], critic, rnn_hidden=hidden)
        active = np.array([1.0, 0.0 if step >= 3 else 1.0], np.float32)
        buffer.store(
            obs.numpy(), critic.numpy(), out["action"].numpy(),
            out["log_prob"].numpy(), np.array([0.2, 0.1], np.float32),
            np.zeros(2, np.float32), out["value"].item(), active,
            next_value=out["value"].item(), env_id=0, env_step_index=step,
            episode_start_masks=np.full(2, float(step == 0), np.float32),
        )
        hidden = out["rnn_hidden"].detach() * torch.as_tensor(active).unsqueeze(-1)
    return buffer


def test_role_importance_ratio_uses_detached_exp_logprob_delta():
    ratio = happo.TAMCategoricalHAPPOTrainer.role_importance_ratio(
        torch.tensor([0.2, -0.4], requires_grad=True),
        torch.tensor([0.0, -0.1]),
    )
    torch.testing.assert_close(ratio, torch.exp(torch.tensor([0.2, -0.3])))
    assert ratio.requires_grad is False


def test_trainer_uses_sequence_replay_and_reports_happo_metrics():
    policy = _policy()
    policy.evaluate_actions = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("one-step replay must not be used")
    )
    trainer = happo.TAMCategoricalHAPPOTrainer(policy, ppo_epochs=1)
    metrics = trainer.update(_buffer(policy))
    required = {
        "actor_loss_mav", "actor_loss_uav", "critic_loss",
        "entropy_mav", "entropy_uav", "approx_kl_mav", "approx_kl_uav",
        "edge_bin_rate", "low_bin_rate", "high_bin_rate",
        "throttle_high_rate", "surface_edge_rate",
        "max_action_prob_mav", "max_action_prob_uav",
        "action_bin_usage_mav", "action_bin_usage_uav",
        "grad_norm_actor", "grad_norm_shared", "grad_norm_mav_head",
        "grad_norm_uav_head", "grad_norm_critic",
        "correction_factor_mean", "correction_factor_max", "correction_factor_min",
        "mav_active_sample_count", "uav_active_sample_count",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[key]) for key in required)
    assert metrics["mav_active_sample_count"] == 5.0
    assert metrics["uav_active_sample_count"] == 3.0
    assert not any("log_std" in key for key in metrics)
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


def _optimizer_parameter_ids(optimizer):
    if optimizer is None:
        return set()
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_actor_optimizers_have_disjoint_parameter_ownership():
    policy = _policy()
    trainer = happo.TAMCategoricalHAPPOTrainer(policy, ppo_epochs=1)
    shared = {id(parameter) for parameter in policy.actor_shared_parameters()}
    mav_head = {id(parameter) for parameter in policy.mav_actor.parameters()}
    uav_head = {id(parameter) for parameter in policy.uav_actor.parameters()}

    assert _optimizer_parameter_ids(trainer.shared_actor_opt) == shared
    assert _optimizer_parameter_ids(trainer.mav_opt) == mav_head
    assert _optimizer_parameter_ids(trainer.uav_opt) == uav_head
    assert shared.isdisjoint(_optimizer_parameter_ids(trainer.mav_opt))
    assert shared.isdisjoint(_optimizer_parameter_ids(trainer.uav_opt))
    assert _optimizer_parameter_ids(trainer.mav_opt).isdisjoint(
        _optimizer_parameter_ids(trainer.uav_opt)
    )


def test_empty_shared_parameter_group_does_not_create_adam_with_no_parameters():
    policy = _policy()
    policy.actor_shared_parameters = lambda: []
    trainer = happo.TAMCategoricalHAPPOTrainer(policy, ppo_epochs=1)
    assert trainer.shared_actor_opt is None


def test_formal_trainer_class_is_distinct_from_reference_trainer():
    assert happo.TAMCategoricalHAPPOTrainer is not happo.HAPPOReferenceTrainer
