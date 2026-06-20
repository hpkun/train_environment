from __future__ import annotations

import numpy as np
import torch

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.happo.happo_trainer import HAPPOReferenceTrainer
from algorithms.happo.rollout_safety import zero_inactive_actions


def _policy():
    torch.manual_seed(11)
    return TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96,
        critic_state_dim=480,
        action_levels=40,
        hidden_dim=32,
        rnn_hidden_size=32,
        num_attention_heads=4,
    )


def _buffer(policy, steps=4):
    buffer = HAPPORolloutBuffer(
        max_len=steps,
        num_red=2,
        actor_dim=96,
        critic_dim=480,
        action_dim=4,
        role_ids=[0, 1],
        rnn_hidden_size=32,
        action_dtype=np.int64,
    )
    hidden = policy.init_hidden(2)
    for step in range(steps):
        obs = torch.randn(2, 96)
        critic = torch.randn(480)
        with torch.no_grad():
            out = policy.act(obs, [0, 1], critic, rnn_hidden=hidden)
        buffer.store(
            obs.numpy(), critic.numpy(), out["action"].numpy(),
            out["log_prob"].numpy(), np.array([0.2, 0.1], np.float32),
            np.array([0.0, 0.0], np.float32), out["value"].item(),
            np.ones(2, np.float32), next_value=out["value"].item(),
            rnn_hidden=hidden.numpy(),
        )
        hidden = out["rnn_hidden"].detach()
    return buffer


def test_categorical_buffer_preserves_integer_actions():
    policy = _policy()
    buffer = _buffer(policy)
    assert buffer.actions.dtype == np.int64
    assert buffer.get("cpu")["actions"].dtype == torch.long


def test_inactive_action_zeroing_preserves_integer_dtype():
    actions = np.array([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.int64)
    result = zero_inactive_actions(actions, np.array([1.0, 0.0]))
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result[0], actions[0])
    np.testing.assert_array_equal(result[1], np.zeros(4, dtype=np.int64))


def test_rollout_log_probs_match_policy_evaluation():
    policy = _policy()
    data = _buffer(policy).get("cpu")
    roles = data["role_ids"].view(1, 2).expand(data["actions"].shape[0], 2)
    with torch.no_grad():
        log_prob, *_ = policy.evaluate_actions(
            data["actor_obs"], roles, data["critic_state"], data["actions"],
            rnn_hidden=data["rnn_hidden"],
        )
    torch.testing.assert_close(log_prob, data["old_log_probs"])


def test_categorical_ppo_update_reports_distribution_metrics():
    policy = _policy()
    trainer = HAPPOReferenceTrainer(policy, ppo_epochs=1)
    metrics = trainer.update(_buffer(policy))
    required = {
        "entropy_mav", "entropy_uav", "approx_kl_mav", "approx_kl_uav",
        "edge_bin_rate", "low_bin_rate", "high_bin_rate",
        "max_action_prob_mav", "max_action_prob_uav",
        "action_bin_usage_mav", "action_bin_usage_uav",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[name]) for name in required)
    assert not any("log_std" in name or "saturation" in name for name in metrics)
