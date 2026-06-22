import numpy as np
import torch

from algorithms.happo import TAMCategoricalHAPPOTrainer, TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer


def _policy():
    return TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96, critic_state_dim=480, action_levels=40,
        hidden_dim=16, rnn_hidden_size=8, num_attention_heads=4,
    )


def _buffer(policy):
    buffer = HAPPORolloutBuffer(2, 2, 96, 480, 4, [0, 1], rnn_hidden_size=8,
                                action_dtype=np.int64)
    hidden = policy.init_hidden(2)
    buffer.set_rnn_hidden_initial(0, hidden.numpy())
    for step in range(2):
        obs, critic = torch.randn(2, 96), torch.randn(480)
        with torch.no_grad():
            out = policy.act(obs, [0, 1], critic, rnn_hidden=hidden)
        buffer.store(obs.numpy(), critic.numpy(), out["action"].numpy(),
                     out["log_prob"].numpy(), np.zeros(2), np.zeros(2),
                     out["value"].item(), np.ones(2), next_value=out["value"].item(),
                     rnn_hidden=hidden.numpy(), env_step_index=step)
        hidden = out["rnn_hidden"].detach()
    return buffer


def test_entropy_and_survival_credit_telemetry_is_finite():
    policy = _policy()
    metrics = TAMCategoricalHAPPOTrainer(policy, ppo_epochs=1).update(_buffer(policy))
    required = {
        "entropy_mav_raw", "entropy_uav_raw",
        "entropy_mav_per_axis_mean", "entropy_uav_per_axis_mean",
        "entropy_bonus_mav", "entropy_bonus_uav",
        "actor_surrogate_loss_mav_abs", "actor_surrogate_loss_uav_abs",
        "entropy_to_policy_loss_ratio_mav", "entropy_to_policy_loss_ratio_uav",
        "advantage_mean_red_0", "advantage_std_red_0",
        "advantage_min_red_0", "advantage_max_red_0",
        "active_sample_count_red_0", "death_transition_count_red_0",
        "death_transition_used_for_actor_red_0",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(metrics[key]) for key in required)
