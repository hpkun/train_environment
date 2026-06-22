import numpy as np
import torch

from algorithms.happo import TAMCategoricalHAPPOTrainer, TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer


def test_agent_sequential_updates_only_each_agents_own_samples():
    torch.manual_seed(3)
    policy = TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96, critic_state_dim=480, action_levels=40,
        hidden_dim=16, rnn_hidden_size=8, num_attention_heads=4,
    )
    buffer = HAPPORolloutBuffer(
        3, 3, 96, 480, 4, [0, 1, 1], rnn_hidden_size=8,
        action_dtype=np.int64,
    )
    hidden = policy.init_hidden(3)
    buffer.set_rnn_hidden_initial(0, hidden.numpy())
    for step in range(3):
        obs = torch.randn(3, 96)
        critic = torch.randn(480)
        with torch.no_grad():
            out = policy.act(obs, [0, 1, 1], critic, rnn_hidden=hidden)
        buffer.store(
            obs.numpy(), critic.numpy(), out["action"].numpy(),
            out["log_prob"].numpy(), np.zeros(3), np.zeros(3),
            out["value"].item(), np.ones(3), next_value=out["value"].item(),
            rnn_hidden=hidden.numpy(), env_step_index=step,
        )
        hidden = out["rnn_hidden"].detach()
    trainer = TAMCategoricalHAPPOTrainer(
        policy, ppo_epochs=1, happo_update_granularity="agent",
        agent_ids=["red_0", "red_1", "red_2"],
    )
    metrics = trainer.update(buffer)
    assert metrics["agent_active_sample_count_red_0"] == 3.0
    assert metrics["agent_active_sample_count_red_1"] == 3.0
    assert metrics["agent_active_sample_count_red_2"] == 3.0
