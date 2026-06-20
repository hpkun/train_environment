from __future__ import annotations

import numpy as np
import torch

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer


def _policy():
    torch.manual_seed(13)
    return TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96, critic_state_dim=480, action_levels=40,
        hidden_dim=32, rnn_hidden_size=16, num_attention_heads=4,
    )


def test_buffer_exposes_time_ordered_sequence_contract():
    buffer = HAPPORolloutBuffer(
        4, 2, 96, 480, 4, [0, 1], rnn_hidden_size=16,
        action_dtype="int64", num_envs=1,
    )
    initial = np.full((2, 16), 0.25, np.float32)
    buffer.set_rnn_hidden_initial(0, initial)
    for t in range(3):
        buffer.store(
            np.zeros((2, 96), np.float32), np.zeros(480, np.float32),
            np.zeros((2, 4), np.int64), np.zeros(2, np.float32),
            np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0,
            np.ones(2, np.float32), env_id=0, env_step_index=t,
            episode_start_masks=np.full(2, float(t == 0), np.float32),
        )
    sequence = buffer.get_sequences("cpu")[0]
    assert sequence["actor_obs"].shape == (3, 2, 96)
    assert torch.equal(sequence["env_step_index"], torch.arange(3))
    torch.testing.assert_close(sequence["rnn_hidden_initial"], torch.full((2, 16), 0.25))
    assert torch.equal(sequence["agent_alive_masks"], sequence["active_masks"])


def test_sequence_replay_advances_hidden_and_resets_at_episode_start():
    policy = _policy()
    obs = torch.randn(4, 2, 96)
    actions = torch.full((4, 2, 4), 20, dtype=torch.long)
    starts = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0]], dtype=torch.float32)
    active = torch.ones(4, 2)
    out = policy.evaluate_action_sequence(
        obs, [0, 1], torch.randn(4, 480), actions,
        initial_hidden=torch.zeros(2, 16),
        episode_start_masks=starts, active_masks=active,
    )
    assert out["log_prob"].shape == (4, 2)
    assert out["hidden_states"].shape == (4, 2, 16)
    assert not torch.allclose(out["hidden_states"][0], out["hidden_states"][1])

    restarted = policy.evaluate_action_sequence(
        obs[2:3], [0, 1], torch.randn(1, 480), actions[2:3],
        initial_hidden=torch.zeros(2, 16),
        episode_start_masks=torch.ones(1, 2), active_masks=torch.ones(1, 2),
    )
    torch.testing.assert_close(out["hidden_states"][2], restarted["hidden_states"][0])


def test_inactive_agent_hidden_is_zero_and_masked_from_sequence():
    policy = _policy()
    active = torch.tensor([[1, 1], [1, 0], [1, 0]], dtype=torch.float32)
    out = policy.evaluate_action_sequence(
        torch.randn(3, 2, 96), [0, 1], torch.randn(3, 480),
        torch.zeros(3, 2, 4, dtype=torch.long),
        initial_hidden=torch.zeros(2, 16),
        episode_start_masks=torch.tensor([[1, 1], [0, 0], [0, 0]], dtype=torch.float32),
        active_masks=active,
    )
    assert torch.count_nonzero(out["hidden_states"][1:, 1]) == 0
    assert torch.equal(out["active_masks"], active)
