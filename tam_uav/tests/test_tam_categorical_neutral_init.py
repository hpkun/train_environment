from __future__ import annotations

import torch

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy


def _policy(**kwargs):
    torch.manual_seed(5)
    return TAMCategoricalRecurrentHAPPOPolicy(
        actor_obs_dim=96, critic_state_dim=480, action_levels=40,
        hidden_dim=32, rnn_hidden_size=32, num_attention_heads=4,
        **kwargs,
    )


def test_neutral_prior_argmax_is_high_throttle_and_middle_surfaces():
    policy = _policy()
    out = policy.act(torch.zeros(2, 96), roles=[0, 1], deterministic=True)
    expected = torch.tensor([[39, 20, 20, 20], [39, 20, 4, 20]])
    assert torch.equal(out["action"], expected)
    assert policy.neutral_action_centers == {
        "mav": [39, 20, 20, 20], "uav": [39, 20, 4, 20]
    }


def test_neutral_prior_remains_stochastic_with_nonzero_entropy_and_multiple_bins():
    policy = _policy(neutral_action_init_std_bins=4.0)
    obs = torch.zeros(512, 96)
    roles = torch.zeros(512, dtype=torch.long)
    out = policy.act(obs, roles=roles)
    assert torch.all(out["entropy"] > 0)
    for axis in range(4):
        assert torch.unique(out["action"][:, axis]).numel() >= 5
    assert float((out["action"][:, 0] >= 32).float().mean()) > 0.8
    lateral = out["action"][:, [1, 3]]
    assert float(((lateral >= 12) & (lateral <= 28)).float().mean()) > 0.9


def test_neutral_prior_can_be_disabled():
    policy = _policy(neutral_action_init=False)
    assert policy.neutral_action_init is False
