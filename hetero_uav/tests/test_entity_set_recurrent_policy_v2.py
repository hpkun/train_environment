"""Tests for HeteroEntityRecurrentPolicy v2 (entity_dim=21, critic_counts)."""
import numpy as np
import torch
import pytest

from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
    _GlobalEntityCritic,
    _EntityAttention,
)


class TestPolicyV2Forward:
    """Verify v2 policy forward pass with entity_dim=21."""

    def test_policy_creation_default_dim_21(self):
        policy = HeteroEntityRecurrentPolicy()
        assert policy.entity_dim == 21

    def test_act_forward_3v2(self):
        """Forward pass with 3v2 entity sets (3 red agents)."""
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        policy.eval()
        batch = 3  # 3 red agents
        n_entities = 9  # 1 self + 4 ally + 4 enemy = 9

        actor_tokens = torch.randn(batch, n_entities, 21)
        actor_mask = torch.ones(batch, n_entities)
        roles = torch.tensor([0, 1, 1])  # MAV + 2 UAV
        critic_tokens = torch.randn(7, 21)  # 3 red + 4 blue
        critic_mask = torch.ones(7)
        critic_counts = torch.tensor([3.0, 3.0, 4.0, 4.0])  # all alive

        with torch.no_grad():
            out = policy.act(actor_tokens, actor_mask, roles,
                           critic_tokens, critic_mask,
                           rnn_hidden=None, critic_counts=critic_counts)

        assert out["action"].shape == (batch, 3)
        assert out["log_prob"].shape == (batch,)
        assert out["value"].shape == (1,)  # single critic value
        assert out["rnn_hidden"].shape == (batch, policy.rnn_hidden_size)

    def test_act_forward_5v4(self):
        """Forward pass with 5v4 entity sets."""
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        policy.eval()
        batch = 5
        n_entities = 12  # 1 self + 7 ally (5-1) + 4 enemy = 12? No - wait...
        # Actually: 1 self + (red_agents-1) allies + blue_agents enemies
        # For 5v4: 1 + 4 + 4 = 9. But max could be more.
        # Let's use the HeteroEntitySetAdapter's actual token layout:
        # num_entities = 1 + max_allies + max_enemies
        # Max depends on adapter config. Let's just test with 9 (3v2 max).
        n_entities = 12  # 1+5+6 or similar

        actor_tokens = torch.randn(batch, n_entities, 21)
        actor_mask = torch.ones(batch, n_entities)
        roles = torch.tensor([0, 1, 1, 1, 1])
        critic_tokens = torch.randn(5 + 4, 21)  # 9 total
        critic_mask = torch.ones(5 + 4)
        critic_counts = torch.tensor([4.0, 5.0, 3.0, 4.0])  # 4 red alive, 3 blue alive

        with torch.no_grad():
            out = policy.act(actor_tokens, actor_mask, roles,
                           critic_tokens, critic_mask,
                           rnn_hidden=None, critic_counts=critic_counts)

        assert out["action"].shape == (batch, 3)
        assert out["value"].shape == (1,)

    def test_evaluate_actions(self):
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        batch = 3
        n_entities = 9

        actor_tokens = torch.randn(batch, n_entities, 21)
        actor_mask = torch.ones(batch, n_entities)
        roles = torch.tensor([0, 1, 1])
        critic_tokens = torch.randn(7, 21)
        critic_mask = torch.ones(7)
        critic_counts = torch.tensor([3.0, 3.0, 4.0, 4.0])
        actions = torch.randn(batch, 3).clamp(-1, 1)
        rnn_hidden = torch.zeros(batch, 128)

        log_probs, entropy, values, means, role_ids, next_hidden = policy.evaluate_actions(
            actor_tokens, actor_mask, roles,
            critic_tokens, critic_mask, actions,
            rnn_hidden=rnn_hidden, critic_counts=critic_counts)

        assert log_probs.shape == (batch,)
        assert entropy.shape == (batch,)
        assert values.shape == (1,)
        assert means.shape == (batch, 3)

    def test_critic_counts_affect_value(self):
        """Different critic_counts should produce different values."""
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        policy.eval()
        tokens = torch.randn(7, 21)
        mask = torch.ones(7)
        with torch.no_grad():
            v_win = policy.value(tokens, mask, critic_counts=torch.tensor([3., 3., 0., 4.]))
            v_lose = policy.value(tokens, mask, critic_counts=torch.tensor([0., 3., 4., 4.]))
        # Different counts -> different values
        assert v_win.item() != v_lose.item()

    def test_save_load_roundtrip(self, tmp_path):
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        path = tmp_path / "test.pt"
        policy.save(str(path))
        policy2 = HeteroEntityRecurrentPolicy(entity_dim=21)
        policy2.load(str(path))
        for p1, p2 in zip(policy.parameters(), policy2.parameters()):
            assert torch.allclose(p1, p2), "parameters should match after load"


class TestCriticV2:
    """Verify _GlobalEntityCritic v2 with counts."""

    def test_critic_with_counts(self):
        critic = _GlobalEntityCritic(entity_dim=21, hidden_dim=128, num_heads=4, count_feat_dim=4)
        tokens = torch.randn(2, 7, 21)  # batch=2, 7 entities
        keep = torch.ones(2, 7)
        counts = torch.tensor([[3., 3., 2., 4.], [1., 3., 4., 4.]])  # (batch, 4)
        out = critic(tokens, keep, counts)
        assert out.shape == (2,)

    def test_critic_without_counts(self):
        """Backward compatibility: critic works without counts."""
        critic = _GlobalEntityCritic(entity_dim=21, hidden_dim=128, num_heads=4, count_feat_dim=4)
        tokens = torch.randn(1, 7, 21)
        keep = torch.ones(1, 7)
        out = critic(tokens, keep, counts=None)
        assert out.shape == (1,)
