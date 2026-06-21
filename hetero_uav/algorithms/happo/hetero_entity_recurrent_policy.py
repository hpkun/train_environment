"""Variable-token recurrent entity policy with an entity-based critic."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


class _EntityAttention(nn.Module):
    def __init__(self, entity_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, keep_mask: torch.Tensor, pool_ego: bool) -> torch.Tensor:
        keep = keep_mask.bool().clone()
        if pool_ego:
            keep[:, 0] = True
        embedded = self.embedding(tokens)
        attended, _ = self.attention(
            embedded, embedded, embedded, key_padding_mask=~keep, need_weights=False,
        )
        if pool_ego:
            return attended[:, 0]
        weights = keep.to(attended.dtype).unsqueeze(-1)
        return (attended * weights).sum(1) / weights.sum(1).clamp(min=1.0)


class _GlobalEntityCritic(nn.Module):
    def __init__(self, entity_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.encoder = _EntityAttention(entity_dim, hidden_dim, num_heads)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, tokens: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encoder(tokens, keep_mask, pool_ego=False)).squeeze(-1)


class HeteroEntityRecurrentPolicy(nn.Module):
    """Shared entity encoder + GRUCell + heterogeneous action heads."""

    def __init__(self, entity_dim=19, action_dim=3, hidden_dim=128,
                 rnn_hidden_size=128, num_attention_heads=4):
        super().__init__()
        if int(action_dim) != 3:
            raise ValueError("hetero_entity_recurrent requires action_dim=3")
        self.entity_dim = int(entity_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.actor_encoder = _EntityAttention(
            self.entity_dim, self.hidden_dim, self.num_attention_heads)
        self.rnn = nn.GRUCell(self.hidden_dim, self.rnn_hidden_size)
        self.mav_actor = self._head()
        self.uav_actor = self._head()
        self.critic = _GlobalEntityCritic(
            self.entity_dim, self.hidden_dim, self.num_attention_heads)
        initial = float(np.log(0.3))
        self.action_log_std_mav = nn.Parameter(torch.full((3,), initial))
        self.action_log_std_uav = nn.Parameter(torch.full((3,), initial))

    def _head(self):
        return nn.Sequential(
            nn.Linear(self.rnn_hidden_size, 128), nn.Tanh(),
            nn.Linear(128, self.action_dim), nn.Tanh(),
        )

    def actor_shared_parameters(self):
        return list(self.actor_encoder.parameters()) + list(self.rnn.parameters())

    def init_hidden(self, batch: int, device=None):
        return torch.zeros(batch, self.rnn_hidden_size, device=device or next(self.parameters()).device)

    def _actor_features(self, tokens, keep_mask, rnn_hidden):
        leading = tokens.shape[:-2]
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        flat_mask = keep_mask.reshape(-1, keep_mask.shape[-1])
        pooled = self.actor_encoder(flat_tokens, flat_mask, pool_ego=True)
        hidden = rnn_hidden.reshape(-1, self.rnn_hidden_size)
        next_hidden = self.rnn(pooled, hidden)
        return next_hidden, leading

    def _distribution(self, hidden, roles):
        flat_roles = roles.reshape(-1).to(hidden.device)
        means = torch.zeros(hidden.shape[0], self.action_dim, device=hidden.device)
        stds = torch.zeros_like(means)
        mav = flat_roles == MAV_ROLE_ID
        uav = flat_roles == UAV_ROLE_ID
        if mav.any():
            means[mav] = self.mav_actor(hidden[mav])
            stds[mav] = self.action_log_std_mav.exp()
        if uav.any():
            means[uav] = self.uav_actor(hidden[uav])
            stds[uav] = self.action_log_std_uav.exp()
        return Normal(means, stds), means

    def act(self, actor_entity_tokens, actor_keep_mask, roles,
            critic_entity_tokens, critic_keep_mask, deterministic=False,
            rnn_hidden=None):
        device = next(self.parameters()).device
        tokens = torch.as_tensor(actor_entity_tokens, dtype=torch.float32, device=device)
        keep = torch.as_tensor(actor_keep_mask, dtype=torch.bool, device=device)
        role_ids = torch.as_tensor(roles, dtype=torch.long, device=device)
        if rnn_hidden is None:
            rnn_hidden = self.init_hidden(tokens.shape[0], device)
        else:
            rnn_hidden = torch.as_tensor(rnn_hidden, dtype=torch.float32, device=device)
        hidden, leading = self._actor_features(tokens, keep, rnn_hidden)
        dist, means = self._distribution(hidden, role_ids)
        actions = means if deterministic else dist.rsample()
        actions = actions.clamp(-1.0, 1.0)
        value = self.value(critic_entity_tokens, critic_keep_mask)
        return {
            "action": actions.reshape(*leading, self.action_dim),
            "log_prob": dist.log_prob(actions).sum(-1).reshape(*leading),
            "entropy": dist.entropy().sum(-1).reshape(*leading),
            "value": value,
            "mean": means.reshape(*leading, self.action_dim),
            "role_mask": role_ids,
            "rnn_hidden": hidden.reshape(*leading, self.rnn_hidden_size),
        }

    def evaluate_actions(self, actor_entity_tokens, actor_keep_mask, roles,
                         critic_entity_tokens, critic_keep_mask, actions,
                         rnn_hidden=None):
        tokens = actor_entity_tokens.float()
        keep = actor_keep_mask.bool()
        role_ids = roles.long()
        if rnn_hidden is None:
            rnn_hidden = self.init_hidden(int(np.prod(tokens.shape[:-2])), tokens.device).reshape(
                *tokens.shape[:-2], self.rnn_hidden_size)
        hidden, leading = self._actor_features(tokens, keep, rnn_hidden)
        dist, means = self._distribution(hidden, role_ids)
        flat_actions = actions.reshape(-1, self.action_dim)
        values = self.value(critic_entity_tokens, critic_keep_mask)
        return (
            dist.log_prob(flat_actions).sum(-1).reshape(*leading),
            dist.entropy().sum(-1).reshape(*leading),
            values,
            means.reshape(*leading, self.action_dim),
            role_ids,
            hidden.reshape(*leading, self.rnn_hidden_size),
        )

    def value(self, critic_entity_tokens, critic_keep_mask):
        device = next(self.parameters()).device
        tokens = torch.as_tensor(critic_entity_tokens, dtype=torch.float32, device=device)
        keep = torch.as_tensor(critic_keep_mask, dtype=torch.bool, device=device)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            keep = keep.unsqueeze(0)
        return self.critic(tokens, keep)

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
