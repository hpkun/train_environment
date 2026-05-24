"""Attention-based MAPPO network components.

These modules are intentionally not connected to the current vanilla MAPPO
training loop.  They provide a baseline EntityObservationEncoder path for later
MAPPO-Attention / BRMA-MAPPO experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityObservationEncoder(nn.Module):
    """Encode entity-wise observations with shared MLP + self-attention."""

    def __init__(self, entity_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, entities: torch.Tensor,
                entity_mask: torch.Tensor | None = None):
        """Encode entities.

        Args:
            entities: Tensor with shape (B, N, entity_dim).
            entity_mask: Optional tensor with shape (B, N), where 1 means
                masked/invalid/dead and 0 means valid.

        Returns:
            encoded_first: Tensor with shape (B, hidden_size), using the first
                entity as the ego entity representation.
            attn_weights: Tensor with shape (B, num_heads, N, N).

        Paper eq.33 concatenates attention output and original entity embedding.
        This initial implementation keeps the output at hidden_size so it can be
        connected to future training code with minimal interface churn.
        """
        embedded = self.entity_mlp(entities)
        key_padding_mask = None
        if entity_mask is not None:
            key_padding_mask = entity_mask.bool().clone()
            key_padding_mask[:, 0] = False

        encoded, attn_weights = self.attention(
            embedded,
            embedded,
            embedded,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        encoded_first = encoded[:, 0, :]
        return encoded_first, attn_weights


class AttentionActor(nn.Module):
    """Attention actor for future MAPPO-Attention experiments."""

    def __init__(self, entity_dim: int = 11, action_dim: int = 3,
                 hidden_size: int = 128, rnn_hidden: int = 128,
                 num_heads: int = 4):
        super().__init__()
        self.encoder = EntityObservationEncoder(
            entity_dim=entity_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.rnn = nn.GRUCell(hidden_size, rnn_hidden)
        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -1.204))

    def forward(self, entities: torch.Tensor, entity_mask: torch.Tensor,
                rnn_hidden: torch.Tensor):
        encoded_first, attn_weights = self.encoder(entities, entity_mask)
        new_rnn_hidden = self.rnn(encoded_first, rnn_hidden)
        mu = self.action_head(new_rnn_hidden)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu = mu.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mu)
        return torch.distributions.Normal(mu, sigma), new_rnn_hidden, attn_weights


class AttentionCritic(nn.Module):
    """Simple attention critic reserved for future centralized MAPPO use."""

    def __init__(self, entity_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4):
        super().__init__()
        self.encoder = EntityObservationEncoder(
            entity_dim=entity_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, entities: torch.Tensor,
                entity_mask: torch.Tensor) -> torch.Tensor:
        encoded_first, _attn_weights = self.encoder(entities, entity_mask)
        return self.value_head(encoded_first)
