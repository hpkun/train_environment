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
    """Encode entity-wise observations with shared MLP + self-attention.

    Paper eq.33 concatenates the attention output with the original entity
    embedding.  Two modes are supported:

    - ``current`` (default):  output = attention[:, 0, :], dim = hidden_size.
      Backward-compatible with existing checkpoints and smoke tests.
    - ``paper_eq33``:  output = concat([entity_embedding, attention_output],
      dim=-1) at the ego entity index, dim = 2 * hidden_size.
    """

    def __init__(self, entity_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, dropout: float = 0.0,
                 encoder_mode: str = "current"):
        super().__init__()
        if encoder_mode not in ("current", "paper_eq33"):
            raise ValueError(
                f"encoder_mode must be 'current' or 'paper_eq33', "
                f"got {encoder_mode!r}")
        self.encoder_mode = encoder_mode

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
        self.output_dim = (2 * hidden_size if encoder_mode == "paper_eq33"
                           else hidden_size)

    def forward(self, entities: torch.Tensor,
                entity_mask: torch.Tensor | None = None,
                soft_keep_mask: torch.Tensor | None = None):
        """Encode entities.

        Args:
            entities: Tensor with shape (B, N, entity_dim).
            entity_mask: Optional tensor with shape (B, N), where 1 means
                masked/invalid/dead and 0 means valid.
            soft_keep_mask: Optional float tensor with shape (B, N), where
                1 means fully visible and 0 means softly suppressed.  Hard
                entity_mask still controls invalid/dead/padded entities.

        Returns:
            encoded_first: Tensor with shape (B, output_dim).
            attn_weights: Tensor with shape (B, num_heads, N, N).
        """
        embedded = self.entity_mlp(entities)
        if soft_keep_mask is not None:
            if soft_keep_mask.shape != entities.shape[:2]:
                raise ValueError(
                    "soft_keep_mask must have shape matching entities[:2]")
            keep = soft_keep_mask.to(dtype=embedded.dtype, device=embedded.device)
            keep = keep.clamp(0.0, 1.0)
            keep = torch.cat([torch.ones_like(keep[:, :1]), keep[:, 1:]], dim=1)
            embedded = embedded * keep.unsqueeze(-1)

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

        if self.encoder_mode == "paper_eq33":
            # Paper eq.33: concat(entity_embedding, attention_output) for each entity
            concat = torch.cat([embedded, encoded], dim=-1)
            encoded_first = concat[:, 0, :]
        else:
            encoded_first = encoded[:, 0, :]
        return encoded_first, attn_weights


class AttentionActor(nn.Module):
    """Attention actor for MAPPO-Attention / BRMA-MAPPO experiments."""

    def __init__(self, entity_dim: int = 11, action_dim: int = 3,
                 hidden_size: int = 128, rnn_hidden: int = 128,
                 num_heads: int = 4, encoder_mode: str = "current"):
        super().__init__()
        self.encoder = EntityObservationEncoder(
            entity_dim=entity_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            encoder_mode=encoder_mode,
        )
        self.encoder_output_dim = self.encoder.output_dim
        self.rnn = nn.GRUCell(self.encoder_output_dim, rnn_hidden)
        self.action_head = nn.Sequential(
            nn.Linear(rnn_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -1.204))

    # ------------------------------------------------------------------
    #  Internal policy helper — single forward pass through encoder + GRU
    # ------------------------------------------------------------------

    def _policy_from_entities(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        rnn_hidden: torch.Tensor,
        soft_keep_mask: torch.Tensor | None = None,
    ) -> dict:
        """Run encoder → GRU → action head and return all intermediates.

        Entity mask convention: (B, N), 0 / False = visible, 1 / True = ignored.
        The encoder internally forces the self entity (index 0) to be visible
        regardless of the mask value.
        """
        encoded_first, attn_weights = self.encoder(
            entities, entity_mask, soft_keep_mask=soft_keep_mask)
        new_rnn_hidden = self.rnn(encoded_first, rnn_hidden)
        mu = self.action_head(new_rnn_hidden)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        mu = mu.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mu)
        return {
            "dist": torch.distributions.Normal(mu, sigma),
            "new_rnn_hidden": new_rnn_hidden,
            "attn_weights": attn_weights,
            "mu": mu,
            "sigma": sigma,
        }

    def forward(self, entities: torch.Tensor, entity_mask: torch.Tensor,
                rnn_hidden: torch.Tensor):
        """Backward-compatible forward.  Returns (dist, new_rnn_hidden, attn_weights)."""
        out = self._policy_from_entities(entities, entity_mask, rnn_hidden)
        return out["dist"], out["new_rnn_hidden"], out["attn_weights"]

    # ------------------------------------------------------------------
    #  Action evaluation (single path)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        rnn_hidden: torch.Tensor,
        actions: torch.Tensor,
        soft_keep_mask: torch.Tensor | None = None,
    ) -> dict:
        """Compute log-prob, entropy, and diagnostics for given actions.

        Useful for BRMA buffer collection and mask loss — not used by the
        current PPO training loop.

        Args:
            entities:     (B, N, D)
            entity_mask:  (B, N)  0=visible, 1=ignored
            rnn_hidden:   (B, rnn_hidden)
            actions:      (B, action_dim)
            soft_keep_mask: optional (B, N) differentiable keep weights for
                BRMA masked policy evaluation.

        Returns dict with log_prob, entropy_mean, entropy_sum, mu, sigma,
        new_rnn_hidden, attn_weights.
        """
        out = self._policy_from_entities(
            entities, entity_mask, rnn_hidden, soft_keep_mask=soft_keep_mask)
        dist = out["dist"]
        log_prob = dist.log_prob(actions).sum(dim=-1)    # (B,)
        entropy_mean = dist.entropy().mean(dim=-1)        # (B,)
        entropy_sum = dist.entropy().sum(dim=-1)          # (B,)
        return {
            "log_prob": log_prob,
            "entropy_mean": entropy_mean,
            "entropy_sum": entropy_sum,
            "mu": out["mu"],
            "sigma": out["sigma"],
            "new_rnn_hidden": out["new_rnn_hidden"],
            "attn_weights": out["attn_weights"],
        }

    # ------------------------------------------------------------------
    #  Dual-action evaluation (masked / unkmasked)
    # ------------------------------------------------------------------

    def evaluate_dual_actions(
        self,
        entities: torch.Tensor,
        unmasked_entity_mask: torch.Tensor,
        masked_entity_mask: torch.Tensor,
        rnn_hidden: torch.Tensor,
        actions: torch.Tensor,
        masked_soft_keep_mask: torch.Tensor | None = None,
    ) -> dict:
        """Evaluate the same actions under two entity masks.

        ``unmasked_entity_mask`` produces p(a | e), i.e. the actor sees all
        valid entities.  ``masked_entity_mask`` produces p(a | e_mask),
        where some entities are dropped according to BRMA mask rules.
        ``masked_soft_keep_mask`` optionally adds differentiable keep weights
        to the masked path only.

        Both paths start from the **same** ``rnn_hidden`` — the hidden
        state is not chained between the two evaluations.  The encoder
        header forces self (index 0) visible regardless of mask value
        (see ``EntityObservationEncoder.forward``).

        Returns a nested dict with "unmasked" / "masked" sub-dicts plus
        top-level convenience keys.
        """
        unmasked = self.evaluate_actions(
            entities, unmasked_entity_mask, rnn_hidden, actions)
        masked = self.evaluate_actions(
            entities,
            masked_entity_mask,
            rnn_hidden,
            actions,
            soft_keep_mask=masked_soft_keep_mask,
        )
        return {
            "unmasked": unmasked,
            "masked": masked,
            "log_prob_unmasked": unmasked["log_prob"],
            "log_prob_masked": masked["log_prob"],
            "entropy_unmasked_mean": unmasked["entropy_mean"],
            "entropy_masked_mean": masked["entropy_mean"],
            "mu_unmasked": unmasked["mu"],
            "mu_masked": masked["mu"],
            "sigma_unmasked": unmasked["sigma"],
            "sigma_masked": masked["sigma"],
            "new_rnn_hidden": unmasked["new_rnn_hidden"],
        }


class AttentionCritic(nn.Module):
    """Simple attention critic reserved for future centralized MAPPO use."""

    def __init__(self, entity_dim: int = 11, hidden_size: int = 128,
                 num_heads: int = 4, encoder_mode: str = "current"):
        super().__init__()
        self.encoder = EntityObservationEncoder(
            entity_dim=entity_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            encoder_mode=encoder_mode,
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, entities: torch.Tensor,
                entity_mask: torch.Tensor) -> torch.Tensor:
        encoded_first, _attn_weights = self.encoder(entities, entity_mask)
        return self.value_head(encoded_first)


class CentralizedAttentionCritic(nn.Module):
    """Paper-style centralized critic using per-agent entity attention.

    Each red agent has a strict entity table.  The critic encodes every
    agent's table with a *shared* ``EntityObservationEncoder``, then
    concatenates the per-agent features and passes them through an MLP
    value head to produce one value per red agent.

    This critic does NOT use a biased random mask (BRMA).  Entity masks
    only mark dead / padded entities.
    """

    def __init__(self, entity_dim: int = 10, hidden_size: int = 128,
                 num_heads: int = 4, num_agents: int = 2,
                 encoder_mode: str = "current"):
        super().__init__()
        self.encoder = EntityObservationEncoder(
            entity_dim=entity_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            encoder_mode=encoder_mode,
        )
        self.num_agents = num_agents
        self.encoder_output_dim = self.encoder.output_dim
        concat_dim = num_agents * self.encoder_output_dim
        self.value_head = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_agents),
        )

    def forward(self, team_entities: torch.Tensor,
                team_masks: torch.Tensor) -> torch.Tensor:
        """Encode each red agent's entity table and predict per-agent values.

        Args:
            team_entities: (B, A, N, D) for A agents, N entities each.
            team_masks:    (B, A, N) entity masks.

        Returns:
            values: (B, A) per-agent value estimates.
        """
        B, A, N, D = team_entities.shape
        # Flatten agent dimension into batch for shared encoder
        flat_entities = team_entities.reshape(B * A, N, D)
        flat_masks = team_masks.reshape(B * A, N)
        encoded, _attn = self.encoder(flat_entities, flat_masks)
        # encoded: (B*A, encoder_output_dim)
        concat = encoded.reshape(B, A * self.encoder_output_dim)
        return self.value_head(concat)
