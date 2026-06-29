"""BRMA-style recurrent actor policy with GRUCell for staged experiments.

This opt-in policy adds a GRU recurrent layer between the BRMA entity encoder
and the MAV/UAV actor heads.  The centralized critic remains a 480-dim MLP
critic (no GRU in critic).

Scope intentionally excludes random masks, biased masks, and strict HAPPO
correction.  This is a smoke-level GRU integration — PPO updates use one-step
GRU state replay, not full TBPTT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .brma_entity_policy import BRMAEntityObservationEncoder
from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, out_dim),
    )


class BRMARecurrentHAPPOReferencePolicy(nn.Module):
    """BRMA entity encoder + GRUCell + MAV/UAV actor heads.

    ``policy_arch = "brma_recurrent"``

    Architecture::

        flat_obs (96-dim)
          → _flat_to_entities() → entity tensor [B, N, entity_dim]
          → BRMAEntityObservationEncoder → pooled [B, 256]
          → nn.GRUCell(256, rnn_hidden_size) → rnn_hidden [B, rnn_hidden_size]
          → MAV actor head (rnn_hidden_size → 3)
          → UAV actor head (rnn_hidden_size → 3)
          → Normal(μ, σ)

    The critic remains the existing 480-dim MLP (no GRU).
    """

    def __init__(
        self,
        entity_dim: int = 30,
        critic_state_dim: int = 480,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        rnn_hidden_size: int = 128,
        max_allies: int = 4,
        max_enemies: int = 4,
    ):
        super().__init__()
        self.entity_dim = int(entity_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.max_allies = int(max_allies)
        self.max_enemies = int(max_enemies)
        self.enemy_flat_dim = 18
        self.flat_actor_obs_dim = 12 + self.max_allies * 9 + self.max_enemies * self.enemy_flat_dim + 20

        self.encoder = BRMAEntityObservationEncoder(
            entity_dim=self.entity_dim,
            hidden_size=self.hidden_dim,
            num_heads=num_attention_heads,
        )
        # GRU cell between encoder output and action heads
        self.rnn = nn.GRUCell(
            input_size=self.encoder.output_dim,   # 256
            hidden_size=self.rnn_hidden_size,     # 128
        )
        self.mav_actor = _mlp(self.rnn_hidden_size, self.action_dim)
        self.uav_actor = _mlp(self.rnn_hidden_size, self.action_dim)
        self.critic = _mlp(self.critic_state_dim, 1)
        init_log_std = float(np.log(0.3))
        self.action_log_std_mav = nn.Parameter(torch.full((self.action_dim,), init_log_std))
        self.action_log_std_uav = nn.Parameter(torch.full((self.action_dim,), init_log_std))

    @staticmethod
    def infer_role_ids(roles: Iterable[str | int] | torch.Tensor | None, batch: int, device) -> torch.Tensor:
        if roles is None:
            ids = torch.full((batch,), UAV_ROLE_ID, dtype=torch.long, device=device)
            if batch > 0:
                ids[0] = MAV_ROLE_ID
            return ids
        if isinstance(roles, torch.Tensor):
            return roles.to(device=device, dtype=torch.long).reshape(-1)
        out = []
        for role in roles:
            if isinstance(role, str):
                out.append(MAV_ROLE_ID if role == "mav" else UAV_ROLE_ID)
            else:
                out.append(int(role))
        return torch.as_tensor(out, dtype=torch.long, device=device)

    def actor_shared_parameters(self):
        return list(self.encoder.parameters()) + list(self.rnn.parameters())

    # ------------------------------------------------------------------
    # Flat-to-entity decoding (shared with brma_entity)
    # ------------------------------------------------------------------
    def _flat_to_entities(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = flat_obs.reshape(-1, flat_obs.shape[-1])
        batch = flat.shape[0]
        entities = torch.zeros(
            (batch, 1 + self.max_allies + self.max_enemies, self.entity_dim),
            dtype=flat.dtype,
            device=flat.device,
        )
        keep = torch.zeros(
            (batch, 1 + self.max_allies + self.max_enemies),
            dtype=torch.bool,
            device=flat.device,
        )

        ego = flat[:, :12]
        allies_start = 12
        enemies_start = allies_start + self.max_allies * 9
        masks_start = enemies_start + self.max_enemies * self.enemy_flat_dim
        allies = flat[:, allies_start:enemies_start].reshape(batch, self.max_allies, 9)
        enemies = flat[:, enemies_start:masks_start].reshape(batch, self.max_enemies, self.enemy_flat_dim)
        masks = flat[:, masks_start:masks_start + 20]
        ally_valid = masks[:, :self.max_allies]
        ally_alive = masks[:, self.max_allies:self.max_allies * 2]
        enemy_valid = masks[:, self.max_allies * 2:self.max_allies * 2 + self.max_enemies]
        enemy_alive = masks[:, self.max_allies * 2 + self.max_enemies:self.max_allies * 2 + self.max_enemies * 2]
        enemy_observed = masks[:, self.max_allies * 2 + self.max_enemies * 2:
                               self.max_allies * 2 + self.max_enemies * 3]

        entities[:, 0, 0] = 1.0
        entities[:, 0, 3:7] = ego[:, 7:11]
        entities[:, 0, 7:14] = ego[:, :7]
        entities[:, 0, 14] = 1.0
        entities[:, 0, 16] = ego[:, 11]
        keep[:, 0] = True

        for i in range(self.max_allies):
            idx = 1 + i
            entities[:, idx, 1] = 1.0
            entities[:, idx, 3:7] = allies[:, i, 5:9]
            entities[:, idx, 7:12] = allies[:, i, :5]
            entities[:, idx, 14] = 1.0
            keep[:, idx] = (ally_valid[:, i] > 0.5) & (ally_alive[:, i] > 0.5)

        for i in range(self.max_enemies):
            idx = 1 + self.max_allies + i
            entities[:, idx, 2] = 1.0
            entities[:, idx, 7:12] = enemies[:, i, :5]
            entities[:, idx, 15] = 1.0
            entities[:, idx, 17:19] = enemies[:, i, 5:7]
            if self.entity_dim > 19:
                n = min(self.entity_dim - 19, self.enemy_flat_dim - 7)
                entities[:, idx, 19:19 + n] = enemies[:, i, 7:7 + n]
            keep[:, idx] = (
                (enemy_valid[:, i] > 0.5)
                & (enemy_alive[:, i] > 0.5)
                & (enemy_observed[:, i] > 0.5)
            )
        return entities, keep

    # ------------------------------------------------------------------
    # Encode (no GRU — used by value/critic path if needed)
    # ------------------------------------------------------------------
    def encode(self, actor_obs) -> tuple[torch.Tensor, tuple[int, ...]]:
        raw_t = torch.as_tensor(actor_obs, dtype=torch.float32, device=next(self.parameters()).device)
        if raw_t.shape[-1] == self.flat_actor_obs_dim:
            leading_shape = tuple(raw_t.shape[:-1])
            entities_t, keep_mask = self._flat_to_entities(raw_t)
        elif raw_t.ndim >= 3 and raw_t.shape[-1] == self.entity_dim:
            leading_shape = tuple(raw_t.shape[:-2])
            entities_t = raw_t.reshape(-1, raw_t.shape[-2], raw_t.shape[-1])
            keep_mask = torch.ones(entities_t.shape[:2], dtype=torch.bool, device=entities_t.device)
        else:
            raise ValueError(
                f"expected flat actor obs dim {self.flat_actor_obs_dim} or entity dim {self.entity_dim}, "
                f"got shape {tuple(raw_t.shape)}"
            )
        pooled, _attn = self.encoder(entities_t, keep_mask)
        return pooled, leading_shape

    # ------------------------------------------------------------------
    # Initialize hidden state
    # ------------------------------------------------------------------
    def init_hidden(self, batch: int, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch, self.rnn_hidden_size, device=device)

    # ------------------------------------------------------------------
    # Means and stds from rnn_hidden (not encoder output)
    # ------------------------------------------------------------------
    def _means_and_stds(self, rnn_hidden: torch.Tensor, role_ids: torch.Tensor):
        role_ids = role_ids.reshape(-1).to(rnn_hidden.device)
        means = torch.zeros((rnn_hidden.shape[0], self.action_dim), device=rnn_hidden.device)
        stds = torch.zeros_like(means)
        mav_mask = role_ids == MAV_ROLE_ID
        uav_mask = ~mav_mask
        if mav_mask.any():
            means[mav_mask] = torch.clamp(self.mav_actor(rnn_hidden[mav_mask]), -0.999, 0.999)
            stds[mav_mask] = self.action_log_std_mav.exp().expand_as(means[mav_mask])
        if uav_mask.any():
            means[uav_mask] = torch.clamp(self.uav_actor(rnn_hidden[uav_mask]), -0.999, 0.999)
            stds[uav_mask] = self.action_log_std_uav.exp().expand_as(means[uav_mask])
        return means, stds

    # ------------------------------------------------------------------
    # act — with recurrent hidden state
    # ------------------------------------------------------------------
    def act(self, actor_obs, roles=None, critic_state=None, deterministic: bool = False,
            rnn_hidden: torch.Tensor | None = None):
        pooled, leading_shape = self.encode(actor_obs)
        batch = pooled.shape[0]

        if rnn_hidden is None:
            rnn_hidden = self.init_hidden(batch, pooled.device)
        if rnn_hidden.dim() == 3:
            rnn_hidden = rnn_hidden.reshape(-1, rnn_hidden.shape[-1])
        rnn_hidden_new = self.rnn(pooled, rnn_hidden)

        role_ids = self.infer_role_ids(roles, batch, pooled.device)
        mean, std = self._means_and_stds(rnn_hidden_new, role_ids)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        value = None
        if critic_state is not None:
            critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=pooled.device)
            if critic_t.ndim == 1:
                critic_t = critic_t.unsqueeze(0)
            value = self.critic(critic_t).squeeze(-1)
        return {
            "action": action.view(*leading_shape, self.action_dim),
            "log_prob": dist.log_prob(action).sum(dim=-1).view(*leading_shape),
            "entropy": dist.entropy().sum(dim=-1).view(*leading_shape),
            "value": value,
            "mean": mean.view(*leading_shape, self.action_dim),
            "role_mask": role_ids.view(*leading_shape),
            "rnn_hidden": rnn_hidden_new.view(*leading_shape, self.rnn_hidden_size),
        }

    # ------------------------------------------------------------------
    # evaluate_actions — with stored rnn_hidden for one-step replay
    # ------------------------------------------------------------------
    def evaluate_actions(self, actor_obs, roles, critic_state, actions,
                         rnn_hidden: torch.Tensor | None = None):
        pooled, leading_shape = self.encode(actor_obs)
        batch = pooled.shape[0]

        if rnn_hidden is None:
            rnn_hidden = self.init_hidden(batch, pooled.device)
        if rnn_hidden.dim() == 3:
            rnn_hidden = rnn_hidden.reshape(-1, rnn_hidden.shape[-1])
        rnn_hidden_new = self.rnn(pooled, rnn_hidden)

        role_ids = self.infer_role_ids(roles, batch, pooled.device)
        mean, std = self._means_and_stds(rnn_hidden_new, role_ids)
        flat_actions = torch.as_tensor(actions, dtype=torch.float32, device=pooled.device).reshape(-1, self.action_dim)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(flat_actions).sum(dim=-1).view(*leading_shape)
        entropy = dist.entropy().sum(dim=-1).view(*leading_shape)
        critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=pooled.device)
        values = self.critic(critic_t).squeeze(-1)
        return (
            log_prob,
            entropy,
            values,
            mean.view(*leading_shape, self.action_dim),
            role_ids.view(*leading_shape),
        )

    def value(self, critic_state):
        critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=next(self.parameters()).device)
        if critic_t.ndim == 1:
            critic_t = critic_t.unsqueeze(0)
        return self.critic(critic_t).squeeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
