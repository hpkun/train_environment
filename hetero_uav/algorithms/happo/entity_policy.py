"""Entity-attention actor variant for staged HAPPO experiments.

This is an opt-in policy. It does not replace HAPPOReferencePolicy and it keeps
the centralized critic as the existing 480-dim MLP for checkpoint compatibility
with the flat baseline path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, out_dim),
    )


class EntityHAPPOReferencePolicy(nn.Module):
    """Minimal entity-attention actor with MAV and shared-UAV heads.

    The actor consumes entity tokens and an alive/observed attention mask. It is
    intentionally no-GRU and no-biased-mask; those remain later-stage changes.
    """

    def __init__(
        self,
        entity_dim: int,
        critic_state_dim: int = 480,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        max_allies: int = 4,
        max_enemies: int = 4,
    ):
        super().__init__()
        self.entity_dim = int(entity_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_allies = int(max_allies)
        self.max_enemies = int(max_enemies)
        self.flat_actor_obs_dim = 12 + self.max_allies * 9 + self.max_enemies * 7 + 20

        self.entity_encoder = nn.Sequential(
            nn.Linear(self.entity_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.mav_actor = _mlp(self.hidden_dim, self.action_dim)
        self.uav_actor = _mlp(self.hidden_dim, self.action_dim)
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

    def _flat_to_entities(self, flat_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = flat_obs.reshape(-1, flat_obs.shape[-1])
        batch = flat.shape[0]
        entities = torch.zeros(
            (batch, 1 + self.max_allies + self.max_enemies, self.entity_dim),
            dtype=flat.dtype,
            device=flat.device,
        )
        keep = torch.zeros((batch, 1 + self.max_allies + self.max_enemies),
                           dtype=torch.bool, device=flat.device)

        ego = flat[:, :12]
        allies_start = 12
        enemies_start = allies_start + self.max_allies * 9
        masks_start = enemies_start + self.max_enemies * 7
        allies = flat[:, allies_start:enemies_start].reshape(batch, self.max_allies, 9)
        enemies = flat[:, enemies_start:masks_start].reshape(batch, self.max_enemies, 7)
        masks = flat[:, masks_start:masks_start + 20]
        ally_valid = masks[:, :self.max_allies]
        ally_alive = masks[:, self.max_allies:self.max_allies * 2]
        enemy_valid = masks[:, self.max_allies * 2:self.max_allies * 2 + self.max_enemies]
        enemy_alive = masks[:, self.max_allies * 2 + self.max_enemies:self.max_allies * 2 + self.max_enemies * 2]
        enemy_observed = masks[:, self.max_allies * 2 + self.max_enemies * 2:
                               self.max_allies * 2 + self.max_enemies * 3]

        # Token layout: kind(3), role(4), geo(7), side(2), missile_warning(1), track(2).
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
            keep[:, idx] = (
                (enemy_valid[:, i] > 0.5)
                & (enemy_alive[:, i] > 0.5)
                & (enemy_observed[:, i] > 0.5)
            )
        return entities, keep

    def _unpack_entities(self, entity_obs):
        if isinstance(entity_obs, dict):
            entities = entity_obs["entities"]
            attention_mask = entity_obs.get("attention_mask")
        else:
            entities = entity_obs
            attention_mask = None
        device = next(self.parameters()).device
        entities_t = torch.as_tensor(entities, dtype=torch.float32, device=device)
        if entities_t.shape[-1] == self.flat_actor_obs_dim:
            return self._flat_to_entities(entities_t)
        if entities_t.ndim == 2:
            entities_t = entities_t.unsqueeze(0)
        if attention_mask is None:
            mask_t = torch.ones(entities_t.shape[:2], dtype=torch.bool, device=device)
        else:
            mask_t = torch.as_tensor(attention_mask, dtype=torch.float32, device=device)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            mask_t = mask_t > 0.5
        mask_t[:, 0] = True
        return entities_t, mask_t

    def encode(self, entity_obs) -> tuple[torch.Tensor, tuple[int, ...]]:
        if isinstance(entity_obs, dict):
            raw = entity_obs["entities"]
        else:
            raw = entity_obs
        raw_t = torch.as_tensor(raw, dtype=torch.float32, device=next(self.parameters()).device)
        if raw_t.shape[-1] == self.flat_actor_obs_dim:
            leading_shape = tuple(raw_t.shape[:-1])
        elif raw_t.ndim == 2:
            leading_shape = (1,)
        else:
            leading_shape = tuple(raw_t.shape[:-2])
        entities_t, keep_mask = self._unpack_entities(entity_obs)
        encoded = self.entity_encoder(entities_t)
        key_padding_mask = ~keep_mask
        attended, _weights = self.attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attended[:, 0, :], leading_shape

    def _means_and_stds(self, pooled: torch.Tensor, role_ids: torch.Tensor):
        role_ids = role_ids.reshape(-1).to(pooled.device)
        means = torch.zeros((pooled.shape[0], self.action_dim), device=pooled.device)
        stds = torch.zeros_like(means)
        mav_mask = role_ids == MAV_ROLE_ID
        uav_mask = ~mav_mask
        if mav_mask.any():
            means[mav_mask] = torch.clamp(self.mav_actor(pooled[mav_mask]), -0.999, 0.999)
            stds[mav_mask] = self.action_log_std_mav.exp().expand_as(means[mav_mask])
        if uav_mask.any():
            means[uav_mask] = torch.clamp(self.uav_actor(pooled[uav_mask]), -0.999, 0.999)
            stds[uav_mask] = self.action_log_std_uav.exp().expand_as(means[uav_mask])
        return means, stds

    def act(self, entity_obs, roles=None, critic_state=None, deterministic: bool = False):
        pooled, leading_shape = self.encode(entity_obs)
        role_ids = self.infer_role_ids(roles, pooled.shape[0], pooled.device)
        mean, std = self._means_and_stds(pooled, role_ids)
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
        }

    def evaluate_actions(self, entity_obs, roles, critic_state, actions):
        pooled, leading_shape = self.encode(entity_obs)
        role_ids = self.infer_role_ids(roles, pooled.shape[0], pooled.device)
        mean, std = self._means_and_stds(pooled, role_ids)
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

    def uav_imitation_loss_from_flat(self, actor_obs: torch.Tensor, oracle_actions: torch.Tensor) -> torch.Tensor:
        pooled, _leading_shape = self.encode(actor_obs)
        roles = torch.full((pooled.shape[0],), UAV_ROLE_ID, dtype=torch.long, device=pooled.device)
        mean, _std = self._means_and_stds(pooled, roles)
        error = mean - oracle_actions.to(mean.device)
        error = error.clone()
        error[..., 1] = torch.remainder(mean[..., 1] - oracle_actions.to(mean.device)[..., 1] + 1.0, 2.0) - 1.0
        return torch.mean(error ** 2)

    def value(self, critic_state):
        critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=next(self.parameters()).device)
        if critic_t.ndim == 1:
            critic_t = critic_t.unsqueeze(0)
        return self.critic(critic_t).squeeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
