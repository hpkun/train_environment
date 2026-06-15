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
    ):
        super().__init__()
        self.entity_dim = int(entity_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

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

    def _unpack_entities(self, entity_obs):
        if isinstance(entity_obs, dict):
            entities = entity_obs["entities"]
            attention_mask = entity_obs.get("attention_mask")
        else:
            entities = entity_obs
            attention_mask = None
        device = next(self.parameters()).device
        entities_t = torch.as_tensor(entities, dtype=torch.float32, device=device)
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

    def encode(self, entity_obs) -> torch.Tensor:
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
        return attended[:, 0, :]

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
        pooled = self.encode(entity_obs)
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
            "action": action,
            "log_prob": dist.log_prob(action).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "value": value,
            "mean": mean,
            "role_mask": role_ids,
        }

    def value(self, critic_state):
        critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=next(self.parameters()).device)
        if critic_t.ndim == 1:
            critic_t = critic_t.unsqueeze(0)
        return self.critic(critic_t).squeeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
