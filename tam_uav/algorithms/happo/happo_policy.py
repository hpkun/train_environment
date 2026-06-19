"""Minimal HAPPO-style policy for 3v2 reference validation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


MAV_ROLE_ID = 0
UAV_ROLE_ID = 1


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, out_dim),
    )


class HAPPOReferencePolicy(nn.Module):
    """Separate MAV/UAV actors with one centralized critic.

    This is a HAPPO-style v0 reference policy, not full TAM-HAPPO. It has no
    attention and no recurrent state.
    """

    def __init__(self, actor_obs_dim: int = 96, critic_state_dim: int = 480,
                 action_dim: int = 3):
        super().__init__()
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.mav_actor = _mlp(self.actor_obs_dim, self.action_dim)
        self.uav_actor = _mlp(self.actor_obs_dim, self.action_dim)
        self.critic = _mlp(self.critic_state_dim, 1)
        init_log_std = float(np.log(0.3))
        self.action_log_std_mav = nn.Parameter(torch.full((self.action_dim,), init_log_std))
        self.action_log_std_uav = nn.Parameter(torch.full((self.action_dim,), init_log_std))

    @staticmethod
    def infer_role_ids(actor_obs: torch.Tensor, roles: Iterable[str | int] | torch.Tensor | None = None) -> torch.Tensor:
        if roles is not None:
            if isinstance(roles, torch.Tensor):
                if roles.dtype == torch.bool:
                    return roles.long()
                return roles.long()
            ids = []
            for role in roles:
                if isinstance(role, str):
                    ids.append(MAV_ROLE_ID if role == "mav" else UAV_ROLE_ID)
                else:
                    ids.append(int(role))
            return torch.as_tensor(ids, device=actor_obs.device, dtype=torch.long)

        if actor_obs.ndim == 3:
            flat = actor_obs.reshape(-1, actor_obs.shape[-1])
        else:
            flat = actor_obs
        role_slice = flat[:, 7:11] if flat.shape[-1] >= 11 else torch.zeros((flat.shape[0], 4), device=flat.device)
        inferred = torch.full((flat.shape[0],), UAV_ROLE_ID, device=flat.device, dtype=torch.long)
        inferred[role_slice[:, 0] > role_slice[:, 1:].max(dim=-1).values] = MAV_ROLE_ID
        if actor_obs.ndim == 3:
            return inferred.view(actor_obs.shape[0], actor_obs.shape[1])
        if inferred.numel() > 0 and not torch.any(inferred == MAV_ROLE_ID):
            inferred[0] = MAV_ROLE_ID
        return inferred

    def _means_and_stds(self, actor_obs: torch.Tensor, role_ids: torch.Tensor):
        flat_obs = actor_obs.reshape(-1, actor_obs.shape[-1])
        flat_roles = role_ids.reshape(-1).to(flat_obs.device)
        means = torch.zeros((flat_obs.shape[0], self.action_dim), device=flat_obs.device)
        stds = torch.zeros_like(means)
        mav_mask = flat_roles == MAV_ROLE_ID
        uav_mask = ~mav_mask
        if mav_mask.any():
            means[mav_mask] = torch.clamp(self.mav_actor(flat_obs[mav_mask]), -0.999, 0.999)
            stds[mav_mask] = self.action_log_std_mav.exp().expand_as(means[mav_mask])
        if uav_mask.any():
            means[uav_mask] = torch.clamp(self.uav_actor(flat_obs[uav_mask]), -0.999, 0.999)
            stds[uav_mask] = self.action_log_std_uav.exp().expand_as(means[uav_mask])
        return means.view(*actor_obs.shape[:-1], self.action_dim), stds.view(*actor_obs.shape[:-1], self.action_dim)

    def act(self, actor_obs, roles=None, critic_state=None, deterministic: bool = False):
        actor_obs_t = torch.as_tensor(actor_obs, dtype=torch.float32, device=next(self.parameters()).device)
        if actor_obs_t.ndim == 1:
            actor_obs_t = actor_obs_t.unsqueeze(0)
        role_ids = self.infer_role_ids(actor_obs_t, roles)
        mean, std = self._means_and_stds(actor_obs_t, role_ids)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = None
        if critic_state is not None:
            critic_t = torch.as_tensor(critic_state, dtype=torch.float32, device=actor_obs_t.device)
            if critic_t.ndim == 1:
                critic_t = critic_t.unsqueeze(0)
            value = self.critic(critic_t).squeeze(-1)
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "mean": mean,
            "role_mask": role_ids,
        }

    def evaluate_actions(self, actor_obs, roles, critic_state, actions):
        role_ids = self.infer_role_ids(actor_obs, roles)
        mean, std = self._means_and_stds(actor_obs, role_ids)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(critic_state).squeeze(-1)
        return log_prob, entropy, values, mean, role_ids

    def value(self, critic_state):
        return self.critic(critic_state).squeeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
