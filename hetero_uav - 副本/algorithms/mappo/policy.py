"""Plain shared-policy MAPPO actor-critic for heterogeneous compositions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPPOActorCritic(nn.Module):
    """Shared-actor centralized-critic for MAPPO baseline.

    Actor:  MLP [140, 256, 128] -> Gaussian(mean(3,), learnable log_std(3,))
    Critic: MLP [700, 256, 128, 1]
    """

    def __init__(self, actor_obs_dim: int = 140, critic_state_dim: int = 700,
                 action_dim: int = 3):
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.critic_state_dim = critic_state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(actor_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.action_log_std = nn.Parameter(
            torch.full((action_dim,), -1.204))  # ln(0.3)

        self.critic = nn.Sequential(
            nn.Linear(critic_state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, actor_obs, critic_state, deterministic: bool = False):
        """Return (action_dist, value, action, log_prob, entropy)."""
        mean = self.actor(actor_obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        mean = mean.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, sigma)

        if deterministic:
            action = mean.clamp(-1.0, 1.0)
        else:
            action = dist.sample().clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().mean(dim=-1)

        value = self.critic(critic_state).squeeze(-1)

        return dist, value, action, log_prob, entropy

    def evaluate_actions(self, actor_obs, critic_state, actions):
        """Evaluate given actions (for PPO update)."""
        mean = self.actor(actor_obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        mean = mean.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, sigma)

        new_log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean(dim=-1)

        value = self.critic(critic_state).squeeze(-1)

        return new_log_prob, entropy, value


class RoleConditionedMAPPOActorCritic(nn.Module):
    """Role-conditioned actor: shared encoder + MAV/UAV role-specific heads.

    Actor: shared_encoder (96→256→Tanh) → role heads (256→128→Tanh→3)
    Critic: same centralized MLP as MAPPOActorCritic (480→256→128→1)
    """

    def __init__(self, actor_obs_dim: int = 96, critic_state_dim: int = 480,
                 action_dim: int = 3,
                 role_start: int = 7, role_dim: int = 4,
                 mav_role_index: int = 0, obs_layout: str = "v2"):
        super().__init__()
        if obs_layout not in ("v2",):
            raise ValueError(f"obs_layout must be 'v2', got {obs_layout!r}")
        if actor_obs_dim < role_start + role_dim:
            raise ValueError(f"actor_obs_dim={actor_obs_dim} < role_start+role_dim={role_start+role_dim}")
        if not (0 <= mav_role_index < role_dim):
            raise ValueError(f"mav_role_index={mav_role_index} out of [0, {role_dim})")
        self.actor_obs_dim = actor_obs_dim
        self.critic_state_dim = critic_state_dim
        self.action_dim = action_dim
        self.role_start = role_start
        self.role_dim = role_dim
        self.mav_role_index = mav_role_index
        self.obs_layout = obs_layout

        self.shared_encoder = nn.Sequential(
            nn.Linear(actor_obs_dim, 256),
            nn.Tanh(),
        )
        self.mav_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.uav_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.action_log_std = nn.Parameter(
            torch.full((action_dim,), -1.204))

        # Compatibility: trainer expects model.actor.parameters()
        self.actor = nn.ModuleList([self.shared_encoder, self.mav_head, self.uav_head])

        self.critic = nn.Sequential(
            nn.Linear(critic_state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def _role_conditioned_mean(self, actor_obs):
        encoded = self.shared_encoder(actor_obs)
        role = actor_obs[:, self.role_start : self.role_start + self.role_dim]
        mav_mask = role[:, self.mav_role_index] > 0.5
        mean = self.uav_head(encoded)  # default
        if mav_mask.any():
            mean = mean.clone()
            mean[mav_mask] = self.mav_head(encoded[mav_mask])
        return mean

    def forward(self, actor_obs, critic_state, deterministic: bool = False):
        mean = self._role_conditioned_mean(actor_obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        mean = mean.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, sigma)

        if deterministic:
            action = mean.clamp(-1.0, 1.0)
        else:
            action = dist.sample().clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean(dim=-1)
        value = self.critic(critic_state).squeeze(-1)

        return dist, value, action, log_prob, entropy

    def evaluate_actions(self, actor_obs, critic_state, actions):
        mean = self._role_conditioned_mean(actor_obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        mean = mean.clamp(-0.999, 0.999)
        sigma = torch.exp(self.action_log_std).clamp(min=1e-4)
        sigma = sigma.unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, sigma)

        new_log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean(dim=-1)
        value = self.critic(critic_state).squeeze(-1)

        return new_log_prob, entropy, value
