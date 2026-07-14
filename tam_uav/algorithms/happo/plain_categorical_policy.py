"""Plain non-recurrent MultiDiscrete policy for paper-environment HAPPO smoke tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .happo_policy import MAV_ROLE_ID, UAV_ROLE_ID


def _actor(obs_dim: int, action_dim: int, levels: int):
    return nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128),
                         nn.Tanh(), nn.Linear(128, action_dim * levels))


class PlainCategoricalHAPPOPolicy(nn.Module):
    """No GRU, attention, masking network, or TAM feature module."""

    action_distribution = "multidiscrete_categorical"

    def __init__(self, actor_obs_dim: int, critic_state_dim: int,
                 action_dim: int = 4, action_levels: int = 40):
        super().__init__()
        self.action_dim = action_dim
        self.action_levels = action_levels
        self.mav_actor = _actor(actor_obs_dim, action_dim, action_levels)
        self.uav_actor = _actor(actor_obs_dim, action_dim, action_levels)
        self.critic = nn.Sequential(nn.Linear(critic_state_dim, 128), nn.Tanh(),
                                    nn.Linear(128, 1))

    def _logits(self, obs, roles):
        flat = obs.reshape(-1, obs.shape[-1])
        role_flat = roles.reshape(-1)
        logits = torch.zeros((flat.shape[0], self.action_dim, self.action_levels),
                             device=flat.device)
        mav = role_flat == MAV_ROLE_ID
        if mav.any():
            logits[mav] = self.mav_actor(flat[mav]).reshape(-1, self.action_dim, self.action_levels)
        if (~mav).any():
            logits[~mav] = self.uav_actor(flat[~mav]).reshape(-1, self.action_dim, self.action_levels)
        return logits.reshape(*obs.shape[:-1], self.action_dim, self.action_levels)

    def act(self, actor_obs, roles, critic_state, deterministic=False):
        device = next(self.parameters()).device
        obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        role_ids = torch.as_tensor(roles, dtype=torch.long, device=device)
        logits = self._logits(obs, role_ids)
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)
        return {"action": actions, "log_prob": dist.log_prob(actions).sum(-1),
                "entropy": dist.entropy().sum(-1), "value": self.value(state)}

    def evaluate_actions(self, actor_obs, roles, critic_state, actions, **_kwargs):
        logits = self._logits(actor_obs, roles)
        dist = Categorical(logits=logits)
        return (dist.log_prob(actions.long()).sum(-1), dist.entropy().sum(-1),
                self.value(critic_state), logits.argmax(-1).float(), roles)

    def action_probabilities(self, actor_obs, roles, _rnn_hidden=None):
        return torch.softmax(self._logits(actor_obs, roles), dim=-1), None

    def value(self, critic_state):
        return self.critic(critic_state).squeeze(-1)
