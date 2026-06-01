from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class ActionBatch:
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = mlp(obs_dim, hidden_dim, action_dim)
        self.critic = mlp(state_dim, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: torch.Tensor) -> ActionBatch:
        dist = self.distribution(obs)
        actions = dist.sample().clamp(-1.0, 1.0)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.value(state)
        return ActionBatch(actions=actions, log_probs=log_probs, values=values)

    def evaluate_actions(self, obs: torch.Tensor, state: torch.Tensor,
                         actions: torch.Tensor):
        dist = self.distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value(state)
        return log_probs, entropy, values
