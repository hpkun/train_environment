"""Full HAPPO baseline: independent per-agent actors + shared V-value critic.

Aligns with ICLR 2022 HAPPO paper: each agent has its own actor parameters,
a global shared critic estimates team value, and actor updates use the
sequential correction factor M.

This is NOT TAM-HAPPO (no GRU, no attention, no entity tokens, no masks)
and NOT the simplified role-wise HAPPOReferencePolicy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.Tanh(),
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, out_dim),
    )


class FullHAPPOPolicy(nn.Module):
    """Per-agent independent actors + shared V-value critic.

    Args:
        actor_obs_dim:    per-agent observation dimension
        critic_state_dim: centralized critic state dimension
        action_dim:       action space dimension (default 3)
        num_agents:       number of agents (e.g. 3 for 3v2)
        hidden_dim:       MLP hidden size (unused; _mlp uses fixed 256->128)
        init_log_std:     initial log std for all agents
    """

    def __init__(self, actor_obs_dim: int = 96, critic_state_dim: int = 480,
                 action_dim: int = 3, num_agents: int = 3,
                 hidden_dim: int = 128, init_log_std: float = -1.204):
        super().__init__()
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)

        initial = float(init_log_std)

        self.actors = nn.ModuleList([
            _mlp(self.actor_obs_dim, self.action_dim)
            for _ in range(self.num_agents)
        ])
        self.action_log_stds = nn.ParameterList([
            nn.Parameter(torch.full((self.action_dim,), initial))
            for _ in range(self.num_agents)
        ])
        self.critic = _mlp(self.critic_state_dim, 1)

    def _distribution(self, obs: torch.Tensor, agent_idx: int):
        """Return Normal distribution for one agent."""
        mean = self.actors[agent_idx](obs)  # [..., action_dim]
        mean = torch.nan_to_num(mean).clamp(-0.999, 0.999)
        std = self.action_log_stds[agent_idx].exp().clamp(min=1e-4)
        while std.dim() < mean.dim():
            std = std.unsqueeze(0)
        return Normal(mean, std), mean

    def act(self, actor_obs: torch.Tensor, critic_state=None,
            deterministic: bool = False, rnn_hidden=None):
        """Sample actions for all agents.

        Args:
            actor_obs: [N, actor_dim] or [B, N, actor_dim]
            critic_state: [critic_dim] or [B, critic_dim]

        Returns dict with keys: action, log_prob, entropy, value, mean
        """
        if actor_obs.dim() == 2:
            # [N, D] -> add batch dim
            actor_obs = actor_obs.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, N, D = actor_obs.shape
        actions = torch.zeros(B, N, self.action_dim, device=actor_obs.device)
        log_probs = torch.zeros(B, N, device=actor_obs.device)
        entropies = torch.zeros(B, N, device=actor_obs.device)
        means = torch.zeros(B, N, self.action_dim, device=actor_obs.device)

        for i in range(self.num_agents):
            dist, mean = self._distribution(actor_obs[:, i, :], i)
            if deterministic:
                a = mean.clamp(-1.0, 1.0)
            else:
                a = dist.rsample().clamp(-1.0, 1.0)
            actions[:, i, :] = a
            log_probs[:, i] = dist.log_prob(a).sum(dim=-1)
            entropies[:, i] = dist.entropy().sum(dim=-1)
            means[:, i, :] = mean

        value = self.value(critic_state) if critic_state is not None else None

        out = {
            "action": actions.squeeze(0) if squeezed else actions,
            "log_prob": log_probs.squeeze(0) if squeezed else log_probs,
            "entropy": entropies.squeeze(0) if squeezed else entropies,
            "value": value,
            "mean": means.squeeze(0) if squeezed else means,
        }
        if rnn_hidden is not None:
            out["rnn_hidden"] = rnn_hidden
        return out

    def evaluate_actions(self, actor_obs: torch.Tensor,
                         critic_state: torch.Tensor,
                         actions: torch.Tensor):
        """Evaluate actions for PPO update.

        Args:
            actor_obs:    [T, N, D]
            critic_state: [T, Dc]
            actions:      [T, N, action_dim]

        Returns: (log_prob [T,N], entropy [T,N], values [T], means [T,N,A])
        """
        T, N = actor_obs.shape[:2]
        log_probs = torch.zeros(T, N, device=actor_obs.device)
        entropies = torch.zeros(T, N, device=actor_obs.device)
        means_t = torch.zeros(T, N, self.action_dim, device=actor_obs.device)

        for i in range(self.num_agents):
            dist, mean = self._distribution(actor_obs[:, i, :], i)
            log_probs[:, i] = dist.log_prob(actions[:, i, :]).sum(dim=-1)
            entropies[:, i] = dist.entropy().sum(dim=-1)
            means_t[:, i, :] = mean

        values = self.value(critic_state)
        return log_probs, entropies, values, means_t

    def evaluate_agent_actions(self, agent_idx: int,
                                actor_obs_i: torch.Tensor,
                                actions_i: torch.Tensor):
        """Evaluate actions for a single agent (used in sequential update).

        Args:
            agent_idx:   which agent
            actor_obs_i: [T, D]
            actions_i:   [T, action_dim]

        Returns: (log_prob_i [T], entropy_i [T], mean_i [T, A])
        """
        dist, mean = self._distribution(actor_obs_i, agent_idx)
        log_prob = dist.log_prob(actions_i).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean

    def value(self, critic_state: torch.Tensor) -> torch.Tensor:
        if critic_state.dim() == 1:
            critic_state = critic_state.unsqueeze(0)
        return self.critic(critic_state).squeeze(-1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
