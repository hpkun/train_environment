"""Paper-aligned HAPPO baseline: independent per-agent actors + shared V critic.

ICLR 2022 HAPPO: each agent has its own actor parameters, a global shared
critic estimates team value, and actor updates use the sequential
correction factor M.

Does NOT include: TAM-HAPPO, GRU, attention, entity tokens, masks,
MAV shared-track special design, BRMA-MAPPO, MAPPO.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.Tanh(),
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, out_dim),
    )


class PureHAPPOPolicy(nn.Module):
    """Paper-aligned HAPPO baseline policy.

    Args:
        actor_obs_dim:    per-agent observation dimension (default 96)
        critic_state_dim: centralized critic state dimension (default 480)
        action_dim:       action space dimension (default 3)
        num_agents:       number of agents (e.g. 3 for 3v2)
        init_log_std:     initial log std for all agents
    """

    def __init__(self, actor_obs_dim: int = 96, critic_state_dim: int = 480,
                 action_dim: int = 3, num_agents: int = 3,
                 init_log_std: float = -1.204):
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

    def _check_agent_count(self, N: int):
        if N != self.num_agents:
            raise ValueError(
                f"PureHAPPOPolicy built for {self.num_agents} agents, got {N}. "
                f"Train separate HAPPO baselines for each scale."
            )

    def _distribution(self, obs: torch.Tensor, agent_idx: int):
        mean = self.actors[agent_idx](obs)
        mean = torch.nan_to_num(mean).clamp(-0.999, 0.999)
        std = self.action_log_stds[agent_idx].exp().clamp(min=1e-4)
        while std.dim() < mean.dim():
            std = std.unsqueeze(0)
        return Normal(mean, std), mean

    def act(self, actor_obs, roles=None, critic_state=None,
            deterministic: bool = False, rnn_hidden=None, **kwargs):
        """Sample actions for all agents.

        Args:
            actor_obs: [N, D] or [B, N, D]
            roles:     ignored (pure HAPPO uses per-agent actors)
            rnn_hidden: ignored (pure HAPPO has no recurrence)

        Returns dict with keys: action, log_prob, entropy, value, mean.
        If rnn_hidden is passed, it is returned unchanged.
        """
        device = next(self.parameters()).device
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        if critic_state is not None:
            critic_state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)

        if actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, N, D = actor_obs.shape
        self._check_agent_count(N)

        actions = torch.zeros(B, N, self.action_dim, device=device)
        log_probs = torch.zeros(B, N, device=device)
        entropies = torch.zeros(B, N, device=device)
        means = torch.zeros(B, N, self.action_dim, device=device)

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

    def evaluate_actions(self, actor_obs, critic_state, actions):
        """Evaluate actions for PPO update.

        Args:
            actor_obs:    [T, N, D]
            critic_state: [T, Dc]
            actions:      [T, N, A]

        Returns: (log_prob [T,N], entropy [T,N], values [T], means [T,N,A])
        """
        device = next(self.parameters()).device
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        critic_state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

        T, N = actor_obs.shape[:2]
        self._check_agent_count(N)

        log_probs = torch.zeros(T, N, device=device)
        entropies = torch.zeros(T, N, device=device)
        means_t = torch.zeros(T, N, self.action_dim, device=device)

        for i in range(self.num_agents):
            dist, mean = self._distribution(actor_obs[:, i, :], i)
            log_probs[:, i] = dist.log_prob(actions[:, i, :]).sum(dim=-1)
            entropies[:, i] = dist.entropy().sum(dim=-1)
            means_t[:, i, :] = mean

        values = self.value(critic_state)
        return log_probs, entropies, values, means_t

    def evaluate_agent_actions(self, agent_idx: int,
                                actor_obs_i, actions_i):
        """Evaluate actions for a single agent (sequential update)."""
        device = next(self.parameters()).device
        actor_obs_i = torch.as_tensor(actor_obs_i, dtype=torch.float32, device=device)
        actions_i = torch.as_tensor(actions_i, dtype=torch.float32, device=device)
        dist, mean = self._distribution(actor_obs_i, agent_idx)
        log_prob = dist.log_prob(actions_i).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean

    def value(self, critic_state):
        device = next(self.parameters()).device
        critic_state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)
        if critic_state.dim() == 1:
            critic_state = critic_state.unsqueeze(0)
        return self.critic(critic_state).squeeze(-1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))


class PureHAPPOTanhPolicy(PureHAPPOPolicy):
    """Pure-HAPPO with tanh-squashed Gaussian action accounting.

    This keeps the independent actor / centralized critic structure from
    ``PureHAPPOPolicy`` but uses the same transformed distribution for rollout
    and PPO replay. The historical ``PureHAPPOPolicy`` clamp behavior is left
    unchanged for baseline compatibility.
    """

    tanh_eps = 1e-6
    raw_action_limit = 4.0

    def _squashed_log_prob(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        correction = torch.log(1.0 - squashed.pow(2) + self.tanh_eps)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def _atanh_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0 + self.tanh_eps, 1.0 - self.tanh_eps)
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    def act(self, actor_obs, roles=None, critic_state=None,
            deterministic: bool = False, rnn_hidden=None, **kwargs):
        device = next(self.parameters()).device
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        if critic_state is not None:
            critic_state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)

        if actor_obs.dim() == 2:
            actor_obs = actor_obs.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, N, _D = actor_obs.shape
        self._check_agent_count(N)

        actions = torch.zeros(B, N, self.action_dim, device=device)
        raw_actions = torch.zeros(B, N, self.action_dim, device=device)
        log_probs = torch.zeros(B, N, device=device)
        entropies = torch.zeros(B, N, device=device)
        means = torch.zeros(B, N, self.action_dim, device=device)

        for i in range(self.num_agents):
            dist, mean = self._distribution(actor_obs[:, i, :], i)
            raw = mean if deterministic else dist.rsample()
            raw = raw.clamp(-self.raw_action_limit, self.raw_action_limit)
            action = torch.tanh(raw)
            actions[:, i, :] = action
            raw_actions[:, i, :] = raw
            log_probs[:, i] = self._squashed_log_prob(dist, raw)
            entropies[:, i] = dist.entropy().sum(dim=-1)
            means[:, i, :] = mean

        value = self.value(critic_state) if critic_state is not None else None
        out = {
            "action": actions.squeeze(0) if squeezed else actions,
            "raw_action": raw_actions.squeeze(0) if squeezed else raw_actions,
            "log_prob": log_probs.squeeze(0) if squeezed else log_probs,
            "entropy": entropies.squeeze(0) if squeezed else entropies,
            "value": value,
            "mean": means.squeeze(0) if squeezed else means,
        }
        if rnn_hidden is not None:
            out["rnn_hidden"] = rnn_hidden
        return out

    def evaluate_actions(self, actor_obs, critic_state, actions):
        device = next(self.parameters()).device
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=device)
        critic_state = torch.as_tensor(critic_state, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

        T, N = actor_obs.shape[:2]
        self._check_agent_count(N)

        log_probs = torch.zeros(T, N, device=device)
        entropies = torch.zeros(T, N, device=device)
        means_t = torch.zeros(T, N, self.action_dim, device=device)
        raw_actions = self._atanh_action(actions)

        for i in range(self.num_agents):
            dist, mean = self._distribution(actor_obs[:, i, :], i)
            log_probs[:, i] = self._squashed_log_prob(dist, raw_actions[:, i, :])
            entropies[:, i] = dist.entropy().sum(dim=-1)
            means_t[:, i, :] = mean

        values = self.value(critic_state)
        return log_probs, entropies, values, means_t

    def evaluate_agent_actions(self, agent_idx: int,
                                actor_obs_i, actions_i):
        device = next(self.parameters()).device
        actor_obs_i = torch.as_tensor(actor_obs_i, dtype=torch.float32, device=device)
        actions_i = torch.as_tensor(actions_i, dtype=torch.float32, device=device)
        dist, mean = self._distribution(actor_obs_i, agent_idx)
        raw_action = self._atanh_action(actions_i)
        log_prob = self._squashed_log_prob(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean
