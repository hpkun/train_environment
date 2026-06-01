from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutStorage:
    def __init__(self, rollout_steps: int, num_agents: int, obs_dim: int,
                 state_dim: int, action_dim: int, gamma: float, gae_lambda: float,
                 device: torch.device):
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs = np.zeros((rollout_steps, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((rollout_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_agents), dtype=np.float32)
        self.step = 0

    def insert(self, obs, state, actions, log_probs, values, rewards, dones) -> None:
        i = self.step
        self.obs[i] = obs
        self.states[i] = state
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = values
        self.rewards[i] = rewards
        self.dones[i] = dones.astype(np.float32)
        self.step += 1

    def compute_batch(self, next_value: np.ndarray) -> RolloutBatch:
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros((self.num_agents,), dtype=np.float32)
        next_values = next_value.astype(np.float32)
        for t in reversed(range(self.rollout_steps)):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * nonterminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_values = self.values[t]
        returns = advantages + self.values
        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states_repeated = np.repeat(self.states[:, None, :], self.num_agents, axis=1)
        return RolloutBatch(
            obs=torch.as_tensor(self.obs.reshape(-1, self.obs.shape[-1]), device=self.device),
            states=torch.as_tensor(states_repeated.reshape(-1, self.states.shape[-1]),
                                   device=self.device),
            actions=torch.as_tensor(self.actions.reshape(-1, self.actions.shape[-1]),
                                    device=self.device),
            old_log_probs=torch.as_tensor(self.log_probs.reshape(-1), device=self.device),
            returns=torch.as_tensor(returns.reshape(-1), device=self.device),
            advantages=torch.as_tensor(adv, device=self.device),
        )
