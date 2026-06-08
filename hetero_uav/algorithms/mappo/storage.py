"""Single-environment rollout buffer for MAPPO baseline."""
from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout trajectories for one env.

    Fields per timestep t:
        actor_obs[t]   : (num_red, 140)
        critic_state[t]: (700,)
        actions[t]     : (num_red, 3)
        log_probs[t]   : (num_red,)
        rewards[t]     : (num_red,)
        dones[t]       : (num_red,)
        value          : scalar (team value)
        red_valid[t]   : (num_red,)
    """

    def __init__(self, max_len: int, num_red: int,
                 actor_dim: int, critic_dim: int, action_dim: int):
        self.max_len = max_len
        self.num_red = num_red
        self.pos = 0

        self.actor_obs     = np.zeros((max_len, num_red, actor_dim), dtype=np.float32)
        self.critic_state  = np.zeros((max_len, critic_dim), dtype=np.float32)
        self.actions       = np.zeros((max_len, num_red, action_dim), dtype=np.float32)
        self.log_probs     = np.zeros((max_len, num_red), dtype=np.float32)
        self.rewards       = np.zeros((max_len, num_red), dtype=np.float32)
        self.dones         = np.zeros((max_len, num_red), dtype=np.float32)
        self.values        = np.zeros(max_len, dtype=np.float32)
        self.red_valid     = np.zeros((max_len, num_red), dtype=np.float32)
        self.full = False

    def store(self, actor_obs, critic_state, actions, log_probs,
              rewards, dones, value, red_valid):
        idx = self.pos
        self.actor_obs[idx]    = actor_obs
        self.critic_state[idx] = critic_state
        self.actions[idx]      = actions
        self.log_probs[idx]    = log_probs
        self.rewards[idx]      = rewards
        self.dones[idx]        = dones
        self.values[idx]       = value
        self.red_valid[idx]    = red_valid
        self.pos += 1
        if self.pos >= self.max_len:
            self.full = True

    def __len__(self):
        return self.pos

    def get(self, device):
        """Return tensors for PPO update."""
        n = self.pos
        return (
            torch.as_tensor(self.actor_obs[:n], device=device),
            torch.as_tensor(self.critic_state[:n], device=device),
            torch.as_tensor(self.actions[:n], device=device),
            torch.as_tensor(self.log_probs[:n], device=device),
            torch.as_tensor(self.rewards[:n], device=device),
            torch.as_tensor(self.dones[:n], device=device),
            torch.as_tensor(self.values[:n], device=device),
            torch.as_tensor(self.red_valid[:n], device=device),
        )
