"""Rollout buffer for HAPPO reference v0."""
from __future__ import annotations

import numpy as np
import torch


class HAPPORolloutBuffer:
    def __init__(self, max_len: int, num_red: int, actor_dim: int,
                 critic_dim: int, action_dim: int, role_ids,
                 rnn_hidden_size: int = 0, action_dtype=np.float32):
        self.max_len = int(max_len)
        self.num_red = int(num_red)
        self.pos = 0
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.actor_obs = np.zeros((max_len, num_red, actor_dim), dtype=np.float32)
        self.critic_state = np.zeros((max_len, critic_dim), dtype=np.float32)
        self.action_dtype = np.dtype(action_dtype)
        self.actions = np.zeros(
            (max_len, num_red, action_dim), dtype=self.action_dtype
        )
        self.log_probs = np.zeros((max_len, num_red), dtype=np.float32)
        self.rewards = np.zeros((max_len, num_red), dtype=np.float32)
        self.dones = np.zeros((max_len, num_red), dtype=np.float32)
        self.values = np.zeros(max_len, dtype=np.float32)
        self.next_values = np.full(max_len, np.nan, dtype=np.float32)
        self.active_masks = np.zeros((max_len, num_red), dtype=np.float32)
        self.env_ids = np.zeros(max_len, dtype=np.int64)
        self.role_ids = np.asarray(role_ids, dtype=np.int64)
        if self.rnn_hidden_size > 0:
            self.rnn_hidden = np.zeros((max_len, num_red, rnn_hidden_size), dtype=np.float32)

    def store(self, actor_obs, critic_state, actions, log_probs,
              rewards, dones, value, active_masks, next_value=None, env_id=0,
              rnn_hidden=None):
        idx = self.pos
        self.actor_obs[idx] = actor_obs
        self.critic_state[idx] = critic_state
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.values[idx] = float(value)
        if next_value is not None:
            self.next_values[idx] = float(next_value)
        self.active_masks[idx] = active_masks
        self.env_ids[idx] = int(env_id)
        if self.rnn_hidden_size > 0 and rnn_hidden is not None:
            self.rnn_hidden[idx] = np.asarray(rnn_hidden, dtype=np.float32)
        self.pos += 1

    def __len__(self):
        return self.pos

    def get(self, device):
        n = self.pos
        data = {
            "actor_obs": torch.as_tensor(self.actor_obs[:n], device=device),
            "critic_state": torch.as_tensor(self.critic_state[:n], device=device),
            "actions": torch.as_tensor(self.actions[:n], device=device),
            "old_log_probs": torch.as_tensor(self.log_probs[:n], device=device),
            "rewards": torch.as_tensor(self.rewards[:n], device=device),
            "dones": torch.as_tensor(self.dones[:n], device=device),
            "values": torch.as_tensor(self.values[:n], device=device),
            "next_values": torch.as_tensor(self.next_values[:n], device=device),
            "active_masks": torch.as_tensor(self.active_masks[:n], device=device),
            "env_ids": torch.as_tensor(self.env_ids[:n], device=device),
            "role_ids": torch.as_tensor(self.role_ids, device=device),
        }
        if self.rnn_hidden_size > 0:
            data["rnn_hidden"] = torch.as_tensor(self.rnn_hidden[:n], device=device)
        return data
