"""Rollout buffer for HAPPO reference v0."""
from __future__ import annotations

import numpy as np
import torch


class HAPPORolloutBuffer:
    def __init__(self, max_len: int, num_red: int, actor_dim: int,
                 critic_dim: int, action_dim: int, role_ids,
                 rnn_hidden_size: int = 0, action_dtype=np.float32,
                 num_envs: int = 1):
        self.max_len = int(max_len)
        self.num_red = int(num_red)
        self.pos = 0
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.num_envs = int(num_envs)
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
        self.agent_alive_masks = np.zeros((max_len, num_red), dtype=np.float32)
        self.death_transition_masks = np.zeros((max_len, num_red), dtype=np.float32)
        self.episode_start_masks = np.zeros((max_len, num_red), dtype=np.float32)
        self.env_step_index = np.zeros(max_len, dtype=np.int64)
        self.env_ids = np.zeros(max_len, dtype=np.int64)
        self._next_env_step = np.zeros(self.num_envs, dtype=np.int64)
        self.role_ids = np.asarray(role_ids, dtype=np.int64)
        if self.rnn_hidden_size > 0:
            self.rnn_hidden = np.zeros((max_len, num_red, rnn_hidden_size), dtype=np.float32)
            self.rnn_hidden_initial = np.zeros(
                (self.num_envs, num_red, rnn_hidden_size), dtype=np.float32
            )

    def set_rnn_hidden_initial(self, env_id: int, hidden) -> None:
        if self.rnn_hidden_size <= 0:
            raise ValueError("buffer has no recurrent hidden state")
        self.rnn_hidden_initial[int(env_id)] = np.asarray(hidden, dtype=np.float32)

    def store(self, actor_obs, critic_state, actions, log_probs,
              rewards, dones, value, active_masks, next_value=None, env_id=0,
              rnn_hidden=None, episode_start_masks=None, env_step_index=None,
              death_transition_masks=None):
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
        self.agent_alive_masks[idx] = active_masks
        if death_transition_masks is not None:
            self.death_transition_masks[idx] = np.asarray(
                death_transition_masks, dtype=np.float32
            )
        if episode_start_masks is not None:
            self.episode_start_masks[idx] = np.asarray(
                episode_start_masks, dtype=np.float32
            )
        self.env_ids[idx] = int(env_id)
        env_id = int(env_id)
        if env_step_index is None:
            env_step_index = self._next_env_step[env_id]
        self.env_step_index[idx] = int(env_step_index)
        self._next_env_step[env_id] = int(env_step_index) + 1
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
            "agent_alive_masks": torch.as_tensor(self.agent_alive_masks[:n], device=device),
            "death_transition_masks": torch.as_tensor(
                self.death_transition_masks[:n], device=device
            ),
            "episode_start_masks": torch.as_tensor(self.episode_start_masks[:n], device=device),
            "env_step_index": torch.as_tensor(self.env_step_index[:n], device=device),
            "env_ids": torch.as_tensor(self.env_ids[:n], device=device),
            "role_ids": torch.as_tensor(self.role_ids, device=device),
        }
        if self.rnn_hidden_size > 0:
            data["rnn_hidden"] = torch.as_tensor(self.rnn_hidden[:n], device=device)
            data["rnn_hidden_initial"] = torch.as_tensor(
                self.rnn_hidden_initial, device=device
            )
        return data

    def get_sequences(self, device):
        data = self.get(device)
        sequences = []
        for env_id in torch.unique(data["env_ids"], sorted=True):
            indices = torch.nonzero(data["env_ids"] == env_id, as_tuple=False).flatten()
            order = torch.argsort(data["env_step_index"][indices], stable=True)
            indices = indices[order]
            sequence = {
                key: value[indices]
                for key, value in data.items()
                if key not in {"role_ids", "rnn_hidden_initial"}
                and value.ndim > 0 and value.shape[0] == len(self)
            }
            sequence["role_ids"] = data["role_ids"]
            sequence["buffer_indices"] = indices
            if self.rnn_hidden_size > 0:
                sequence["rnn_hidden_initial"] = data["rnn_hidden_initial"][int(env_id)]
            sequences.append(sequence)
        return sequences
