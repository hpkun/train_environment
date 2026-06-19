"""Array-based adapter for MAPPO-style runners."""

from __future__ import annotations

import numpy as np


class MAPPOEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents
        self.n_agents = env.n_agents
        self.obs_shape = env.obs_shape
        self.state_shape = env.state_shape
        self.action_shape = env.action_shape

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._stack_obs(obs), self.env.get_state(), info

    def step(self, actions):
        action_dict = {
            aid: np.asarray(actions[i], dtype=np.float32)
            for i, aid in enumerate(self.env.agent_ids)
        }
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        dones = np.array([
            bool(terminated.get(aid, False) or truncated.get(aid, False))
            for aid in self.env.agent_ids
        ], dtype=np.bool_)
        reward_arr = np.array([rewards.get(aid, 0.0) for aid in self.env.agent_ids],
                              dtype=np.float32)
        return self._stack_obs(obs), self.env.get_state(), reward_arr, dones, info

    def get_avail_actions(self):
        return np.stack([self.env.get_avail_actions()[aid] for aid in self.env.agent_ids])

    def close(self):
        self.env.close()

    def _stack_obs(self, obs):
        return np.stack([obs[aid]["flat"] for aid in self.env.agent_ids]).astype(np.float32)
