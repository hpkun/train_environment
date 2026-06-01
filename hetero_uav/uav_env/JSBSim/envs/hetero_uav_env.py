"""Heterogeneous MAV-UAV cooperative air combat environment."""

from __future__ import annotations

import numpy as np

from ..core.utils import make_box, make_dict_space
from ..tasks.hetero_combat_task import HeteroCombatTask


class HeteroUAVEnv:
    """Multi-agent environment with a Gymnasium-like API.

    reset returns ``(obs, info)`` and step returns
    ``(obs, rewards, terminated, truncated, info)``, matching the current
    train_environment worker style. ``wrappers.MAPPOEnvWrapper`` exposes a
    stacked array API for MAPPO runners that expect centralized state.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, config: dict, config_path: str | None = None):
        self.config = dict(config)
        self.config_path = str(config_path) if config_path is not None else None
        self.task = HeteroCombatTask(self.config)
        self.controlled_side = self.task.controlled_side
        self.rng = np.random.default_rng(self.config.get("seed", None))
        self._obs: dict | None = None
        self._info: dict | None = None

        self.agent_ids = self.task.controlled_agent_ids_from_config()
        self.num_agents = len(self.agent_ids)
        self.n_agents = self.num_agents
        self.action_shape = int(self.config.get("action_shape", 3))
        self.obs_shape = self.task.observation.obs_dim
        self.state_shape = self.task.observation.state_dim
        self.action_space = make_dict_space({
            aid: make_box(-1.0, 1.0, (self.action_shape,)) for aid in self.agent_ids
        })
        self.observation_space = make_dict_space({
            aid: make_box(-np.inf, np.inf, (self.obs_shape,)) for aid in self.agent_ids
        })

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._obs, self._info = self.task.reset(self.rng)
        self.agent_ids = [a.agent_id for a in self.task.controlled_agents()]
        self.num_agents = len(self.agent_ids)
        self.n_agents = self.num_agents
        return self.get_obs(), self._info

    def step(self, actions):
        self._obs, rewards, terminated, truncated, self._info = self.task.step(actions)
        return (
            self.get_obs(),
            self._filter_controlled(rewards),
            self._filter_controlled(terminated),
            self._filter_controlled(truncated),
            self._info,
        )

    def close(self):
        return None

    def render(self, mode=None):
        if self._info is None:
            return "HeteroUAVEnv(not reset)"
        text = (
            f"step={self._info['episode_step']} "
            f"red_alive={self._info['red_alive']} "
            f"blue_alive={self._info['blue_alive']} "
            f"mav_alive={self._info['mav_alive']}"
        )
        if mode == "text" or mode is None:
            print(text)
        return text

    def get_obs(self):
        if self._obs is None:
            self.reset()
        return self._filter_controlled(self._obs)

    def get_state(self):
        return self.task.get_state()

    def get_avail_actions(self):
        return {aid: np.ones(self.action_shape, dtype=np.float32) for aid in self.agent_ids}

    def sample_actions(self):
        return {aid: self.action_space[aid].sample() for aid in self.agent_ids}

    def _filter_controlled(self, data: dict) -> dict:
        return {aid: data[aid] for aid in self.agent_ids if aid in data}
