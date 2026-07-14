"""Gymnasium-style facade for the isolated TAM paper environment."""

from __future__ import annotations

import numpy as np

from .task import TAMPaperTask


class TAMPaperEnv:
    metadata = {"render_modes": ["text"]}

    def __init__(self, **config):
        if config.get("paper_environment_mode") != "tam_paper_env_v1":
            raise ValueError("jsbsim_tam_paper requires paper_environment_mode=tam_paper_env_v1")
        self.config = dict(config)
        published = dict(self.config["published_parameters"])
        for key in tuple(published):
            if key in self.config:
                published[key] = self.config[key]
        self.config["published_parameters"] = published
        inferred = dict(self.config["inferred_parameters"])
        for key in tuple(inferred):
            if key in self.config:
                inferred[key] = self.config[key]
        self.config["inferred_parameters"] = inferred
        self.task = TAMPaperTask(self.config)
        self.rng = np.random.default_rng(config.get("seed"))
        self.agent_ids = self.task.controlled_agent_ids_from_config()
        self.red_ids = list(self.agent_ids)
        self.blue_ids = [str(a.get("id", f"blue_{i}"))
                         for i, a in enumerate(config["blue_agents"])]
        self.agent_roles = {str(a.get("id", f"red_{i}")): a["role"]
                            for i, a in enumerate(config["red_agents"])}
        self.num_agents = self.n_agents = len(self.agent_ids)
        self.tam_action_distribution = "multidiscrete_categorical"
        self.tam_action_levels = 40
        self.action_interface = "tam_direct_fcs_4d"
        self.paper_environment_mode = "tam_paper_env_v1"
        self._obs = None
        self._make_spaces()

    def _make_spaces(self):
        from gymnasium import spaces
        self.action_space = spaces.Dict({aid: spaces.MultiDiscrete([40, 40, 40, 40])
                                         for aid in self.agent_ids})
        per_agent = {}
        for aid in self.agent_ids:
            side = "red"
            allies, enemies = self.task.observation.shapes_for(side)
            per_agent[aid] = spaces.Dict({
                "ego_state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
                "ally_states": spaces.Box(-np.inf, np.inf, (allies, 5), dtype=np.float32),
                "enemy_states": spaces.Box(-np.inf, np.inf, (enemies, 5), dtype=np.float32),
                "incoming_missile_states": spaces.Box(
                    -np.inf, np.inf, (self.task.observation.max_incoming, 5), dtype=np.float32),
                "ally_mask": spaces.Box(0.0, 1.0, (allies,), dtype=np.float32),
                "enemy_mask": spaces.Box(0.0, 1.0, (enemies,), dtype=np.float32),
                "incoming_missile_mask": spaces.Box(
                    0.0, 1.0, (self.task.observation.max_incoming,), dtype=np.float32),
            })
        self.observation_space = spaces.Dict(per_agent)

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        all_obs, info = self.task.reset(self.rng)
        self._obs = {aid: all_obs[aid] for aid in self.agent_ids}
        return self._obs, info

    def step(self, actions):
        all_obs, rewards, terminated, truncated, info = self.task.step(actions)
        self._obs = {aid: all_obs[aid] for aid in self.agent_ids}
        return (self._obs,
                {aid: rewards[aid] for aid in self.agent_ids},
                {aid: terminated[aid] for aid in self.agent_ids},
                {aid: truncated[aid] for aid in self.agent_ids}, info)

    def map_action(self, indices):
        return self.task.map_action(indices)

    def flatten_observation(self, observation):
        return self.task.observation.flatten(observation)

    def get_flat_obs(self):
        if self._obs is None:
            self.reset()
        return {aid: self.flatten_observation(self._obs[aid]) for aid in self.agent_ids}

    def get_state(self):
        flat = self.get_flat_obs()
        return np.concatenate([flat[aid] for aid in self.agent_ids]).astype(np.float32)

    def get_avail_actions(self):
        result = {}
        by_id = {a.agent_id: a for a in self.task.agents}
        for aid in self.agent_ids:
            mask = np.ones((4, 40), dtype=np.float32)
            if self.task.agents and not by_id[aid].alive:
                mask.fill(0.0)
                mask[:, 20] = 1.0
            result[aid] = mask
        return result

    def sample_actions(self):
        return {aid: self.action_space[aid].sample() for aid in self.agent_ids}

    def close(self):
        for agent in self.task.agents:
            close = getattr(agent, "close", None)
            if close:
                close()

    def render(self, mode=None):
        del mode
        info = self.task.last_info
        text = (f"tam_paper_env_v1 step={info.get('episode_step', 0)} "
                f"red_alive={info.get('red_alive', 0)} blue_alive={info.get('blue_alive', 0)}")
        print(text)
        return text
