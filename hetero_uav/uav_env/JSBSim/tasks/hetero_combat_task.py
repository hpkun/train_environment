"""Task logic for heterogeneous MAV-UAV cooperative combat."""

from __future__ import annotations

import numpy as np

from ..core.aircraft import AircraftPlatform
from ..core.info import InfoBuilder
from ..core.missile import MissileManager
from ..core.observation import ObservationBuilder
from ..core.opponent_policy import make_opponent_policy
from ..core.reward import RewardBuilder
from ..core.scenario import ScenarioBuilder
from ..core.sensor import SensorSuite
from ..core.termination import TerminationChecker


class HeteroCombatTask:
    def __init__(self, config: dict):
        self.config = config
        self.decision_dt = 1.0 / float(config.get("decision_frequency", 5.0))
        self.speed_range = tuple(config.get("speed_range", [102.0, 408.0]))
        self.map_boundary = config.get("map_boundary", {})
        self.controlled_side = str(config.get("controlled_side", "red"))
        self.scenario = ScenarioBuilder(config)
        self.sensor = SensorSuite(config)
        self.missiles = MissileManager(config)
        self.observation = ObservationBuilder(config, self.sensor)
        self.reward = RewardBuilder(config)
        self.termination = TerminationChecker(config)
        self.info = InfoBuilder()
        self.opponent_policy = make_opponent_policy(str(config.get("opponent_policy", "rule_nearest")))
        self.agents: list[AircraftPlatform] = []
        self.step_count = 0
        self.episode_return: dict[str, float] = {}
        self.last_win_flag: str | None = None

    def reset(self, rng: np.random.Generator) -> tuple[dict, dict]:
        self.agents = self.scenario.build(rng)
        self.step_count = 0
        self.last_win_flag = None
        self.episode_return = {a.agent_id: 0.0 for a in self.agents}
        obs = self.observation.build_obs(self.agents)
        info = self.info.build(self.agents, self.step_count, self.episode_return, None)
        return obs, info

    def step(self, actions) -> tuple[dict, dict[str, float], dict[str, bool], dict[str, bool], dict]:
        self.step_count += 1
        action_map = self._normalize_actions(actions)
        action_map.update(self._opponent_actions(action_map))
        for agent in self.agents:
            agent.step(action_map.get(agent.agent_id, np.zeros(3, dtype=np.float32)),
                       self.decision_dt, self.speed_range)
        self._apply_boundary_and_crash()
        events = self._resolve_missiles()
        terminated_env, truncated_env, win_flag = self.termination.check(self.agents, self.step_count)
        done_env = terminated_env or truncated_env
        rewards = self.reward.compute(self.agents, events, done_env, win_flag)
        for aid, value in rewards.items():
            self.episode_return[aid] = self.episode_return.get(aid, 0.0) + float(value)
        self.last_win_flag = win_flag
        obs = self.observation.build_obs(self.agents)
        terminated = {a.agent_id: bool(terminated_env or not a.alive) for a in self.agents}
        truncated = {a.agent_id: bool(truncated_env) for a in self.agents}
        info = self.info.build(self.agents, self.step_count, self.episode_return, win_flag)
        return obs, rewards, terminated, truncated, info

    def get_state(self) -> np.ndarray:
        return self.observation.build_state(self.agents)

    def _normalize_actions(self, actions) -> dict[str, np.ndarray]:
        if isinstance(actions, dict):
            return {str(k): np.asarray(v, dtype=np.float32) for k, v in actions.items()}
        arr = np.asarray(actions, dtype=np.float32)
        controlled = self.controlled_agents()
        return {agent.agent_id: arr[i] for i, agent in enumerate(controlled[: len(arr)])}

    def controlled_agents(self) -> list[AircraftPlatform]:
        if self.controlled_side == "all":
            return list(self.agents)
        return [a for a in self.agents if a.side == self.controlled_side]

    def controlled_agent_ids_from_config(self) -> list[str]:
        if self.controlled_side == "all":
            sides = ("red", "blue")
        else:
            sides = (self.controlled_side,)
        ids = []
        for side in sides:
            for idx, entry in enumerate(self.config.get(f"{side}_agents", [])):
                ids.append(str(entry.get("id", f"{side}_{idx}")))
        return ids

    def _opponent_actions(self, action_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.controlled_side == "all":
            return {}
        result = {}
        for agent in self.agents:
            if agent.side == self.controlled_side or agent.agent_id in action_map:
                continue
            result[agent.agent_id] = self.opponent_policy.act(agent, self.agents)
        return result

    def _resolve_missiles(self):
        events = []
        for shooter in self.agents:
            if not shooter.alive:
                continue
            enemies = [a for a in self.agents if a.side != shooter.side and a.alive]
            if not enemies:
                continue
            target = min(enemies, key=lambda e: np.linalg.norm(e.position - shooter.position))
            event = self.missiles.try_launch(shooter, target, self.sensor)
            if event is not None:
                events.append(event)
        return events

    def _apply_boundary_and_crash(self) -> None:
        xlim = self.map_boundary.get("x", [-50000.0, 50000.0])
        ylim = self.map_boundary.get("y", [-50000.0, 50000.0])
        zlim = self.map_boundary.get("altitude", [750.0, 10000.0])
        for agent in self.agents:
            if not agent.alive:
                continue
            x, y, z = agent.position
            if z < zlim[0]:
                agent.kill("crash")
            elif z > zlim[1] or x < xlim[0] or x > xlim[1] or y < ylim[0] or y > ylim[1]:
                agent.kill("boundary")
