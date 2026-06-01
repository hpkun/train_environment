"""Reward calculation for the debug heterogeneous combat task."""

from __future__ import annotations

import numpy as np

from .aircraft import AircraftPlatform


class RewardBuilder:
    def __init__(self, config: dict):
        weights = config.get("reward_weights", {})
        self.kill_reward = float(weights.get("kill_reward", 20.0))
        self.death_penalty = float(weights.get("death_penalty", -20.0))
        self.mav_death_penalty = float(weights.get("mav_death_penalty", -50.0))
        self.boundary_penalty = float(weights.get("boundary_penalty", -5.0))
        self.crash_penalty = float(weights.get("crash_penalty", -10.0))
        self.win_reward = float(weights.get("win_reward", 30.0))
        self.lose_penalty = float(weights.get("lose_penalty", -30.0))
        self.survival_reward = float(weights.get("survival_reward", 0.02))

    def compute(self, agents: list[AircraftPlatform], events: list, done: bool,
                win_flag: str | None) -> dict[str, float]:
        rewards = {a.agent_id: (self.survival_reward if a.alive else 0.0) for a in agents}
        by_id = {a.agent_id: a for a in agents}
        for event in events:
            if event.hit:
                rewards[event.shooter_id] += self.kill_reward
                target = by_id.get(event.target_id)
                if target is not None:
                    rewards[event.target_id] += self.death_penalty
                    if target.aircraft_type.reward_role == "leader_survival":
                        rewards[event.target_id] += self.mav_death_penalty
        for agent in agents:
            if agent.out_of_boundary:
                rewards[agent.agent_id] += self.boundary_penalty
            if agent.crashed:
                rewards[agent.agent_id] += self.crash_penalty
        if done and win_flag in ("red", "blue"):
            for agent in agents:
                if agent.side == win_flag:
                    rewards[agent.agent_id] += self.win_reward
                else:
                    rewards[agent.agent_id] += self.lose_penalty
        return rewards
