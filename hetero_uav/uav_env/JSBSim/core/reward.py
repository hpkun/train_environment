"""Reward calculation for the debug heterogeneous combat task."""

from __future__ import annotations

import numpy as np

from .aircraft import AircraftPlatform
from .sensor import SensorSuite
from .utils import heading_to_unit, los_angle, safe_norm


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
        self.mav_survival_reward = float(weights.get("mav_survival_reward", 0.06))
        self.scout_detect_reward = float(weights.get("scout_detect_reward", 0.025))
        self.attack_window_reward = float(weights.get("attack_window_reward", 0.04))
        self.interceptor_pressure_reward = float(weights.get("interceptor_pressure_reward", 0.035))

    def compute(self, agents: list[AircraftPlatform], events: list, done: bool,
                win_flag: str | None, sensor: SensorSuite, attack_range: float,
                max_los_angle: float) -> dict[str, float]:
        rewards = {}
        for agent in agents:
            if not agent.alive:
                rewards[agent.agent_id] = 0.0
            elif agent.aircraft_type.reward_role == "leader_survival":
                rewards[agent.agent_id] = self.mav_survival_reward
            elif agent.aircraft_type.reward_role == "scout":
                rewards[agent.agent_id] = self.survival_reward * 1.5
            else:
                rewards[agent.agent_id] = self.survival_reward

        rewards = self._role_shaping(agents, rewards, sensor, attack_range, max_los_angle)
        by_id = {a.agent_id: a for a in agents}
        for event in events:
            if event.fired and not event.hit:
                rewards[event.shooter_id] -= 0.02
            if event.hit and event.target_id is not None:
                shooter = by_id.get(event.shooter_id)
                role = shooter.aircraft_type.reward_role if shooter is not None else ""
                role_bonus = 1.2 if role in ("attack", "intercept") else 1.0
                rewards[event.shooter_id] += self.kill_reward * role_bonus
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
            if done and not agent.alive and agent.aircraft_type.reward_role == "leader_survival":
                rewards[agent.agent_id] += self.mav_death_penalty
        if done and win_flag in ("red_win", "blue_win"):
            winning_side = "red" if win_flag == "red_win" else "blue"
            for agent in agents:
                if agent.side == winning_side:
                    rewards[agent.agent_id] += self.win_reward
                else:
                    rewards[agent.agent_id] += self.lose_penalty
        return rewards

    def _role_shaping(self, agents: list[AircraftPlatform], rewards: dict[str, float],
                      sensor: SensorSuite, attack_range: float,
                      max_los_angle: float) -> dict[str, float]:
        for agent in agents:
            if not agent.alive:
                continue
            enemies = [a for a in agents if a.side != agent.side and a.alive]
            visible = [e for e in enemies if sensor.can_detect(agent, e)]
            role = agent.aircraft_type.reward_role
            if role == "scout":
                rewards[agent.agent_id] += min(0.08, self.scout_detect_reward * len(visible))
            if role in ("attack", "intercept"):
                for enemy in visible:
                    rel = enemy.position - agent.position
                    distance = safe_norm(rel)
                    if distance <= attack_range:
                        angle = los_angle(heading_to_unit(agent.heading, agent.pitch), rel)
                        if angle <= max_los_angle:
                            rewards[agent.agent_id] += self.attack_window_reward
                            break
            if role == "intercept" and enemies:
                nearest = min(safe_norm(e.position - agent.position) for e in enemies)
                rewards[agent.agent_id] += self.interceptor_pressure_reward * max(
                    0.0, 1.0 - min(nearest, attack_range) / attack_range)
        return rewards
