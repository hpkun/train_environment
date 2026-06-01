"""Observation and global-state builders."""

from __future__ import annotations

import numpy as np

from .aircraft import AircraftPlatform
from .sensor import SensorSuite
from .utils import heading_to_unit, los_angle, safe_norm

EGO_DIM = 12
ENTITY_DIM = 12
STATE_ENTITY_DIM = 10


class ObservationBuilder:
    def __init__(self, config: dict, sensor: SensorSuite):
        self.config = config
        self.sensor = sensor
        self.max_red = int(config.get("max_red_agents", max(1, len(config.get("red_agents", [])))))
        self.max_blue = int(config.get("max_blue_agents", max(1, len(config.get("blue_agents", [])))))
        max_contacts = max(self.max_red - 1 + self.max_blue,
                           self.max_blue - 1 + self.max_red)
        self.obs_dim = EGO_DIM + max_contacts * ENTITY_DIM
        self.state_dim = (self.max_red + self.max_blue) * STATE_ENTITY_DIM

    def build_obs(self, agents: list[AircraftPlatform]) -> dict[str, dict[str, np.ndarray]]:
        by_side = {
            "red": [a for a in agents if a.side == "red"],
            "blue": [a for a in agents if a.side == "blue"],
        }
        obs = {}
        for agent in agents:
            allies = [a for a in by_side[agent.side] if a.agent_id != agent.agent_id]
            enemies = by_side["blue" if agent.side == "red" else "red"]
            max_allies = self.max_red - 1 if agent.side == "red" else self.max_blue - 1
            max_enemies = self.max_blue if agent.side == "red" else self.max_red
            ally_states = self._padded_entities(agent, allies, max_allies, side_id=0)
            enemy_states = self._padded_entities(agent, enemies, max_enemies, side_id=1)
            ego_state = self._ego(agent)
            flat_contacts = np.concatenate([ally_states.reshape(-1), enemy_states.reshape(-1)])
            flat = np.zeros(self.obs_dim, dtype=np.float32)
            raw_flat = np.concatenate([ego_state, flat_contacts])
            flat[: raw_flat.size] = raw_flat
            death_mask = np.array([1.0 if a.alive else 0.0 for a in agents], dtype=np.float32)
            obs[agent.agent_id] = {
                "flat": flat.astype(np.float32),
                "ego_state": ego_state.astype(np.float32),
                "ally_states": ally_states.astype(np.float32),
                "enemy_states": enemy_states.astype(np.float32),
                "death_mask": death_mask,
                "missile_warning": np.array([self.sensor.missile_warning(agent)], dtype=np.float32),
                "altitude": np.array([agent.position[2] if agent.alive else 0.0], dtype=np.float32),
                "velocity": agent.velocity.astype(np.float32) if agent.alive else np.zeros(3, dtype=np.float32),
                "type_id": np.array([agent.type_id], dtype=np.float32),
            }
        return obs

    def build_state(self, agents: list[AircraftPlatform]) -> np.ndarray:
        ordered = [a for a in agents if a.side == "red"] + [a for a in agents if a.side == "blue"]
        max_total = self.max_red + self.max_blue
        rows = np.zeros((max_total, STATE_ENTITY_DIM), dtype=np.float32)
        for i, agent in enumerate(ordered[:max_total]):
            rows[i] = np.array([
                agent.position[0] / 50000.0,
                agent.position[1] / 50000.0,
                agent.position[2] / 10000.0,
                agent.velocity[0] / 500.0,
                agent.velocity[1] / 500.0,
                agent.velocity[2] / 500.0,
                agent.heading / np.pi,
                1.0 if agent.alive else 0.0,
                agent.missile_left / 4.0,
                float(agent.type_id),
            ], dtype=np.float32)
        return rows.reshape(-1)

    def _ego(self, agent: AircraftPlatform) -> np.ndarray:
        if not agent.alive:
            return np.zeros(EGO_DIM, dtype=np.float32)
        return np.array([
            agent.position[0] / 50000.0,
            agent.position[1] / 50000.0,
            agent.position[2] / 10000.0,
            agent.velocity[0] / 500.0,
            agent.velocity[1] / 500.0,
            agent.velocity[2] / 500.0,
            agent.heading / np.pi,
            agent.pitch / (np.pi / 2.0),
            agent.roll / np.pi,
            1.0,
            agent.missile_left / 4.0,
            float(agent.type_id),
        ], dtype=np.float32)

    def _padded_entities(self, ego: AircraftPlatform, entities: list[AircraftPlatform],
                         count: int, side_id: int) -> np.ndarray:
        rows = np.zeros((count, ENTITY_DIM), dtype=np.float32)
        for i, target in enumerate(entities[:count]):
            rows[i] = self._relative_entity(ego, target, side_id)
        return rows

    def _relative_entity(self, ego: AircraftPlatform, target: AircraftPlatform,
                         side_id: int) -> np.ndarray:
        if not ego.alive or not target.alive:
            return np.zeros(ENTITY_DIM, dtype=np.float32)
        observable = target.side == ego.side or self.sensor.can_detect(ego, target)
        if not observable:
            return np.zeros(ENTITY_DIM, dtype=np.float32)
        rel_pos = target.position - ego.position
        rel_vel = target.velocity - ego.velocity
        distance = safe_norm(rel_pos)
        rel_alt = float(target.position[2] - ego.position[2])
        forward = heading_to_unit(ego.heading, ego.pitch)
        angle = los_angle(forward, rel_pos)
        return np.array([
            rel_pos[0] / 50000.0,
            rel_pos[1] / 50000.0,
            rel_pos[2] / 10000.0,
            distance / 100000.0,
            rel_alt / 10000.0,
            rel_vel[0] / 500.0,
            rel_vel[1] / 500.0,
            rel_vel[2] / 500.0,
            angle / np.pi,
            float(side_id),
            float(target.type_id),
            float(target.missile_left),
        ], dtype=np.float32)
