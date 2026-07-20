"""Unique 7D/5D paper observation with fixed slots and masks."""

from __future__ import annotations

import numpy as np

from .situation import assess_pair
from .protocol import environment_values


OBS_KEYS = ("ego_state", "ally_states", "enemy_states", "incoming_missile_states",
            "ally_mask", "enemy_mask", "incoming_missile_mask")


class PaperObservation:
    def __init__(self, published: dict, unpublished: dict):
        self.published = published
        self.unpublished = unpublished
        self.derived = environment_values(unpublished)
        self.max_red = self.derived["max_red_agents"]
        self.max_blue = self.derived["max_blue_agents"]
        self.max_incoming = int(self.derived["max_incoming_missiles"])
        self.position_norm_m = float(self.derived["position_norm_m"])
        self.altitude_norm_m = float(self.derived["altitude_norm_m"])

    def shapes_for(self, side: str):
        allies = (self.max_red if side == "red" else self.max_blue) - 1
        enemies = self.max_blue if side == "red" else self.max_red
        return allies, enemies

    def build(self, agents, missiles, selected_targets=None) -> dict[str, dict[str, np.ndarray]]:
        by_side = {side: sorted([a for a in agents if a.side == side], key=lambda a: a.agent_id)
                   for side in ("red", "blue")}
        result = {}
        for ego in agents:
            allies = [a for a in by_side[ego.side] if a.agent_id != ego.agent_id]
            enemies = by_side["blue" if ego.side == "red" else "red"]
            selected = None if selected_targets is None else selected_targets.get(ego.agent_id)
            enemies = sorted(enemies, key=lambda item: (
                item.agent_id != selected, item.agent_id))
            max_allies, max_enemies = self.shapes_for(ego.side)
            ally_states = np.zeros((max_allies, 5), dtype=np.float32)
            enemy_states = np.zeros((max_enemies, 5), dtype=np.float32)
            ally_mask = np.zeros(max_allies, dtype=np.float32)
            enemy_mask = np.zeros(max_enemies, dtype=np.float32)
            if ego.alive:
                for idx, target in enumerate(allies[:max_allies]):
                    if target.alive:
                        ally_states[idx] = self._relative(ego, target.position, target.velocity)
                        ally_mask[idx] = 1.0
                for idx, target in enumerate(enemies[:max_enemies]):
                    if target.alive:
                        enemy_states[idx] = self._relative(ego, target.position, target.velocity)
                        enemy_mask[idx] = 1.0
            incoming = [m for m in missiles if m.alive and m.target_id == ego.agent_id]
            incoming.sort(key=lambda m: (np.linalg.norm(m.position - ego.position)
                                         / max(m.speed_mps, 1.0), m.missile_id))
            missile_states = np.zeros((self.max_incoming, 5), dtype=np.float32)
            missile_mask = np.zeros(self.max_incoming, dtype=np.float32)
            if ego.alive:
                for idx, missile in enumerate(incoming[:self.max_incoming]):
                    missile_states[idx] = self._relative(ego, missile.position, missile.velocity)
                    missile_mask[idx] = 1.0
            item = {
                "ego_state": self._ego(ego),
                "ally_states": ally_states,
                "enemy_states": enemy_states,
                "incoming_missile_states": missile_states,
                "ally_mask": ally_mask,
                "enemy_mask": enemy_mask,
                "incoming_missile_mask": missile_mask,
            }
            if not all(np.isfinite(value).all() for value in item.values()):
                raise FloatingPointError(f"non-finite paper observation for {ego.agent_id}")
            result[ego.agent_id] = item
        return result

    def flatten(self, item: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(item[key], dtype=np.float32).reshape(-1)
                               for key in OBS_KEYS]).astype(np.float32)

    def _ego(self, agent) -> np.ndarray:
        if not agent.alive:
            return np.zeros(7, dtype=np.float32)
        return np.array([
            agent.position[0] / self.position_norm_m,
            agent.position[1] / self.position_norm_m,
            agent.position[2] / self.altitude_norm_m,
            agent.speed / float(self.published["maximum_speed_mps"]),
            agent.pitch / np.pi,
            agent.heading / np.pi,
            agent.roll / np.pi,
        ], dtype=np.float32)

    def _relative(self, ego, position, velocity) -> np.ndarray:
        pair = assess_pair(ego.position, ego.velocity, position, velocity,
                           self.published["maximum_attack_range_m"],
                           self.derived["situation_height_norm_m"],
                           self.published["maximum_speed_mps"])
        return np.array([
            pair.relative_speed_mps / self.published["maximum_speed_mps"],
            pair.relative_altitude_m / self.altitude_norm_m,
            pair.distance_m / self.position_norm_m,
            pair.ata_rad / np.pi,
            pair.aa_rad / np.pi,
        ], dtype=np.float32)
