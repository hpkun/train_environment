"""HeteroObsAdapter v2 for MAV-shared geometric observations."""
from __future__ import annotations

import numpy as np


REQUIRED_V2_KEYS = {
    "ego_geo_state",
    "ally_geo_states",
    "enemy_geo_states",
    "enemy_observed_mask",
    "enemy_track_source",
}


class HeteroObsAdapterV2:
    """Convert mav_shared_geo raw obs to fixed-dim actor/critic inputs."""

    def __init__(self, max_red: int = 5, max_blue: int = 4, role_dim: int = 4):
        self.max_red = max_red
        self.max_blue = max_blue
        self.max_allies = max_red - 1
        self.max_enemies = max_blue
        self.role_dim = role_dim

        self.ego_geo_dim = 7
        self.relative_geo_dim = 5
        self.track_source_dim = 2
        self.ego_feature_dim = 12
        self.ally_entity_dim = 9
        self.enemy_entity_dim = 7
        self.mask_dim = 20
        self.flat_actor_obs_dim = (
            self.ego_feature_dim
            + self.max_allies * self.ally_entity_dim
            + self.max_enemies * self.enemy_entity_dim
            + self.mask_dim
        )
        self.critic_state_dim = self.flat_actor_obs_dim * self.max_red

    def adapt_agent(
        self,
        agent_id: str,
        obs: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
    ) -> dict:
        self._require_v2_obs(obs)
        red_ids, blue_ids = self._resolve_team_ids(agent_id, red_ids, blue_ids)
        is_red = agent_id.startswith("red_")
        ally_ids = ([aid for aid in red_ids if aid != agent_id]
                    if is_red else [aid for aid in blue_ids if aid != agent_id])
        enemy_ids = blue_ids if is_red else red_ids

        ego_feature = self._build_ego(obs)
        ally_entities, ally_valid, ally_alive = self._build_ally_entities(obs, ally_ids)
        enemy_entities, enemy_valid, enemy_alive, enemy_observed = self._build_enemy_entities(
            obs, enemy_ids)

        flat = np.concatenate([
            ego_feature,
            ally_entities.ravel(),
            enemy_entities.ravel(),
            ally_valid,
            ally_alive,
            enemy_valid,
            enemy_alive,
            enemy_observed,
        ]).astype(np.float32)

        return {
            "ego_feature": ego_feature,
            "ally_entities": ally_entities,
            "enemy_entities": enemy_entities,
            "ally_valid_mask": ally_valid,
            "ally_alive_mask": ally_alive,
            "enemy_valid_mask": enemy_valid,
            "enemy_alive_mask": enemy_alive,
            "enemy_observed_mask": enemy_observed,
            "flat_actor_obs": flat,
        }

    def adapt_all(
        self,
        obs_dict: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
        controlled_side: str = "red",
    ) -> dict:
        del info
        if controlled_side != "red":
            raise NotImplementedError(
                "HeteroObsAdapterV2 only supports controlled_side='red'")

        if red_ids is None:
            red_ids = sorted([k for k in obs_dict if k.startswith("red_")],
                             key=lambda x: int(x.split("_")[1]))
        if blue_ids is None:
            blue_ids = sorted([k for k in obs_dict if k.startswith("blue_")],
                              key=lambda x: int(x.split("_")[1]))

        actor_obs: dict[str, np.ndarray] = {}
        structured: dict[str, dict] = {}
        for rid in red_ids:
            out = self.adapt_agent(rid, obs_dict[rid], red_ids=red_ids, blue_ids=blue_ids)
            actor_obs[rid] = out["flat_actor_obs"]
            structured[rid] = out

        critic_parts = []
        for i in range(self.max_red):
            if i < len(red_ids):
                critic_parts.append(actor_obs[red_ids[i]])
            else:
                critic_parts.append(np.zeros(self.flat_actor_obs_dim, dtype=np.float32))
        critic_state = np.concatenate(critic_parts).astype(np.float32)

        red_valid_mask = np.zeros(self.max_red, dtype=np.float32)
        red_valid_mask[:min(len(red_ids), self.max_red)] = 1.0

        return {
            "actor_obs": actor_obs,
            "structured_actor_obs": structured,
            "critic_state": critic_state,
            "red_valid_mask": red_valid_mask,
        }

    def _build_ego(self, obs: dict) -> np.ndarray:
        ego_geo = np.asarray(obs["ego_geo_state"], dtype=np.float32).reshape(-1)
        ego_role = np.asarray(obs["ego_role"], dtype=np.float32).reshape(-1)
        missile_warning = np.asarray(obs["missile_warning"], dtype=np.float32).reshape(-1)
        return np.concatenate([
            self._pad_1d(ego_geo, self.ego_geo_dim),
            self._pad_1d(ego_role, self.role_dim),
            self._pad_1d(missile_warning, 1),
        ]).astype(np.float32)

    def _build_ally_entities(self, obs: dict, ally_ids: list[str]) -> tuple:
        geo = self._pad_2d(obs["ally_geo_states"], self.max_allies, self.relative_geo_dim)
        roles = self._pad_2d(obs.get("ally_roles", []), self.max_allies, self.role_dim)
        entities = np.zeros((self.max_allies, self.ally_entity_dim), dtype=np.float32)
        valid = np.zeros(self.max_allies, dtype=np.float32)
        alive = np.zeros(self.max_allies, dtype=np.float32)
        for i, _aid in enumerate(ally_ids[:self.max_allies]):
            valid[i] = 1.0
            if not np.allclose(geo[i], 0.0):
                alive[i] = 1.0
                entities[i] = np.concatenate([geo[i], roles[i]]).astype(np.float32)
        return entities, valid, alive

    def _build_enemy_entities(self, obs: dict, enemy_ids: list[str]) -> tuple:
        geo = self._pad_2d(obs["enemy_geo_states"], self.max_enemies, self.relative_geo_dim)
        source = self._pad_2d(obs["enemy_track_source"], self.max_enemies, self.track_source_dim)
        observed = self._pad_1d(obs["enemy_observed_mask"], self.max_enemies)
        entities = np.zeros((self.max_enemies, self.enemy_entity_dim), dtype=np.float32)
        valid = np.zeros(self.max_enemies, dtype=np.float32)
        alive = np.zeros(self.max_enemies, dtype=np.float32)
        for i, _eid in enumerate(enemy_ids[:self.max_enemies]):
            valid[i] = 1.0
            if observed[i] > 0.5:
                alive[i] = 1.0
                entities[i] = np.concatenate([geo[i], source[i]]).astype(np.float32)
        return entities, valid, alive, observed.astype(np.float32)

    @staticmethod
    def _require_v2_obs(obs: dict) -> None:
        missing = sorted(REQUIRED_V2_KEYS.difference(obs))
        if missing:
            raise ValueError(
                "HeteroObsAdapterV2 requires observation_mode='mav_shared_geo'; "
                f"missing keys: {missing}"
            )

    @staticmethod
    def _pad_1d(value, length: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros(length, dtype=np.float32)
        n = min(length, arr.shape[0])
        if n:
            out[:n] = arr[:n]
        return out

    @staticmethod
    def _pad_2d(value, rows: int, cols: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(0, cols) if arr.size == 0 else arr.reshape(-1, cols)
        out = np.zeros((rows, cols), dtype=np.float32)
        r = min(rows, arr.shape[0])
        c = min(cols, arr.shape[1]) if arr.ndim == 2 else 0
        if r and c:
            out[:r, :c] = arr[:r, :c]
        return out

    @staticmethod
    def _resolve_team_ids(agent_id: str, red_ids: list[str] | None,
                          blue_ids: list[str] | None):
        if red_ids is None or blue_ids is None:
            raise ValueError("red_ids and blue_ids are required for HeteroObsAdapterV2")
        if not red_ids or not blue_ids:
            raise ValueError(f"empty team ids for agent {agent_id}")
        return red_ids, blue_ids
