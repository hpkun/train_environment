"""HeteroObsAdapter v3 for mav_shared_geo_v2 full shared geometry."""
from __future__ import annotations

import numpy as np

from .hetero_obs_adapter_v2 import HeteroObsAdapterV2, REQUIRED_V2_KEYS


REQUIRED_V3_KEYS = REQUIRED_V2_KEYS.union({
    "enemy_relative_pos_xyz",
    "enemy_relative_vel_xyz",
    "enemy_bearing_elevation",
    "enemy_speed_heading",
    "enemy_full_geo_valid_mask",
})


class HeteroObsAdapterV3(HeteroObsAdapterV2):
    """Fixed-size adapter including direct/MAV-shared full enemy geometry."""

    schema_version = "hetero_obs_adapter_v3_mav_shared_full_geo"

    def __init__(self, max_red: int = 5, max_blue: int = 4, role_dim: int = 4):
        super().__init__(max_red=max_red, max_blue=max_blue, role_dim=role_dim)
        self.enemy_full_geo_dim = 10
        self.enemy_entity_dim = 7 + self.enemy_full_geo_dim + 1
        self.flat_actor_obs_dim = (
            self.ego_feature_dim
            + self.max_allies * self.ally_entity_dim
            + self.max_enemies * self.enemy_entity_dim
            + self.mask_dim
        )
        self.critic_state_dim = self.flat_actor_obs_dim * self.max_red

    def _build_enemy_entities(self, obs: dict, enemy_ids: list[str]) -> tuple:
        base_geo = self._pad_2d(obs["enemy_geo_states"], self.max_enemies, self.relative_geo_dim)
        source = self._pad_2d(obs["enemy_track_source"], self.max_enemies, self.track_source_dim)
        rel_pos = self._pad_2d(obs["enemy_relative_pos_xyz"], self.max_enemies, 3)
        rel_vel = self._pad_2d(obs["enemy_relative_vel_xyz"], self.max_enemies, 3)
        bearing_el = self._pad_2d(obs["enemy_bearing_elevation"], self.max_enemies, 2)
        speed_heading = self._pad_2d(obs["enemy_speed_heading"], self.max_enemies, 2)
        full_valid = self._pad_1d(obs["enemy_full_geo_valid_mask"], self.max_enemies)
        observed = self._pad_1d(obs["enemy_observed_mask"], self.max_enemies)
        raw_alive = self._pad_1d(obs["enemy_alive_mask"], self.max_enemies)

        entities = np.zeros((self.max_enemies, self.enemy_entity_dim), dtype=np.float32)
        valid = np.zeros(self.max_enemies, dtype=np.float32)
        alive = np.zeros(self.max_enemies, dtype=np.float32)
        observed_mask = np.zeros(self.max_enemies, dtype=np.float32)
        for i, _eid in enumerate(enemy_ids[:self.max_enemies]):
            valid[i] = 1.0
            alive[i] = 1.0 if raw_alive[i] > 0.5 else 0.0
            if alive[i] > 0.5 and observed[i] > 0.5:
                observed_mask[i] = 1.0
                entities[i] = np.concatenate([
                    base_geo[i],
                    source[i],
                    rel_pos[i],
                    rel_vel[i],
                    bearing_el[i],
                    speed_heading[i],
                    np.asarray([full_valid[i]], dtype=np.float32),
                ]).astype(np.float32)
        return entities, valid, alive, observed_mask

    @staticmethod
    def _require_v2_obs(obs: dict) -> None:
        missing = sorted(REQUIRED_V3_KEYS.difference(obs))
        if missing:
            raise ValueError(
                "HeteroObsAdapterV3 requires observation_mode='mav_shared_geo_v2'; "
                f"missing keys: {missing}"
            )
