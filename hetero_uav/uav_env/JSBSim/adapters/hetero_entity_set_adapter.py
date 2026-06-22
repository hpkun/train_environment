"""Variable-size entity-set adapter for heterogeneous scale transfer.

v2 differences from v1:
  - entity_dim 19 -> 21: adds alive_flag[1] + observed_flag[1] to every token
  - critic_keep_mask = "valid slot exists" (not "alive")
  - dead entities remain in critic tokens; alive_flag encodes alive/dead
  - actor keep_mask continues to filter unobserved/dead for attention
  - v1 checkpoints are rejected with a clear error message
"""
from __future__ import annotations

import numpy as np


ROLE_VOCAB = ("mav", "attack_uav", "scout_uav", "interceptor_uav")
ROLE_DIM = len(ROLE_VOCAB)
# v1: 3(kind) + 4(role) + 7(geo) + 2(side) + 1(mw) + 2(track) = 19
# v2: +1(alive_flag) + 1(observed_flag) = 21
ENTITY_DIM = 21
FEATURE_SCHEMA_VERSION = "hetero_entity_set_v2"
REQUIRED_ENTITY_SET_KEYS = {
    "ego_geo_state",
    "ego_role",
    "missile_warning",
    "ally_geo_states",
    "ally_roles",
    "ally_alive_mask",
    "enemy_geo_states",
    "enemy_alive_mask",
    "enemy_observed_mask",
    "enemy_track_source",
}
GLOBAL_ENTITY_KEYS = {"ego_geo_state", "ego_role", "missile_warning"}
# V1_SCHEMA for checkpoint rejection
V1_SCHEMA_VERSION = "hetero_entity_set_v1"


class HeteroEntitySetAdapter:
    """Build actor-local and critic-global entity sets from mav_shared_geo.

    v2 token layout (21 dims):
      [0:3]   kind one-hot  (self=0, ally=1, enemy=2)
      [3:7]   role one-hot  (mav, attack_uav, scout, interceptor)
      [7:14]  geo7          (ego: x,y,z,speed,pitch,yaw,roll; rel: speed_diff,delta_h,dist,ata,aa,0,0)
      [14:16] side one-hot  (red=0, blue=1)
      [16]    missile_warning
      [17:19] track_source  (own_sensor, mav_shared)
      [19]    alive_flag    (1=alive, 0=dead)
      [20]    observed_flag (1=currently observed, 0=not observed / dead)
    """

    entity_dim = ENTITY_DIM
    role_dim = ROLE_DIM
    role_vocab = ROLE_VOCAB
    feature_schema_version = FEATURE_SCHEMA_VERSION

    # Offsets for new v2 fields
    ALIVE_IDX = 19
    OBSERVED_IDX = 20

    def adapt_all(
        self,
        obs_dict: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
        controlled_side: str = "red",
    ) -> dict:
        if controlled_side != "red":
            raise NotImplementedError("only controlled_side='red' is supported")
        red_ids = red_ids or self._ids(obs_dict, "red_")
        blue_ids = blue_ids or self._ids(obs_dict, "blue_")
        if not red_ids or not blue_ids:
            raise ValueError("entity-set adapter requires non-empty red and blue teams")
        for rid in red_ids:
            missing = sorted(REQUIRED_ENTITY_SET_KEYS.difference(obs_dict.get(rid, {})))
            if missing:
                raise ValueError(
                    "HeteroEntitySetAdapter requires "
                    "observation_mode='mav_shared_geo'; "
                    f"agent {rid} missing keys: {missing}"
                )
        for aid in red_ids + blue_ids:
            missing = sorted(GLOBAL_ENTITY_KEYS.difference(obs_dict.get(aid, {})))
            if missing:
                raise ValueError(
                    "HeteroEntitySetAdapter global critic requires "
                    "observation_mode='mav_shared_geo'; "
                    f"agent {aid} missing keys: {missing}"
                )

        actor_tokens = []
        actor_masks = []
        role_ids = []
        for rid in red_ids:
            obs = obs_dict[rid]
            allies = [aid for aid in red_ids if aid != rid]
            actor_tokens.append(self._actor_tokens(obs, len(allies), len(blue_ids)))
            actor_masks.append(self._actor_mask(obs, len(allies), len(blue_ids)))
            role_ids.append(self._role_id(obs.get("ego_role", [])))

        # --- Critic: all entities, dead included ---
        global_ids = red_ids + blue_ids
        critic_tokens = []
        for aid in global_ids:
            obs = obs_dict[aid]
            alive = self._alive(info, aid)
            token = self._global_token(obs, side=0 if aid.startswith("red_") else 1)
            token[self.ALIVE_IDX] = float(alive)
            # ego always "observes" itself when alive
            token[self.OBSERVED_IDX] = float(alive)
            critic_tokens.append(token)
        critic_tokens = np.stack(critic_tokens).astype(np.float32)
        # critic_keep_mask = valid slot exists (always 1 for all global_ids)
        critic_keep_mask = np.ones(len(global_ids), dtype=np.float32)

        # --- Extra critic features: alive counts per side ---
        red_alive = sum(1 for aid in red_ids if self._alive(info, aid))
        blue_alive = sum(1 for aid in blue_ids if self._alive(info, aid))
        critic_counts = np.array(
            [red_alive, len(red_ids), blue_alive, len(blue_ids)], dtype=np.float32
        )

        return {
            "actor_entity_tokens": np.stack(actor_tokens).astype(np.float32),
            "actor_keep_mask": np.stack(actor_masks).astype(np.float32),
            "critic_entity_tokens": critic_tokens,
            "critic_keep_mask": critic_keep_mask,
            "critic_counts": critic_counts,
            "role_ids": np.asarray(role_ids, dtype=np.int64),
            "entity_dim": self.entity_dim,
            "feature_schema_version": self.feature_schema_version,
        }

    def _actor_tokens(self, obs: dict, ally_count: int, enemy_count: int) -> np.ndarray:
        tokens = [self._self_token(obs)]
        ally_geo = self._rows(obs.get("ally_geo_states", []), ally_count, 5)
        ally_roles = self._rows(obs.get("ally_roles", []), ally_count, self.role_dim)
        ally_alive = self._vector(obs.get("ally_alive_mask", []), ally_count)
        for i in range(ally_count):
            t = self._relative_token(1, ally_geo[i], ally_roles[i], side=0)
            t[self.ALIVE_IDX] = float(ally_alive[i] > 0.5)
            t[self.OBSERVED_IDX] = t[self.ALIVE_IDX]  # allies always visible when alive
            tokens.append(t)
        enemy_geo = self._rows(obs.get("enemy_geo_states", []), enemy_count, 5)
        sources = self._rows(obs.get("enemy_track_source", []), enemy_count, 2)
        enemy_alive = self._vector(obs.get("enemy_alive_mask", []), enemy_count)
        enemy_obs = self._vector(obs.get("enemy_observed_mask", []), enemy_count)
        for i in range(enemy_count):
            t = self._relative_token(2, enemy_geo[i], np.zeros(self.role_dim), side=1)
            t[17:19] = sources[i]
            t[self.ALIVE_IDX] = float(enemy_alive[i] > 0.5)
            t[self.OBSERVED_IDX] = float(enemy_obs[i] > 0.5)
            tokens.append(t)
        return np.stack(tokens)

    def _actor_mask(self, obs: dict, ally_count: int, enemy_count: int) -> np.ndarray:
        """Actor keep_mask: only alive+observed entities visible to the policy."""
        ally_alive = self._vector(obs.get("ally_alive_mask", []), ally_count)
        enemy_alive = self._vector(obs.get("enemy_alive_mask", []), enemy_count)
        enemy_observed = self._vector(obs.get("enemy_observed_mask", []), enemy_count)
        return np.concatenate([
            np.ones(1, dtype=np.float32),  # self always kept
            (ally_alive > 0.5).astype(np.float32),
            ((enemy_alive > 0.5) & (enemy_observed > 0.5)).astype(np.float32),
        ])

    def _self_token(self, obs: dict) -> np.ndarray:
        token = np.zeros(self.entity_dim, dtype=np.float32)
        token[0] = 1.0  # kind=self
        token[3:7] = self._vector(obs.get("ego_role", []), self.role_dim)
        token[7:14] = self._vector(obs.get("ego_geo_state", []), 7)
        token[14] = 1.0  # side=red
        token[16] = float(self._vector(obs.get("missile_warning", []), 1)[0])
        token[self.ALIVE_IDX] = 1.0
        token[self.OBSERVED_IDX] = 1.0
        return token

    def _global_token(self, obs: dict, side: int) -> np.ndarray:
        token = self._self_token(obs)
        token[14:16] = 0.0
        token[14 + side] = 1.0
        # alive/observed set by caller
        return token

    def _relative_token(self, kind: int, geo: np.ndarray, role: np.ndarray, side: int) -> np.ndarray:
        token = np.zeros(self.entity_dim, dtype=np.float32)
        token[kind] = 1.0
        token[3:7] = role
        token[7:12] = geo
        token[14 + side] = 1.0
        return token

    @staticmethod
    def _alive(info: dict | None, aid: str) -> float:
        if info is None or aid not in info:
            return 1.0
        return float(bool(info[aid].get("alive", True)))

    @staticmethod
    def _role_id(role) -> int:
        arr = np.asarray(role, dtype=np.float32).reshape(-1)
        return int(np.argmax(arr)) if arr.size and np.max(arr) > 0 else 1

    @staticmethod
    def _ids(obs_dict: dict, prefix: str) -> list[str]:
        return sorted(
            [aid for aid in obs_dict if aid.startswith(prefix)],
            key=lambda aid: int(aid.split("_")[1]),
        )

    @staticmethod
    def _vector(value, size: int) -> np.ndarray:
        src = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros(size, dtype=np.float32)
        out[:min(size, src.size)] = src[:size]
        return out

    @staticmethod
    def _rows(value, rows: int, cols: int) -> np.ndarray:
        src = np.asarray(value, dtype=np.float32)
        if src.size == 0:
            return np.zeros((rows, cols), dtype=np.float32)
        src = src.reshape(-1, cols)
        out = np.zeros((rows, cols), dtype=np.float32)
        out[:min(rows, len(src))] = src[:rows]
        return out
