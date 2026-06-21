"""Variable-size entity-set adapter for heterogeneous scale transfer."""
from __future__ import annotations

import numpy as np


ROLE_VOCAB = ("mav", "attack_uav", "scout_uav", "interceptor_uav")
ROLE_DIM = len(ROLE_VOCAB)
ENTITY_DIM = 19
FEATURE_SCHEMA_VERSION = "hetero_entity_set_v1"


class HeteroEntitySetAdapter:
    """Build actor-local and critic-global entity sets from mav_shared_geo."""

    entity_dim = ENTITY_DIM
    role_dim = ROLE_DIM
    role_vocab = ROLE_VOCAB
    feature_schema_version = FEATURE_SCHEMA_VERSION

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

        actor_tokens = []
        actor_masks = []
        role_ids = []
        for rid in red_ids:
            obs = obs_dict[rid]
            allies = [aid for aid in red_ids if aid != rid]
            actor_tokens.append(self._actor_tokens(obs, len(allies), len(blue_ids)))
            actor_masks.append(self._actor_mask(obs, len(allies), len(blue_ids)))
            role_ids.append(self._role_id(obs.get("ego_role", [])))

        global_ids = red_ids + blue_ids
        critic_tokens = np.stack([
            self._global_token(obs_dict[aid], side=0 if aid.startswith("red_") else 1)
            for aid in global_ids
        ]).astype(np.float32)
        critic_mask = np.asarray([
            self._alive(info, aid) for aid in global_ids
        ], dtype=np.float32)

        return {
            "actor_entity_tokens": np.stack(actor_tokens).astype(np.float32),
            "actor_keep_mask": np.stack(actor_masks).astype(np.float32),
            "critic_entity_tokens": critic_tokens,
            "critic_keep_mask": critic_mask,
            "role_ids": np.asarray(role_ids, dtype=np.int64),
            "entity_dim": self.entity_dim,
            "feature_schema_version": self.feature_schema_version,
        }

    def _actor_tokens(self, obs: dict, ally_count: int, enemy_count: int) -> np.ndarray:
        tokens = [self._self_token(obs)]
        ally_geo = self._rows(obs.get("ally_geo_states", []), ally_count, 5)
        ally_roles = self._rows(obs.get("ally_roles", []), ally_count, self.role_dim)
        for i in range(ally_count):
            tokens.append(self._relative_token(1, ally_geo[i], ally_roles[i], side=0))
        enemy_geo = self._rows(obs.get("enemy_geo_states", []), enemy_count, 5)
        sources = self._rows(obs.get("enemy_track_source", []), enemy_count, 2)
        for i in range(enemy_count):
            token = self._relative_token(2, enemy_geo[i], np.zeros(self.role_dim), side=1)
            token[17:19] = sources[i]
            tokens.append(token)
        return np.stack(tokens)

    def _actor_mask(self, obs: dict, ally_count: int, enemy_count: int) -> np.ndarray:
        ally_alive = self._vector(obs.get("ally_alive_mask", []), ally_count)
        enemy_alive = self._vector(obs.get("enemy_alive_mask", []), enemy_count)
        enemy_observed = self._vector(obs.get("enemy_observed_mask", []), enemy_count)
        return np.concatenate([
            np.ones(1, dtype=np.float32),
            (ally_alive > 0.5).astype(np.float32),
            ((enemy_alive > 0.5) & (enemy_observed > 0.5)).astype(np.float32),
        ])

    def _self_token(self, obs: dict) -> np.ndarray:
        token = np.zeros(self.entity_dim, dtype=np.float32)
        token[0] = 1.0
        token[3:7] = self._vector(obs.get("ego_role", []), self.role_dim)
        token[7:14] = self._vector(obs.get("ego_geo_state", []), 7)
        token[14] = 1.0
        token[16] = float(self._vector(obs.get("missile_warning", []), 1)[0])
        return token

    def _global_token(self, obs: dict, side: int) -> np.ndarray:
        token = self._self_token(obs)
        token[14:16] = 0.0
        token[14 + side] = 1.0
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

