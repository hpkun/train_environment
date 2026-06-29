"""Entity-set view over canonical HeteroObsAdapterV2 outputs."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - supports direct file loading in lightweight tests.
    from .hetero_obs_adapter_v2 import HeteroObsAdapterV2
except ImportError:  # pragma: no cover
    from hetero_obs_adapter_v2 import HeteroObsAdapterV2


ROLE_NAMES = ("mav", "uav", "scout", "interceptor")
ROLE_MAV = 0
ROLE_UAV = 1


class EntitySetAdapter:
    """Convert mav_shared_geo observations into fixed-width entity tokens.

    Token layout is shared for self, allies, and enemies:
    kind one-hot(3), role one-hot(4), geo/full-geometry features(18),
    side one-hot(2), missile-warning(1).

    Enemy tokens use the canonical 18-dim enemy entity from HeteroObsAdapterV2:
    compact geo(5), track-source(2), relative position(3), relative velocity(3),
    bearing/elevation(2), speed/heading(2), full-geometry valid mask(1).
    """

    def __init__(self, max_red: int = 5, max_blue: int = 4, role_dim: int = 4):
        self.v2 = HeteroObsAdapterV2(max_red=max_red, max_blue=max_blue, role_dim=role_dim)
        self.max_red = self.v2.max_red
        self.max_blue = self.v2.max_blue
        self.max_allies = self.v2.max_allies
        self.max_enemies = self.v2.max_enemies
        self.role_dim = role_dim
        self.entity_feature_dim = self.v2.enemy_entity_dim
        self.entity_dim = 3 + role_dim + self.entity_feature_dim + 2 + 1
        self.num_entities = 1 + self.max_allies + self.max_enemies

    @property
    def flat_actor_obs_dim(self) -> int:
        return self.v2.flat_actor_obs_dim

    @property
    def critic_state_dim(self) -> int:
        return self.v2.critic_state_dim

    def adapt_agent(
        self,
        agent_id: str,
        obs: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
    ) -> dict:
        flat = self.v2.adapt_agent(agent_id, obs, info=info, red_ids=red_ids, blue_ids=blue_ids)
        return self.from_v2_structured(agent_id, flat)

    def adapt_all(
        self,
        obs_dict: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
        controlled_side: str = "red",
    ) -> dict:
        adapted = self.v2.adapt_all(
            obs_dict,
            info=info,
            red_ids=red_ids,
            blue_ids=blue_ids,
            controlled_side=controlled_side,
        )
        entity_actor_obs = {
            aid: self.from_v2_structured(aid, structured)
            for aid, structured in adapted["structured_actor_obs"].items()
        }
        return {
            **adapted,
            "entity_actor_obs": entity_actor_obs,
            "entity_dim": self.entity_dim,
            "num_entities": self.num_entities,
        }

    def from_v2_structured(self, agent_id: str, structured: dict) -> dict:
        role_onehot = np.asarray(structured["ego_feature"][7:11], dtype=np.float32)
        role_id = self._role_id(role_onehot)
        role_name = ROLE_NAMES[role_id] if role_id < len(ROLE_NAMES) else "uav"

        self_entity = self._token(
            kind=0,
            role=role_onehot,
            features=structured["ego_feature"][:7],
            side=0,
            missile_warning=float(structured["ego_feature"][11]),
        )

        ally_entities = np.zeros((self.max_allies, self.entity_dim), dtype=np.float32)
        for i in range(self.max_allies):
            ally = structured["ally_entities"][i]
            ally_entities[i] = self._token(
                kind=1,
                role=ally[5:9],
                features=ally[:5],
                side=0,
                missile_warning=0.0,
            )

        enemy_entities = np.zeros((self.max_enemies, self.entity_dim), dtype=np.float32)
        for i in range(self.max_enemies):
            enemy = structured["enemy_entities"][i]
            enemy_entities[i] = self._token(
                kind=2,
                role=np.zeros(self.role_dim, dtype=np.float32),
                features=enemy,
                side=1,
                missile_warning=0.0,
            )

        entities = np.vstack([
            self_entity.reshape(1, -1),
            ally_entities,
            enemy_entities,
        ]).astype(np.float32)

        entity_valid = np.concatenate([
            np.ones(1, dtype=np.float32),
            structured["ally_valid_mask"],
            structured["enemy_valid_mask"],
        ]).astype(np.float32)
        alive = np.concatenate([
            np.ones(1, dtype=np.float32),
            structured["ally_alive_mask"],
            structured["enemy_alive_mask"],
        ]).astype(np.float32)
        observed = np.concatenate([
            np.ones(1, dtype=np.float32),
            structured["ally_valid_mask"] * structured["ally_alive_mask"],
            structured["enemy_observed_mask"],
        ]).astype(np.float32)
        attention_mask = (entity_valid * alive * observed).astype(np.float32)
        attention_mask[0] = 1.0

        entity_type_id = np.concatenate([
            np.array([0], dtype=np.int64),
            np.ones(self.max_allies, dtype=np.int64),
            np.full(self.max_enemies, 2, dtype=np.int64),
        ])

        return {
            "agent_id": agent_id,
            "role_id": role_id,
            "role_name": role_name,
            "entity_type_id": entity_type_id,
            "self_entity": self_entity,
            "ally_entities": ally_entities,
            "enemy_entities": enemy_entities,
            "entities": entities,
            "entity_valid_mask": entity_valid,
            "alive_mask": alive,
            "observed_mask": observed,
            "attention_mask": attention_mask,
            "flat_actor_obs": structured["flat_actor_obs"],
        }

    def _token(
        self,
        kind: int,
        role: np.ndarray,
        features: np.ndarray,
        side: int,
        missile_warning: float,
    ) -> np.ndarray:
        kind_oh = np.zeros(3, dtype=np.float32)
        kind_oh[kind] = 1.0
        side_oh = np.zeros(2, dtype=np.float32)
        side_oh[side] = 1.0
        return np.concatenate([
            kind_oh,
            self._pad_1d(role, self.role_dim),
            self._pad_1d(features, self.entity_feature_dim),
            side_oh,
            np.array([missile_warning], dtype=np.float32),
        ]).astype(np.float32)

    @staticmethod
    def _pad_1d(value, length: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out = np.zeros(length, dtype=np.float32)
        n = min(length, arr.shape[0])
        if n:
            out[:n] = arr[:n]
        return out

    @staticmethod
    def _role_id(role_onehot: np.ndarray) -> int:
        if role_onehot.size == 0 or float(np.max(role_onehot)) <= 0.0:
            return ROLE_UAV
        return int(np.argmax(role_onehot))
