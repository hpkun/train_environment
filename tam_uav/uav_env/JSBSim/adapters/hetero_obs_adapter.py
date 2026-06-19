"""HeteroObsAdapter v1 — raw dict obs → fixed-dim actor / critic input.

Does NOT modify the environment, reward, missile, PID, or training.
"""
from __future__ import annotations

import numpy as np


class HeteroObsAdapter:
    """Convert HeteroUavCombatEnv raw dict obs to fixed-dim vectors.

    Follows HeteroObsSpec v1 (docs/hetero_obs_spec_v1.md).
    """

    def __init__(
        self,
        max_red: int = 5,
        max_blue: int = 4,
        role_dim: int = 4,
        ego_state_dim: int = 11,
        entity_state_dim: int = 11,
    ):
        self.max_red = max_red
        self.max_blue = max_blue
        self.max_allies = max_red - 1   # 4
        self.max_enemies = max_blue      # 4
        self.role_dim = role_dim
        self.ego_state_dim = ego_state_dim
        self.entity_state_dim = entity_state_dim

        # ---- computed dimensions ----
        self.ego_feature_dim = (ego_state_dim + role_dim + 1 + 1 + 3)  # 20
        self.ally_entity_dim = (entity_state_dim + role_dim)            # 15
        self.enemy_entity_dim = entity_state_dim                        # 11
        self.mask_dim = (self.max_allies + self.max_allies
                         + self.max_enemies + self.max_enemies)         # 16
        self.flat_actor_obs_dim = (
            self.ego_feature_dim
            + self.max_allies * self.ally_entity_dim
            + self.max_enemies * self.enemy_entity_dim
            + self.mask_dim)                                          # 140
        self.critic_state_dim = self.flat_actor_obs_dim * self.max_red  # 700

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def adapt_agent(
        self,
        agent_id: str,
        obs: dict,
        info: dict | None = None,
        red_ids: list[str] | None = None,
        blue_ids: list[str] | None = None,
    ) -> dict:
        """Build structured and flat actor features for one agent."""
        red_ids, blue_ids = self._resolve_team_ids(agent_id, obs, red_ids, blue_ids)

        is_red = agent_id.startswith("red")
        ally_ids = ([aid for aid in red_ids if aid != agent_id]
                    if is_red else [aid for aid in blue_ids if aid != agent_id])
        enemy_ids = blue_ids if is_red else red_ids

        # ---- ego ----
        ego_feature = self._build_ego(obs)

        # ---- ally entities ----
        ally_ents, ally_valid, ally_alive = self._build_ally_entities(
            obs, ally_ids, info)

        # ---- enemy entities ----
        enemy_ents, enemy_valid, enemy_alive = self._build_enemy_entities(
            obs, enemy_ids, info)

        # ---- flat ----
        flat = np.concatenate([
            ego_feature,
            ally_ents.ravel(),
            enemy_ents.ravel(),
            ally_valid.ravel(),
            ally_alive.ravel(),
            enemy_valid.ravel(),
            enemy_alive.ravel(),
        ]).astype(np.float32)

        return {
            "ego_feature": ego_feature,
            "ally_entities": ally_ents,
            "enemy_entities": enemy_ents,
            "ally_valid_mask": ally_valid,
            "ally_alive_mask": ally_alive,
            "enemy_valid_mask": enemy_valid,
            "enemy_alive_mask": enemy_alive,
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
        """Build actor and critic inputs for the controlled team."""
        if controlled_side != "red":
            raise NotImplementedError(
                "HeteroObsAdapter v1 only supports controlled_side='red'")

        if red_ids is None:
            red_ids = sorted(
                [k for k in obs_dict if k.startswith("red_")],
                key=lambda x: int(x.split("_")[1]))
        if blue_ids is None:
            blue_ids = sorted(
                [k for k in obs_dict if k.startswith("blue_")],
                key=lambda x: int(x.split("_")[1]))

        actor_obs: dict[str, np.ndarray] = {}
        structured: dict[str, dict] = {}
        for rid in red_ids:
            out = self.adapt_agent(
                rid, obs_dict[rid], info=info,
                red_ids=red_ids, blue_ids=blue_ids)
            actor_obs[rid] = out["flat_actor_obs"]
            structured[rid] = out

        # critic state: concatenate padded red flat actor obs
        critic_parts = []
        for i in range(self.max_red):
            if i < len(red_ids):
                critic_parts.append(actor_obs[red_ids[i]])
            else:
                critic_parts.append(
                    np.zeros(self.flat_actor_obs_dim, dtype=np.float32))
        critic_state = np.concatenate(critic_parts).astype(np.float32)

        red_valid_mask = np.zeros(self.max_red, dtype=np.float32)
        red_valid_mask[:len(red_ids)] = 1.0

        return {
            "actor_obs": actor_obs,
            "structured_actor_obs": structured,
            "critic_state": critic_state,
            "red_valid_mask": red_valid_mask,
        }

    # ------------------------------------------------------------------
    #  Internal builders
    # ------------------------------------------------------------------

    def _build_ego(self, obs: dict) -> np.ndarray:
        alt = np.asarray(obs["altitude"], dtype=np.float32).ravel()
        vel = np.asarray(obs["velocity"], dtype=np.float32).ravel()
        mw = np.asarray(obs["missile_warning"], dtype=np.float32).ravel()
        ego_state = np.asarray(obs["ego_state"], dtype=np.float32).ravel()
        ego_role = np.asarray(obs["ego_role"], dtype=np.float32).ravel()
        return np.concatenate([
            ego_state,
            ego_role,
            mw,
            alt / 10000.0,
            vel / 600.0,
        ]).astype(np.float32)

    def _build_ally_entities(self, obs: dict, ally_ids: list[str],
                             info: dict | None) -> tuple:
        return self._build_entities(
            obs, ally_ids, self.max_allies, info,
            state_key="ally_states", role_key="ally_roles", include_role=True)

    def _build_enemy_entities(self, obs: dict, enemy_ids: list[str],
                              info: dict | None) -> tuple:
        return self._build_entities(
            obs, enemy_ids, self.max_enemies, info,
            state_key="enemy_states", role_key="enemy_roles", include_role=False)

    def _build_entities(self, obs: dict, entity_ids: list[str],
                        max_slots: int, info: dict | None,
                        state_key: str, role_key: str,
                        include_role: bool) -> tuple:
        entity_dim = (self.ally_entity_dim if include_role
                      else self.enemy_entity_dim)

        entities = np.zeros((max_slots, entity_dim), dtype=np.float32)
        valid_mask = np.zeros(max_slots, dtype=np.float32)
        alive_mask = np.zeros(max_slots, dtype=np.float32)

        state_array = np.asarray(obs.get(state_key, []), dtype=np.float32)
        if state_array.ndim == 1 and state_array.size > 0:
            state_array = state_array.reshape(-1, self.entity_state_dim)
        role_array = None
        if include_role:
            role_array = np.asarray(obs.get(role_key, []), dtype=np.float32)

        # Try to get alive info from env info dict
        alive_agent_ids: set[str] = set()
        if info is not None:
            for key in ("alive_agents", "agent_alive"):
                if key in info and isinstance(info[key], (list, set)):
                    alive_agent_ids = set(info[key])
                    break
            if not alive_agent_ids:
                per_agent = info.get("agent_alive", None)
                if isinstance(per_agent, dict):
                    alive_agent_ids = {aid for aid, al in per_agent.items()
                                       if al is True or al == 1}

        for i, eid in enumerate(entity_ids):
            if i >= max_slots:
                break
            # valid: slot corresponds to a real agent
            valid_mask[i] = 1.0

            # alive: from info if available, else infer from non-zero state
            if alive_agent_ids:
                is_alive = eid in alive_agent_ids
            else:
                # Fallback: infer alive from non-zero state vector.
                # Assumption: dead agents have all-zero entity state in BRMA obs.
                if i < state_array.shape[0]:
                    is_alive = not np.allclose(state_array[i], 0.0)
                else:
                    is_alive = False

            if is_alive:
                alive_mask[i] = 1.0
                if i < state_array.shape[0]:
                    state_vec = state_array[i]
                    if include_role and role_array is not None \
                            and i < role_array.shape[0]:
                        entities[i] = np.concatenate([
                            state_vec, role_array[i]]).astype(np.float32)
                    else:
                        entities[i, :self.entity_state_dim] = state_vec
            # else: dead but valid — feature stays zero, valid=1, alive=0

        return entities, valid_mask, alive_mask

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_team_ids(agent_id: str, obs: dict,
                          red_ids: list[str] | None,
                          blue_ids: list[str] | None):
        if red_ids is None:
            red_ids = sorted(
                [k for k in obs if k.startswith("red_")],
                key=lambda x: int(x.split("_")[1]))
        if blue_ids is None:
            blue_ids = sorted(
                [k for k in obs if k.startswith("blue_")],
                key=lambda x: int(x.split("_")[1]))
        if not red_ids:
            raise ValueError(
                f"Cannot infer red_ids from obs keys for agent {agent_id}")
        if not blue_ids:
            raise ValueError(
                f"Cannot infer blue_ids from obs keys for agent {agent_id}")
        return red_ids, blue_ids
