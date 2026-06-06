"""Scripted opponent policies for Stage 1 MAPPO baseline runners.

These policies live at the training/evaluation script layer. They do not modify
the environment, reward, missile, PID, termination, action mapping, or aircraft
models.
"""
from __future__ import annotations

import numpy as np


class OpponentPolicy:
    """Generate blue-side high-level actions for baseline evaluation.

    Supported modes:
    - ``zero``: all blue actions are zero, for smoke/debug only.
    - ``random``: sample uniformly from [-1, 1].
    - ``rule_nearest``: steer toward the nearest non-zero red entity in each
      blue agent's own ``enemy_states`` observation.
    - ``greedy_fsm``: low-intrusion finite-state scripted blue intent policy.
    """

    MODES = {"zero", "random", "rule_nearest", "greedy_fsm"}

    def __init__(self, mode: str = "zero", seed: int | None = None):
        if mode not in self.MODES:
            raise ValueError(f"unknown opponent policy mode: {mode}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.last_states: dict[str, str] = {}

    def act(self, obs_dict: dict, blue_ids: list[str],
            deterministic: bool = True) -> dict[str, np.ndarray]:
        del deterministic
        self.last_states = {}
        if self.mode == "zero":
            return {
                bid: np.zeros(3, dtype=np.float32)
                for bid in blue_ids
            }
        if self.mode == "random":
            return {
                bid: self.rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
                for bid in blue_ids
            }
        if self.mode == "greedy_fsm":
            actions = {}
            for index, bid in enumerate(blue_ids):
                action, state = self._greedy_fsm_action(
                    obs_dict.get(bid, {}), agent_index=index)
                actions[bid] = action
                self.last_states[bid] = state
            return actions
        return {
            bid: self._rule_nearest_action(obs_dict.get(bid, {}))
            for bid in blue_ids
        }

    @staticmethod
    def _rule_nearest_action(obs: dict) -> np.ndarray:
        enemy_states = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
        if enemy_states.ndim != 2 or enemy_states.shape[0] == 0:
            return np.array([0.0, 0.0, 0.3], dtype=np.float32)

        best_state = None
        best_dist = float("inf")
        for state in enemy_states:
            if state.size < 3 or np.allclose(state, 0.0):
                continue
            rel = state[:3].astype(np.float32)
            dist = float(np.linalg.norm(rel))
            if dist < best_dist:
                best_dist = dist
                best_state = state

        if best_state is None:
            return np.array([0.0, 0.0, 0.3], dtype=np.float32)

        # Existing BRMA observations place enemy states in ego body-relative
        # order. y > 0 means target is to the right; z > 0 means target is above.
        pitch = float(best_state[2]) * 2.0
        heading = float(best_state[1]) * 2.0
        speed = 0.8

        action = np.array([pitch, heading, speed], dtype=np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # Per-agent target persistence (slot index → last targeted slot)
    _last_targets: dict[int, int] = {}

    @classmethod
    def _greedy_fsm_action(
        cls, obs: dict, agent_index: int = 0
    ) -> tuple[np.ndarray, str]:
        if cls._scalar(obs.get("missile_warning", 0.0)) > 0.0:
            heading = cls._fallback_heading(obs, agent_index, scale=0.8)
            return cls._clip_action([0.6, heading, 1.0]), "evade"

        altitude = cls._altitude_value(obs)
        if altitude is not None and altitude < 0.2:
            return cls._clip_action([0.7, 0.0, 0.8]), "recover_altitude"

        # Target persistence: prefer last-targeted slot if still visible
        last_slot = cls._last_targets.get(agent_index)
        if last_slot is not None:
            enemy_states = cls._enemy_states(obs)
            visible = cls._visible_mask(obs, enemy_states.shape[0])
            if (last_slot < enemy_states.shape[0] and visible[last_slot]
                    and cls._state_is_valid(enemy_states[last_slot])):
                action = cls._attack_action_from_obs(
                    obs, enemy_states[last_slot], agent_index)
                return action, "attack_nearest"

        target = cls._select_mav_target(obs)
        if target is not None:
            action = cls._attack_action_from_obs(obs, target, agent_index)
            # Record slot for persistence
            enemy_states = cls._enemy_states(obs)
            for idx, es in enumerate(enemy_states):
                if np.array_equal(es, target):
                    cls._last_targets[agent_index] = idx
                    break
            return action, "attack_mav_priority"

        target = cls._select_nearest_target(obs)
        if target is not None:
            action = cls._attack_action_from_obs(obs, target, agent_index)
            enemy_states = cls._enemy_states(obs)
            for idx, es in enumerate(enemy_states):
                if np.array_equal(es, target):
                    cls._last_targets[agent_index] = idx
                    break
            return action, "attack_nearest"

        cls._last_targets.pop(agent_index, None)
        return cls._search_acquire_action(obs, agent_index), "search_acquire"

    @classmethod
    def _get_current_heading_norm(cls, obs_agent: dict, fallback: float = 0.0) -> float:
        """Return current heading as normalized [-1, 1] value.

        Action mapping in env.py: target_heading = action[1] * pi.
        So action[1] represents absolute heading in [-1, 1] where 0=north,
        0.5=east, 1/-1=south, -0.5=west.

        The ego_geo_state stores yaw_norm = yaw / pi where heading=0 rad ⇛ north
        maps to yaw_norm=0, heading=pi rad ⇛ south maps to yaw_norm=1 or -1.
        So yaw_norm from ego_geo_state[5] is directly usable as heading action.
        """
        ego_geo = np.asarray(obs_agent.get("ego_geo_state", []), dtype=np.float32).ravel()
        if ego_geo.size >= 6:
            yaw_norm = float(ego_geo[5])
            if np.isfinite(yaw_norm):
                return float(np.clip(yaw_norm, -1.0, 1.0))
        # Fallback: cannot read heading — use small offset, not absolute 0
        # Note: this is unreliable; search_acquire should always receive obs
        # with ego_geo_state when using the mav_shared_geo observation mode.
        return float(np.clip(fallback + 0.05, -1.0, 1.0))

    @classmethod
    def _search_acquire_action(cls, obs_agent: dict | None = None,
                                agent_index: int = 0) -> np.ndarray:
        """Keep current heading + minimal deconfliction offset at high speed.

        The env action[1] is an *absolute* target heading (action[1] * pi rad).
        Search-acquire must preserve the current heading so blue continues
        toward the red formation instead of turning to absolute 0° (north).
        """
        if obs_agent is not None:
            base_heading = cls._get_current_heading_norm(obs_agent)
        else:
            base_heading = 0.0
        offset = 0.02 if agent_index % 2 == 0 else -0.02
        heading = float(np.clip(base_heading + offset, -1.0, 1.0))
        return cls._clip_action([0.0, heading, 1.0])

    @classmethod
    def _patrol_action(cls, agent_index: int = 0) -> np.ndarray:
        heading = 0.2 if agent_index % 2 == 0 else -0.2
        return cls._clip_action([0.0, heading, 0.6])

    @staticmethod
    def _scalar(value) -> float:
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        out = float(np.nan_to_num(arr.reshape(-1)[0], nan=0.0))
        return out

    @classmethod
    def _altitude_value(cls, obs: dict) -> float | None:
        # Low-intrusion heuristic: current observation variants do not expose a
        # single canonical altitude scale to this script-layer policy. Revisit
        # this threshold against the environment's real altitude field before
        # using greedy_fsm as a training opponent.
        for key in ("altitude", "altitude_norm"):
            if key in obs:
                return cls._scalar(obs[key])
        ego_geo = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32)
        if ego_geo.size >= 3:
            return float(np.nan_to_num(ego_geo.reshape(-1)[2], nan=0.0))
        ego_state = np.asarray(obs.get("ego_state", []), dtype=np.float32)
        if ego_state.size >= 3:
            return float(np.nan_to_num(ego_state.reshape(-1)[2], nan=0.0))
        return None

    @classmethod
    def _fallback_heading(cls, obs: dict, agent_index: int, scale: float) -> float:
        target = cls._select_nearest_target(obs)
        if target is not None and target.size >= 2:
            return float(np.clip(np.sign(float(target[1])) * scale, -1.0, 1.0))
        return scale if agent_index % 2 == 0 else -scale

    @classmethod
    def _select_mav_target(cls, obs: dict) -> np.ndarray | None:
        enemy_states = cls._enemy_states(obs)
        if enemy_states.shape[0] == 0:
            return None

        role_indices = cls._mav_indices(obs.get("enemy_roles", None))
        if not role_indices:
            role_indices = cls._mav_indices(obs.get("enemy_types", None))
        if not role_indices:
            return None

        visible = cls._visible_mask(obs, enemy_states.shape[0])
        candidates: list[tuple[float, np.ndarray]] = []
        for idx in role_indices:
            if idx >= enemy_states.shape[0] or not visible[idx]:
                continue
            state = enemy_states[idx]
            if cls._state_is_valid(state):
                candidates.append((cls._distance(state), state))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @classmethod
    def _select_nearest_target(cls, obs: dict) -> np.ndarray | None:
        enemy_states = cls._enemy_states(obs)
        if enemy_states.shape[0] == 0:
            return None
        visible = cls._visible_mask(obs, enemy_states.shape[0])
        candidates: list[tuple[float, np.ndarray]] = []
        for idx, state in enumerate(enemy_states):
            if not visible[idx] or not cls._state_is_valid(state):
                continue
            candidates.append((cls._distance(state), state))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _enemy_states(obs: dict) -> np.ndarray:
        for key in ("enemy_states", "enemy_geo_states"):
            arr = np.asarray(obs.get(key, []), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return np.nan_to_num(arr, nan=0.0)
        return np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _visible_mask(obs: dict, count: int) -> np.ndarray:
        for key in ("enemy_observed_mask", "enemy_visible_mask", "enemy_alive_mask"):
            arr = np.asarray(obs.get(key, []), dtype=np.float32).reshape(-1)
            if arr.size >= count:
                return arr[:count] > 0.0
        return np.ones(count, dtype=bool)

    @staticmethod
    def _mav_indices(value) -> list[int]:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return []
        return [
            int(i) for i, row in enumerate(arr)
            if row.size > 0 and float(np.nan_to_num(row[0], nan=0.0)) > 0.5
        ]

    @staticmethod
    def _state_is_valid(state: np.ndarray) -> bool:
        return state.size >= 2 and not np.allclose(state, 0.0)

    @staticmethod
    def _distance(state: np.ndarray) -> float:
        if state.size >= 3:
            return float(np.linalg.norm(state[:3]))
        return float(np.linalg.norm(state[:2]))

    @classmethod
    def _attack_action(cls, target: np.ndarray) -> np.ndarray:
        """Legacy attack — does NOT use current heading (preserved for rule_nearest)."""
        pitch = float(target[2]) * 2.0 if target.size >= 3 else 0.0
        heading = float(target[1]) * 2.0 if target.size >= 2 else 0.0
        dist = cls._distance(target)
        if dist > 0.6:
            speed = 1.0
        elif dist > 0.25:
            speed = 0.8
        else:
            speed = 0.5
        return cls._clip_action([pitch, heading, speed])

    @classmethod
    def _attack_action_from_obs(cls, obs_agent: dict, target: np.ndarray,
                                 agent_index: int = 0) -> np.ndarray:
        """Attack with absolute-heading-aware correction.

        target comes from enemy_states (BRMA body-frame relative):
          target[0]=Δx(forward), target[1]=Δy(right), target[2]=Δz(up).
        target[1] is a signed lateral offset — NOT an absolute bearing.
        env.py maps action[1] * pi → absolute target heading.

        This method reads current heading from ego_geo_state and adds a
        signed correction based on target[1], preserving the current
        heading as baseline rather than forcing absolute ~0°.
        """
        current_heading = cls._get_current_heading_norm(obs_agent)
        # Pitch: use relative vertical delta (positive = target above)
        pitch = float(target[2]) * 2.0 if target.size >= 3 else 0.0
        # Heading: signed bearing correction from lateral offset
        # target[1] > 0 → target right → positive heading correction
        bearing_correction = float(target[1]) * 0.3 if target.size >= 2 else 0.0
        heading = float(np.clip(current_heading + bearing_correction, -1.0, 1.0))
        # Speed by distance
        dist = cls._distance(target)
        if dist > 0.6:
            speed = 1.0
        elif dist > 0.25:
            speed = 0.8
        else:
            speed = 0.5
        return cls._clip_action([pitch, heading, speed])

    @staticmethod
    def _clip_action(values) -> np.ndarray:
        action = np.asarray(values, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
