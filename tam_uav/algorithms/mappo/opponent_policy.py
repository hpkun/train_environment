"""Scripted opponent policies for Stage 1 MAPPO baseline runners.

These policies live at the training/evaluation script layer. They do not modify
the environment, reward, missile, PID, termination, action mapping, or aircraft
models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _wrap_heading_norm(value: float) -> float:
    """Wrap normalized heading to [-1, 1] where +/-1 are the same direction."""
    wrapped = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
    while wrapped > 1.0:
        wrapped -= 2.0
    while wrapped < -1.0:
        wrapped += 2.0
    return wrapped


class OpponentPolicy:
    """Generate blue-side high-level actions for baseline evaluation.

    Supported modes:
    - ``zero``: all blue actions are zero, for smoke/debug only.
    - ``random``: sample uniformly from [-1, 1].
    - ``rule_nearest``: steer toward the nearest non-zero red entity in each
      blue agent's own ``enemy_states`` observation.
    - ``greedy_fsm``: low-intrusion finite-state scripted blue intent policy.
    - ``brma_rule``: delegates to the parent project's ``rule_based_agent.py``
      (BRMA-MAPPO paper-aligned blue opponent).
    - ``tam_direct_fsm``: nearest-target rule with 4D direct-FCS output.
    """

    MODES = {
        "zero", "random", "rule_nearest", "greedy_fsm", "brma_rule",
        "tam_direct_fsm",
    }

    def __init__(self, mode: str = "zero", seed: int | None = None):
        if mode not in self.MODES:
            raise ValueError(f"unknown opponent policy mode: {mode}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.last_states: dict[str, str] = {}
        # Instance-level target persistence (slot index → last targeted slot)
        self.last_targets: dict[int, int] = {}
        # Per-agent distance memory and lost-target counter
        self.last_target_distances: dict[int, float] = {}
        self.lost_target_steps: dict[int, int] = {}
        self.last_assigned_targets: dict[str, int] = {}
        self.used_env_refresh_engaged_targets = False
        self.used_env_own_kinematics = False
        self.used_env_own_positions = False

    def reset_memory(self) -> None:
        """Clear per-agent target persistence and state history."""
        self.last_targets.clear()
        self.last_target_distances.clear()
        self.lost_target_steps.clear()
        self.last_states.clear()
        self.last_assigned_targets.clear()
        self.used_env_refresh_engaged_targets = False
        self.used_env_own_kinematics = False
        self.used_env_own_positions = False

    def act(self, obs_dict: dict, blue_ids: list[str],
            deterministic: bool = True, env=None) -> dict[str, np.ndarray]:
        del deterministic
        self.last_states = {}
        self.last_assigned_targets: dict[str, int] = {}
        self.used_env_refresh_engaged_targets = False
        self.used_env_own_kinematics = False
        self.used_env_own_positions = False
        own_kinematics = self._env_blue_own_kinematics(env)
        own_positions = self._env_blue_own_positions(env)
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
            assigned_targets: set[int] = set()
            engaged_targets = self._env_engaged_target_slots(env)
            assigned_targets.update(engaged_targets)
            for index, bid in enumerate(blue_ids):
                ownship = self._ownship_context(bid, own_kinematics, own_positions)
                action, state = self._greedy_fsm_action(
                    obs_dict.get(bid, {}),
                    agent_index=index,
                    assigned_targets=assigned_targets,
                    ownship=ownship,
                )
                actions[bid] = action
                self.last_states[bid] = state
                target_slot = self.last_targets.get(index)
                if target_slot is not None:
                    assigned_targets.add(target_slot)
                    self.last_assigned_targets[bid] = target_slot
            return actions
        if self.mode == "brma_rule":
            return self._brma_rule_actions(obs_dict, blue_ids, env)
        if self.mode == "tam_direct_fsm":
            actions = {}
            for index, bid in enumerate(blue_ids):
                ownship = self._ownship_context(bid, own_kinematics, own_positions)
                actions[bid] = self._tam_direct_action(
                    obs_dict.get(bid, {}), blue_id=bid, env=env,
                    ownship=ownship, agent_index=index,
                )
            return actions
        return {
            bid: self._rule_nearest_action(obs_dict.get(bid, {}))
            for bid in blue_ids
        }

    def _brma_rule_actions(self, obs_dict, blue_ids, env) -> dict[str, np.ndarray]:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from rule_based_agent import blue_coordinated_actions
        except ImportError as e:
            raise ImportError(
                "brma_rule requires parent project rule_based_agent.py. "
                "Ensure the parent repo root is accessible. "
                f"Original error: {e}"
            ) from e

        num_blue = len(blue_ids)
        num_red = 0
        if env is not None:
            num_red = len(getattr(env, "red_ids", []))

        engaged_targets = set()
        if env is not None and hasattr(env, "refresh_engaged_targets"):
            try:
                engaged_targets = env.refresh_engaged_targets()
            except Exception:
                pass
        if engaged_targets is None:
            engaged_targets = set()

        own_positions = {}
        if env is not None and hasattr(env, "get_blue_own_positions"):
            try:
                own_positions = env.get_blue_own_positions()
            except Exception:
                pass
        if own_positions is None:
            own_positions = {}

        own_headings = {}
        if env is not None and hasattr(env, "get_blue_own_kinematics"):
            try:
                kin = env.get_blue_own_kinematics()
                own_headings = {bid: float(k.get("heading", 0.0)) for bid, k in kin.items()}
            except Exception:
                pass
        if own_headings is None:
            own_headings = {}

        blue_obs = {bid: obs_dict.get(bid, {}) for bid in blue_ids}
        return blue_coordinated_actions(
            blue_obs=blue_obs,
            num_blue=num_blue,
            num_red=num_red,
            engaged_targets=engaged_targets,
            own_positions=own_positions,
            own_headings=own_headings,
        )

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

    @classmethod
    def _tam_obs_source_and_target(cls, obs: dict):
        """Return (source_name, target_state_or_None, nearest_target_idx_or_None)."""
        # Check enemy_states first (11-dim body-frame)
        es = np.asarray(obs.get("enemy_states", []), dtype=np.float32)
        if es.ndim == 2 and es.shape[0] > 0:
            target, tidx = cls._select_nearest_target(obs)
            return ("enemy_states", target, tidx)
        # Check enemy_geo_states (5-dim geometric)
        egs = np.asarray(obs.get("enemy_geo_states", []), dtype=np.float32)
        if egs.ndim == 2 and egs.shape[0] > 0:
            target, tidx = cls._select_nearest_target(obs)
            return ("enemy_geo_states", target, tidx)
        return ("none", None, None)

    @classmethod
    def _tam_blue_altitude_m(cls, obs: dict, env=None, blue_id: str = "") -> float:
        """Return blue altitude in meters. Prefer env sim, then obs."""
        if env is not None and blue_id and hasattr(env, "blue_planes"):
            sim = env.blue_planes.get(blue_id)
            if sim is not None and sim.is_alive:
                return float(sim.get_geodetic()[2])
        alt_arr = np.asarray(obs.get("altitude", []), dtype=np.float32).reshape(-1)
        if alt_arr.size > 0 and alt_arr[0] > 100:
            return float(alt_arr[0])
        ego = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32).reshape(-1)
        if ego.size >= 3:
            return float(ego[2]) * 10000.0
        return 6000.0

    @classmethod
    def _tam_blue_speed_ms(cls, obs: dict, env=None, blue_id: str = "") -> float:
        if env is not None and blue_id and hasattr(env, "blue_planes"):
            sim = env.blue_planes.get(blue_id)
            if sim is not None and sim.is_alive:
                return float(np.linalg.norm(sim.get_velocity()))
        vel = np.asarray(obs.get("velocity", []), dtype=np.float32).reshape(-1)
        if vel.size >= 3:
            return float(np.linalg.norm(vel))
        return 250.0

    @classmethod
    def _tam_blue_rpy_rad(cls, obs: dict, env=None, blue_id: str = ""):
        if env is not None and blue_id and hasattr(env, "blue_planes"):
            sim = env.blue_planes.get(blue_id)
            if sim is not None and sim.is_alive:
                return sim.get_rpy()
        ego = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32).reshape(-1)
        if ego.size >= 7:
            return np.array([float(ego[6])*np.pi, float(ego[4])*np.pi, float(ego[5])*np.pi], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    @classmethod
    def _tam_direct_action(cls, obs: dict, blue_id: str = "", env=None,
                           ownship: dict | None = None, agent_index: int = 0) -> np.ndarray:
        # ---- Read blue state from env or obs ----
        alt_m = cls._tam_blue_altitude_m(obs, env, blue_id)
        speed_ms = cls._tam_blue_speed_ms(obs, env, blue_id)
        rpy = cls._tam_blue_rpy_rad(obs, env, blue_id)
        roll_deg = float(np.rad2deg(rpy[0]))
        pitch_deg = float(np.rad2deg(rpy[1]))
        # Estimate vertical speed from env sim if available
        vs_ms = 0.0
        if env is not None and blue_id and hasattr(env, "blue_planes"):
            sim = env.blue_planes.get(blue_id)
            if sim is not None and sim.is_alive:
                vs_ms = float(sim.get_velocity()[2])
        sim_time = 0.0
        if env is not None:
            sim_time = float(getattr(env, "current_step", 0)) * float(getattr(env, "env_dt", 0.2))

        # ---- Source-aware target acquisition ----
        source_name, target, _tidx = cls._tam_obs_source_and_target(obs)

        # ---- Stability gate conditions ----
        cold_start = sim_time < 40.0
        speed_low = speed_ms < 180.0
        alt_low = alt_m < 3500.0
        roll_high = abs(roll_deg) > 45.0
        pitch_high = abs(pitch_deg) > 30.0
        descending_fast = vs_ms < -20.0
        in_stability = cold_start or speed_low or alt_low or roll_high or pitch_high or descending_fast

        throttle = 1.0  # always max during stability; pursuit may reduce
        aileron = 0.0
        elevator = 0.0
        rudder = 0.0

        if in_stability:
            # ---- L1: Flight-stability controller ----
            # Target pitch: climb to survive
            if cold_start or speed_low:
                target_pitch_deg = 10.0
            elif alt_low or descending_fast:
                target_pitch_deg = 15.0
            else:
                target_pitch_deg = 5.0
            pitch_error_rad = np.deg2rad(target_pitch_deg) - rpy[1]
            # F16: negative elevator = nose up
            # Damping: descending → need more nose-up (more negative elevator)
            damping = 0.1 * vs_ms if vs_ms < 0 else 0.0
            elevator = float(np.clip(-0.5 * pitch_error_rad + damping, -0.35, 0.35))
            # Roll leveling: negative aileron for positive roll in F16?
            # From calibration: F16 positive aileron → positive roll
            if abs(roll_deg) > 10.0:
                aileron = float(np.clip(-0.02 * roll_deg, -0.25, 0.25))
        else:
            # ---- L2: Combat pursuit controller ----
            throttle = 0.9
            if target is not None:
                if source_name == "enemy_geo_states":
                    # Layout: [0]=rel_speed, [1]=delta_h/10000, [2]=dist/40000, [3]=ATA/π, [4]=AA/π
                    distance = cls._distance(target)
                    throttle = 1.0 if distance > 0.3 else 0.8
                    # ATA → lateral pursuit (limited)
                    if target.size >= 4:
                        aileron = float(np.clip(target[3] * 0.4, -0.25, 0.25))
                    # delta_h → elevator (negative = nose up, positive target[1] = target above)
                    if target.size >= 2:
                        elevator = float(np.clip(-target[1] * 1.5, -0.25, 0.25))
                elif source_name == "enemy_states":
                    # 11-dim body-frame: [dx,dy,dz_body, AO,TA,R, Vtgt, sin(roll),cos(roll),sin(pitch),cos(pitch)]
                    distance = cls._distance(target)
                    throttle = 1.0 if distance > 0.3 else 0.8
                    if target.size >= 3:
                        aileron = float(np.clip(target[1] * 0.3, -0.25, 0.25))
                    if target.size >= 3:
                        elevator = float(np.clip(-target[2] * 0.3, -0.25, 0.25))
            # Small pitch-up bias for safety
            elevator = float(np.clip(elevator - 0.02, -0.30, 0.30))

        return cls._clip_action([throttle, aileron, elevator, rudder])

    LOST_TARGET_TURN_BACK_LIMIT = 50  # env steps before giving up

    def _greedy_fsm_action(
        self, obs: dict, agent_index: int = 0,
        assigned_targets: set[int] | None = None,
        ownship: dict | None = None,
    ) -> tuple[np.ndarray, str]:
        if not obs:
            return np.array([0.0, 0.0, 0.3], dtype=np.float32), "missing_obs"

        assigned_targets = assigned_targets or set()
        if self._scalar(obs.get("missile_warning", 0.0)) > 0.0:
            self.lost_target_steps.pop(agent_index, None)
            heading = self._fallback_heading(obs, agent_index, scale=0.8)
            return self._clip_action([0.6, heading, 1.0]), "evade"

        altitude = self._altitude_value(obs)
        if altitude is not None and altitude < 0.2:
            self.lost_target_steps.pop(agent_index, None)
            return self._clip_action([0.7, 0.0, 0.8]), "recover_altitude"

        enemy_states, source_name = self._get_attack_targets(obs)

        # Turn-back: lost target but have recent memory
        lost_steps = self.lost_target_steps.get(agent_index, 0)
        visible_target, _visible_idx = self._select_nearest_target(obs)
        has_visible = visible_target is not None

        if not has_visible and lost_steps > 0:
            lost_steps += 1
            self.lost_target_steps[agent_index] = lost_steps
            if lost_steps <= self.LOST_TARGET_TURN_BACK_LIMIT:
                return self._turn_back_action(obs, agent_index, ownship), "turn_back"

        # Target persistence: prefer last-targeted slot if still visible
        last_slot = self.last_targets.get(agent_index)
        if last_slot is not None and enemy_states.shape[0] > 0:
            visible = self._visible_mask(obs, enemy_states.shape[0])
            if (last_slot < enemy_states.shape[0] and visible[last_slot]
                    and self._state_is_valid(enemy_states[last_slot])):
                action = self._attack_action_from_obs(
                    obs, enemy_states[last_slot], source_name, agent_index)
                # Record distance and reset lost counter
                self.last_target_distances[agent_index] = self._distance(
                    enemy_states[last_slot])
                self.lost_target_steps[agent_index] = 0
                return action, "attack_nearest"

        target, target_idx = self._select_mav_target(obs, assigned_targets)
        if target is not None:
            action = self._attack_action_from_obs(
                obs, target, source_name, agent_index)
            self.last_targets[agent_index] = target_idx
            self.last_target_distances[agent_index] = self._distance(target)
            self.lost_target_steps[agent_index] = 0
            return action, "attack_mav_priority"

        target, target_idx = self._select_nearest_target(obs, assigned_targets)
        if target is not None:
            action = self._attack_action_from_obs(
                obs, target, source_name, agent_index)
            self.last_targets[agent_index] = target_idx
            self.last_target_distances[agent_index] = self._distance(target)
            self.lost_target_steps[agent_index] = 0
            return action, "attack_nearest"

        # No visible target → start counting lost steps if we had a target
        if self.last_targets.get(agent_index) is not None:
            self.lost_target_steps[agent_index] = (
                self.lost_target_steps.get(agent_index, 0) + 1)
            if self.lost_target_steps[agent_index] <= self.LOST_TARGET_TURN_BACK_LIMIT:
                return self._turn_back_action(obs, agent_index, ownship), "turn_back"

        self.last_targets.pop(agent_index, None)
        self.lost_target_steps.pop(agent_index, None)
        self.last_target_distances.pop(agent_index, None)
        return self._search_acquire_action(obs, agent_index, ownship), "search_acquire"

    @classmethod
    def _turn_back_action(
        cls, obs_agent: dict, agent_index: int = 0, ownship: dict | None = None
    ) -> np.ndarray:
        """Conservative turn-back when target is lost after close approach.

        Executes ~90° turn (heading +0.5 or -0.5) to sweep back toward
        the last known target direction.  Direction alternates per agent.
        """
        current_heading = cls._own_heading_norm(obs_agent, ownship)
        direction = 0.5 if agent_index % 2 == 0 else -0.5
        heading = _wrap_heading_norm(current_heading + direction)
        return cls._clip_action([0.05, heading, 0.8])

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
                                agent_index: int = 0,
                                ownship: dict | None = None) -> np.ndarray:
        """Keep current heading + minimal deconfliction offset at high speed.

        The env action[1] is an *absolute* target heading (action[1] * pi rad).
        Search-acquire must preserve the current heading so blue continues
        toward the red formation instead of turning to absolute 0° (north).
        """
        base_heading = cls._own_heading_norm(obs_agent or {}, ownship)
        boundary_heading = cls._boundary_return_heading_norm(ownship)
        if boundary_heading is not None:
            base_heading = boundary_heading
        offset = 0.02 if agent_index % 2 == 0 else -0.02
        heading = _wrap_heading_norm(base_heading + offset)
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
        target_state, _target_idx = target
        if target_state is not None and target_state.size >= 2:
            return float(np.clip(np.sign(float(target_state[1])) * scale, -1.0, 1.0))
        return scale if agent_index % 2 == 0 else -scale

    @classmethod
    def _own_heading_norm(cls, obs_agent: dict, ownship: dict | None = None) -> float:
        if ownship:
            heading_norm = ownship.get("heading_norm")
            if heading_norm is not None and np.isfinite(float(heading_norm)):
                return _wrap_heading_norm(float(heading_norm))
        return cls._get_current_heading_norm(obs_agent)

    @classmethod
    def _boundary_return_heading_norm(cls, ownship: dict | None) -> float | None:
        if not ownship:
            return None
        position = ownship.get("position")
        center = ownship.get("blue_center_position")
        if position is None or center is None:
            return None
        rel = np.asarray(center, dtype=np.float32)[:2] - np.asarray(position, dtype=np.float32)[:2]
        distance = float(np.linalg.norm(rel))
        if distance < 12000.0:
            return None
        north, east = float(rel[0]), float(rel[1])
        return _wrap_heading_norm(float(np.arctan2(east, north) / np.pi))

    @classmethod
    def _select_mav_target(
        cls, obs: dict, assigned_targets: set[int] | None = None
    ) -> tuple[np.ndarray | None, int | None]:
        enemy_states = cls._enemy_states(obs)
        if enemy_states.shape[0] == 0:
            return None, None

        role_indices = cls._mav_indices(obs.get("enemy_roles", None))
        if not role_indices:
            role_indices = cls._mav_indices(obs.get("enemy_types", None))
        if not role_indices:
            return None, None

        assigned_targets = assigned_targets or set()
        visible = cls._visible_mask(obs, enemy_states.shape[0])
        candidates: list[tuple[float, np.ndarray]] = []
        for idx in role_indices:
            if idx in assigned_targets:
                continue
            if idx >= enemy_states.shape[0] or not visible[idx]:
                continue
            state = enemy_states[idx]
            if cls._state_is_valid(state):
                candidates.append((cls._distance(state), state, idx))
        if not candidates:
            return None, None
        _, state, idx = min(candidates, key=lambda item: item[0])
        return state, idx

    @classmethod
    def _select_nearest_target(
        cls, obs: dict, assigned_targets: set[int] | None = None
    ) -> tuple[np.ndarray | None, int | None]:
        enemy_states = cls._enemy_states(obs)
        if enemy_states.shape[0] == 0:
            return None, None
        visible = cls._visible_mask(obs, enemy_states.shape[0])
        assigned_targets = assigned_targets or set()
        candidates: list[tuple[float, np.ndarray, int]] = []
        fallback_candidates: list[tuple[float, np.ndarray, int]] = []
        for idx, state in enumerate(enemy_states):
            if not visible[idx] or not cls._state_is_valid(state):
                continue
            item = (cls._distance(state), state, idx)
            fallback_candidates.append(item)
            if idx not in assigned_targets:
                candidates.append(item)
        if not candidates:
            candidates = fallback_candidates
        if not candidates:
            return None, None
        _, state, idx = min(candidates, key=lambda item: item[0])
        return state, idx

    def _env_engaged_target_slots(self, env) -> set[int]:
        if env is None or not hasattr(env, "refresh_engaged_targets"):
            return set()
        try:
            engaged = env.refresh_engaged_targets()
            self.used_env_refresh_engaged_targets = True
        except Exception:
            return set()
        if isinstance(engaged, dict):
            values = engaged.values()
        else:
            values = engaged or []
        slots: set[int] = set()
        for value in values:
            if isinstance(value, str) and value.startswith("red_"):
                try:
                    slots.add(int(value.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
            elif isinstance(value, (int, np.integer)):
                slots.add(int(value))
        return slots

    def _env_blue_own_kinematics(self, env) -> dict:
        if env is None or not hasattr(env, "get_blue_own_kinematics"):
            return {}
        try:
            data = env.get_blue_own_kinematics()
        except Exception:
            return {}
        self.used_env_own_kinematics = True
        return data or {}

    def _env_blue_own_positions(self, env) -> dict:
        if env is None or not hasattr(env, "get_blue_own_positions"):
            return {}
        try:
            data = env.get_blue_own_positions()
        except Exception:
            return {}
        self.used_env_own_positions = True
        return data or {}

    @staticmethod
    def _ownship_context(
        blue_id: str, own_kinematics: dict, own_positions: dict
    ) -> dict:
        context: dict = {}
        kin = own_kinematics.get(blue_id, {}) if own_kinematics else {}
        if kin:
            if "heading" in kin:
                context["heading_norm"] = float(kin["heading"]) / np.pi
            if "position" in kin:
                context["position"] = np.asarray(kin["position"], dtype=np.float32)
        if blue_id in own_positions:
            context["position"] = np.asarray(own_positions[blue_id], dtype=np.float32)
        positions = [
            np.asarray(pos, dtype=np.float32)
            for pos in (own_positions or {}).values()
            if np.asarray(pos).size >= 2
        ]
        if positions:
            context["blue_center_position"] = np.mean(np.stack(positions), axis=0)
        return context

    @staticmethod
    def _enemy_states(obs: dict) -> np.ndarray:
        for key in ("enemy_states", "enemy_geo_states"):
            arr = np.asarray(obs.get(key, []), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return np.nan_to_num(arr, nan=0.0)
        return np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _get_attack_targets(obs: dict) -> tuple[np.ndarray, str]:
        """Return (target_states, source_name).

        source_name is "enemy_states" for BRMA body-frame relative vectors
        (target[1] can be used as signed lateral correction), or
        "enemy_geo_states" for V2 geometric vectors (no signed bearing).
        """
        for key in ("enemy_states", "enemy_geo_states"):
            arr = np.asarray(obs.get(key, []), dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return np.nan_to_num(arr, nan=0.0), key
        return np.zeros((0, 0), dtype=np.float32), "none"

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
                                 source_name: str = "enemy_states",
                                 agent_index: int = 0) -> np.ndarray:
        """Attack with absolute-heading-aware correction.

        target source:
        - enemy_states (BRMA body-frame): target[1] = Δy (right+) →
          signed lateral correction can be used.
        - enemy_geo_states (V2 geometric): target[1] is NOT body-frame
          lateral — no signed bearing available.  Falls back to holding
          current heading.

        env.py maps action[1] * pi → absolute target heading.
        """
        current_heading = cls._get_current_heading_norm(obs_agent)
        pitch = float(target[2]) * 2.0 if target.size >= 3 else 0.0

        if source_name == "enemy_states":
            # target[1] > 0 → target right → positive heading correction
            bearing_correction = float(target[1]) * 0.3 if target.size >= 2 else 0.0
        else:
            # enemy_geo_states: no signed bearing available;
            # hold current heading, close distance
            bearing_correction = 0.0

        heading = _wrap_heading_norm(current_heading + bearing_correction)
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
