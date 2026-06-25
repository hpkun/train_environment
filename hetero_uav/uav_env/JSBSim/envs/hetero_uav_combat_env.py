"""Minimal MAV/UAV heterogeneous extension of the BRMA environment."""

from __future__ import annotations

from copy import deepcopy

import gymnasium
import numpy as np

from ..env import UavCombatEnv
from ..utils import get2d_AO_TA_R

FT_PER_M = 1.0 / 0.3048
FPS_PER_MPS = 1.0 / 0.3048
TYPE_VOCAB = ["mav", "attack_uav", "scout_uav", "interceptor_uav"]
ROLE_VOCAB = ["mav", "attack_uav", "scout_uav", "interceptor_uav"]


def _type_onehot(type_name: str) -> np.ndarray:
    vec = np.zeros(len(TYPE_VOCAB), dtype=np.float32)
    if type_name in TYPE_VOCAB:
        vec[TYPE_VOCAB.index(type_name)] = 1.0
    return vec


def _role_onehot(role_name: str) -> np.ndarray:
    vec = np.zeros(len(ROLE_VOCAB), dtype=np.float32)
    if role_name in ROLE_VOCAB:
        vec[ROLE_VOCAB.index(role_name)] = 1.0
    return vec


def _metadata_matrix(agent_ids: list[str], values: dict[str, str], kind: str) -> np.ndarray:
    width = len(TYPE_VOCAB) if kind == "type" else len(ROLE_VOCAB)
    if not agent_ids:
        return np.zeros((0, width), dtype=np.float32)
    onehot = _type_onehot if kind == "type" else _role_onehot
    return np.stack([onehot(values.get(aid, "")) for aid in agent_ids], axis=0).astype(np.float32)

DEFAULT_AIRCRAFT_TYPE_PARAMS = {
    "mav": {
        "aircraft_model": "A-4",
        "role": "mav",
        "num_missiles": 0,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "attack_uav": {
        "aircraft_model": "f16",
        "role": "attack_uav",
        "num_missiles": 2,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "scout_uav": {
        "aircraft_model": "f16",
        "role": "scout_uav",
        "num_missiles": 0,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
    "interceptor_uav": {
        "aircraft_model": "f16",
        "role": "interceptor_uav",
        "num_missiles": 2,
        "init_altitude_offset_m": 0.0,
        "init_speed_offset_mps": 0.0,
    },
}

HAPPO_REF_V0_REWARD_COMPONENT_KEYS = (
    "mav_survival",
    "mav_support",
    "mav_attack",
    "mav_dodge",
    "uav_attack_window",
    "uav_fire",
    "uav_hit",
    "uav_dodge",
    "event",
    "safety",
    "death_penalty",
)


class HeteroUavCombatEnv(UavCombatEnv):
    """BRMA environment with per-agent aircraft model, role, and missile count.

    This first heterogeneous version deliberately preserves the original BRMA
    observation, reward, missile, action, PID, and termination logic.
    """

    def __init__(
        self,
        *args,
        red_agent_types: list[str] | None = None,
        blue_agent_types: list[str] | None = None,
        aircraft_type_params: dict | None = None,
        observation_mode: str = "brma_sensor",
        uav_direct_observation_range_m: float = 10000.0,
        mav_observation_range_m: float = 80000.0,
        action_trim_by_role: dict | None = None,
        action_trim_by_type: dict | None = None,
        action_trim_by_agent: dict | None = None,
        hetero_reward_mode: str = "brma_legacy",
        **kwargs,
    ):
        self._initial_states = kwargs.pop("initial_states", None) or {}
        if hetero_reward_mode not in {"brma_legacy", "minimal_v1", "role_v1", "happo_ref_v0", "paper_role_reward_v1"}:
            raise ValueError(f"unknown hetero_reward_mode: {hetero_reward_mode}")
        self.hetero_reward_mode = hetero_reward_mode
        self._tam_reward_scale = float(kwargs.pop("tam_reward_scale", 0.05))
        # Cached per-step obs for reward overlay (minimal_v1 / role_v1)
        self._last_step_obs: dict = {}
        # First-death detection for MAV — penalize once per episode
        self._mav_death_penalized: bool = False
        # First-death detection per UAV (role_v1)
        self._uav_death_penalized: set[str] = set()
        if observation_mode not in {"brma_sensor", "mav_shared_geo"}:
            raise ValueError(f"unknown observation_mode: {observation_mode}")
        self.observation_mode = observation_mode
        self.uav_direct_observation_range_m = float(uav_direct_observation_range_m)
        self.mav_observation_range_m = float(mav_observation_range_m)
        self.action_trim_by_role = self._normalize_action_trim_map(action_trim_by_role)
        self.action_trim_by_type = self._normalize_action_trim_map(action_trim_by_type)
        self.action_trim_by_agent = self._normalize_action_trim_map(action_trim_by_agent)
        self.action_trim_enabled = True
        self._last_action_trim_applied: dict[str, list[float]] = {}
        self._last_effective_actions: dict[str, list[float]] = {}
        super().__init__(*args, **kwargs)
        self.aircraft_type_params = deepcopy(DEFAULT_AIRCRAFT_TYPE_PARAMS)
        if aircraft_type_params:
            for name, params in aircraft_type_params.items():
                merged = dict(self.aircraft_type_params.get(name, {}))
                merged.update(params or {})
                self.aircraft_type_params[name] = merged

        self.red_agent_types = self._fit_agent_types(
            red_agent_types, self.max_num_red, ["mav", "attack_uav"]
        )
        self.blue_agent_types = self._fit_agent_types(
            blue_agent_types, self.max_num_blue, ["attack_uav", "attack_uav"]
        )
        self.agent_types: dict[str, str] = {}
        self.agent_roles: dict[str, str] = {}
        self.agent_models: dict[str, str] = {}
        self._refresh_agent_metadata()
        self._extend_hetero_observation_space()

    @staticmethod
    def _normalize_action_trim_map(values: dict | None) -> dict[str, np.ndarray]:
        if not values:
            return {}
        out: dict[str, np.ndarray] = {}
        for key, raw in values.items():
            if isinstance(raw, dict):
                trim = [
                    float(raw.get("pitch", 0.0)),
                    float(raw.get("heading", 0.0)),
                    float(raw.get("speed", 0.0)),
                ]
            else:
                trim = list(raw)
                if len(trim) != 3:
                    raise ValueError(f"action trim for {key!r} must have 3 values")
            out[str(key)] = np.asarray(trim, dtype=np.float32)
        return out

    def set_action_trim_enabled(self, enabled: bool) -> None:
        self.action_trim_enabled = bool(enabled)

    def _action_trim_for_agent(self, agent_id: str) -> np.ndarray:
        if not self.action_trim_enabled:
            return np.zeros(3, dtype=np.float32)
        if agent_id in self.action_trim_by_agent:
            return self.action_trim_by_agent[agent_id]
        role = self.agent_roles.get(agent_id, "")
        if role in self.action_trim_by_role:
            return self.action_trim_by_role[role]
        type_name = self.agent_types.get(agent_id, "")
        if type_name in self.action_trim_by_type:
            return self.action_trim_by_type[type_name]
        return np.zeros(3, dtype=np.float32)

    def _apply_action_trim(self, actions: dict) -> dict:
        trimmed = dict(actions)
        self._last_action_trim_applied = {}
        self._last_effective_actions = {}
        for aid, action in actions.items():
            trim = self._action_trim_for_agent(aid)
            raw = np.asarray(action, dtype=np.float32)
            effective = np.clip(raw + trim, -1.0, 1.0).astype(np.float32)
            trimmed[aid] = effective
            self._last_action_trim_applied[aid] = [
                round(float(value), 6) for value in trim
            ]
            self._last_effective_actions[aid] = [
                round(float(value), 6) for value in effective
            ]
        return trimmed

    def step(self, actions: dict):
        trimmed = self._apply_action_trim(actions)
        obs, rewards, terminated, truncated, info = super().step(trimmed)
        if self.hetero_reward_mode in {"minimal_v1", "role_v1", "happo_ref_v0", "paper_role_reward_v1"}:
            self._last_step_obs = obs
        return obs, rewards, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._last_step_obs = {}
        self._mav_death_penalized = False
        self._uav_death_penalized = set()
        self._paper_reset_reward_state()
        self._paper_terminal_applied = False
        obs, info = super().reset(*args, **kwargs)
        if self.hetero_reward_mode in {"minimal_v1", "role_v1", "happo_ref_v0", "paper_role_reward_v1"}:
            self._last_step_obs = obs
        return obs, info

    def _extend_hetero_observation_space(self) -> None:
        metadata_spaces = {
            "ego_type": gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(len(TYPE_VOCAB),), dtype=np.float32),
            "ego_role": gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(len(ROLE_VOCAB),), dtype=np.float32),
        }

        for aid in self.blue_ids:
            spaces = dict(self.observation_space.spaces[aid].spaces)
            spaces.update(metadata_spaces)
            spaces["ally_types"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_blue - 1, len(TYPE_VOCAB)), dtype=np.float32)
            spaces["ally_roles"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_blue - 1, len(ROLE_VOCAB)), dtype=np.float32)
            spaces["enemy_types"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_red, len(TYPE_VOCAB)), dtype=np.float32)
            spaces["enemy_roles"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_red, len(ROLE_VOCAB)), dtype=np.float32)
            if self.observation_mode == "mav_shared_geo":
                self._add_mav_shared_geo_spaces(
                    spaces, self.max_num_blue - 1, self.max_num_red)
            self.observation_space.spaces[aid] = gymnasium.spaces.Dict(spaces)

        for aid in self.red_ids:
            spaces = dict(self.observation_space.spaces[aid].spaces)
            spaces.update(metadata_spaces)
            spaces["ally_types"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_red - 1, len(TYPE_VOCAB)), dtype=np.float32)
            spaces["ally_roles"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_red - 1, len(ROLE_VOCAB)), dtype=np.float32)
            spaces["enemy_types"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_blue, len(TYPE_VOCAB)), dtype=np.float32)
            spaces["enemy_roles"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_num_blue, len(ROLE_VOCAB)), dtype=np.float32)
            if self.observation_mode == "mav_shared_geo":
                self._add_mav_shared_geo_spaces(
                    spaces, self.max_num_red - 1, self.max_num_blue)
            self.observation_space.spaces[aid] = gymnasium.spaces.Dict(spaces)

    @staticmethod
    def _add_mav_shared_geo_spaces(spaces: dict, max_allies: int, max_enemies: int) -> None:
        spaces["ego_geo_state"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        spaces["ally_geo_states"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_allies, 5), dtype=np.float32)
        spaces["enemy_geo_states"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_enemies, 5), dtype=np.float32)
        spaces["ally_alive_mask"] = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(max_allies,), dtype=np.float32)
        spaces["enemy_alive_mask"] = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(max_enemies,), dtype=np.float32)
        spaces["enemy_observed_mask"] = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(max_enemies,), dtype=np.float32)
        spaces["enemy_track_source"] = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(max_enemies, 2), dtype=np.float32)

    @staticmethod
    def _fit_agent_types(values: list[str] | None, count: int, default: list[str]) -> list[str]:
        selected = list(values) if values is not None else list(default)
        if len(selected) < count:
            selected.extend([selected[-1] if selected else "attack_uav"] * (count - len(selected)))
        return selected[:count]

    def _refresh_agent_metadata(self) -> None:
        self.agent_types.clear()
        self.agent_roles.clear()
        self.agent_models.clear()
        for i, aid in enumerate(self.red_ids):
            self._set_agent_metadata(aid, self.red_agent_types[i])
        for i, aid in enumerate(self.blue_ids):
            self._set_agent_metadata(aid, self.blue_agent_types[i])

    def _set_agent_metadata(self, agent_id: str, type_name: str) -> None:
        params = self.aircraft_type_params.get(type_name, self.aircraft_type_params["attack_uav"])
        self.agent_types[agent_id] = type_name
        self.agent_roles[agent_id] = str(params.get("role", type_name))
        self.agent_models[agent_id] = str(params.get("aircraft_model", "f16"))

    def _get_agent_obs(self, agent_id: str) -> dict:
        obs = super()._get_agent_obs(agent_id)
        if agent_id.startswith("blue"):
            ally_ids = [aid for aid in self.blue_ids if aid != agent_id]
            enemy_ids = list(self.red_ids)
        else:
            ally_ids = [aid for aid in self.red_ids if aid != agent_id]
            enemy_ids = list(self.blue_ids)

        obs["ego_type"] = _type_onehot(self.agent_types.get(agent_id, ""))
        obs["ego_role"] = _role_onehot(self.agent_roles.get(agent_id, ""))
        obs["ally_types"] = _metadata_matrix(ally_ids, self.agent_types, "type")
        obs["ally_roles"] = _metadata_matrix(ally_ids, self.agent_roles, "role")
        obs["enemy_types"] = _metadata_matrix(enemy_ids, self.agent_types, "type")
        obs["enemy_roles"] = _metadata_matrix(enemy_ids, self.agent_roles, "role")
        if self.observation_mode == "mav_shared_geo":
            obs.update(self._build_mav_shared_geo_obs(agent_id, ally_ids, enemy_ids))
        return obs

    def _build_mav_shared_geo_obs(
        self, agent_id: str, ally_ids: list[str], enemy_ids: list[str]
    ) -> dict:
        ego_sim = self._get_sim(agent_id)
        max_allies = len(ally_ids)
        max_enemies = len(enemy_ids)
        ego_alive = ego_sim is not None and ego_sim.is_alive

        ego_geo_state = np.zeros(7, dtype=np.float32)
        ally_geo_states = np.zeros((max_allies, 5), dtype=np.float32)
        enemy_geo_states = np.zeros((max_enemies, 5), dtype=np.float32)
        ally_alive_mask = np.zeros(max_allies, dtype=np.float32)
        enemy_alive_mask = np.zeros(max_enemies, dtype=np.float32)
        enemy_observed_mask = np.zeros(max_enemies, dtype=np.float32)
        enemy_track_source = np.zeros((max_enemies, 2), dtype=np.float32)

        if not ego_alive:
            return {
                "ego_geo_state": ego_geo_state,
                "ally_geo_states": ally_geo_states,
                "enemy_geo_states": enemy_geo_states,
                "ally_alive_mask": ally_alive_mask,
                "enemy_alive_mask": enemy_alive_mask,
                "enemy_observed_mask": enemy_observed_mask,
                "enemy_track_source": enemy_track_source,
            }

        ego_geo_state = self._ego_geo_state(ego_sim)

        for i, ally_id in enumerate(ally_ids):
            ally_sim = self._get_sim(ally_id)
            if ally_sim is not None and ally_sim.is_alive:
                ally_alive_mask[i] = 1.0
                ally_geo_states[i] = self._relative_geo_state(ego_sim, ally_sim)

        mav_sim = self._get_red_mav_sim()
        ego_is_red = agent_id.startswith("red_")
        ego_is_mav = self.agent_roles.get(agent_id) == "mav"

        for i, enemy_id in enumerate(enemy_ids):
            enemy_sim = self._get_sim(enemy_id)
            if enemy_sim is None or not enemy_sim.is_alive:
                continue
            enemy_alive_mask[i] = 1.0

            own_direct = (
                self._distance_m(ego_sim, enemy_sim)
                <= self.uav_direct_observation_range_m
            )
            mav_shared = False
            if ego_is_red and not ego_is_mav and mav_sim is not None and mav_sim.is_alive:
                mav_shared = (
                    self._distance_m(mav_sim, enemy_sim)
                    <= self.mav_observation_range_m
                )
            if ego_is_red and ego_is_mav:
                own_direct = (
                    self._distance_m(ego_sim, enemy_sim)
                    <= self.mav_observation_range_m
                )

            if own_direct:
                enemy_geo_states[i] = self._relative_geo_state(ego_sim, enemy_sim)
                enemy_observed_mask[i] = 1.0
                enemy_track_source[i] = np.array([1.0, 0.0], dtype=np.float32)
            elif mav_shared:
                enemy_geo_states[i] = self._relative_geo_state(ego_sim, enemy_sim)
                enemy_observed_mask[i] = 1.0
                enemy_track_source[i] = np.array([0.0, 1.0], dtype=np.float32)

        return {
            "ego_geo_state": ego_geo_state,
            "ally_geo_states": ally_geo_states,
            "enemy_geo_states": enemy_geo_states,
            "ally_alive_mask": ally_alive_mask,
            "enemy_alive_mask": enemy_alive_mask,
            "enemy_observed_mask": enemy_observed_mask,
            "enemy_track_source": enemy_track_source,
        }

    @staticmethod
    def _distance_m(a, b) -> float:
        return float(np.linalg.norm(a.get_position() - b.get_position()))

    def _get_red_mav_sim(self):
        for aid in self.red_ids:
            if self.agent_roles.get(aid) == "mav":
                return self.red_planes.get(aid)
        return None

    @staticmethod
    def _ego_geo_state(sim) -> np.ndarray:
        pos = sim.get_position()
        vel = sim.get_velocity()
        roll, pitch, yaw = sim.get_rpy()
        speed = float(np.linalg.norm(vel))
        return np.array([
            pos[0] / 40000.0,
            pos[1] / 40000.0,
            pos[2] / 10000.0,
            speed / 600.0,
            pitch / np.pi,
            yaw / np.pi,
            roll / np.pi,
        ], dtype=np.float32)

    @staticmethod
    def _relative_geo_state(observer, target) -> np.ndarray:
        obs_pos = observer.get_position()
        obs_vel = observer.get_velocity()
        tgt_pos = target.get_position()
        tgt_vel = target.get_velocity()
        obs_speed = float(np.linalg.norm(obs_vel))
        tgt_speed = float(np.linalg.norm(tgt_vel))
        distance = float(np.linalg.norm(tgt_pos - obs_pos))
        delta_h = float(tgt_pos[2] - obs_pos[2])

        obs_feat = np.array([
            obs_pos[0], obs_pos[1], -obs_pos[2],
            obs_vel[0], obs_vel[1], -obs_vel[2],
        ], dtype=np.float64)
        tgt_feat = np.array([
            tgt_pos[0], tgt_pos[1], -tgt_pos[2],
            tgt_vel[0], tgt_vel[1], -tgt_vel[2],
        ], dtype=np.float64)
        ata, aa, _range = get2d_AO_TA_R(obs_feat, tgt_feat)
        return np.array([
            (tgt_speed - obs_speed) / 600.0,
            delta_h / 10000.0,
            distance / 40000.0,
            ata / np.pi,
            aa / np.pi,
        ], dtype=np.float32)

    def _aircraft_model_for(self, agent_id: str, color: str, index: int) -> str:
        return self.agent_models.get(agent_id, "f16")

    def _num_missiles_for(self, agent_id: str) -> int:
        type_name = self.agent_types.get(agent_id, "attack_uav")
        params = self.aircraft_type_params.get(type_name, self.aircraft_type_params["attack_uav"])
        return int(params.get("num_missiles", self.num_missiles_per_plane))

    def _ensure_happo_ref_v0_reward_component_keys(self, components: dict) -> None:
        """Expose stable per-agent HAPPO reward component fields for audits."""
        for aid in self.agent_ids:
            comp = components.setdefault(aid, {})
            for key in HAPPO_REF_V0_REWARD_COMPONENT_KEYS:
                comp.setdefault(key, 0.0)

    PAPER_ROLE_REWARD_PROFILE = "brma_uav_tam_mav_event_v1"
    PAPER_MAV_SHARED_TRACK_LOOKBACK = 15  # env steps for MAV-guided fire/hit history

    def _build_launch_quality_record(self, shooter, target, range_m=None, target_selection=None):
        record = super()._build_launch_quality_record(shooter, target, range_m=range_m, target_selection=target_selection)
        if str(shooter.uid).startswith("red_") and self.agent_roles.get(shooter.uid, "") != "mav":
            tid = target.uid
            was_guided = self._paper_mav_shared_track_history.get((shooter.uid, tid), -999)
            record["mav_guided_at_launch"] = (self.current_step - was_guided <= self.PAPER_MAV_SHARED_TRACK_LOOKBACK)
            record["mav_guided_lookback_steps"] = self.PAPER_MAV_SHARED_TRACK_LOOKBACK
            record["mav_guided_source"] = "mav_shared_track_history" if record["mav_guided_at_launch"] else ""
            mav_observed = self._paper_mav_observed_history.get(tid, -999)
            record["mav_observed_at_launch"] = (self.current_step - mav_observed <= self.PAPER_MAV_SHARED_TRACK_LOOKBACK)
            record["mav_observed_source"] = "mav_observed_history" if record["mav_observed_at_launch"] else ""
        else:
            record["mav_guided_at_launch"] = False
            record["mav_guided_lookback_steps"] = self.PAPER_MAV_SHARED_TRACK_LOOKBACK
            record["mav_guided_source"] = ""
            record["mav_observed_at_launch"] = False
            record["mav_observed_source"] = ""
        return record

    def _paper_add_capped_reward(self, agent_id, key, delta, low, high):
        """Add delta to cumulative, return clipped actual amount added."""
        c = self._paper_reward_cumulative.setdefault(agent_id, {})
        old = c.get(key, 0.0)
        new = float(np.clip(old + delta, low, high))
        c[key] = new
        return new - old

    def _paper_reset_reward_state(self):
        self._paper_reward_cumulative = {}
        self._paper_out_zone_penalized = set()
        self._paper_mav_shared_track_history: dict[tuple, int] = {}
        self._paper_mav_observed_history: dict[str, float] = {}
        self._paper_reward_targets_current_step: dict = {}

    def _compute_brma_uav_tam_mav_event_v1(self, base_rewards, components, mav_id):
        """paper_role_reward_v1: BRMA flight + UAV keeps r_adv + MAV TAM dense + events + hetero terminal."""
        tam_scale = float(getattr(self, "_tam_reward_scale", 0.05))

        # ── Remove r_end for ALL red ──
        for rid in self.red_ids:
            old = components[rid].get("r_end", 0.0)
            if old != 0.0:
                components[rid]["r_end_raw_removed"] = float(old)
                base_rewards[rid] = base_rewards.get(rid, 0.0) - old
                components[rid]["r_end"] = 0.0

        # ── MAV only: remove r_adv. UAV keeps BRMA r_adv. ──
        if mav_id and mav_id in components:
            old = components[mav_id].get("r_adv", 0.0)
            if old != 0.0:
                components[mav_id]["r_adv_removed"] = float(old)
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) - old
                components[mav_id]["r_adv"] = 0.0

        # ── MAV dense: R_MAV_dense = 0.12*R_safety + 0.08*R_support + 0.01*I(alive) ──
        if mav_id and mav_id in self.red_planes:
            mav = self.red_planes[mav_id]
            comp = components.setdefault(mav_id, {})
            if mav.is_alive:
                mav_pos = mav.get_position()
                blue_sims = [s for s in self.blue_planes.values() if s.is_alive]
                # R_dist (D_danger=8000m, D_safe=15000m)
                if blue_sims:
                    d_min = min(np.linalg.norm(mav_pos - b.get_position()) for b in blue_sims)
                else:
                    d_min = 15000.0
                D_d, D_s = 8000.0, 15000.0
                if d_min < D_d: R_dist = -1.0
                elif d_min < D_s: R_dist = -1.0 + 1.2 * (d_min - D_d) / (D_s - D_d)
                else: R_dist = 0.2
                R_dist = float(np.clip(R_dist, -1.0, 0.2))
                # R_threat: missile_warning or blue has BRMA launch window on MAV
                R_threat = 0.0
                if mav.check_missile_warning() is not None:
                    R_threat -= 1.0
                for b in blue_sims:
                    m = self._missile_candidate_metrics(b, mav)
                    if m["range_ok"] and m["ao_ok"] and m["ta_ok"]:
                        R_threat -= 0.5; break
                R_threat = float(np.clip(R_threat, -1.0, 0.0))
                # R_aspect using BRMA-style A(alpha)*D(d)
                max_AD = 0.0
                for b in blue_sims:
                    d_b = float(np.linalg.norm(mav_pos - b.get_position()))
                    if d_b <= 0: continue
                    m = self._missile_candidate_metrics(b, mav)
                    alpha_deg = np.rad2deg(m["AO_rad"])
                    if alpha_deg <= 4: A_val = 1.0
                    elif alpha_deg < 35: A_val = 1.0 - (alpha_deg - 4.0) / 31.0
                    else: A_val = 0.0
                    D_val = 1.0 if d_b <= 10000 else np.exp(1.0 - d_b / 10000.0)
                    max_AD = max(max_AD, A_val * D_val)
                R_aspect = -max_AD
                R_safety = 0.5*R_dist + 0.3*R_threat + 0.2*R_aspect
                r_safe = 0.12 * R_safety
                comp["tam_mav_safety_raw"] = R_safety
                comp["tam_mav_safety_dist"] = R_dist
                comp["tam_mav_safety_threat"] = R_threat
                comp["tam_mav_safety_aspect"] = R_aspect
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_safe

                # R_support = 0.5*R_pos + 0.3*R_aware + 0.2*R_shared
                uav_poses = [self.red_planes[r].get_position() for r in self.red_ids
                             if r != mav_id and self.red_planes[r].is_alive]
                if uav_poses:
                    d_c = float(np.linalg.norm(mav_pos - np.mean(uav_poses, axis=0)))
                else:
                    d_c = 8000.0
                D_near, D_opt, D_far = 3000.0, 8000.0, 20000.0
                if d_c < D_near: R_pos = -0.5
                elif d_c < D_opt: R_pos = (d_c - D_near) / (D_opt - D_near)
                elif d_c < D_far: R_pos = 1.0 - (d_c - D_opt) / (D_far - D_opt)
                else: R_pos = -0.5
                n_blue = max(sum(1 for s in self.blue_planes.values() if s.is_alive), 1)
                mav_obs = np.asarray(self._last_step_obs.get(mav_id, {}).get("enemy_observed_mask", []), dtype=np.float32)
                S_observe = float(np.sum(mav_obs > 0.5)) / n_blue
                shared_total = 0.0
                n_uav = max(sum(1 for r in self.red_ids if r != mav_id and self.red_planes[r].is_alive), 1)
                for rid in self.red_ids:
                    if rid == mav_id: continue
                    src = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
                    if src.ndim == 2 and src.shape[1] >= 2:
                        shared_total += float(np.sum(src[:, 1] > 0.5))
                S_shared = shared_total / max(n_uav * n_blue, 1)
                R_aware = S_observe  # v1: observed ratio
                R_support = 0.5*R_pos + 0.3*R_aware + 0.2*S_shared
                r_sup = 0.08 * R_support
                comp["tam_mav_support_raw"] = R_support
                comp["tam_mav_support_pos"] = R_pos
                comp["tam_mav_support_aware"] = R_aware
                comp["tam_mav_support_shared"] = S_shared
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_sup
                # alive bonus
                r_alive = 0.01
                comp["tam_mav_alive_bonus"] = r_alive
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_alive

                comp["tam_mav_dense_reward"] = float(base_rewards.get(mav_id, 0.0) - max(0.0, base_rewards.get(mav_id, 0.0) - r_safe - r_sup - r_alive))

            # MAV death event (once)
            elif not mav.is_alive and not self._mav_death_penalized:
                self._mav_death_penalized = True
                comp["event_mav_death"] = -6.0
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) - 6.0
                for rid in self.red_ids:
                    if rid == mav_id: continue
                    if self.red_planes[rid].is_alive:
                        components.setdefault(rid, {})["event_mav_loss_team"] = -1.0
                        base_rewards[rid] = base_rewards.get(rid, 0.0) - 1.0

        # ── Events: UAV kill + UAV death/crash + out_zone ──
        step = getattr(self, "current_step", 0)
        for rid in self.red_ids:
            if self.agent_roles.get(rid, "") == "mav": continue
            comp = components.setdefault(rid, {})
            sim = self.red_planes.get(rid)
            # UAV kill from done hit records
            done_hits = [r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
                         if str(r.get("shooter_id","")) == str(rid) and str(r.get("raw_termination_reason","")) == "hit"]
            if done_hits:
                kill_r = 4.0 * len(done_hits)
                comp["event_uav_kill"] = kill_r
                comp["uav_hit_direct_count"] = sum(1 for r in done_hits if not bool(r.get("mav_guided_at_launch", False)))
                comp["uav_hit_mav_guided_count"] = sum(1 for r in done_hits if bool(r.get("mav_guided_at_launch", False)))
                base_rewards[rid] = base_rewards.get(rid, 0.0) + kill_r
                # team kill bonus for all alive red
                for rid2 in self.red_ids:
                    if self.red_planes.get(rid2) and self.red_planes[rid2].is_alive:
                        components.setdefault(rid2, {})["event_team_kill"] = components.setdefault(rid2, {}).get("event_team_kill", 0.0) + 0.5 * len(done_hits)
                        base_rewards[rid2] = base_rewards.get(rid2, 0.0) + 0.5 * len(done_hits)
            # UAV death/crash
            if sim is None or not sim.is_alive:
                if rid not in self._uav_death_penalized:
                    self._uav_death_penalized.add(rid)
                    crash = bool(rid in getattr(self, "_crashed_this_step", set()))
                    death_r = -5.0 if crash else -4.0
                    comp["event_uav_death" if not crash else "event_uav_crash"] = death_r
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + death_r
            # Out zone (once)
            if sim and sim.is_alive and rid not in self._paper_out_zone_penalized:
                pos = sim.get_position()
                if abs(float(pos[0])) > 40000 or abs(float(pos[1])) > 40000 or sim.get_geodetic()[2] > 10000:
                    self._paper_out_zone_penalized.add(rid)
                    comp["event_out_zone"] = -2.0
                    base_rewards[rid] = base_rewards.get(rid, 0.0) - 2.0

        # ── Hetero terminal (once per episode end) ──
        if not getattr(self, "_paper_terminal_applied", False):
            n_red = len(self.red_ids); n_blue = len(self.blue_ids)
            n_red_a = sum(1 for s in self.red_planes.values() if s.is_alive)
            n_blue_a = sum(1 for s in self.blue_planes.values() if s.is_alive)
            is_end = (n_blue_a == 0 or n_red_a == 0 or step >= self.max_steps)
            if is_end:
                self._paper_terminal_applied = True
                blue_d = n_blue - n_blue_a; red_d = n_red - n_red_a
                red_win = (n_blue_a == 0 and n_red_a > 0)
                red_fail = (n_red_a == 0)
                mutual = (n_blue_a == 0 and n_red_a == 0)
                timeout = (not red_win and not red_fail and not mutual)
                if red_win: R_win = 8.0
                elif red_fail: R_win = -8.0
                elif mutual: R_win = 0.0
                elif timeout: R_win = 4.0 * (blue_d / max(n_blue, 1) - red_d / max(n_red, 1))
                else: R_win = 0.0
                R_surv = 2.0 * (n_red_a / max(n_red, 1) - n_blue_a / max(n_blue, 1))
                mav_alive = bool(mav_id and self.red_planes.get(mav_id) and self.red_planes[mav_id].is_alive)
                R_mav = 0.0
                if not red_fail:
                    R_mav = 1.5 if mav_alive else -2.0
                R_term = R_win + R_surv + R_mav
                for rid in self.red_ids:
                    comp = components.setdefault(rid, {})
                    comp["terminal_hetero_raw"] = R_term
                    comp["terminal_win_component"] = R_win
                    comp["terminal_survival_component"] = R_surv
                    comp["terminal_mav_component"] = R_mav
                    comp["terminal_applied"] = 1
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + R_term

        # ── Log-only fields (active=0) ──
        for rid in self.red_ids:
            comp = components.setdefault(rid, {})
            comp["uav_attack"] = 0.0
            comp["uav_fire"] = 0.0
            comp["uav_hit"] = 0.0
            comp["uav_fire_log"] = 0.0
            comp["uav_attack_mav_shared_multiplier"] = 0
            comp["mav_assist"] = 0.0
            comp["uav_fire_direct_count"] = comp.get("uav_fire_direct_count", 0)
            comp["uav_fire_mav_guided_count"] = comp.get("uav_fire_mav_guided_count", 0)
            comp.setdefault("event_total", 0.0)

        # ── Final clipping [-10, 10] for red agents ──
        for rid in self.red_ids:
            pre = base_rewards.get(rid, 0.0)
            base_rewards[rid] = float(np.clip(pre, -10.0, 10.0))
            components.setdefault(rid, {})["reward_pre_clip"] = pre
            components.setdefault(rid, {})["reward_clip_delta"] = base_rewards[rid] - pre

        return base_rewards, components

    def _compute_rewards(self) -> tuple[dict, dict]:
        """Override to add minimal hetero role-aware overlay."""
        base_rewards, components = super()._compute_rewards()

        if self.hetero_reward_mode not in {"minimal_v1", "role_v1", "happo_ref_v0", "paper_role_reward_v1"}:
            return base_rewards, components

        mav_id = self.red_ids[0] if self.red_ids else None

        # ---- minimal_v1 overlay ----
        if self.hetero_reward_mode == "minimal_v1":
            for aid in self.agent_ids:
                comp = components.setdefault(aid, {})
                for key in ("r_mav_survival", "r_mav_death", "r_mav_support",
                            "r_shared_track_used", "r_attack_kill_bonus"):
                    comp.setdefault(key, 0.0)

            if mav_id and mav_id in self.red_planes:
                mav = self.red_planes[mav_id]
                r_mav_survival = 0.005 if mav.is_alive else 0.0
                if mav.is_alive:
                    self._mav_death_penalized = False
                    r_mav_death = 0.0
                elif not self._mav_death_penalized:
                    r_mav_death = -2.0
                    self._mav_death_penalized = True
                else:
                    r_mav_death = 0.0
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_mav_survival + r_mav_death
                components[mav_id]["r_mav_survival"] = float(r_mav_survival)
                components[mav_id]["r_mav_death"] = float(r_mav_death)

            for rid in self.red_ids:
                if rid not in self._last_step_obs:
                    continue
                o = self._last_step_obs[rid]
                shared_count = 0
                src = np.asarray(o.get("enemy_track_source", []), dtype=np.float32)
                if src.ndim == 2 and src.shape[1] >= 2:
                    shared_count = int(np.sum(src[:, 1] > 0.5))
                if shared_count > 0 and mav_id and mav_id != rid:
                    support = min(0.01 * shared_count, 0.05)
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + support
                    components[mav_id]["r_mav_support"] = float(support)
                    used = min(0.005 * shared_count, 0.02)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + used
                    components[rid]["r_shared_track_used"] = float(used)

            return base_rewards, components

        # ---- happo_ref_v0 overlay ----
        if self.hetero_reward_mode == "happo_ref_v0":
            self._ensure_happo_ref_v0_reward_component_keys(components)

            # Safety terms are deliberately small. The base BRMA reward still
            # carries the primary shaping; this mode only adds role signals.
            for rid in self.red_ids:
                sim = self.red_planes.get(rid)
                if sim is None:
                    continue
                comp = components.setdefault(rid, {})
                obs = self._last_step_obs.get(rid, {})
                safety = 0.0
                if sim.is_alive:
                    altitude = float(np.asarray(obs.get("altitude", [0.0])).reshape(-1)[0]) if obs else 0.0
                    velocity = np.asarray(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
                    speed = float(np.linalg.norm(velocity)) if velocity.size else 0.0
                    if 2500.0 <= altitude <= 12000.0:
                        safety += 0.002
                    else:
                        safety -= 0.003
                    if 120.0 <= speed <= 420.0:
                        safety += 0.002
                    else:
                        safety -= 0.003
                comp["safety"] = float(np.clip(safety, -0.01, 0.01))
                base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["safety"]

            if mav_id and mav_id in self.red_planes:
                mav = self.red_planes[mav_id]
                comp = components.setdefault(mav_id, {})
                if mav.is_alive:
                    comp["mav_survival"] = 0.01
                    self._mav_death_penalized = False
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + 0.01
                elif not self._mav_death_penalized:
                    comp["death_penalty"] = -4.0
                    self._mav_death_penalized = True
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) - 4.0

                support = 0.0
                mav_obs = self._last_step_obs.get(mav_id, {})
                observed = np.asarray(mav_obs.get("enemy_observed_mask", []), dtype=np.float32)
                if observed.size:
                    support += min(0.01 * float(np.sum(observed > 0.5)), 0.04)
                for rid in self.red_ids:
                    if rid == mav_id:
                        continue
                    uav_obs = self._last_step_obs.get(rid, {})
                    src = np.asarray(uav_obs.get("enemy_track_source", []), dtype=np.float32)
                    if src.ndim == 2 and src.shape[1] >= 2:
                        support += min(0.005 * float(np.sum(src[:, 1] > 0.5)), 0.02)
                support = float(np.clip(support, 0.0, 0.08))
                comp["mav_support"] = support
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + support

                team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
                if team_kills > 0 and mav.is_alive:
                    event = min(0.5 * team_kills, 1.0)
                    comp["event"] = float(event)
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + event

            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav":
                    continue
                sim = self.red_planes.get(rid)
                comp = components.setdefault(rid, {})
                if sim is None:
                    continue
                if not sim.is_alive:
                    if rid not in self._uav_death_penalized:
                        comp["death_penalty"] = -2.0
                        self._uav_death_penalized.add(rid)
                        base_rewards[rid] = base_rewards.get(rid, 0.0) - 2.0
                    continue

                obs = self._last_step_obs.get(rid, {})
                enemy_geo = np.asarray(obs.get("enemy_geo_states", []), dtype=np.float32)
                enemy_alive = np.asarray(obs.get("enemy_alive_mask", []), dtype=np.float32)
                window = 0.0
                if enemy_geo.ndim == 2 and enemy_alive.ndim == 1:
                    for i in range(min(enemy_geo.shape[0], enemy_alive.shape[0])):
                        if enemy_alive[i] < 0.5:
                            continue
                        distance_norm = abs(float(enemy_geo[i, 2]))
                        ata_norm = abs(float(enemy_geo[i, 3]))
                        aa_norm = abs(float(enemy_geo[i, 4]))
                        if distance_norm < 0.35:
                            window += 0.005
                        if ata_norm < 0.25:
                            window += 0.005
                        if aa_norm < 0.35:
                            window += 0.003
                window = float(np.clip(window, 0.0, 0.04))
                comp["uav_attack_window"] = window
                base_rewards[rid] = base_rewards.get(rid, 0.0) + window

                fired = int(self._missile_launch_counts.get(rid, 0))
                if fired > 0:
                    comp["uav_fire"] = min(0.02 * fired, 0.04)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["uav_fire"]

                kills = int(self._step_kill_count.get(rid, 0))
                if kills > 0:
                    comp["uav_hit"] = min(2.0 * kills, 4.0)
                    comp["event"] = min(1.0 * kills, 2.0)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["uav_hit"] + comp["event"]

                mw = np.asarray(obs.get("missile_warning", [0.0]), dtype=np.float32).reshape(-1)
                if mw.size and mw[0] > 0.5:
                    comp["uav_dodge"] = 0.005
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["uav_dodge"]

            return base_rewards, components

        # ---- paper_role_reward_v1: brma_uav_tam_mav_event_v1 ----
        if self.hetero_reward_mode == "paper_role_reward_v1":
            return self._compute_brma_uav_tam_mav_event_v1(
                base_rewards, components, mav_id)

        # role_v1 (default fallback)
        if self.hetero_reward_mode == "role_v1":
            ROLE_MAV_KEYS = [
            "r_role_mav_survival", "r_role_mav_death",
            "r_role_mav_support", "r_role_mav_team_contribution",
        ]
        ROLE_UAV_KEYS = [
            "r_role_uav_attack_window", "r_role_uav_kill_bonus",
            "r_role_uav_death_penalty", "r_role_uav_missile_warning",
        ]
        for aid in self.agent_ids:
            comp = components.setdefault(aid, {})
            role = self.agent_roles.get(aid, "")
            if role == "mav":
                for key in ROLE_MAV_KEYS:
                    comp.setdefault(key, 0.0)
            elif role == "attack_uav":
                for key in ROLE_UAV_KEYS:
                    comp.setdefault(key, 0.0)

        # --- A. MAV rewards ---
        if mav_id and mav_id in self.red_planes:
            mav = self.red_planes[mav_id]

            # A1. Survival (+0.01/step)
            if mav.is_alive:
                r = 0.01
                self._mav_death_penalized = False
            else:
                r = 0.0
            base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r
            components[mav_id]["r_role_mav_survival"] = float(r)

            # A2. Death penalty (-10, once)
            if not mav.is_alive and not self._mav_death_penalized:
                d = -10.0
                self._mav_death_penalized = True
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + d
                components[mav_id]["r_role_mav_death"] = float(d)
            else:
                components[mav_id]["r_role_mav_death"] = 0.0

        # A3. MAV support: bonus when MAV actually observes enemies (not just alive)
        if mav_id and mav_id in self._last_step_obs:
            o = self._last_step_obs.get(mav_id, {})
            observed_mask = np.asarray(o.get("enemy_observed_mask", []), dtype=np.float32)
            enemy_seen = int(np.sum(observed_mask > 0.5))
            support = min(0.005 * enemy_seen, 0.03)
            components[mav_id].setdefault("r_role_mav_support", 0.0)
            if support > 0:
                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + support
                components[mav_id]["r_role_mav_support"] = float(support)

            # A3b. Extra support when UAVs use MAV shared tracks
            for rid in self.red_ids:
                if rid == mav_id or rid not in self._last_step_obs:
                    continue
                uav_obs = self._last_step_obs[rid]
                src = np.asarray(uav_obs.get("enemy_track_source", []), dtype=np.float32)
                if src.ndim == 2 and src.shape[1] >= 2:
                    shared = int(np.sum(src[:, 1] > 0.5))
                    if shared > 0:
                        extra = min(0.005 * shared, 0.02)
                        base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + extra
                        components[mav_id]["r_role_mav_support"] = (
                            float(components[mav_id].get("r_role_mav_support", 0.0)) + extra
                        )

        # A4. MAV team contribution: when red kills, MAV alive gets bonus
        for aid in self.red_ids:
            kills = self._step_kill_count.get(aid, 0)
            if kills > 0 and mav_id and mav_id != aid:
                mav_sim = self.red_planes.get(mav_id)
                if mav_sim is not None and mav_sim.is_alive:
                    bonus = min(1.0 * kills, 5.0)
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + bonus
                    components[mav_id].setdefault("r_role_mav_team_contribution", 0.0)
                    components[mav_id]["r_role_mav_team_contribution"] = (
                        float(components[mav_id]["r_role_mav_team_contribution"]) + bonus
                    )

        # --- B. UAV rewards ---
        for aid in self.red_ids:
            role = self.agent_roles.get(aid, "")
            if role != "attack_uav":
                continue
            sim = self.red_planes.get(aid)
            if sim is None or not sim.is_alive:
                # UAV death penalty (once)
                if aid not in self._uav_death_penalized:
                    pid = -5.0
                    self._uav_death_penalized.add(aid)
                    base_rewards[aid] = base_rewards.get(aid, 0.0) + pid
                    components[aid]["r_role_uav_death_penalty"] = float(pid)
                continue

            # B1. Attack window shaping
            if aid in self._last_step_obs:
                uav_obs = self._last_step_obs[aid]
                ego_geo = np.asarray(uav_obs.get("ego_geo_state", []), dtype=np.float32)
                enemy_geo = np.asarray(uav_obs.get("enemy_geo_states", []), dtype=np.float32)
                enemy_alive = np.asarray(uav_obs.get("enemy_alive_mask", []), dtype=np.float32)

                window_reward = 0.0
                if enemy_geo.ndim == 2 and enemy_alive.ndim == 1:
                    for i in range(min(len(enemy_alive), enemy_geo.shape[0])):
                        if enemy_alive[i] < 0.5:
                            continue
                        eg = enemy_geo[i]
                        # eg = [speed_diff, delta_h, distance, ata/pi, aa/pi]
                        distance_norm = abs(float(eg[2]))  # distance/40000
                        ata_norm = abs(float(eg[3]))        # ata/pi
                        aa_norm = abs(float(eg[4]))         # aa/pi
                        # Reward if within reasonable engagement parameters
                        if distance_norm < 0.5 and ata_norm < 0.3:
                            window_reward += 0.005
                if window_reward > 0:
                    window_reward = min(window_reward, 0.03)
                    base_rewards[aid] = base_rewards.get(aid, 0.0) + window_reward
                    components[aid]["r_role_uav_attack_window"] = float(window_reward)
                else:
                    components[aid]["r_role_uav_attack_window"] = 0.0

            # B2. Kill bonus
            kills = self._step_kill_count.get(aid, 0)
            if kills > 0:
                kb = min(8.0 * kills, 10.0)
                base_rewards[aid] = base_rewards.get(aid, 0.0) + kb
                components[aid]["r_role_uav_kill_bonus"] = float(kb)
            else:
                components[aid]["r_role_uav_kill_bonus"] = 0.0

            # B3. Missile warning (light penalty)
            mw = 1.0
            if aid in self._last_step_obs:
                mw_arr = np.asarray(
                    self._last_step_obs[aid].get("missile_warning", [0.0]),
                    dtype=np.float32,
                ).ravel()
                if len(mw_arr) > 0 and mw_arr[0] > 0.5:
                    mw = -0.005
                else:
                    mw = 0.0
            components[aid]["r_role_uav_missile_warning"] = float(mw)
            if mw != 0.0:
                base_rewards[aid] = base_rewards.get(aid, 0.0) + mw

            return base_rewards, components

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = super()._get_info(reward_components)
        info["reward_mode"] = self.hetero_reward_mode
        if reward_components is not None:
            info["reward_components"] = {
                aid: dict(values) for aid, values in reward_components.items()
            }
        info["agent_types"] = dict(self.agent_types)
        info["agent_roles"] = dict(self.agent_roles)
        info["agent_models"] = dict(self.agent_models)
        info["observation_mode"] = self.observation_mode
        info["action_trim_enabled"] = bool(self.action_trim_enabled)
        info["action_trim_by_role"] = {
            key: value.tolist() for key, value in self.action_trim_by_role.items()
        }
        info["last_action_trim_applied"] = dict(self._last_action_trim_applied)
        info["last_effective_actions"] = dict(self._last_effective_actions)
        info["agent_init_offsets"] = {}
        for aid in self.agent_ids:
            info["agent_init_offsets"][aid] = self._init_offsets_for(aid)
        return info

    def _init_offsets_for(self, agent_id: str) -> dict:
        type_name = self.agent_types.get(agent_id, "attack_uav")
        params = self.aircraft_type_params.get(
            type_name, self.aircraft_type_params["attack_uav"])
        return {
            "altitude_offset_m": float(params.get("init_altitude_offset_m", 0.0)),
            "speed_offset_mps": float(params.get("init_speed_offset_mps", 0.0)),
        }

    def _make_init_state(self, color: str, index: int) -> dict:
        init = super()._make_init_state(color, index)

        agent_id = f"{color.lower()}_{index}"

        # ---- per-agent initial_states override (paper-aligned configs) ----
        override = self._initial_states.get(agent_id, {})
        if "lon" in override:
            init["ic\\long-gc-deg" if "ic\\long-gc-deg" in init
                 else "ic/long-gc-deg"] = float(override["lon"])
        if "lat" in override:
            init["ic\\lat-geod-deg" if "ic\\lat-geod-deg" in init
                 else "ic/lat-geod-deg"] = float(override["lat"])
        if "altitude_m" in override:
            alt_ft = float(override["altitude_m"]) * FT_PER_M
            if "ic\\h-sl-ft" in init:
                init["ic\\h-sl-ft"] = alt_ft
            elif "ic/h-sl-ft" in init:
                init["ic/h-sl-ft"] = alt_ft
        if "speed_mps" in override:
            speed_fps = float(override["speed_mps"]) * FPS_PER_MPS
            if "ic\\u-fps" in init:
                init["ic\\u-fps"] = speed_fps
            elif "ic/u-fps" in init:
                init["ic/u-fps"] = speed_fps
        if "yaw_deg" in override:
            if "ic\\psi-true-deg" in init:
                init["ic\\psi-true-deg"] = float(override["yaw_deg"])
            elif "ic/psi-true-deg" in init:
                init["ic/psi-true-deg"] = float(override["yaw_deg"])

        # ---- type-based offsets ----
        offsets = self._init_offsets_for(agent_id)
        alt_offset_m = offsets["altitude_offset_m"]
        speed_offset_mps = offsets["speed_offset_mps"]

        if alt_offset_m != 0.0:
            alt_offset_ft = alt_offset_m * FT_PER_M
            if "ic\\h-sl-ft" in init:
                init["ic\\h-sl-ft"] = float(init["ic\\h-sl-ft"]) + alt_offset_ft
            elif "ic/h-sl-ft" in init:
                init["ic/h-sl-ft"] = float(init["ic/h-sl-ft"]) + alt_offset_ft

        if speed_offset_mps != 0.0:
            speed_offset_fps = speed_offset_mps * FPS_PER_MPS
            if "ic\\u-fps" in init:
                init["ic\\u-fps"] = float(init["ic\\u-fps"]) + speed_offset_fps
            elif "ic/u-fps" in init:
                init["ic/u-fps"] = float(init["ic/u-fps"]) + speed_offset_fps

        return init


__all__ = [
    "HeteroUavCombatEnv",
    "ROLE_VOCAB",
    "TYPE_VOCAB",
    "_role_onehot",
    "_type_onehot",
]
