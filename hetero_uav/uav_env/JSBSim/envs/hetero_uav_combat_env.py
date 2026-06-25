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

    PAPER_ROLE_REWARD_PROFILE = "brma_flight_tam_role_aligned_v1"
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

        # ---- paper_role_reward_v1: brma_flight_tam_role_aligned_v1 ----
        if self.hetero_reward_mode == "paper_role_reward_v1":
            tam_scale = float(getattr(self, "_tam_reward_scale", 0.05))

            # ── Remove BRMA r_adv (situation) and r_end (terminal) for all red ──
            for rid in self.red_ids:
                for key_del, log_key in [("r_adv", "r_adv_removed"), ("r_end", "r_end_raw_removed")]:
                    old = components[rid].get(key_del, 0.0)
                    if old != 0.0:
                        components[rid][log_key] = float(old)
                        base_rewards[rid] = base_rewards.get(rid, 0.0) - old
                        components[rid][key_del] = 0.0

            # ── Set up tam_* component keys ──
            for rid in self.red_ids:
                comp = components.setdefault(rid, {})
                role = self.agent_roles.get(rid, "")
                if role == "mav":
                    for k in ("tam_mav_safety_raw","tam_mav_safety","tam_mav_support_raw",
                              "tam_mav_support","tam_mav_event_raw","tam_mav_event",
                              "tam_mav_death_event","tam_mav_team_contribution_event",
                              "mav_out_zone_log"):
                        comp.setdefault(k, 0.0)
                else:
                    for k in ("tam_uav_angle_raw","tam_uav_angle","tam_uav_distance_raw",
                              "tam_uav_distance","tam_uav_dodge_raw","tam_uav_dodge",
                              "tam_uav_event_raw","tam_uav_event",
                              "tam_uav_kill_event_count","tam_uav_death_event",
                              "tam_uav_out_zone_log","uav_fire_log",
                              "uav_fire_direct_count","uav_fire_mav_guided_count",
                              "uav_hit_direct_count","uav_hit_mav_guided_count",
                              "tam_uav_reward_target_id","tam_uav_reward_target_track_source",
                              "tam_uav_ATA_3d_rad","tam_uav_AA_3d_rad","tam_uav_range_3d_m"):
                        comp.setdefault(k, 0.0)
                # Active reward summary
                comp.setdefault("active_brma_flight", float(base_rewards.get(rid, 0.0)))

            # ── Update MAV shared track history ──
            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav": continue
                uav_obs = self._last_step_obs.get(rid, {})
                src = np.asarray(uav_obs.get("enemy_track_source", []), dtype=np.float32)
                for bi, bid in enumerate(self.blue_ids):
                    if bi < src.shape[0] and src[bi, 1] > 0.5:
                        self._paper_mav_shared_track_history[(rid, bid)] = self.current_step
            if mav_id:
                mav_obs = self._last_step_obs.get(mav_id, {})
                obs_mask = np.asarray(mav_obs.get("enemy_observed_mask", []), dtype=np.float32)
                for bi, bid in enumerate(self.blue_ids):
                    if bi < obs_mask.size and obs_mask[bi] > 0.5:
                        self._paper_mav_observed_history[bid] = self.current_step

            # ── MAV: TAM R_safety + R_support + R_event ──
            if mav_id and mav_id in self.red_planes:
                mav = self.red_planes[mav_id]
                comp = components.setdefault(mav_id, {})
                if mav.is_alive:
                    mav_pos = mav.get_position()
                    blue_sims = [s for s in self.blue_planes.values() if s.is_alive]
                    # R_safety = 0.5*R_dist + 0.3*R_threat + 0.2*R_aspect
                    if blue_sims:
                        d_MB_km = min(np.linalg.norm(mav_pos - b.get_position()) for b in blue_sims) / 1000.0
                    else:
                        d_MB_km = 40.0
                    if d_MB_km < 8: S_dist = -1.0
                    elif d_MB_km < 15: S_dist = -1.0 + 2.0*(d_MB_km-8)/7.0
                    elif d_MB_km <= 30: S_dist = 1.0
                    elif d_MB_km <= 40: S_dist = 1.0 - 1.5*(d_MB_km-30)/10.0
                    else: S_dist = -1.0
                    S_dist = float(np.clip(S_dist, -1.0, 1.0))
                    mw = int(mav.check_missile_warning() is not None)
                    blue_can_launch = 0
                    for b in blue_sims:
                        if self._build_launch_geometry_3d(b, mav).get("launch_geometry_ok_3d", False):
                            blue_can_launch = 1; break
                    S_threat = -1.0 if (mw or blue_can_launch) else 0.5
                    S_AO_vals = []; S_TA_vals = []
                    for b in blue_sims:
                        g = self._build_launch_geometry_3d(b, mav)
                        S_AO_vals.append(1.0 - np.clip(g.get("ATA_3d_rad", np.pi)/np.deg2rad(45), 0, 1))
                        S_TA_vals.append(np.clip((g.get("TA_3d_rad", 0)-np.pi/2)/(np.pi/2), 0, 1))
                    G_blue = max(0.5*a+0.5*t for a,t in zip(S_AO_vals, S_TA_vals)) if S_AO_vals else 0.0
                    S_aspect = 1.0 - 2.0*G_blue
                    R_safety_raw = 0.5*S_dist + 0.3*S_threat + 0.2*S_aspect
                    r_safety = tam_scale * R_safety_raw
                    comp["tam_mav_safety_raw"] = R_safety_raw
                    comp["tam_mav_safety"] = r_safety
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_safety

                    # R_support = 0.6*R_pos + 0.4*R_aware
                    uav_poses = [self.red_planes[r].get_position() for r in self.red_ids
                                 if r != mav_id and self.red_planes[r].is_alive]
                    if uav_poses:
                        c_uav = np.mean(uav_poses, axis=0)
                        d_MU_km = float(np.linalg.norm(mav_pos - c_uav))/1000.0
                    else:
                        d_MU_km = 15.0
                    if d_MU_km < 5: S_pos = -0.2
                    elif d_MU_km < 8: S_pos = -0.2 + 1.2*(d_MU_km-5)/3.0
                    elif d_MU_km <= 22: S_pos = 1.0
                    elif d_MU_km <= 35: S_pos = 1.0 - 1.5*(d_MU_km-22)/13.0
                    else: S_pos = -1.0
                    S_pos = float(np.clip(S_pos, -1.0, 1.0))
                    n_blue_alive = max(sum(1 for s in self.blue_planes.values() if s.is_alive), 1)
                    mav_obs_mask = np.asarray(self._last_step_obs.get(mav_id, {}).get("enemy_observed_mask", []), dtype=np.float32)
                    S_observe = float(np.sum(mav_obs_mask > 0.5)) / n_blue_alive
                    shared_total = 0.0
                    n_uav_alive = max(sum(1 for r in self.red_ids if r != mav_id and self.red_planes[r].is_alive), 1)
                    for rid in self.red_ids:
                        if rid == mav_id: continue
                        src = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
                        if src.ndim == 2 and src.shape[1] >= 2:
                            shared_total += float(np.sum(src[:, 1] > 0.5))
                    S_shared = shared_total / max(n_uav_alive * n_blue_alive, 1)
                    guided_count = sum(
                        1 for r in self.red_ids if r != mav_id
                        and self._paper_reward_targets_current_step.get(r, {}).get("best_metrics", {}).get("S_mav", 0) > 0.5)
                    S_guided = guided_count / max(n_uav_alive, 1)
                    S_info_raw = 0.4*S_observe + 0.4*S_shared + 0.2*S_guided
                    S_aware = 2.0*S_info_raw - 1.0
                    R_support_raw = 0.6*S_pos + 0.4*S_aware
                    r_support = tam_scale * R_support_raw
                    comp["tam_mav_support_raw"] = R_support_raw
                    comp["tam_mav_support"] = r_support
                    comp["tam_mav_support_pos"] = S_pos
                    comp["tam_mav_support_aware"] = S_aware
                    comp["mav_support_observe_log"] = S_observe
                    comp["mav_support_shared_log"] = S_shared
                    comp["mav_support_guided_log"] = S_guided
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_support

                    # R_event = -I(death)*C_d + bounded team contribution
                    # Death handled below; assist from done hit records
                    for rid in self.red_ids:
                        if rid == mav_id: continue
                        done_hits = [r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
                                     if str(r.get("shooter_id","")) == str(rid) and str(r.get("raw_termination_reason","")) == "hit"]
                        for _ in done_hits:
                            raw_event = 200.0  # per kill, capped
                            r_event = tam_scale * raw_event
                            comp["tam_mav_team_contribution_event"] = comp.get("tam_mav_team_contribution_event", 0.0) + raw_event
                            capped = min(comp["tam_mav_team_contribution_event"], 200.0)
                            comp["tam_mav_event_raw"] = comp.get("tam_mav_event_raw", 0.0) + raw_event
                            comp["tam_mav_event"] = tam_scale * capped
                            base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_event
                elif not mav.is_alive:
                    if not self._mav_death_penalized:
                        self._mav_death_penalized = True
                        raw_death = -200.0
                        r_death = tam_scale * raw_death
                        comp["tam_mav_death_event"] = raw_death
                        comp["tam_mav_event_raw"] = comp.get("tam_mav_event_raw", 0.0) + raw_death
                        comp["tam_mav_event"] = tam_scale * comp.get("tam_mav_event_raw", 0.0)
                        base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_death
                comp["active_tam_mav"] = float(base_rewards.get(mav_id, 0.0) - comp.get("active_brma_flight", 0.0))

            # ── UAV: TAM R_A + R_D + R_DM + R_E ──
            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav": continue
                sim = self.red_planes.get(rid)
                comp = components.setdefault(rid, {})
                if sim is None or not sim.is_alive:
                    if rid not in self._uav_death_penalized:
                        self._uav_death_penalized.add(rid)
                        raw_death = -200.0
                        r_death = tam_scale * raw_death
                        comp["tam_uav_death_event"] = raw_death
                        comp["tam_uav_event_raw"] = comp.get("tam_uav_event_raw", 0.0) + raw_death
                        comp["tam_uav_event"] = tam_scale * comp.get("tam_uav_event_raw", 0.0)
                        base_rewards[rid] = base_rewards.get(rid, 0.0) + r_death
                    continue

                obs = self._last_step_obs.get(rid, {})
                enemy_geo = np.asarray(obs.get("enemy_geo_states", []), dtype=np.float32)
                enemy_alive = np.asarray(obs.get("enemy_alive_mask", []), dtype=np.float32)
                enemy_obs_mask = np.asarray(obs.get("enemy_observed_mask", []), dtype=np.float32)
                uav_src = np.asarray(obs.get("enemy_track_source", []), dtype=np.float32)

                # TAM R_A + R_D: select best observed target
                best_score, best_j, best_ata, best_aa, best_range_3d = -1e9, -1, 0.0, 0.0, 0.0
                if enemy_geo.ndim == 2 and enemy_alive.ndim == 1:
                    for j in range(min(enemy_geo.shape[0], enemy_alive.shape[0])):
                        if enemy_alive[j] < 0.5: continue
                        observed = (enemy_obs_mask.size > j and enemy_obs_mask[j] > 0.5)
                        if uav_src.ndim == 2 and uav_src.shape[1] >= 1 and not observed:
                            observed = uav_src[j, 0] > 0.5 or (uav_src.shape[1] >= 2 and uav_src[j, 1] > 0.5)
                        if not observed: continue
                        if j < len(self.blue_ids):
                            b_sim = self.blue_planes.get(self.blue_ids[j])
                            if b_sim and b_sim.is_alive:
                                g = self._build_launch_geometry_3d(sim, b_sim)
                                ATA = g["ATA_3d_rad"]; AA = g["TA_3d_rad"]
                            else:
                                ATA = abs(float(enemy_geo[j, 3])) * np.pi; AA = abs(float(enemy_geo[j, 4])) * np.pi
                        else:
                            ATA = abs(float(enemy_geo[j, 3])) * np.pi; AA = abs(float(enemy_geo[j, 4])) * np.pi
                        R_A = 1.0 - (ATA + AA) / np.pi
                        R = float(g.get("range_3d_m", abs(float(enemy_geo[j, 2])) * 40000)) if j < len(self.blue_ids) else abs(float(enemy_geo[j, 2])) * 40000
                        R_km = R / 1000.0
                        if R_km <= 5: R_D = 1.0
                        elif R_km < 10: R_D = np.exp(-0.921*(R_km - 5.0))
                        else: R_D = -1.0
                        score = 15.0 * R_A + 10.0 * R_D
                        if score > best_score:
                            best_score, best_j, best_ata, best_aa, best_range_3d = score, j, ATA, AA, R

                if best_j >= 0:
                    raw_angle = 15.0 * (1.0 - (best_ata + best_aa) / np.pi)
                    raw_dist = 10.0 * (1.0 if best_range_3d/1000.0 <= 5 else (np.exp(-0.921*(best_range_3d/1000.0-5.0)) if best_range_3d/1000.0 < 10 else -1.0))
                    r_angle = tam_scale * raw_angle
                    r_dist = tam_scale * raw_dist
                    comp["tam_uav_angle_raw"] = raw_angle; comp["tam_uav_angle"] = r_angle
                    comp["tam_uav_distance_raw"] = raw_dist; comp["tam_uav_distance"] = r_dist
                    comp["tam_uav_ATA_3d_rad"] = best_ata; comp["tam_uav_AA_3d_rad"] = best_aa
                    comp["tam_uav_range_3d_m"] = best_range_3d
                    comp["tam_uav_reward_target_id"] = self.blue_ids[best_j] if best_j < len(self.blue_ids) else ""
                    comp["tam_uav_reward_target_track_source"] = "direct" if (uav_src.ndim == 2 and uav_src.shape[1] >= 1 and uav_src[best_j, 0] > 0.5) else "mav_shared" if (uav_src.ndim == 2 and uav_src.shape[1] >= 2 and uav_src[best_j, 1] > 0.5) else "unknown"
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + r_angle + r_dist
                # R_DM = 0 for now
                comp["tam_uav_dodge_raw"] = 0.0; comp["tam_uav_dodge"] = 0.0
                comp["tam_uav_dodge_unavailable_reason"] = "missile_state_not_observable_or_not_stable"
                # uav_fire = 0 (log only)
                comp["uav_fire_log"] = 0.0
                # UAV hit event from done records
                done_hits = [r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
                             if str(r.get("shooter_id","")) == str(rid) and str(r.get("raw_termination_reason","")) == "hit"]
                if done_hits:
                    raw_kill = 200.0 * len(done_hits)
                    comp["tam_uav_kill_event_count"] = len(done_hits)
                    comp["tam_uav_event_raw"] = comp.get("tam_uav_event_raw", 0.0) + raw_kill
                    comp["tam_uav_event"] = tam_scale * comp.get("tam_uav_event_raw", 0.0)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + tam_scale * raw_kill
                # Log guided/direct hit counts (no extra active reward)
                guided_h = sum(1 for r in done_hits if bool(r.get("mav_guided_at_launch", False)))
                comp["uav_hit_direct_count"] = len(done_hits) - guided_h
                comp["uav_hit_mav_guided_count"] = guided_h
                comp["active_tam_uav"] = float(base_rewards.get(rid, 0.0) - comp.get("active_brma_flight", 0.0))
            for rid in self.red_ids:
                sim = self.red_planes.get(rid)
                if sim is None or not sim.is_alive:
                    continue
                comp = components.setdefault(rid, {})
                pos = sim.get_position()
                alt_m = sim.get_geodetic()[2]
                n_m, e_m = float(pos[0]), float(pos[1])
                out_zone = abs(n_m) > 40000 or abs(e_m) > 40000 or alt_m > 10000
                f_boundary = -10.0 if out_zone else 0.0
                for old_k in ("r_boundary", "r_bound"):
                    old_b = comp.get(old_k, 0.0)
                    if old_b != 0.0:
                        comp["r_boundary_raw_removed"] = comp.get("r_boundary_raw_removed", 0.0) + float(old_b)
                        base_rewards[rid] = base_rewards.get(rid, 0.0) - old_b
                r_bound = 0.04 * f_boundary
                comp["r_boundary"] = float(r_bound)
                base_rewards[rid] = base_rewards.get(rid, 0.0) + r_bound
                # Altitude envelope
                old_alt = comp.get("r_alt", 0.0)
                if old_alt != 0.0:
                    comp["r_alt_removed"] = float(old_alt)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) - old_alt
                if alt_m < 3000: f_alt = -1.0
                elif alt_m <= 9000: f_alt = 0.0
                elif alt_m <= 10000: f_alt = -(alt_m - 9000) / 1000.0
                else: f_alt = -1.0
                f_alt = float(np.clip(f_alt, -1.0, 0.0))
                r_alt_env = 0.01 * f_alt
                comp["r_altitude_envelope"] = float(r_alt_env)
                comp["out_zone"] = int(out_zone)
                comp["altitude_m"] = round(alt_m, 1)
                comp["north_m"] = round(n_m, 1)
                comp["east_m"] = round(e_m, 1)
                base_rewards[rid] = base_rewards.get(rid, 0.0) + r_alt_env

            # ---- Update MAV shared track history ----
            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav":
                    continue
                uav_obs = self._last_step_obs.get(rid, {})
                src = np.asarray(uav_obs.get("enemy_track_source", []), dtype=np.float32)
                for bi, bid in enumerate(self.blue_ids):
                    if bi < src.shape[0] and src[bi, 1] > 0.5:
                        self._paper_mav_shared_track_history[(rid, bid)] = self.current_step
            if mav_id:
                mav_obs = self._last_step_obs.get(mav_id, {})
                obs_mask = np.asarray(mav_obs.get("enemy_observed_mask", []), dtype=np.float32)
                for bi, bid in enumerate(self.blue_ids):
                    if bi < obs_mask.size and obs_mask[bi] > 0.5:
                        self._paper_mav_observed_history[bid] = self.current_step

            # ---- UAV target selection (BEFORE MAV support for S_guided) ----
            self._paper_reward_targets_current_step = {}
            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav":
                    continue
                sim_u = self.red_planes.get(rid)
                if sim_u is None or not sim_u.is_alive:
                    continue
                obs_u = self._last_step_obs.get(rid, {})
                enemy_geo = np.asarray(obs_u.get("enemy_geo_states", []), dtype=np.float32)
                enemy_alive = np.asarray(obs_u.get("enemy_alive_mask", []), dtype=np.float32)
                enemy_obs = np.asarray(obs_u.get("enemy_observed_mask", []), dtype=np.float32)
                uav_src = np.asarray(obs_u.get("enemy_track_source", []), dtype=np.float32)
                best_score, best_j, best_metrics = -1e9, -1, {}
                if enemy_geo.ndim == 2 and enemy_alive.ndim == 1:
                    for j in range(min(enemy_geo.shape[0], enemy_alive.shape[0])):
                        if enemy_alive[j] < 0.5:
                            continue
                        observed = (enemy_obs[j] > 0.5 if enemy_obs.size > j else False)
                        if uav_src.ndim == 2 and uav_src.shape[1] >= 2:
                            observed = observed or uav_src[j, 0] > 0.5 or uav_src[j, 1] > 0.5
                        if not observed:
                            continue
                        d_norm = abs(float(enemy_geo[j, 2]))
                        ao_n = abs(float(enemy_geo[j, 3]))
                        ta_n = abs(float(enemy_geo[j, 4]))
                        d_km = d_norm * 40000.0 / 1000.0
                        if d_km <= 5: R_D = 1.0
                        elif d_km < 10: R_D = 2.0 * np.exp(-0.921 * (d_km - 5.0)) - 1.0
                        else: R_D = -1.0
                        S_AO = 1.0 - np.clip(ao_n * 180.0 / 45.0, 0, 1)
                        S_TA = np.clip((ta_n * 180.0 - 90.0) / 90.0, 0, 1)
                        R_A = 2.0 * (0.6 * S_AO + 0.4 * S_TA) - 1.0
                        I_gate = 1.0 if (0.0125 < d_norm < 0.25 and ao_n < 0.25 and ta_n > 0.5) else 0.0
                        S_mav = 1.0 if (uav_src.ndim == 2 and uav_src.shape[1] >= 2 and uav_src[j, 1] > 0.5) else 0.0
                        Q_D = (R_D + 1.0) / 2.0; Q_A = (R_A + 1.0) / 2.0
                        score = 0.40 * Q_D + 0.35 * Q_A + 0.10 * I_gate + 0.15 * S_mav
                        if score > best_score:
                            best_score, best_j = score, j
                            best_metrics = {"R_D": R_D, "R_A": R_A, "I_gate": I_gate, "S_mav": S_mav,
                                            "d_norm": d_norm, "ao_n": ao_n, "ta_n": ta_n}
                self._paper_reward_targets_current_step[rid] = {
                    "best_j": best_j, "best_score": best_score, "best_metrics": best_metrics}

            PAPER_MAV_KEYS = [
                "mav_safety", "mav_safety_dist", "mav_safety_threat", "mav_safety_aspect",
                "mav_support", "mav_support_position", "mav_support_information",
                "mav_support_observe", "mav_support_shared", "mav_support_guided",
                "mav_event", "mav_death", "mav_out_zone", "mav_assist",
            ]
            PAPER_UAV_KEYS = [
                "uav_attack", "uav_attack_distance", "uav_attack_angle", "uav_attack_gate",
                "uav_attack_mav_shared_multiplier", "uav_fire", "uav_hit", "uav_dodge",
                "uav_death", "uav_out_zone",
            ]
            for aid in self.agent_ids:
                comp = components.setdefault(aid, {})
                role = self.agent_roles.get(aid, "")
                for key in (PAPER_MAV_KEYS if role == "mav" else PAPER_UAV_KEYS if role == "attack_uav" else []):
                    comp.setdefault(key, 0.0)

            # ---- MAV rewards ----
            if mav_id and mav_id in self.red_planes:
                mav = self.red_planes[mav_id]
                comp = components.setdefault(mav_id, {})
                if mav.is_alive:
                    mav_pos = mav.get_position()
                    # --- mav_safety ---
                    blue_sims = [s for s in self.blue_planes.values() if s.is_alive]
                    if blue_sims:
                        d_MB_km = min(np.linalg.norm(mav_pos - b.get_position()) for b in blue_sims) / 1000.0
                    else:
                        d_MB_km = 40.0
                    comp["mav_nearest_blue_range_m"] = round(d_MB_km * 1000, 0)
                    if d_MB_km < 8: S_dist = -1.0
                    elif d_MB_km < 15: S_dist = -1.0 + 2.0 * (d_MB_km - 8.0) / 7.0
                    elif d_MB_km <= 30: S_dist = 1.0
                    elif d_MB_km <= 40: S_dist = 1.0 - 1.5 * (d_MB_km - 30.0) / 10.0
                    else: S_dist = -1.0
                    S_dist = float(np.clip(S_dist, -1.0, 1.0))
                    # S_threat — use environment's _missile_candidate_metrics
                    mw = int(mav.check_missile_warning() is not None)
                    blue_can_launch = 0
                    S_AO_vals, S_TA_vals = [], []
                    for b in blue_sims:
                        metrics = self._missile_candidate_metrics(b, mav)
                        if metrics["range_ok"] and metrics["ao_ok"] and metrics["ta_ok"]:
                            blue_can_launch = 1
                        AO_rad = metrics.get("AO_rad", np.pi)
                        TA_rad = metrics.get("TA_rad", 0.0)
                        S_AO_vals.append(1.0 - np.clip(AO_rad / np.deg2rad(45), 0, 1))
                        S_TA_vals.append(np.clip((TA_rad - np.pi / 2) / (np.pi / 2), 0, 1))
                    S_threat = -1.0 if (mw or blue_can_launch) else 0.5
                    G_blue = max([0.5 * a + 0.5 * t for a, t in zip(S_AO_vals, S_TA_vals)]) if S_AO_vals else 0.0
                    S_aspect = 1.0 - 2.0 * G_blue
                    R_safety = 0.5 * S_dist + 0.3 * S_threat + 0.2 * S_aspect
                    r_mav_safety = self._paper_add_capped_reward(mav_id, "mav_safety", 0.006 * R_safety, -6.0, 6.0)
                    comp["mav_safety"] = r_mav_safety
                    comp["mav_safety_dist"] = float(S_dist)
                    comp["mav_safety_threat"] = float(S_threat)
                    comp["mav_safety_aspect"] = float(S_aspect)
                    comp["mav_blue_can_launch"] = blue_can_launch
                    comp["mav_missile_warning"] = mw
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_mav_safety

                    # --- mav_support ---
                    uav_poses = [self.red_planes[r].get_position() for r in self.red_ids
                                 if r != mav_id and self.red_planes[r].is_alive]
                    if uav_poses:
                        c_uav = np.mean(uav_poses, axis=0)
                        d_MU_km = float(np.linalg.norm(mav_pos - c_uav)) / 1000.0
                    else:
                        d_MU_km = 15.0
                    comp["mav_uav_center_range_m"] = round(d_MU_km * 1000, 0)
                    if d_MU_km < 5: S_pos = -0.2
                    elif d_MU_km < 8: S_pos = -0.2 + 1.2 * (d_MU_km - 5) / 3.0
                    elif d_MU_km <= 22: S_pos = 1.0
                    elif d_MU_km <= 35: S_pos = 1.0 - 1.5 * (d_MU_km - 22) / 13.0
                    else: S_pos = -1.0
                    S_pos = float(np.clip(S_pos, -1.0, 1.0))
                    # S_information
                    mav_obs_mask = np.asarray(self._last_step_obs.get(mav_id, {}).get("enemy_observed_mask", []), dtype=np.float32)
                    n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
                    S_observe = float(np.sum(mav_obs_mask > 0.5)) / max(n_blue_alive, 1)
                    shared_total, guided_total = 0, 0
                    n_uav_alive = sum(1 for r in self.red_ids if r != mav_id and self.red_planes[r].is_alive)
                    for rid in self.red_ids:
                        if rid == mav_id: continue
                        src = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
                        if src.ndim == 2 and src.shape[1] >= 2:
                            shared_total += float(np.sum(src[:, 1] > 0.5))
                    S_shared = shared_total / max(n_uav_alive * max(n_blue_alive, 1), 1)
                    # S_guided: UAV reward targets from MAV-shared tracks
                    guided_count = sum(
                        1 for r in self.red_ids if r != mav_id
                        and self._paper_reward_targets_current_step.get(r, {}).get("best_metrics", {}).get("S_mav", 0) > 0.5
                    )
                    S_guided = guided_count / max(n_uav_alive, 1) if n_uav_alive > 0 else 0.0
                    S_info_raw = 0.4 * S_observe + 0.4 * S_shared + 0.2 * S_guided
                    S_information = 2.0 * S_info_raw - 1.0
                    R_support = 0.6 * S_pos + 0.4 * S_information
                    r_mav_support = self._paper_add_capped_reward(mav_id, "mav_support", 0.006 * R_support, -6.0, 6.0)
                    comp["mav_support"] = r_mav_support
                    comp["mav_support_position"] = float(S_pos)
                    comp["mav_support_information"] = float(S_information)
                    comp["mav_support_observe"] = float(S_observe)
                    comp["mav_support_shared"] = float(S_shared)
                    comp["mav_support_guided"] = float(S_guided)
                    base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_mav_support

                    # --- mav event: death, out_zone, assist ---
                    if mav_id not in self._paper_out_zone_penalized:
                        n_m = float(mav_pos[0]); e_m = float(mav_pos[1])
                        if abs(n_m) > 40000 or abs(e_m) > 40000 or mav.get_geodetic()[2] > 10000:
                            self._paper_out_zone_penalized.add(mav_id)
                            r_oz = self._paper_add_capped_reward(mav_id, "mav_out_zone", -15.0, -15.0, 0.0)
                            comp["mav_out_zone"] = r_oz
                            base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_oz
                    # assist: use launch record mav_guided_at_launch or mav_observed_at_launch
                    for rid in self.red_ids:
                        if rid == mav_id: continue
                        done_hits = [
                            r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
                            if str(r.get("shooter_id", "")) == str(rid)
                            and str(r.get("raw_termination_reason", "")) == "hit"]
                        for lr in done_hits:
                            if bool(lr.get("mav_guided_at_launch", False)) or bool(lr.get("mav_observed_at_launch", False)):
                                r_asst = self._paper_add_capped_reward(mav_id, "mav_assist", 2.5, 0.0, 5.0)
                                comp["mav_assist"] = comp.get("mav_assist", 0.0) + r_asst
                                base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_asst
                elif not mav.is_alive:
                    if not self._mav_death_penalized:
                        self._mav_death_penalized = True
                        r_d = self._paper_add_capped_reward(mav_id, "mav_death", -20.0, -20.0, 0.0)
                        comp["mav_death"] = r_d
                        base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_d

            # ---- UAV rewards ----
            for rid in self.red_ids:
                if self.agent_roles.get(rid, "") == "mav":
                    continue
                sim = self.red_planes.get(rid)
                comp = components.setdefault(rid, {})
                if sim is None or not sim.is_alive:
                    if not sim or not sim.is_alive:
                        if rid not in self._uav_death_penalized:
                            self._uav_death_penalized.add(rid)
                            r_d = self._paper_add_capped_reward(rid, "uav_death", -15.0, -15.0, 0.0)
                            comp["uav_death"] = r_d
                            base_rewards[rid] = base_rewards.get(rid, 0.0) + r_d
                    continue

                pos = sim.get_position()
                n_m, e_m, alt_m = float(pos[0]), float(pos[1]), sim.get_geodetic()[2]
                if rid not in self._paper_out_zone_penalized:
                    if abs(n_m) > 40000 or abs(e_m) > 40000 or alt_m > 10000:
                        self._paper_out_zone_penalized.add(rid)
                        r_oz = self._paper_add_capped_reward(rid, "uav_out_zone", -10.0, -10.0, 0.0)
                        comp["uav_out_zone"] = r_oz
                        base_rewards[rid] = base_rewards.get(rid, 0.0) + r_oz

                obs = self._last_step_obs.get(rid, {})
                enemy_geo = np.asarray(obs.get("enemy_geo_states", []), dtype=np.float32)
                enemy_alive = np.asarray(obs.get("enemy_alive_mask", []), dtype=np.float32)
                uav_src = np.asarray(obs.get("enemy_track_source", []), dtype=np.float32)
                ego_geo = np.asarray(obs.get("ego_geo_state", []), dtype=np.float32)

                # --- uav_attack: use pre-computed reward target ---
                tgt = self._paper_reward_targets_current_step.get(rid, {})
                best_j = tgt.get("best_j", -1)
                best_metrics = tgt.get("best_metrics", {})
                if best_j >= 0 and best_metrics:
                    R_D = best_metrics["R_D"]; R_A = best_metrics["R_A"]
                    I_gate = best_metrics["I_gate"]; S_mav = best_metrics["S_mav"]
                    r_attack_raw = (0.003 * R_D + 0.003 * R_A + 0.002 * I_gate) * (1.0 + 0.25 * S_mav)
                    r_attack = self._paper_add_capped_reward(rid, "uav_attack", r_attack_raw, -3.0, 5.0)
                    comp["uav_attack"] = r_attack
                    comp["uav_attack_raw"] = round(r_attack_raw, 5)
                    comp["uav_attack_distance"] = round(R_D, 3)
                    comp["uav_attack_angle"] = round(R_A, 3)
                    comp["uav_attack_gate"] = int(I_gate)
                    comp["uav_attack_mav_shared_multiplier"] = int(S_mav)
                    comp["uav_reward_target_id"] = self.blue_ids[best_j] if best_j < len(self.blue_ids) else ""
                    comp["uav_reward_target_score"] = round(tgt.get("best_score", -1), 3)
                    comp["uav_reward_target_mav_shared"] = int(S_mav)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + r_attack
                else:
                    comp["uav_attack"] = self._paper_add_capped_reward(rid, "uav_attack", -0.001, -3.0, 5.0)
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["uav_attack"]

                # --- uav_fire: current-step only, no speed filter for scripted AAM ---
                step_launches = [
                    r for r in (getattr(self, "_launch_quality_step_records", None) or [])
                    if str(r.get("shooter_id", "")) == str(rid)]
                direct_fire = 0; guided_fire = 0
                for lr in step_launches:
                    tid = lr.get("target_id", "")
                    was_guided = self._paper_mav_shared_track_history.get((rid, tid), -999)
                    if self.current_step - was_guided <= self.PAPER_MAV_SHARED_TRACK_LOOKBACK:
                        guided_fire += 1
                    else:
                        direct_fire += 1
                r_fire = min(0.10 * direct_fire + 0.15 * guided_fire, 0.30)
                capped = self._paper_add_capped_reward(rid, "uav_fire", r_fire, 0.0, 0.30)
                comp["uav_fire"] = capped
                comp["uav_fire_direct_count"] = direct_fire
                comp["uav_fire_mav_guided_count"] = guided_fire
                base_rewards[rid] = base_rewards.get(rid, 0.0) + capped

                # --- uav_hit: use launch record mav_guided_at_launch ---
                done_launches = [
                    r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
                    if str(r.get("shooter_id", "")) == str(rid)
                    and str(r.get("raw_termination_reason", "")) == "hit"]
                direct_hit = 0; guided_hit = 0
                for lr in done_launches:
                    if bool(lr.get("mav_guided_at_launch", False)):
                        guided_hit += 1
                    else:
                        direct_hit += 1
                if direct_hit + guided_hit > 0:
                    r_hit = 12.0 * direct_hit + 15.0 * guided_hit
                    comp["uav_hit"] = r_hit
                    comp["uav_hit_direct_count"] = direct_hit
                    comp["uav_hit_mav_guided_count"] = guided_hit
                    base_rewards[rid] = base_rewards.get(rid, 0.0) + r_hit

                # --- uav_dodge: approach angle only, no missile_warning reward ---
                r_dodge = 0.0
                mw_val = float(np.asarray(obs.get("missile_warning", [0.0])).reshape(-1)[0])
                if mw_val > 0.5:
                    comp["uav_dodge_unavailable_reason"] = "missile_velocity_unavailable"
                comp["uav_dodge"] = 0.0

            # Update guided fraction for MAV support
            if mav_id:
                for rid in self.red_ids:
                    if rid == mav_id: continue
                    uav_src2 = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
                    if uav_src2.ndim == 2 and uav_src2.shape[1] >= 2:
                        comp_m = components.setdefault(mav_id, {})
                        comp_m["mav_support_guided"] = float(np.sum(uav_src2[:, 1] > 0.5)) / max(1, np.sum(uav_src2 > 0.5) + 1e-6)

            return base_rewards, components

        # role_v1 (default fallback)
        else:
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
