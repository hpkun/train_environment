"""Minimal MAV/UAV heterogeneous extension of the BRMA environment."""

from __future__ import annotations

import math
from copy import deepcopy

import gymnasium
import numpy as np

from ..env import UavCombatEnv
from ..utils import get2d_AO_TA_R
from ..alignment.los_geometry import compute_3d_range, compute_body_x_q_los
from ..alignment.reward_utils import (
    altitude_reward_pairwise_mean_eq17,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)

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

TAM_PAPER_V2_MAV_COMPONENT_KEYS = (
    "tam_v2_mav_safety", "tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect",
    "tam_v2_mav_support", "tam_v2_mav_pos", "tam_v2_mav_aware",
    "tam_v2_mav_event", "tam_v2_mav_death", "tam_v2_mav_team_bonus",
    "tam_v2_total",
)

TAM_PAPER_V2_UAV_COMPONENT_KEYS = (
    "tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
    "tam_v2_uav_angle_raw", "tam_v2_uav_distance",
    "tam_v2_uav_dodge", "tam_v2_uav_dodge_angle", "tam_v2_uav_dodge_speed",
    "tam_v2_uav_event", "tam_v2_uav_kill", "tam_v2_uav_death",
    "tam_v2_uav_out_of_zone", "tam_v2_total",
)

TAM_PAPER_V2_LOG_ONLY_KEYS = (
    "tam_v2_mav_shared_log", "tam_v2_mav_assist_log",
    "tam_v2_uav_fire_log", "tam_v2_uav_mav_shared_track_log",
    "brma_r_adv_log", "brma_r_pitch_log", "brma_r_roll_log",
    "brma_r_alt_log", "brma_r_bound_log", "brma_r_vel_log",
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
        if hetero_reward_mode not in {"brma_legacy", "minimal_v1", "role_v1", "happo_ref_v0", "happo_ref_v1_mav_support", "paper_role_reward_v1", "tam_paper_reward_v2", "tam_paper_reward_v3", "tam_paper_reward_v4", "tam_paper_reward_v6_jsbsim_aligned_v3", "tam_paper_reward_v7_role_aligned", "tam_brma_scripted_reward_v1", "brma_paper_homogeneous_v1", "brma_role_no_missile_reward_v8", "tam_brma_paper_aligned_v1", "tam_happo_table1_v1", "brma_tam_scripted_composite_v1", "brma_tam_scale_aligned_v1"}:
            raise ValueError(f"unknown hetero_reward_mode: {hetero_reward_mode}")
        self.hetero_reward_mode = hetero_reward_mode
        self._tam_reward_scale = float(kwargs.pop("tam_reward_scale", 0.05))
        _happo_v1_cfg = kwargs.pop("happo_ref_v1_mav_support", None) or {}
        self.happo_ref_v1_mav_support_config = deepcopy(_happo_v1_cfg)
        if hetero_reward_mode == "happo_ref_v1_mav_support" and not self.happo_ref_v1_mav_support_config:
            raise ValueError("happo_ref_v1_mav_support mode requires happo_ref_v1_mav_support config block")
        # TAM paper reward v2 config
        _tam_cfg = kwargs.pop("tam_paper_reward_v2", None) or {}
        self.tam_paper_reward_v2_config = deepcopy(_tam_cfg)
        if hetero_reward_mode == "tam_paper_reward_v2" and not self.tam_paper_reward_v2_config:
            raise ValueError("tam_paper_reward_v2 mode requires tam_paper_reward_v2 config block")
        # TAM paper reward v3 config (env-consistent)
        _tam_v3_cfg = kwargs.pop("tam_paper_reward_v3", None) or {}
        self.tam_paper_reward_v3_config = deepcopy(_tam_v3_cfg)
        if hetero_reward_mode == "tam_paper_reward_v3" and not self.tam_paper_reward_v3_config:
            raise ValueError("tam_paper_reward_v3 mode requires tam_paper_reward_v3 config block")
        # TAM paper reward v4 config (BRMA flight status + situation + terminal outcome)
        _tam_v4_cfg = kwargs.pop("tam_paper_reward_v4", None) or {}
        self.tam_paper_reward_v4_config = deepcopy(_tam_v4_cfg)
        if hetero_reward_mode == "tam_paper_reward_v4" and not self.tam_paper_reward_v4_config:
            raise ValueError("tam_paper_reward_v4 mode requires tam_paper_reward_v4 config block")
        # TAM paper reward v6 JSBSim-aligned v3 config
        _tam_v6v3_cfg = kwargs.pop("tam_paper_reward_v6_jsbsim_aligned_v3", None) or {}
        self.tam_paper_reward_v6_jsbsim_aligned_v3_config = deepcopy(_tam_v6v3_cfg)
        if hetero_reward_mode == "tam_paper_reward_v6_jsbsim_aligned_v3" and not self.tam_paper_reward_v6_jsbsim_aligned_v3_config:
            raise ValueError("tam_paper_reward_v6_jsbsim_aligned_v3 mode requires tam_paper_reward_v6_jsbsim_aligned_v3 config block")
        # TAM paper reward v7 role-aligned config
        _tam_v7_cfg = kwargs.pop("tam_paper_reward_v7_role_aligned", None) or {}
        self.tam_paper_reward_v7_role_aligned_config = deepcopy(_tam_v7_cfg)
        if hetero_reward_mode == "tam_paper_reward_v7_role_aligned" and not self.tam_paper_reward_v7_role_aligned_config:
            raise ValueError("tam_paper_reward_v7_role_aligned mode requires tam_paper_reward_v7_role_aligned config block")
        # TAM-BRMA scripted reward v1 config
        _tam_brma_s1_cfg = kwargs.pop("tam_brma_scripted_reward_v1", None) or {}
        self.tam_brma_scripted_reward_v1_config = deepcopy(_tam_brma_s1_cfg)
        if hetero_reward_mode == "tam_brma_scripted_reward_v1" and not self.tam_brma_scripted_reward_v1_config:
            raise ValueError("tam_brma_scripted_reward_v1 mode requires config block")
        _paper_v1_cfg = kwargs.pop("tam_brma_paper_aligned_v1", None) or {}
        self.tam_brma_paper_aligned_v1_config = deepcopy(_paper_v1_cfg)
        if hetero_reward_mode == "tam_brma_paper_aligned_v1" and not self.tam_brma_paper_aligned_v1_config:
            raise ValueError("tam_brma_paper_aligned_v1 mode requires config block")
        _tam_table1_cfg = kwargs.pop("tam_happo_table1_v1", None) or {}
        self.tam_happo_table1_v1_config = deepcopy(_tam_table1_cfg)
        if hetero_reward_mode == "tam_happo_table1_v1" and not self.tam_happo_table1_v1_config:
            raise ValueError("tam_happo_table1_v1 mode requires config block")
        _brma_tam_composite_cfg = kwargs.pop("brma_tam_scripted_composite_v1", None) or {}
        self.brma_tam_scripted_composite_v1_config = deepcopy(_brma_tam_composite_cfg)
        if hetero_reward_mode == "brma_tam_scripted_composite_v1" and not self.brma_tam_scripted_composite_v1_config:
            raise ValueError("brma_tam_scripted_composite_v1 mode requires config block")
        _scale_v1_cfg = kwargs.pop("brma_tam_scale_aligned_v1", None) or {}
        self.brma_tam_scale_aligned_v1_config = deepcopy(_scale_v1_cfg)
        if hetero_reward_mode == "brma_tam_scale_aligned_v1" and not self.brma_tam_scale_aligned_v1_config:
            raise ValueError("brma_tam_scale_aligned_v1 mode requires config block")
        # Cached per-step obs for reward overlay (minimal_v1 / role_v1)
        self._last_step_obs: dict = {}
        # First-death detection for MAV — penalize once per episode
        self._mav_death_penalized: bool = False
        # First-death detection per UAV (role_v1)
        self._uav_death_penalized: set[str] = set()
        # TAM paper v2 per-episode state
        self._tam_v2_out_of_zone_penalized: set[str] = set()
        self._tam_v2_missile_speed_cache: dict[str, float] = {}
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
        self.missile_evasion_mode = str(getattr(self, "missile_evasion_config", {}).get("mode", ""))
        self.missile_evasion_teams = str(getattr(self, "missile_evasion_config", {}).get("teams", ""))
        if self.hetero_reward_mode == "brma_tam_scripted_composite_v1":
            self._validate_brma_tam_scripted_composite_v1_contract()
        if self.hetero_reward_mode == "brma_tam_scale_aligned_v1":
            self._validate_brma_tam_scale_aligned_v1_contract()

    # -- TAM Paper Reward v6 JSBSim-aligned v3 -------------------------------

    def _tam_v6v3_reset_episode_state(self) -> None:
        self._tam_v6v3_terminal_applied = False
        self._tam_v6v3_uav_death_penalized: set[str] = set()
        self._tam_v6v3_mav_death_penalized = False
        self._tam_v6v3_mav_team_credit_used = 0.0

    # -- BRMA-MAPPO paper homogeneous diagnostic reward ----------------------

    def _brma_homo_reset_episode_state(self) -> None:
        self._brma_homo_terminal_applied = False

    def _happo_ref_v1_reset_episode_state(self) -> None:
        self._happo_v1_mav_death_penalized = False
        self._happo_v1_mav_team_credit_used = 0.0
        self._paper_aligned_v1_mav_death_penalized = False
        self._paper_aligned_v1_mav_team_credit_used = 0.0

    def _tam_brma_paper_v1_reset_episode_state(self) -> None:
        self._paper_aligned_v1_mav_death_penalized = False
        self._paper_aligned_v1_mav_team_credit_used = 0.0

    def _tam_happo_table1_v1_reset_episode_state(self) -> None:
        self._tam_table1_uav_death_penalized: set[str] = set()
        self._tam_table1_uav_out_of_zone_penalized: set[str] = set()
        self._tam_table1_mav_death_penalized = False
        self._tam_table1_mav_team_credit_used = 0.0
        self._tam_table1_missile_speed_cache: dict[str, float] = {}

    def _brma_tam_scripted_reset_episode_state(self) -> None:
        self._brma_tam_uav_death_penalized: set[str] = set()
        self._brma_tam_uav_horizontal_oob_penalized: set[str] = set()
        self._brma_tam_mav_death_penalized = False
        self._brma_tam_mav_team_credit_used = 0.0
        self._brma_tam_alive_before_step: dict[str, bool] = {}
        self._brma_tam_all_attack_uav_dead_steps = 0
        self._brma_tam_reward_target_switch_counts: dict[str, int] = {}
        self._brma_tam_last_reward_target: dict[str, str] = {}
        self._brma_tam_missile_speed_cache: dict[str, float] = {}
        self._reward_target_diagnostic_records: list[dict] = []

    def _brma_tam_scale_v1_reset_episode_state(self) -> None:
        self._scale_v1_uav_death_penalized: set[str] = set()
        self._scale_v1_uav_horizontal_oob_penalized: set[str] = set()
        self._scale_v1_mav_death_penalized = False
        self._scale_v1_mav_team_credit_used = 0.0
        self._scale_v1_terminal_applied = False
        self._scale_v1_progress_cache: dict[str, dict] = {}
        self._scale_v1_reward_target_switch_counts: dict[str, int] = {}
        self._scale_v1_alive_before_step: dict[str, bool] = {}

    @staticmethod
    def _brma_homo_td15(distance_m: float) -> float:
        distance = float(distance_m)
        if not np.isfinite(distance):
            return 0.0
        if distance <= 15000.0:
            return 1.0
        return float(np.exp(1.0 - distance / 15000.0))

    def _brma_homo_boundary(self, sim) -> float:
        pos = np.asarray(sim.get_position(), dtype=np.float64)
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        if abs(float(pos[0])) > half or abs(float(pos[1])) > half:
            return -10.0
        return 0.0

    def _brma_homo_altitude(self, sim, alive_blue: list) -> float:
        if not alive_blue:
            return 0.0
        alt = float(sim.get_geodetic()[2])
        enemy_alts = [float(blue.get_geodetic()[2]) for blue in alive_blue]
        return altitude_reward_pairwise_mean_eq17(alt, enemy_alts)

    def _brma_homo_situation_reward(self, sim, alive_blue: list) -> tuple[float, float, float]:
        own_adv = 0.0
        enemy_threat = 0.0
        ego_pos = sim.get_position()
        ego_rpy = sim.get_rpy()
        for blue in alive_blue:
            blue_pos = blue.get_position()
            blue_rpy = blue.get_rpy()
            d_3d = compute_3d_range(ego_pos, blue_pos)
            q_red_to_blue = compute_body_x_q_los(ego_pos, ego_rpy, blue_pos)
            q_blue_to_red = compute_body_x_q_los(blue_pos, blue_rpy, ego_pos)
            own_adv += ta_angle_advantage_fixed(np.rad2deg(q_red_to_blue)) * self._brma_homo_td15(d_3d)
            enemy_threat += ta_angle_advantage_fixed(np.rad2deg(q_blue_to_red)) * self._brma_homo_td15(d_3d)
        r_adv = own_adv - 0.8 * enemy_threat
        return float(r_adv), float(own_adv), float(enemy_threat)

    def _brma_homo_terminal_outcome(self) -> float:
        n_blue_alive = sum(1 for sim in self.blue_planes.values() if sim.is_alive)
        n_red_alive = sum(1 for sim in self.red_planes.values() if sim.is_alive)
        if n_blue_alive == n_red_alive:
            return 0.0
        return float(30.0 * (n_red_alive - n_blue_alive))

    def _compute_brma_paper_homogeneous_v1(self, base_rewards: dict, components: dict):
        alive_blue = [sim for sim in self.blue_planes.values() if sim.is_alive]
        n_blue_alive = len(alive_blue)
        n_red_alive = sum(1 for sim in self.red_planes.values() if sim.is_alive)
        round_over = n_blue_alive == 0 or n_red_alive == 0 or self.current_step >= self.max_steps
        terminal_applied = 0.0
        r_end = 0.0
        if round_over and not getattr(self, "_brma_homo_terminal_applied", False):
            r_end = self._brma_homo_terminal_outcome()
            self._brma_homo_terminal_applied = True
            terminal_applied = 1.0

        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            comp = components.setdefault(rid, {})
            if sim.is_alive:
                r_pitch = float(self._pitch_penalty(sim))
                r_roll = float(self._roll_penalty(sim))
                r_altitude = float(self._brma_homo_altitude(sim, alive_blue))
                r_boundary = float(self._brma_homo_boundary(sim))
                r_speed = float(self._speed_penalty(sim))
                r_adv, own_adv, enemy_threat = self._brma_homo_situation_reward(sim, alive_blue)
            else:
                r_pitch = r_roll = r_altitude = r_boundary = r_speed = r_adv = 0.0
                own_adv = enemy_threat = 0.0

            total = (
                0.01 * r_pitch
                + 0.002 * r_roll
                + 0.04 * r_altitude
                + 0.04 * r_boundary
                + 0.02 * r_speed
                + 0.15 * r_adv
                + r_end
            )
            comp.update({
                "brma_homo_r_pitch": r_pitch,
                "brma_homo_r_roll": r_roll,
                "brma_homo_r_altitude": r_altitude,
                "brma_homo_r_boundary": r_boundary,
                "brma_homo_r_speed": r_speed,
                "brma_homo_r_adv": r_adv,
                "brma_homo_r_end": r_end,
                "brma_homo_total": float(total),
                "brma_homo_own_adv_sum": own_adv,
                "brma_homo_enemy_threat_sum": enemy_threat,
                "brma_homo_td15_used": 1.0,
                "brma_homo_role_agnostic": 1.0,
                "brma_homo_terminal_applied": terminal_applied,
            })
            base_rewards[rid] = float(total)
            components[rid] = comp
        return base_rewards, components

    def _compute_brma_role_no_missile_reward_v8(self, base_rewards: dict, components: dict):
        """BRMA trunk reward with MAV attack-situation disabled.

        ``super()._compute_rewards`` already stores the BRMA flight, situation,
        and terminal components after their BRMA weights have been applied.
        This overlay only removes the already-weighted situation term from MAVs.
        Attack UAV rewards remain the parent BRMA flight + situation + terminal
        reward.  No TAM event/safety/support or missile-process reward helper is
        called here, so no v7 state can be mutated.
        """
        for rid in self.red_ids:
            comp = components.setdefault(rid, {})
            role = self.agent_roles.get(rid, "")
            comp["brma_role_no_missile_active"] = 1.0
            comp["brma_role_active_brma_flight"] = 1.0
            comp["brma_role_active_brma_terminal"] = 1.0
            comp["brma_role_removed_situation_is_weighted"] = 1.0
            comp["brma_role_is_mav"] = 1.0 if role == "mav" else 0.0
            if role == "mav":
                removed = float(comp.get("r_adv", 0.0))
                base_rewards[rid] = float(base_rewards.get(rid, 0.0)) - removed
                comp["r_adv"] = 0.0
                comp["brma_role_removed_situation"] = removed
                comp["brma_role_situation_active"] = 0.0
                comp["brma_role_active_brma_situation"] = 0.0
            else:
                comp["brma_role_removed_situation"] = 0.0
                comp["brma_role_situation_active"] = 1.0
                comp["brma_role_active_brma_situation"] = 1.0
            comp["brma_role_no_missile_total"] = float(base_rewards.get(rid, 0.0))
            comp["total"] = float(base_rewards.get(rid, 0.0))
            components[rid] = comp
        return base_rewards, components

    @staticmethod
    def _body_x_los_angle(observer, target) -> float:
        try:
            angle = float(compute_body_x_q_los(
                observer.get_position(), observer.get_rpy(), target.get_position()))
        except Exception:
            return float(np.pi)
        if not np.isfinite(angle):
            return float(np.pi)
        return float(np.clip(angle, 0.0, np.pi))

    def _paper_v1_mav_position_support(self, mav, cfg: dict) -> tuple[float, dict]:
        sp = cfg.get("mav_support", {})
        d_opt = float(sp.get("d_opt_m", 8000.0))
        d_max = float(sp.get("d_max_m", 25000.0))
        points = []
        for rid in self.red_ids:
            if self.agent_roles.get(rid) != "attack_uav":
                continue
            sim = self.red_planes.get(rid)
            if sim is not None and getattr(sim, "is_alive", False):
                points.append(np.asarray(sim.get_position(), dtype=np.float64)[:2])
        for bid in self.blue_ids:
            sim = self.blue_planes.get(bid)
            if sim is not None and getattr(sim, "is_alive", False):
                points.append(np.asarray(sim.get_position(), dtype=np.float64)[:2])
        if mav is None or not getattr(mav, "is_alive", False) or not points:
            return 0.0, {
                "paper_v1_mav_battlefield_center_x": 0.0,
                "paper_v1_mav_battlefield_center_y": 0.0,
                "paper_v1_mav_pos_distance_m": 0.0,
            }
        center = np.mean(np.stack(points, axis=0), axis=0)
        mav_xy = np.asarray(mav.get_position(), dtype=np.float64)[:2]
        d_b = float(np.linalg.norm(mav_xy - center))
        if d_b < d_opt:
            r_pos = d_b / max(d_opt, 1e-6) - 1.0
        elif d_b < d_max:
            r_pos = 1.0 - (d_b - d_opt) / max(d_max - d_opt, 1e-6)
        else:
            r_pos = -0.5
        return float(r_pos), {
            "paper_v1_mav_battlefield_center_x": float(center[0]),
            "paper_v1_mav_battlefield_center_y": float(center[1]),
            "paper_v1_mav_pos_distance_m": d_b,
        }

    def _paper_v1_mav_awareness(self, mav, cfg: dict) -> tuple[float, dict]:
        if mav is None or not getattr(mav, "is_alive", False):
            return 0.0, {"paper_v1_mav_aware_observed_count": 0.0}
        obs_range = float(getattr(self, "mav_observation_range_m", 80000.0))
        reward = 0.0
        observed_count = 0
        mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
        for blue in self.blue_planes.values():
            if not getattr(blue, "is_alive", False):
                continue
            distance = float(np.linalg.norm(mav_pos - np.asarray(blue.get_position(), dtype=np.float64)))
            angle = self._body_x_los_angle(mav, blue)
            if angle < np.pi / 2.0 and distance < obs_range:
                observed_count += 1
                reward += 0.3 * (1.0 - angle / (np.pi / 2.0))
        return float(reward), {"paper_v1_mav_aware_observed_count": float(observed_count)}

    def _paper_v1_blue_aspect_threat(self, blue, mav) -> float:
        if blue is None or mav is None:
            return 0.0
        if not getattr(blue, "is_alive", False) or not getattr(mav, "is_alive", False):
            return 0.0
        angle = self._body_x_los_angle(blue, mav)
        if angle < np.pi / 4.0:
            return float(-(1.0 - angle / (np.pi / 4.0)))
        return 0.0

    def _paper_v1_mav_safety(self, mav, cfg: dict) -> tuple[float, dict]:
        scfg = cfg.get("mav_safety", {})
        d_danger = float(scfg.get("d_danger_m", 8000.0))
        d_safe = float(scfg.get("d_safe_m", 15000.0))
        alive_blue = [s for s in self.blue_planes.values() if getattr(s, "is_alive", False)]
        if mav is None or not getattr(mav, "is_alive", False):
            logs = {
                "paper_v1_mav_dist": 0.0,
                "paper_v1_mav_threat": 0.0,
                "paper_v1_mav_aspect": 0.0,
                "paper_v1_mav_safety_danger_m": d_danger,
                "paper_v1_mav_safety_safe_m": d_safe,
            }
            logs["paper_v1_mav_safety"] = 0.0
            return 0.0, logs
        mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
        if alive_blue:
            near_d = min(float(np.linalg.norm(mav_pos - np.asarray(b.get_position(), dtype=np.float64)))
                         for b in alive_blue)
            if near_d < d_danger:
                r_dist = -(1.0 - near_d / max(d_danger, 1e-6))
            elif near_d < d_safe:
                r_dist = -0.5 * (1.0 - (near_d - d_danger) / max(d_safe - d_danger, 1e-6))
            else:
                r_dist = 0.2
        else:
            near_d = 0.0
            r_dist = 0.0
        r_threat = -1.0 if mav.check_missile_warning() is not None else 0.0
        r_aspect = sum(self._paper_v1_blue_aspect_threat(blue, mav) for blue in alive_blue)
        safety = (
            float(scfg.get("dist_weight", 0.5)) * r_dist
            + float(scfg.get("threat_weight", 0.3)) * r_threat
            + float(scfg.get("aspect_weight", 0.2)) * r_aspect
        )
        logs = {
            "paper_v1_mav_safety": float(safety),
            "paper_v1_mav_dist": float(r_dist),
            "paper_v1_mav_threat": float(r_threat),
            "paper_v1_mav_aspect": float(r_aspect),
            "paper_v1_mav_nearest_blue_distance_m": float(near_d),
            "paper_v1_mav_safety_danger_m": d_danger,
            "paper_v1_mav_safety_safe_m": d_safe,
        }
        return float(safety), logs

    def _paper_v1_mav_support(self, mav, cfg: dict) -> tuple[float, dict]:
        sp = cfg.get("mav_support", {})
        r_pos, pos_logs = self._paper_v1_mav_position_support(mav, cfg)
        r_aware, aware_logs = self._paper_v1_mav_awareness(mav, cfg)
        support = (
            float(sp.get("pos_weight", 0.6)) * r_pos
            + float(sp.get("aware_weight", 0.4)) * r_aware
        )
        logs = {
            "paper_v1_mav_support": float(support),
            "paper_v1_mav_pos": float(r_pos),
            "paper_v1_mav_aware": float(r_aware),
        }
        logs.update(pos_logs)
        logs.update(aware_logs)
        return float(support), logs

    def _paper_v1_mav_event(self, mav_id: str, mav, cfg: dict) -> tuple[float, dict]:
        ecfg = cfg.get("mav_event", {})
        death = 0.0
        if mav is not None and (not mav.is_alive) and not self._paper_aligned_v1_mav_death_penalized:
            death = -float(ecfg.get("death_penalty_raw", 200.0))
            self._paper_aligned_v1_mav_death_penalized = True
        mav_alive = bool(mav is not None and mav.is_alive)
        kills = 0
        for rid in self.red_ids:
            if rid == mav_id or self.agent_roles.get(rid) != "attack_uav":
                continue
            kills += int(self._step_kill_count.get(rid, 0))
        cap = float(ecfg.get("team_credit_cap_raw", 200.0))
        if mav_alive and kills > 0:
            available = max(0.0, cap - self._paper_aligned_v1_mav_team_credit_used)
            credit = min(float(ecfg.get("team_credit_per_kill_raw", 100.0)) * kills, available)
            self._paper_aligned_v1_mav_team_credit_used += credit
        else:
            credit = 0.0
        event = death + credit
        logs = {
            "paper_v1_mav_event_raw": float(event),
            "paper_v1_mav_event_death_raw": float(death),
            "paper_v1_mav_event_team_credit_delta_raw": float(credit),
            "paper_v1_mav_event_team_credit_used_raw": float(self._paper_aligned_v1_mav_team_credit_used),
            "paper_v1_mav_event_team_credit_cap_raw": cap,
        }
        return float(event), logs

    def _paper_v1_shared_track_logs(self, mav_id: str) -> dict:
        shared_slots = 0.0
        for rid in self.red_ids:
            if rid == mav_id or self.agent_roles.get(rid) != "attack_uav":
                continue
            src = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
            if src.ndim == 2 and src.shape[1] >= 2:
                shared_slots += float(np.sum(src[:, 1] > 0.5))
        launches = [
            r for r in (getattr(self, "_launch_quality_step_records", None) or [])
            if str(r.get("shooter_id", "")).startswith("red_")
            and self.agent_roles.get(str(r.get("shooter_id", ""))) == "attack_uav"
        ]
        hits = [
            r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
            if str(r.get("shooter_id", "")).startswith("red_")
            and self.agent_roles.get(str(r.get("shooter_id", ""))) == "attack_uav"
            and str(r.get("raw_termination_reason", "")) == "hit"
        ]
        return {
            "paper_v1_mav_shared_track_log": float(shared_slots),
            "paper_v1_red_launch_with_mav_shared_track_log": float(
                sum(1 for r in launches if str(r.get("launch_track_source", "")) == "mav_shared")
            ),
            "paper_v1_red_hit_with_mav_shared_track_log": float(
                sum(1 for r in hits if str(r.get("launch_track_source", "")) == "mav_shared")
            ),
        }

    def _tam_table1_shared_track_logs(self, mav_id: str) -> dict:
        paper_logs = self._paper_v1_shared_track_logs(mav_id)
        return {
            "tam_table1_mav_shared_track_slots_log": paper_logs.get("paper_v1_mav_shared_track_log", 0.0),
            "tam_table1_red_launch_with_mav_shared_track_log": paper_logs.get(
                "paper_v1_red_launch_with_mav_shared_track_log", 0.0
            ),
            "tam_table1_red_hit_with_mav_shared_track_log": paper_logs.get(
                "paper_v1_red_hit_with_mav_shared_track_log", 0.0
            ),
        }

    def _tam_table1_uav_height_raw(self, sim, cfg: dict) -> tuple[float, dict]:
        """JSBSim safety adaptation of TAM-HAPPO Table 1 R_H = P_V + P_H.

        The paper does not map directly to this JSBSim state interface here, so
        normal flight returns zero and only unsafe altitude / vertical speed
        excursions receive negative shaping.
        """
        ucfg = cfg.get("uav", {})
        alt = float(sim.get_geodetic()[2])
        vel = np.asarray(sim.get_velocity(), dtype=np.float64)
        vertical_speed = abs(float(vel[2])) if vel.size >= 3 and np.isfinite(vel[2]) else 0.0
        alt_min = float(ucfg.get("altitude_min_m", 2500.0))
        alt_max = float(ucfg.get("altitude_max_m", 10000.0))
        vz_limit = float(ucfg.get("vertical_speed_limit_mps", 150.0))
        ph = -1.0 if (not np.isfinite(alt) or alt < alt_min or alt > alt_max) else 0.0
        pv = -1.0 if vertical_speed > vz_limit else 0.0
        return float(ph + pv), {
            "tam_table1_uav_height_pv": float(pv),
            "tam_table1_uav_height_ph": float(ph),
        }

    @staticmethod
    def _tam_table1_speed_raw(speed_mps: float, reference_speed_mps: float) -> float:
        speed = float(speed_mps)
        ref = max(float(reference_speed_mps), 1e-6)
        if speed < 0.5 * ref:
            return 1.0
        if speed <= 1.5 * ref:
            return 2.0 - 2.0 * speed / ref
        return -1.0

    @staticmethod
    def _tam_table1_distance_raw(distance_m: float) -> float:
        d_km = float(distance_m) / 1000.0
        if d_km <= 5.0:
            return 1.0
        if d_km < 10.0:
            return float(np.exp(-0.921 * (d_km - 5.0)))
        return -1.0

    def _tam_table1_best_uav_target(self, sim) -> tuple[object | None, dict]:
        alive_blue = self._tam_v2_alive_blue()
        if not alive_blue or sim is None or not getattr(sim, "is_alive", False):
            return None, {
                "tam_table1_uav_target_id_log": "",
                "tam_table1_uav_target_distance_km": 0.0,
                "tam_table1_uav_target_ata_rad": 0.0,
                "tam_table1_uav_target_aa_rad": 0.0,
                "tam_table1_uav_angle_raw": 0.0,
                "tam_table1_uav_distance_raw": 0.0,
            }
        red_feat = self._tam_v2_feature(sim)
        best = None
        best_logs = {}
        best_score = -float("inf")
        for bid in self.blue_ids:
            blue = self.blue_planes.get(bid)
            if blue not in alive_blue:
                continue
            blue_feat = self._tam_v2_feature(blue)
            ata, ta, distance_m = get2d_AO_TA_R(red_feat, blue_feat)
            aa = float(np.pi - ta)
            r_angle = float(1.0 - (ata + aa) / np.pi)
            r_dist = self._tam_table1_distance_raw(distance_m)
            score = 0.6 * r_angle + 0.4 * r_dist
            if score > best_score:
                best_score = score
                best = blue
                best_logs = {
                    "tam_table1_uav_target_id_log": bid,
                    "tam_table1_uav_target_distance_km": float(distance_m) / 1000.0,
                    "tam_table1_uav_target_ata_rad": float(ata),
                    "tam_table1_uav_target_aa_rad": float(aa),
                    "tam_table1_uav_angle_raw": r_angle,
                    "tam_table1_uav_distance_raw": r_dist,
                }
        return best, best_logs

    def _tam_table1_dodge_raw(self, sim, cfg: dict) -> tuple[float, float, float, float]:
        threat = getattr(sim, "under_missiles", None)
        warning = sim.check_missile_warning() if hasattr(sim, "check_missile_warning") else None
        if not threat and warning is None:
            return 0.0, 0.0, 0.0, 0.0
        ucfg = cfg.get("uav", {})
        v_norm = float(ucfg.get("dodge_speed_norm_mps", 1000.0))
        candidates = []
        for missile in list(threat or []):
            if not getattr(missile, "is_alive", False):
                continue
            try:
                uid = str(getattr(missile, "uid", getattr(missile, "_uid", id(missile))))
                mv = np.asarray(missile.get_velocity(), dtype=np.float64)
                sp = float(np.linalg.norm(mv))
                los = np.asarray(sim.get_position(), dtype=np.float64) - np.asarray(missile.get_position(), dtype=np.float64)
                los_norm = float(np.linalg.norm(los))
                if sp <= 1e-6 or los_norm <= 1e-6:
                    continue
                cos_lambda = float(np.clip(np.dot(mv, los) / (sp * los_norm), -1.0, 1.0))
                r_angle = -cos_lambda
                prev_sp = self._tam_table1_missile_speed_cache.get(uid)
                r_speed = 0.0 if prev_sp is None else (prev_sp - sp) / max(v_norm, 1e-6)
                self._tam_table1_missile_speed_cache[uid] = sp
                candidates.append((r_angle + r_speed, r_angle, r_speed))
            except Exception:
                continue
        if not candidates:
            return 0.0, 0.0, 0.0, 1.0
        total, angle, speed = max(candidates, key=lambda item: item[0])
        return float(total), float(angle), float(speed), 0.0

    def _tam_table1_uav_event(self, aid: str, sim, cfg: dict) -> tuple[float, dict]:
        ev = cfg.get("uav", {}).get("event", {})
        kills = int(self._step_kill_count.get(aid, 0))
        kill_reward = float(ev.get("kill_enemy", 200.0)) * kills
        death = 0.0
        if sim is not None and not getattr(sim, "is_alive", False) and aid not in self._tam_table1_uav_death_penalized:
            death = float(ev.get("death", -200.0))
            self._tam_table1_uav_death_penalized.add(aid)
        out_of_zone = 0.0
        if sim is not None and self._tam_v7_out_of_zone(sim) and aid not in self._tam_table1_uav_out_of_zone_penalized:
            out_of_zone = float(ev.get("first_out_of_zone", -100.0))
            self._tam_table1_uav_out_of_zone_penalized.add(aid)
        event = kill_reward + death + out_of_zone
        return float(event), {
            "tam_table1_uav_event": float(event),
            "tam_table1_uav_kill": float(kill_reward),
            "tam_table1_uav_death": float(death),
            "tam_table1_uav_out_of_zone": float(out_of_zone),
        }

    def _tam_table1_uav_reward(self, aid: str, sim, cfg: dict, base_components: dict) -> tuple[float, dict]:
        vals: dict[str, float | str] = {}
        weights = cfg.get("uav", {}).get("weights", {})
        if sim is not None and getattr(sim, "is_alive", False):
            height_raw, height_logs = self._tam_table1_uav_height_raw(sim, cfg)
            vals.update(height_logs)
            vals["tam_table1_uav_height"] = float(weights.get("height", 10.0)) * height_raw
            speed = float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))
            speed_raw = self._tam_table1_speed_raw(speed, cfg.get("uav", {}).get("reference_speed_mps", 300.0))
            vals["tam_table1_uav_speed"] = float(weights.get("speed", 10.0)) * speed_raw
            _target, target_logs = self._tam_table1_best_uav_target(sim)
            vals.update(target_logs)
            vals["tam_table1_uav_angle"] = float(weights.get("angle", 15.0)) * float(target_logs["tam_table1_uav_angle_raw"])
            vals["tam_table1_uav_distance"] = float(weights.get("distance", 10.0)) * float(target_logs["tam_table1_uav_distance_raw"])
            dodge_raw, dodge_angle, dodge_speed, missing = self._tam_table1_dodge_raw(sim, cfg)
            vals["tam_table1_uav_dodge"] = float(weights.get("dodge", 30.0)) * dodge_raw
            vals["tam_table1_uav_dodge_angle"] = dodge_angle
            vals["tam_table1_uav_dodge_speed"] = dodge_speed
            vals["tam_table1_uav_missing_dodge_geometry"] = missing
        else:
            vals.update({
                "tam_table1_uav_height": 0.0,
                "tam_table1_uav_height_pv": 0.0,
                "tam_table1_uav_height_ph": 0.0,
                "tam_table1_uav_speed": 0.0,
                "tam_table1_uav_angle": 0.0,
                "tam_table1_uav_distance": 0.0,
                "tam_table1_uav_dodge": 0.0,
                "tam_table1_uav_dodge_angle": 0.0,
                "tam_table1_uav_dodge_speed": 0.0,
                "tam_table1_uav_target_id_log": "",
                "tam_table1_uav_target_distance_km": 0.0,
                "tam_table1_uav_target_ata_rad": 0.0,
                "tam_table1_uav_target_aa_rad": 0.0,
                "tam_table1_uav_missing_dodge_geometry": 0.0,
            })
        event, event_logs = self._tam_table1_uav_event(aid, sim, cfg)
        vals.update(event_logs)
        total = (
            float(vals["tam_table1_uav_height"])
            + float(vals["tam_table1_uav_speed"])
            + float(vals["tam_table1_uav_angle"])
            + float(vals["tam_table1_uav_distance"])
            + float(vals["tam_table1_uav_dodge"])
            + event
        )
        orig = base_components.get(aid, {})
        vals["tam_table1_uav_brma_adv_log"] = float(orig.get("r_adv", 0.0))
        vals["tam_table1_uav_brma_end_log"] = float(orig.get("r_end", 0.0))
        vals["tam_table1_uav_launch_with_mav_shared_track_log"] = 0.0
        vals["tam_table1_uav_hit_with_mav_shared_track_log"] = 0.0
        vals["tam_table1_uav_total"] = float(total)
        return float(total), vals

    def _tam_table1_mav_dist(self, mav, cfg: dict) -> tuple[float, float]:
        alive_blue = self._tam_v2_alive_blue()
        if mav is None or not getattr(mav, "is_alive", False) or not alive_blue:
            return 0.0, 0.0
        mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
        near_d = min(float(np.linalg.norm(mav_pos - np.asarray(blue.get_position(), dtype=np.float64))) for blue in alive_blue)
        scfg = cfg.get("mav", {}).get("safety", {})
        d_danger = float(scfg.get("d_danger_m", 8000.0))
        d_safe = float(scfg.get("d_safe_m", 15000.0))
        if near_d < d_danger:
            return float(-(1.0 - near_d / max(d_danger, 1e-6))), near_d
        if near_d < d_safe:
            return float(-0.5 * (1.0 - (near_d - d_danger) / max(d_safe - d_danger, 1e-6))), near_d
        return 0.0, near_d

    def _tam_table1_mav_threat(self, mav) -> float:
        if mav is None or not getattr(mav, "is_alive", False):
            return 0.0
        if mav.check_missile_warning() is not None:
            return -1.0
        for blue in self.blue_planes.values():
            if not getattr(blue, "is_alive", False):
                continue
            metrics = self._missile_candidate_metrics(blue, mav)
            if bool(metrics.get("range_ok")) and bool(metrics.get("ao_ok")) and bool(metrics.get("ta_ok")):
                return -1.0
        return 0.0

    def _tam_table1_mav_aspect(self, mav) -> float:
        if mav is None or not getattr(mav, "is_alive", False):
            return 0.0
        mav_feat = self._tam_v2_feature(mav)
        aspect = 0.0
        for blue in self._tam_v2_alive_blue():
            blue_feat = self._tam_v2_feature(blue)
            _ao, ta, _r = get2d_AO_TA_R(mav_feat, blue_feat)
            if ta < np.pi / 4.0:
                aspect -= 1.0 - ta / (np.pi / 4.0)
        return float(aspect)

    def _tam_table1_mav_awareness(self, mav) -> tuple[float, float]:
        if mav is None or not getattr(mav, "is_alive", False):
            return 0.0, 0.0
        obs_range = float(getattr(self, "mav_observation_range_m", 80000.0))
        mav_feat = self._tam_v2_feature(mav)
        reward = 0.0
        observed = 0.0
        for blue in self._tam_v2_alive_blue():
            blue_feat = self._tam_v2_feature(blue)
            ao, _ta, distance_m = get2d_AO_TA_R(mav_feat, blue_feat)
            if distance_m < obs_range and ao < np.pi / 2.0:
                observed += 1.0
                reward += 0.3 * (1.0 - ao / (np.pi / 2.0))
        return float(reward), float(observed)

    def _tam_table1_mav_position(self, mav, cfg: dict) -> tuple[float, dict]:
        sp = cfg.get("mav", {}).get("support", {})
        d_opt = float(sp.get("d_opt_m", 8000.0))
        d_max = float(sp.get("d_max_m", 25000.0))
        center = np.asarray([0.0, 0.0], dtype=np.float64)
        if mav is None or not getattr(mav, "is_alive", False):
            d_b = 0.0
            r_pos = 0.0
        else:
            mav_xy = np.asarray(mav.get_position(), dtype=np.float64)[:2]
            d_b = float(np.linalg.norm(mav_xy - center))
            if d_b < d_opt:
                r_pos = d_b / max(d_opt, 1e-6) - 1.0
            elif d_b < d_max:
                r_pos = 1.0 - (d_b - d_opt) / max(d_max - d_opt, 1e-6)
            else:
                r_pos = -0.5
        return float(r_pos), {
            "tam_table1_mav_support_anchor_x": 0.0,
            "tam_table1_mav_support_anchor_y": 0.0,
            "tam_table1_mav_support_distance_m": float(d_b),
        }

    def _tam_table1_mav_event(self, mav_id: str, mav, cfg: dict) -> tuple[float, dict]:
        ecfg = cfg.get("mav", {}).get("event", {})
        death = 0.0
        if mav is not None and not getattr(mav, "is_alive", False) and not self._tam_table1_mav_death_penalized:
            death = -float(ecfg.get("death_penalty", 200.0))
            self._tam_table1_mav_death_penalized = True
        team_kills = sum(
            int(self._step_kill_count.get(rid, 0))
            for rid in self.red_ids
            if rid != mav_id and self.agent_roles.get(rid) == "attack_uav"
        )
        cap = float(ecfg.get("team_credit_cap", 200.0))
        available = max(0.0, cap - float(self._tam_table1_mav_team_credit_used))
        credit = min(float(ecfg.get("team_credit_per_kill", 100.0)) * team_kills, available)
        self._tam_table1_mav_team_credit_used += credit
        event = death + credit
        return float(event), {
            "tam_table1_mav_event": float(event),
            "tam_table1_mav_death": float(death),
            "tam_table1_mav_team_credit_delta": float(credit),
            "tam_table1_mav_team_credit_used": float(self._tam_table1_mav_team_credit_used),
            "tam_table1_mav_team_credit_cap": cap,
        }

    def _tam_table1_mav_reward(self, mav_id: str, mav, cfg: dict, base_components: dict) -> tuple[float, dict]:
        scfg = cfg.get("mav", {}).get("safety", {})
        spcfg = cfg.get("mav", {}).get("support", {})
        r_dist, _near_d = self._tam_table1_mav_dist(mav, cfg)
        r_threat = self._tam_table1_mav_threat(mav)
        r_aspect = self._tam_table1_mav_aspect(mav)
        safety = (
            float(scfg.get("dist_weight", 0.5)) * r_dist
            + float(scfg.get("threat_weight", 0.3)) * r_threat
            + float(scfg.get("aspect_weight", 0.2)) * r_aspect
        )
        r_pos, pos_logs = self._tam_table1_mav_position(mav, cfg)
        r_aware, observed_count = self._tam_table1_mav_awareness(mav)
        support = (
            float(spcfg.get("pos_weight", 0.6)) * r_pos
            + float(spcfg.get("aware_weight", 0.4)) * r_aware
        )
        event, event_logs = self._tam_table1_mav_event(mav_id, mav, cfg)
        total = safety + support + event
        orig = base_components.get(mav_id, {})
        vals = {
            "tam_table1_mav_safety": float(safety),
            "tam_table1_mav_dist": float(r_dist),
            "tam_table1_mav_threat": float(r_threat),
            "tam_table1_mav_aspect": float(r_aspect),
            "tam_table1_mav_support": float(support),
            "tam_table1_mav_pos": float(r_pos),
            "tam_table1_mav_aware": float(r_aware),
            "tam_table1_mav_observed_count": float(observed_count),
            "tam_table1_mav_removed_brma_adv_log": float(orig.get("r_adv", 0.0)),
            "tam_table1_mav_removed_brma_end_log": float(orig.get("r_end", 0.0)),
            "tam_table1_mav_total": float(total),
        }
        vals.update(pos_logs)
        vals.update(event_logs)
        return float(total), vals

    def _compute_tam_happo_table1_v1(self, base_rewards: dict, components: dict):
        """Rebuild red rewards from TAM-HAPPO Table 1 role terms only."""
        cfg = self.tam_happo_table1_v1_config
        mav_id = next(
            (rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"),
            self.red_ids[0] if self.red_ids else None,
        )
        shared_logs = self._tam_table1_shared_track_logs(mav_id or "")
        team_total = 0.0
        for rid in self.red_ids:
            comp = components.setdefault(rid, {})
            sim = self.red_planes.get(rid)
            if self.agent_roles.get(rid) == "mav":
                total, vals = self._tam_table1_mav_reward(rid, sim, cfg, components)
                vals.update(shared_logs)
            else:
                total, vals = self._tam_table1_uav_reward(rid, sim, cfg, components)
                vals["tam_table1_uav_launch_with_mav_shared_track_log"] = shared_logs[
                    "tam_table1_red_launch_with_mav_shared_track_log"
                ]
                vals["tam_table1_uav_hit_with_mav_shared_track_log"] = shared_logs[
                    "tam_table1_red_hit_with_mav_shared_track_log"
                ]
            vals["tam_table1_total"] = float(total)
            comp.update(vals)
            comp["total"] = float(total)
            base_rewards[rid] = float(total)
            team_total += float(total)
            components[rid] = comp
        if mav_id and mav_id in components:
            components[mav_id]["tam_table1_total"] = float(components[mav_id]["tam_table1_mav_total"])
        return base_rewards, components

    # -- BRMA/TAM scale-aligned reward v1 ------------------------------------

    def _validate_brma_tam_scale_aligned_v1_contract(self) -> None:
        cfg = self.brma_tam_scale_aligned_v1_config
        evasion = getattr(self, "missile_evasion_config", {}) or {}
        mode = str(evasion.get("mode", "")).lower()
        teams = str(evasion.get("teams", "")).lower()
        if mode != "brma_scripted":
            raise ValueError(
                "brma_tam_scale_aligned_v1 requires missile_evasion.mode='brma_scripted'"
            )
        if teams not in {"red_only", "both"}:
            raise ValueError(
                "brma_tam_scale_aligned_v1 requires missile_evasion.teams in {'red_only', 'both'}"
            )
        required = (
            ("reward_contract_revision",),
            ("uav", "progress", "distance_weight"),
            ("uav", "progress", "angle_weight"),
            ("uav", "progress", "speed_weight"),
            ("uav", "progress", "clip_min"),
            ("uav", "progress", "clip_max"),
            ("uav", "progress", "distance_optimal_km"),
            ("uav", "progress", "distance_decay_km"),
            ("uav", "event", "kill_enemy"),
            ("uav", "event", "death_or_crash"),
            ("uav", "event", "first_horizontal_out_of_zone"),
            ("mav", "role_scale"),
            ("mav", "safety", "dist_weight"),
            ("mav", "safety", "threat_weight"),
            ("mav", "safety", "aspect_weight"),
            ("mav", "safety", "d_danger_m"),
            ("mav", "safety", "d_safe_m"),
            ("mav", "support", "pos_weight"),
            ("mav", "support", "aware_weight"),
            ("mav", "support", "d_opt_m"),
            ("mav", "support", "d_max_m"),
            ("mav", "event", "death_penalty"),
            ("mav", "event", "full_enemy_team_credit"),
            ("terminal", "coefficient"),
            ("terminal", "mav_loss_weight_mode"),
            ("logging", "log_raw_potentials"),
            ("logging", "log_progress_reset_reason"),
            ("logging", "log_scale_diagnostics"),
        )
        for path in required:
            value = cfg
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(
                        "brma_tam_scale_aligned_v1 requires explicit config key "
                        + ".".join(path)
                    )
                value = value[key]
        if int(cfg["reward_contract_revision"]) != 3:
            raise ValueError("brma_tam_scale_aligned_v1 requires reward_contract_revision=3")
        if cfg["terminal"]["mav_loss_weight_mode"] != "match_initial_attack_uav_count":
            raise ValueError("unsupported scale-aligned MAV terminal weighting")
        if self.observation_mode != "mav_shared_geo":
            raise ValueError("brma_tam_scale_aligned_v1 requires observation_mode='mav_shared_geo'")
        if getattr(self, "red_target_selection_mode", "closest") != "closest":
            raise ValueError("brma_tam_scale_aligned_v1 requires red_target_selection_mode='closest'")
        if int(self.aircraft_type_params.get("mav", {}).get("num_missiles", 0)) != 0:
            raise ValueError("brma_tam_scale_aligned_v1 requires MAV num_missiles == 0")

    @staticmethod
    def _scale_v1_distance_potential(distance_m: float, optimal_km: float = 5.0,
                                     decay_km: float = 10.0) -> float:
        d_km = float(distance_m) / 1000.0
        if not np.isfinite(d_km):
            return 0.0
        return float(np.exp(-max(d_km - float(optimal_km), 0.0) / max(float(decay_km), 1e-6)))

    @staticmethod
    def _scale_v1_angle_potential(angle_raw: float) -> float:
        return float(np.clip((float(angle_raw) + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _scale_v1_speed_potential(speed_raw: float) -> float:
        return float(np.clip((float(speed_raw) + 1.0) / 2.0, 0.0, 1.0))

    def _scale_v1_uav_progress(self, aid: str, target_id: str | None,
                               geom: dict, speed_logs: dict) -> dict:
        pcfg = self.brma_tam_scale_aligned_v1_config["uav"]["progress"]
        cache = self._scale_v1_progress_cache.get(aid)
        target = self.blue_planes.get(target_id) if target_id else None
        reason = "none"
        valid = bool(target is not None and getattr(target, "is_alive", False))
        if not valid:
            reason = "no_alive_target" if target_id is None else "target_dead"
        elif float(geom.get("tam_geometry_valid", 0.0)) <= 0.5:
            reason = "invalid_geometry"
            valid = False
        elif float(speed_logs.get("speed_ratio_valid", 0.0)) <= 0.5:
            reason = "invalid_speed"
            valid = False

        if not valid:
            self._scale_v1_progress_cache.pop(aid, None)
            return {
                "scale_v1_phi_distance": 0.0, "scale_v1_phi_angle": 0.0,
                "scale_v1_phi_speed": 0.0, "scale_v1_delta_distance": 0.0,
                "scale_v1_delta_angle": 0.0, "scale_v1_delta_speed": 0.0,
                "scale_v1_progress_raw": 0.0, "scale_v1_progress_clipped": 0.0,
                "scale_v1_progress_reset_flag": 1.0,
                "scale_v1_progress_reset_reason": reason,
            }

        phi_d = self._scale_v1_distance_potential(
            geom["target_distance_m"], pcfg["distance_optimal_km"], pcfg["distance_decay_km"]
        )
        phi_a = self._scale_v1_angle_potential(geom["tam_angle_raw"])
        phi_v = self._scale_v1_speed_potential(speed_logs["tam_speed_raw"])
        reset = False
        if cache is None:
            reason, reset = "episode_start", True
        elif cache.get("target_id") != target_id:
            previous = self.blue_planes.get(cache.get("target_id"))
            reason = "target_dead" if previous is None or not getattr(previous, "is_alive", False) else "target_switch"
            reset = True
            self._scale_v1_reward_target_switch_counts[aid] = (
                self._scale_v1_reward_target_switch_counts.get(aid, 0) + 1
            )
        if reset:
            delta_d = delta_a = delta_v = raw = clipped = 0.0
        else:
            delta_d = phi_d - float(cache["phi_distance"])
            delta_a = phi_a - float(cache["phi_angle"])
            delta_v = phi_v - float(cache["phi_speed"])
            raw = (
                float(pcfg["distance_weight"]) * delta_d
                + float(pcfg["angle_weight"]) * delta_a
                + float(pcfg["speed_weight"]) * delta_v
            )
            clipped = float(np.clip(raw, float(pcfg["clip_min"]), float(pcfg["clip_max"])))
        self._scale_v1_progress_cache[aid] = {
            "target_id": target_id, "phi_distance": phi_d,
            "phi_angle": phi_a, "phi_speed": phi_v,
        }
        return {
            "scale_v1_phi_distance": phi_d, "scale_v1_phi_angle": phi_a,
            "scale_v1_phi_speed": phi_v, "scale_v1_delta_distance": delta_d,
            "scale_v1_delta_angle": delta_a, "scale_v1_delta_speed": delta_v,
            "scale_v1_progress_raw": raw, "scale_v1_progress_clipped": clipped,
            "scale_v1_progress_reset_flag": float(reset),
            "scale_v1_progress_reset_reason": reason,
        }

    def _scale_v1_uav_event(self, aid: str, sim) -> tuple[float, dict]:
        cfg = self.brma_tam_scale_aligned_v1_config["uav"]["event"]
        kills = int(getattr(self, "_step_kill_count", {}).get(aid, 0))
        kill = float(cfg["kill_enemy"]) * kills
        death = oob = 0.0
        alive_before = bool(self._scale_v1_alive_before_step.get(aid, getattr(sim, "is_alive", False)))
        if alive_before and not getattr(sim, "is_alive", False) and aid not in self._scale_v1_uav_death_penalized:
            death = float(cfg["death_or_crash"])
            self._scale_v1_uav_death_penalized.add(aid)
        elif self._brma_tam_horizontal_oob(sim) and aid not in self._scale_v1_uav_horizontal_oob_penalized:
            oob = float(cfg["first_horizontal_out_of_zone"])
            self._scale_v1_uav_horizontal_oob_penalized.add(aid)
        total = kill + death + oob
        return total, {
            "scale_v1_uav_event_kill": kill, "scale_v1_uav_event_death": death,
            "scale_v1_uav_event_oob": oob, "scale_v1_uav_event_total": total,
        }

    def _scale_v1_mav_role(self, mav) -> tuple[float, dict]:
        cfg = self.brma_tam_scale_aligned_v1_config["mav"]
        _unused, safety = self._brma_tam_mav_safety(mav, {"mav": {"safety": cfg["safety"]}})
        alive_blue = [
            self.blue_planes[bid] for bid in self.blue_ids
            if self.blue_planes.get(bid) is not None and self.blue_planes[bid].is_alive
        ]
        alive_uav = [
            self.red_planes[rid] for rid in self.red_ids
            if self.agent_roles.get(rid) == "attack_uav"
            and self.red_planes.get(rid) is not None and self.red_planes[rid].is_alive
        ]
        spcfg = cfg["support"]
        r_pos = center_distance = 0.0
        if alive_blue and alive_uav:
            center = np.mean([
                self._brma_tam_safe_vec(sim, "get_position")[:2]
                for sim in alive_blue + alive_uav
            ], axis=0)
            center_distance = float(np.linalg.norm(
                self._brma_tam_safe_vec(mav, "get_position")[:2] - center
            ))
            r_pos = self._brma_tam_mav_pos_raw(center_distance, spcfg["d_opt_m"], spcfg["d_max_m"])
        aware_sum = 0.0
        mav_pos = self._brma_tam_safe_vec(mav, "get_position")
        mav_vel = self._brma_tam_safe_vec(mav, "get_velocity")
        mav_speed = float(np.linalg.norm(mav_vel))
        mav_id = str(getattr(mav, "uid", "red_0"))
        for bid in self.blue_ids:
            blue = self.blue_planes.get(bid)
            if blue is None or not blue.is_alive or not self._mav_shared_track_state(mav_id, bid)["observed"]:
                continue
            los = self._brma_tam_safe_vec(blue, "get_position") - mav_pos
            distance = float(np.linalg.norm(los))
            if distance > 1e-8 and mav_speed > 1e-8 and np.isfinite(distance):
                ao = float(np.arccos(np.clip(np.dot(los / distance, mav_vel / mav_speed), -1.0, 1.0)))
                if ao < np.pi / 2.0:
                    aware_sum += 0.3 * (1.0 - ao / (np.pi / 2.0))
        blue_count = len(alive_blue)
        aspect_sum = float(safety["mav_aspect_raw_sum"])
        aspect_mean = aspect_sum / max(float(blue_count), 1.0)
        aware_mean = aware_sum / max(float(blue_count), 1.0)
        raw = (
            float(cfg["safety"]["dist_weight"]) * float(safety["mav_dist_raw"])
            + float(cfg["safety"]["threat_weight"]) * float(safety["mav_threat_raw"])
            + float(cfg["safety"]["aspect_weight"]) * aspect_mean
            + float(spcfg["pos_weight"]) * r_pos
            + float(spcfg["aware_weight"]) * aware_mean
        )
        role = float(cfg["role_scale"]) * raw
        return role, {
            "scale_v1_mav_dist_raw": float(safety["mav_dist_raw"]),
            "scale_v1_mav_threat_raw": float(safety["mav_threat_raw"]),
            "scale_v1_mav_aspect_raw_sum": aspect_sum,
            "scale_v1_mav_aspect_mean": aspect_mean,
            "scale_v1_mav_pos_raw": r_pos,
            "scale_v1_mav_aware_raw_sum": aware_sum,
            "scale_v1_mav_aware_mean": aware_mean,
            "scale_v1_mav_center_distance_m": center_distance,
            "scale_v1_mav_alive_blue_count": float(blue_count),
            "scale_v1_mav_role_raw": raw, "scale_v1_mav_role": role,
        }

    def _scale_v1_mav_event(self, aid: str, mav) -> tuple[float, dict]:
        cfg = self.brma_tam_scale_aligned_v1_config["mav"]["event"]
        alive_before = bool(self._scale_v1_alive_before_step.get(aid, getattr(mav, "is_alive", False)))
        death = 0.0
        if alive_before and not getattr(mav, "is_alive", False) and not self._scale_v1_mav_death_penalized:
            death = float(cfg["death_penalty"])
            self._scale_v1_mav_death_penalized = True
        new_kills = sum(
            int(getattr(self, "_step_kill_count", {}).get(rid, 0))
            for rid in self.red_ids if self.agent_roles.get(rid) == "attack_uav"
        ) if alive_before else 0
        full_credit = float(cfg["full_enemy_team_credit"])
        per_kill = full_credit / max(float(len(self.blue_ids)), 1.0)
        available = max(0.0, full_credit - self._scale_v1_mav_team_credit_used)
        credit = min(per_kill * new_kills, available) if alive_before else 0.0
        self._scale_v1_mav_team_credit_used += credit
        total = death + credit
        return total, {
            "scale_v1_mav_event_death": death,
            "scale_v1_mav_team_credit_delta": credit,
            "scale_v1_mav_team_credit_used": self._scale_v1_mav_team_credit_used,
            "scale_v1_mav_event_total": total,
        }

    def _scale_v1_terminal_value(self) -> tuple[float, float, float]:
        initial_blue = max(len(self.blue_ids), 1)
        alive_blue = sum(1 for bid in self.blue_ids if self.blue_planes.get(bid) and self.blue_planes[bid].is_alive)
        attack_ids = [rid for rid in self.red_ids if self.agent_roles.get(rid) == "attack_uav"]
        initial_attack = len(attack_ids)
        alive_attack = sum(1 for rid in attack_ids if self.red_planes.get(rid) and self.red_planes[rid].is_alive)
        mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"), None)
        mav_dead = float(bool(mav_id and (not self.red_planes.get(mav_id) or not self.red_planes[mav_id].is_alive)))
        blue_loss = (initial_blue - alive_blue) / float(initial_blue)
        mav_weight = float(initial_attack)
        red_loss = (mav_weight * mav_dead + (initial_attack - alive_attack)) / max(mav_weight + initial_attack, 1.0)
        coefficient = float(self.brma_tam_scale_aligned_v1_config["terminal"]["coefficient"])
        terminal = float(np.clip(coefficient * (blue_loss - red_loss), -coefficient, coefficient))
        return terminal, float(blue_loss), float(red_loss)

    def _compute_brma_tam_scale_aligned_v1(self, base_rewards: dict, components: dict):
        n_blue_alive = sum(1 for sim in self.blue_planes.values() if sim.is_alive)
        n_red_alive = sum(1 for sim in self.red_planes.values() if sim.is_alive)
        round_over = n_blue_alive == 0 or n_red_alive == 0 or self.current_step >= self.max_steps
        terminal = blue_loss = red_loss = terminal_applied = 0.0
        if round_over and not self._scale_v1_terminal_applied:
            terminal, blue_loss, red_loss = self._scale_v1_terminal_value()
            self._scale_v1_terminal_applied = True
            terminal_applied = 1.0
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            comp = components.setdefault(rid, {})
            role = self.agent_roles.get(rid, "")
            alive_before = bool(self._scale_v1_alive_before_step.get(rid, getattr(sim, "is_alive", False)))
            vals = {
                "reward_contract_revision": 3.0,
                "scale_v1_flight_pitch": float(comp.get("r_pitch", 0.0)),
                "scale_v1_flight_roll": float(comp.get("r_roll", 0.0)),
                "scale_v1_flight_altitude": float(comp.get("r_alt", 0.0)),
                "scale_v1_flight_boundary": float(comp.get("r_bound", 0.0)),
                "scale_v1_flight_velocity": float(comp.get("r_vel", 0.0)),
                "scale_v1_brma_adv_log_only": float(comp.get("r_adv", 0.0)),
                "scale_v1_brma_end_log_only": float(comp.get("r_end", 0.0)),
                "scale_v1_brma_death_log_only": float(comp.get("r_death", 0.0)),
                "scale_v1_terminal": terminal if alive_before else 0.0,
                "scale_v1_blue_loss_fraction": blue_loss,
                "scale_v1_red_loss_fraction": red_loss,
                "scale_v1_terminal_applied": terminal_applied,
            }
            if not alive_before or sim is None:
                for key in tuple(vals):
                    if key.startswith("scale_v1_flight_") or key == "scale_v1_terminal":
                        vals[key] = 0.0
                total = 0.0
                vals.update({
                    "scale_v1_flight_total": 0.0, "scale_v1_total": 0.0,
                    "scale_v1_identity_error": 0.0,
                    "scale_v1_uav_total": 0.0, "scale_v1_mav_total": 0.0,
                })
            else:
                flight = sum(vals[key] for key in (
                    "scale_v1_flight_pitch", "scale_v1_flight_roll",
                    "scale_v1_flight_altitude", "scale_v1_flight_boundary",
                    "scale_v1_flight_velocity",
                ))
                vals["scale_v1_flight_total"] = flight
                if role == "mav":
                    role_reward, role_logs = self._scale_v1_mav_role(sim)
                    event, event_logs = self._scale_v1_mav_event(rid, sim)
                    vals.update(role_logs)
                    vals.update(event_logs)
                    total = flight + role_reward + event + vals["scale_v1_terminal"]
                    vals["scale_v1_mav_total"] = total
                    vals["scale_v1_uav_total"] = 0.0
                else:
                    target_id, target, _distance = self._brma_tam_closest_alive_blue(sim)
                    if target is None:
                        geom = {"target_distance_m": 0.0, "tam_angle_raw": 0.0, "tam_geometry_valid": 0.0}
                        speed_logs = self._brma_tam_speed_raw(0.0, 0.0)
                    else:
                        geom = self._brma_tam_3d_geometry(sim, target)
                        speed_logs = self._brma_tam_speed_raw(
                            float(np.linalg.norm(self._brma_tam_safe_vec(sim, "get_velocity"))),
                            float(np.linalg.norm(self._brma_tam_safe_vec(target, "get_velocity"))),
                        )
                    progress = self._scale_v1_uav_progress(rid, target_id, geom, speed_logs)
                    event, event_logs = self._scale_v1_uav_event(rid, sim)
                    vals.update(progress)
                    vals.update(event_logs)
                    vals.update({
                        "scale_v1_reward_target_id": target_id or "",
                        "scale_v1_reward_target_distance_m": float(geom["target_distance_m"]),
                        "scale_v1_reward_target_valid": float(target is not None),
                        "scale_v1_geometry_valid": float(geom.get("tam_geometry_valid", 0.0)),
                        "scale_v1_reward_target_switch_count": float(self._scale_v1_reward_target_switch_counts.get(rid, 0)),
                    })
                    total = flight + progress["scale_v1_progress_clipped"] + event + vals["scale_v1_terminal"]
                    vals["scale_v1_uav_total"] = total
                    vals["scale_v1_mav_total"] = 0.0
                vals["scale_v1_total"] = total
                expected = (
                    flight
                    + (vals.get("scale_v1_mav_role", 0.0) if role == "mav" else vals.get("scale_v1_progress_clipped", 0.0))
                    + (vals.get("scale_v1_mav_event_total", 0.0) if role == "mav" else vals.get("scale_v1_uav_event_total", 0.0))
                    + vals["scale_v1_terminal"]
                )
                vals["scale_v1_identity_error"] = float(total - expected)
                if abs(vals["scale_v1_identity_error"]) > 1e-6:
                    raise ValueError(f"scale-aligned reward identity failure agent={rid}")
            comp.update(vals)
            comp["total"] = float(total)
            base_rewards[rid] = float(total)
        return base_rewards, components

    # -- BRMA/TAM scripted composite reward v1 -------------------------------

    def _validate_brma_tam_scripted_composite_v1_contract(self) -> None:
        evasion = getattr(self, "missile_evasion_config", {}) or {}
        mode = str(evasion.get("mode", "")).lower()
        teams = str(evasion.get("teams", "")).lower()
        if mode != "brma_scripted" or teams not in {"red_only", "both"}:
            raise ValueError(
                "brma_tam_scripted_composite_v1 assumes red evasion is scripted; "
                "R_DM is diagnostic, not active. Use missile_evasion.mode='brma_scripted' "
                "and teams in {'red_only', 'both'}."
            )
        if self.observation_mode != "mav_shared_geo":
            raise ValueError("brma_tam_scripted_composite_v1 requires observation_mode='mav_shared_geo'")
        if getattr(self, "red_target_selection_mode", "closest") != "closest":
            raise ValueError("brma_tam_scripted_composite_v1 requires red_target_selection_mode='closest'")
        mav_params = self.aircraft_type_params.get("mav", {})
        if int(mav_params.get("num_missiles", 0)) != 0:
            raise ValueError("brma_tam_scripted_composite_v1 requires MAV num_missiles == 0")

    @staticmethod
    def _brma_tam_safe_vec(sim, getter: str) -> np.ndarray:
        try:
            value = getattr(sim, getter)()
        except Exception:
            return np.full(3, np.nan, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            out = np.full(3, np.nan, dtype=np.float64)
            out[:arr.size] = arr
            return out
        return arr[:3]

    @staticmethod
    def _brma_tam_3d_geometry(red, blue, eps: float = 1e-8) -> dict:
        red_pos = HeteroUavCombatEnv._brma_tam_safe_vec(red, "get_position")
        blue_pos = HeteroUavCombatEnv._brma_tam_safe_vec(blue, "get_position")
        red_vel = HeteroUavCombatEnv._brma_tam_safe_vec(red, "get_velocity")
        blue_vel = HeteroUavCombatEnv._brma_tam_safe_vec(blue, "get_velocity")
        d_vec = blue_pos - red_pos
        distance = float(np.linalg.norm(d_vec))
        red_speed = float(np.linalg.norm(red_vel))
        blue_speed = float(np.linalg.norm(blue_vel))
        out = {
            "target_distance_m": distance if np.isfinite(distance) else 0.0,
            "tam_ata_rad": 0.0,
            "tam_aa_rad": 0.0,
            "tam_angle_raw": 0.0,
            "tam_geometry_valid": 0.0,
        }
        if (
            not np.isfinite(distance) or not np.isfinite(red_speed) or not np.isfinite(blue_speed)
            or distance < eps or red_speed < eps or blue_speed < eps
        ):
            return out
        los = d_vec / distance
        ata_cos = float(np.clip(np.dot(los, red_vel / red_speed), -1.0, 1.0))
        aa_cos = float(np.clip(np.dot(los, blue_vel / blue_speed), -1.0, 1.0))
        ata = float(np.arccos(ata_cos))
        aa = float(np.arccos(aa_cos))
        raw = float(1.0 - (ata + aa) / np.pi)
        if not np.isfinite(raw):
            raw = 0.0
        out.update({
            "tam_ata_rad": ata,
            "tam_aa_rad": aa,
            "tam_angle_raw": raw,
            "tam_geometry_valid": 1.0,
        })
        return out

    @staticmethod
    def _brma_tam_speed_raw(red_speed_mps: float, target_speed_mps: float, eps: float = 1e-8) -> dict:
        vr = float(red_speed_mps)
        vb = float(target_speed_mps)
        if not np.isfinite(vr) or not np.isfinite(vb) or vr < eps:
            return {
                "tam_speed_raw": 0.0,
                "own_speed_mps": vr if np.isfinite(vr) else 0.0,
                "target_speed_mps": vb if np.isfinite(vb) else 0.0,
                "speed_ratio": 0.0,
                "speed_ratio_valid": 0.0,
            }
        ratio = vb / vr
        if vb < 0.5 * vr:
            raw = 1.0
        elif vb <= 1.5 * vr:
            raw = 2.0 - 2.0 * ratio
        else:
            raw = -1.0
        return {
            "tam_speed_raw": float(raw),
            "own_speed_mps": vr,
            "target_speed_mps": vb,
            "speed_ratio": float(ratio),
            "speed_ratio_valid": 1.0,
        }

    @staticmethod
    def _brma_tam_distance_raw(distance_m: float) -> dict:
        d = float(distance_m)
        if not np.isfinite(d):
            return {"tam_distance_raw": 0.0, "target_distance_m": 0.0, "reward_distance_zone_code": -1.0}
        km = d / 1000.0
        if km <= 5.0:
            raw, zone = 1.0, 0.0
        elif km < 10.0:
            raw, zone = float(np.exp(-0.921 * (km - 5.0))), 1.0
        else:
            raw, zone = -1.0, 2.0
        return {"tam_distance_raw": float(raw), "target_distance_m": d, "reward_distance_zone_code": zone}

    def _brma_tam_closest_alive_blue(self, sim):
        pos = self._brma_tam_safe_vec(sim, "get_position")
        best = (None, None, float("inf"))
        for bid in self.blue_ids:
            blue = self.blue_planes.get(bid)
            if blue is None or not getattr(blue, "is_alive", False):
                continue
            dist = float(np.linalg.norm(self._brma_tam_safe_vec(blue, "get_position") - pos))
            if np.isfinite(dist) and dist < best[2]:
                best = (bid, blue, dist)
        return best

    def _mav_shared_track_state(self, observer_id: str, target_id: str) -> dict:
        """Return current-state visibility without changing observation semantics."""
        observer = self._get_sim(observer_id)
        target = self._get_sim(target_id)
        target_alive = bool(target is not None and getattr(target, "is_alive", False))
        observer_alive = bool(observer is not None and getattr(observer, "is_alive", False))
        direct = False
        shared = False
        if target_alive and observer_alive:
            if observer_id.startswith("red_") and self.agent_roles.get(observer_id) == "mav":
                direct = self._distance_m(observer, target) <= float(self.mav_observation_range_m)
            else:
                direct = self._distance_m(observer, target) <= float(
                    getattr(self, "uav_direct_observation_range_m", 10000.0)
                )
                if observer_id.startswith("red_"):
                    mav = self._get_red_mav_sim()
                    shared = bool(
                        self.agent_roles.get(observer_id) != "mav"
                        and mav is not None
                        and getattr(mav, "is_alive", False)
                        and self._distance_m(mav, target) <= float(self.mav_observation_range_m)
                    )
        observed = bool(direct or shared)
        if direct and shared:
            source = "direct_and_mav_shared"
        elif direct:
            source = "direct"
        elif shared:
            source = "mav_shared"
        else:
            source = "unobserved" if target_alive else "target_dead"
        return {
            "target_alive": target_alive,
            "direct_visible": bool(direct),
            "mav_shared_visible": bool(shared),
            "observed": observed,
            "track_source": source,
        }

    def _brma_tam_track_logs(self, aid: str, target_id: str | None) -> dict:
        logs = {
            "reward_target_observed": 0.0,
            "reward_target_direct_visible": 0.0,
            "reward_target_mav_shared_visible": 0.0,
            "reward_target_unavailable": 1.0,
            "reward_target_matches_lock": 0.0,
            "reward_target_matches_launch": 0.0,
            "reward_target_track_source_direct": 0.0,
            "reward_target_track_source_mav_shared": 0.0,
            "reward_target_track_source_direct_and_mav_shared": 0.0,
            "reward_target_track_source_unknown": 1.0,
        }
        if target_id is None:
            return logs
        state = self._mav_shared_track_state(aid, target_id)
        direct = float(state["direct_visible"])
        shared = float(state["mav_shared_visible"])
        observed = float(state["observed"])
        logs.update({
            "reward_target_observed": observed,
            "reward_target_direct_visible": direct,
            "reward_target_mav_shared_visible": shared,
            "reward_target_unavailable": 0.0 if observed else 1.0,
            "reward_target_track_source_unknown": 0.0 if observed else 1.0,
        })
        if direct and shared:
            logs["reward_target_track_source_direct_and_mav_shared"] = 1.0
        elif direct:
            logs["reward_target_track_source_direct"] = 1.0
        elif shared:
            logs["reward_target_track_source_mav_shared"] = 1.0
        return logs

    def _brma_tam_dodge_diagnostic(self, aid: str, sim) -> tuple[dict, str]:
        logs = {
            "script_selected_missile_numeric": 0.0,
            "incoming_range_m": 0.0,
            "incoming_closing_speed_mps": 0.0,
            "incoming_t_go_sec": 0.0,
            "tam_dodge_raw_log": 0.0,
            "tam_dodge_angle_log": 0.0,
            "tam_dodge_speed_log": 0.0,
            "tam_dodge_geometry_valid": 0.0,
            "tam_dodge_missing_reason": "no_scripted_override",
            "evasion_override_active": 0.0,
        }
        selected = ""
        for rec in getattr(self, "_evasion_step_records", []) or []:
            if rec.get("evasion_agent_id") != aid:
                continue
            selected = str(rec.get("incoming_missile_id", "") or "")
            logs["evasion_override_active"] = 1.0
            logs["script_selected_missile_numeric"] = 1.0 if selected else 0.0
            if not selected:
                logs["tam_dodge_missing_reason"] = "missing_missile_id"
                return logs, selected
            missile = getattr(self, "_missiles_in_flight", {}).get(selected)
            if missile is None:
                logs["tam_dodge_missing_reason"] = "missile_not_found"
                return logs, selected
            if not getattr(missile, "is_alive", False):
                logs["tam_dodge_missing_reason"] = "missile_inactive"
                return logs, selected
            aircraft_pos = self._brma_tam_safe_vec(sim, "get_position")
            aircraft_vel = self._brma_tam_safe_vec(sim, "get_velocity")
            missile_pos = self._brma_tam_safe_vec(missile, "get_position")
            missile_vel = self._brma_tam_safe_vec(missile, "get_velocity")
            los = aircraft_pos - missile_pos
            los_norm = float(np.linalg.norm(los))
            missile_speed = float(np.linalg.norm(missile_vel))
            if not np.isfinite(los_norm) or los_norm < 1e-8:
                logs["tam_dodge_missing_reason"] = "invalid_los"
                return logs, selected
            if not np.isfinite(missile_speed) or missile_speed < 1e-8:
                logs["tam_dodge_missing_reason"] = "invalid_missile_speed"
                return logs, selected
            los_hat = los / los_norm
            angle_raw = -float(np.clip(np.dot(missile_vel / missile_speed, los_hat), -1.0, 1.0))
            prev_speed = getattr(self, "_brma_tam_missile_speed_cache", {}).get(selected)
            norm = float(
                self.brma_tam_scripted_composite_v1_config.get("uav", {}).get(
                    "dodge_speed_norm_mps", 1000.0
                )
            )
            speed_raw = 0.0 if prev_speed is None else (float(prev_speed) - missile_speed) / max(norm, 1e-6)
            self._brma_tam_missile_speed_cache[selected] = missile_speed
            closing = float(np.dot(missile_vel - aircraft_vel, los_hat))
            logs.update({
                "incoming_range_m": los_norm,
                "incoming_closing_speed_mps": closing if np.isfinite(closing) else 0.0,
                "incoming_t_go_sec": los_norm / closing if np.isfinite(closing) and closing > 1e-8 else 0.0,
                "tam_dodge_raw_log": angle_raw + speed_raw,
                "tam_dodge_angle_log": angle_raw,
                "tam_dodge_speed_log": speed_raw,
                "tam_dodge_geometry_valid": 1.0,
                "tam_dodge_missing_reason": "",
            })
            return logs, selected
        return logs, selected

    def _brma_tam_horizontal_oob(self, sim) -> bool:
        pos = self._brma_tam_safe_vec(sim, "get_position")
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        return bool(np.isfinite(pos[0]) and np.isfinite(pos[1]) and (abs(pos[0]) > half or abs(pos[1]) > half))

    def _brma_tam_death_reason(self, aid: str) -> str:
        reason = getattr(self, "_death_reasons", {}).get(aid, "") or ""
        if reason:
            return str(reason)
        for ev in getattr(self, "_death_events_step", []) or []:
            if isinstance(ev, dict) and ev.get("agent_id") == aid:
                return str(ev.get("death_reason", ""))
        return ""

    def _brma_tam_uav_event(self, aid: str, sim, cfg: dict) -> tuple[float, dict]:
        ecfg = cfg.get("uav", {})
        kills = int(getattr(self, "_step_kill_count", {}).get(aid, 0))
        kill_reward = float((ecfg.get("kill_enemy", 200.0))) * kills
        loss = 0.0
        oob = 0.0
        alive_before = bool(getattr(self, "_brma_tam_alive_before_step", {}).get(aid, getattr(sim, "is_alive", False)))
        died_this_step = alive_before and not getattr(sim, "is_alive", False)
        if died_this_step and aid not in getattr(self, "_brma_tam_uav_death_penalized", set()):
            loss = float(ecfg.get("death_or_crash", -200.0))
            self._brma_tam_uav_death_penalized.add(aid)
        elif self._brma_tam_horizontal_oob(sim) and aid not in getattr(self, "_brma_tam_uav_horizontal_oob_penalized", set()):
            oob = float(ecfg.get("first_horizontal_out_of_zone", -100.0))
            self._brma_tam_uav_horizontal_oob_penalized.add(aid)
        total = kill_reward + loss + oob
        return total, {
            "uav_event_kill": float(kill_reward),
            "uav_event_loss": float(loss),
            "uav_event_first_horizontal_out_of_zone": float(oob),
            "uav_event_total": float(total),
            "above_altitude_max_steps": 1.0 if self._brma_tam_safe_vec(sim, "get_position")[2] > float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0)) else 0.0,
            "max_altitude_m": float(self._brma_tam_safe_vec(sim, "get_position")[2]),
            "above_altitude_max_episode_flag": 1.0 if self._brma_tam_safe_vec(sim, "get_position")[2] > float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0)) else 0.0,
        }

    @staticmethod
    def _brma_tam_mav_dist_raw(distance_m: float, d_danger: float, d_safe: float) -> float:
        d = float(distance_m)
        if not np.isfinite(d):
            return 0.0
        if d < d_danger:
            return float(-(1.0 - d / max(d_danger, 1e-6)))
        if d < d_safe:
            return float(-0.5 * (1.0 - (d - d_danger) / max(d_safe - d_danger, 1e-6)))
        return 0.2

    @staticmethod
    def _brma_tam_mav_pos_raw(distance_m: float, d_opt: float, d_max: float) -> float:
        d = float(distance_m)
        if not np.isfinite(d):
            return 0.0
        if d < d_opt:
            return float(d / max(d_opt, 1e-6) - 1.0)
        if d < d_max:
            return float(1.0 - (d - d_opt) / max(d_max - d_opt, 1e-6))
        return -0.5

    def _brma_tam_mav_safety(self, mav, cfg: dict) -> tuple[float, dict]:
        scfg = cfg.get("mav", {}).get("safety", {})
        d_danger = float(scfg.get("d_danger_m", 8000.0))
        d_safe = float(scfg.get("d_safe_m", 15000.0))
        mav_pos = self._brma_tam_safe_vec(mav, "get_position")
        alive_blues = [self.blue_planes[bid] for bid in self.blue_ids if self.blue_planes.get(bid) is not None and self.blue_planes[bid].is_alive]
        near_d = 0.0
        r_dist = 0.0
        if alive_blues:
            near_d = min(float(np.linalg.norm(self._brma_tam_safe_vec(b, "get_position") - mav_pos)) for b in alive_blues)
            r_dist = self._brma_tam_mav_dist_raw(near_d, d_danger, d_safe)
        incoming = [
            m for m in getattr(mav, "under_missiles", []) or []
            if getattr(m, "is_alive", False)
            and (
                getattr(m, "target_aircraft", None) is mav
                or str(getattr(m, "_target_id", "") or "") == str(getattr(mav, "uid", ""))
            )
        ]
        r_threat = -1.0 if incoming else 0.0
        prelaunch_count = 0
        for blue in alive_blues:
            try:
                metrics = self._missile_candidate_metrics(blue, mav)
            except Exception:
                continue
            if bool(metrics.get("launch_geometry_ok_3d", False)):
                prelaunch_count += 1
        r_aspect = 0.0
        if alive_blues:
            for blue in alive_blues:
                vec = mav_pos - self._brma_tam_safe_vec(blue, "get_position")
                blue_vel = self._brma_tam_safe_vec(blue, "get_velocity")
                dist = float(np.linalg.norm(vec))
                spd = float(np.linalg.norm(blue_vel))
                if dist < 1e-8 or spd < 1e-8 or not np.isfinite(dist) or not np.isfinite(spd):
                    continue
                alpha = float(np.arccos(np.clip(np.dot(vec / dist, blue_vel / spd), -1.0, 1.0)))
                if alpha < np.pi / 4.0:
                    r_aspect -= float(1.0 - alpha / (np.pi / 4.0))
        weighted = (
            float(scfg.get("dist_weight", 0.5)) * r_dist
            + float(scfg.get("threat_weight", 0.3)) * r_threat
            + float(scfg.get("aspect_weight", 0.2)) * r_aspect
        )
        return weighted, {
            "mav_dist_raw": float(r_dist),
            "mav_dist_weighted": float(scfg.get("dist_weight", 0.5)) * float(r_dist),
            "mav_nearest_blue_distance_m": float(near_d),
            "mav_threat_raw": float(r_threat),
            "mav_threat_weighted": float(scfg.get("threat_weight", 0.3)) * float(r_threat),
            "mav_actual_incoming_missile_count": float(len(incoming)),
            "mav_prelaunch_geometry_threat_log": -float(prelaunch_count),
            "mav_prelaunch_geometry_threat_count_log": float(prelaunch_count),
            "mav_aspect_raw_sum": float(r_aspect),
            "mav_aspect_weighted": float(scfg.get("aspect_weight", 0.2)) * float(r_aspect),
            "mav_aspect_per_blue_mean": float(r_aspect) / max(float(len(alive_blues)), 1.0),
            "alive_blue_count": float(len(alive_blues)),
        }

    def _brma_tam_mav_support(self, mav, cfg: dict) -> tuple[float, dict]:
        spcfg = cfg.get("mav", {}).get("support", {})
        alive_blue = [self.blue_planes[bid] for bid in self.blue_ids if self.blue_planes.get(bid) is not None and self.blue_planes[bid].is_alive]
        alive_uav = [
            self.red_planes[rid] for rid in self.red_ids
            if self.agent_roles.get(rid) == "attack_uav"
            and self.red_planes.get(rid) is not None
            and self.red_planes[rid].is_alive
        ]
        center_valid = 0.0
        center = np.zeros(2, dtype=np.float64)
        r_pos = 0.0
        mav_center_distance = 0.0
        all_attack_dead = 1.0 if not alive_uav else 0.0
        if alive_blue and alive_uav:
            pts = [self._brma_tam_safe_vec(s, "get_position")[:2] for s in alive_blue + alive_uav]
            center = np.mean(pts, axis=0)
            mav_xy = self._brma_tam_safe_vec(mav, "get_position")[:2]
            mav_center_distance = float(np.linalg.norm(mav_xy - center))
            d_opt = float(spcfg.get("d_opt_m", 8000.0))
            d_max = float(spcfg.get("d_max_m", 25000.0))
            r_pos = self._brma_tam_mav_pos_raw(mav_center_distance, d_opt, d_max)
            center_valid = 1.0
        r_aware = 0.0
        mav_observed = 0
        mav_pos = self._brma_tam_safe_vec(mav, "get_position")
        mav_vel = self._brma_tam_safe_vec(mav, "get_velocity")
        mav_speed = float(np.linalg.norm(mav_vel))
        for bid in self.blue_ids:
            blue = self.blue_planes.get(bid)
            if blue is None or not getattr(blue, "is_alive", False):
                continue
            visibility = self._mav_shared_track_state(getattr(mav, "uid", "red_0"), bid)
            if not visibility["observed"]:
                continue
            mav_observed += 1
            d_vec = self._brma_tam_safe_vec(blue, "get_position") - mav_pos
            dist = float(np.linalg.norm(d_vec))
            if dist < 1e-8 or mav_speed < 1e-8 or not np.isfinite(dist) or not np.isfinite(mav_speed):
                continue
            ao = float(np.arccos(np.clip(np.dot(d_vec / dist, mav_vel / mav_speed), -1.0, 1.0)))
            if ao < np.pi / 2.0:
                r_aware += 0.3 * (1.0 - ao / (np.pi / 2.0))
        shared_slots = 0
        shared_unique: set[str] = set()
        for rid in self.red_ids:
            if self.agent_roles.get(rid) != "attack_uav":
                continue
            red = self.red_planes.get(rid)
            if red is None or not getattr(red, "is_alive", False):
                continue
            for bid in self.blue_ids:
                blue = self.blue_planes.get(bid)
                if blue is None or not getattr(blue, "is_alive", False):
                    continue
                if self._mav_shared_track_state(rid, bid)["mav_shared_visible"]:
                    shared_slots += 1
                    shared_unique.add(bid)
        if all_attack_dead:
            self._brma_tam_all_attack_uav_dead_steps = getattr(self, "_brma_tam_all_attack_uav_dead_steps", 0) + 1
        weighted = float(spcfg.get("pos_weight", 0.6)) * r_pos + float(spcfg.get("aware_weight", 0.4)) * r_aware
        return weighted, {
            "mav_pos_raw": float(r_pos),
            "mav_pos_weighted": float(spcfg.get("pos_weight", 0.6)) * float(r_pos),
            "battlefield_center_x": float(center[0]),
            "battlefield_center_y": float(center[1]),
            "battlefield_center_valid": float(center_valid),
            "attack_uav_alive_count": float(len(alive_uav)),
            "all_attack_uav_dead": float(all_attack_dead),
            "steps_after_all_attack_uav_dead": float(getattr(self, "_brma_tam_all_attack_uav_dead_steps", 0)),
            "mav_reward_after_all_attack_uav_dead": float(weighted if all_attack_dead else 0.0),
            "mav_support_after_all_attack_uav_dead": float(weighted if all_attack_dead else 0.0),
            "mav_center_distance_m": float(mav_center_distance),
            "mav_aware_raw": float(r_aware),
            "mav_aware_raw_sum": float(r_aware),
            "mav_aware_per_blue_mean": float(r_aware) / max(float(len(alive_blue)), 1.0),
            "mav_aware_weighted": float(spcfg.get("aware_weight", 0.4)) * float(r_aware),
            "mav_observed_blue_count": float(mav_observed),
            "mav_alive_blue_count": float(len(alive_blue)),
            "mav_observation_coverage_log": float(mav_observed) / max(float(len(alive_blue)), 1.0),
            "mav_shared_track_slot_count_log": float(shared_slots),
            "mav_shared_track_unique_blue_count_log": float(len(shared_unique)),
            "mav_shared_track_count_log": float(shared_slots),
        }

    def _brma_tam_mav_event(self, aid: str, mav, cfg: dict) -> tuple[float, dict]:
        ecfg = cfg.get("mav", {}).get("event", {})
        alive_before = bool(getattr(self, "_brma_tam_alive_before_step", {}).get(aid, getattr(mav, "is_alive", False)))
        death = 0.0
        if alive_before and not getattr(mav, "is_alive", False) and not getattr(self, "_brma_tam_mav_death_penalized", False):
            death = -float(ecfg.get("death_penalty", 200.0))
            self._brma_tam_mav_death_penalized = True
        team_kills = 0
        if alive_before:
            team_kills = sum(
                int(getattr(self, "_step_kill_count", {}).get(rid, 0))
                for rid in self.red_ids
                if self.agent_roles.get(rid) == "attack_uav"
            )
        if not any(
            self.agent_roles.get(rid) == "attack_uav" and self.red_planes.get(rid) and self.red_planes[rid].is_alive
            for rid in self.red_ids
        ) and not team_kills:
            team_kills = 0
        cap = float(ecfg.get("team_credit_cap", 200.0))
        available = max(0.0, cap - float(getattr(self, "_brma_tam_mav_team_credit_used", 0.0)))
        credit = min(float(ecfg.get("team_credit_per_kill", 100.0)) * team_kills, available)
        if not alive_before:
            credit = 0.0
        self._brma_tam_mav_team_credit_used = float(getattr(self, "_brma_tam_mav_team_credit_used", 0.0)) + credit
        total = death + credit
        return total, {
            "mav_event_death": float(death),
            "mav_team_credit_delta": float(credit),
            "mav_team_credit_used": float(self._brma_tam_mav_team_credit_used),
            "mav_event_total": float(total),
        }

    def _compute_brma_tam_scripted_composite_v1(self, base_rewards: dict, components: dict):
        cfg = self.brma_tam_scripted_composite_v1_config
        selected_missile_ids = set()
        for rec in getattr(self, "_evasion_step_records", []) or []:
            mid = str(rec.get("incoming_missile_id", "") or "")
            if mid:
                selected_missile_ids.add(mid)
        alive_missile_ids = {
            str(mid) for mid, missile in getattr(self, "_missiles_in_flight", {}).items()
            if getattr(missile, "is_alive", False)
        }
        self._brma_tam_missile_speed_cache = {
            mid: speed for mid, speed in getattr(self, "_brma_tam_missile_speed_cache", {}).items()
            if mid in alive_missile_ids and mid in selected_missile_ids
        }
        _any_evasion_override = 0.0
        _any_above_altitude_max = 0.0
        for rid in self.red_ids:
            comp = components.setdefault(rid, {})
            sim = self.red_planes.get(rid)
            role = self.agent_roles.get(rid, "")
            alive_before = bool(getattr(self, "_brma_tam_alive_before_step", {}).get(rid, getattr(sim, "is_alive", False)))
            brma_pitch = float(comp.get("r_pitch", 0.0))
            brma_roll = float(comp.get("r_roll", 0.0))
            brma_vel = float(comp.get("r_vel", 0.0))
            vals = {
                "reward_contract_revision": 2.0,
                "brma_pitch": brma_pitch,
                "brma_roll": brma_roll,
                "brma_vel": brma_vel,
                "brma_alt_log_only": float(comp.get("r_alt", 0.0)),
                "brma_bound_log_only": float(comp.get("r_bound", 0.0)),
                "brma_adv_log_only": float(comp.get("r_adv", 0.0)),
                "brma_end_log_only": float(comp.get("r_end", 0.0)),
                "brma_death_log_only": float(comp.get("r_death", 0.0)),
            }
            altitude = float(self._brma_tam_safe_vec(sim, "get_position")[2]) if sim is not None else 0.0
            altitude_finite = bool(np.isfinite(altitude))
            above_altitude = bool(
                altitude_finite
                and altitude > float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
            )
            vals.update({
                "above_altitude_max_steps": float(above_altitude),
                "max_altitude_m": altitude if altitude_finite else 0.0,
                "above_altitude_max_episode_flag": float(above_altitude),
            })
            if sim is None or not alive_before:
                total = 0.0
                vals.update({"brma_pitch": 0.0, "brma_roll": 0.0, "brma_vel": 0.0})
                if role == "mav":
                    vals.update({
                        "mav_dist_weighted": 0.0,
                        "mav_threat_weighted": 0.0,
                        "mav_aspect_weighted": 0.0,
                        "mav_pos_weighted": 0.0,
                        "mav_aware_weighted": 0.0,
                        "mav_event_total": 0.0,
                        "mav_team_credit_delta": 0.0,
                        "mav_total": 0.0,
                    })
                else:
                    vals.update({
                        "tam_speed_weighted": 0.0,
                        "tam_angle_weighted": 0.0,
                        "tam_distance_weighted": 0.0,
                        "uav_event_total": 0.0,
                        "uav_event_kill": 0.0,
                        "uav_event_loss": 0.0,
                        "uav_total": 0.0,
                    })
                comp.update(vals)
                comp["total"] = total
                base_rewards[rid] = total
                continue

            flight = brma_pitch + brma_roll + brma_vel
            if role == "mav":
                safety, safety_logs = self._brma_tam_mav_safety(sim, cfg)
                support, support_logs = self._brma_tam_mav_support(sim, cfg)
                event, event_logs = self._brma_tam_mav_event(rid, sim, cfg)
                total = flight + safety + support + event
                vals.update(safety_logs)
                vals.update(support_logs)
                vals.update(event_logs)
                vals["mav_total"] = float(total)
                _all_attack_dead = float(support_logs.get("all_attack_uav_dead", 0.0))
                vals["mav_safety_after_all_attack_uav_dead"] = float(safety if _all_attack_dead > 0.5 else 0.0)
                vals["mav_flight_after_all_attack_uav_dead"] = float(flight if _all_attack_dead > 0.5 else 0.0)
                vals["mav_event_after_all_attack_uav_dead"] = float(event if _all_attack_dead > 0.5 else 0.0)
                vals["mav_total_after_all_attack_uav_dead"] = float(total if _all_attack_dead > 0.5 else 0.0)
            else:
                target_id, target, _dist = self._brma_tam_closest_alive_blue(sim)
                if target is None:
                    geom = {
                        "tam_angle_raw": 0.0,
                        "tam_ata_rad": 0.0,
                        "tam_aa_rad": 0.0,
                        "tam_geometry_valid": 0.0,
                        "target_distance_m": 0.0,
                    }
                    speed_logs = self._brma_tam_speed_raw(0.0, 0.0)
                    dist_logs = self._brma_tam_distance_raw(float("nan"))
                else:
                    geom = self._brma_tam_3d_geometry(sim, target)
                    red_speed = float(np.linalg.norm(self._brma_tam_safe_vec(sim, "get_velocity")))
                    target_speed = float(np.linalg.norm(self._brma_tam_safe_vec(target, "get_velocity")))
                    speed_logs = self._brma_tam_speed_raw(red_speed, target_speed)
                    dist_logs = self._brma_tam_distance_raw(geom["target_distance_m"])
                track_logs = self._brma_tam_track_logs(rid, target_id)
                dodge_logs, missile_id = self._brma_tam_dodge_diagnostic(rid, sim)
                event, event_logs = self._brma_tam_uav_event(rid, sim, cfg)
                ucfg = cfg.get("uav", {})
                speed_w = float(ucfg.get("speed_weight", 10.0))
                angle_w = float(ucfg.get("angle_weight", 15.0))
                dist_w = float(ucfg.get("distance_weight", 10.0))
                switch_count = self._brma_tam_reward_target_switch_counts.get(rid, 0)
                if target_id and self._brma_tam_last_reward_target.get(rid) not in {"", target_id, None}:
                    switch_count += 1
                if target_id:
                    self._brma_tam_last_reward_target[rid] = target_id
                self._brma_tam_reward_target_switch_counts[rid] = switch_count
                launch_records = [
                    rec for rec in getattr(self, "_launch_quality_step_records", []) or []
                    if str(rec.get("shooter_id", "")) == rid
                ]
                launch_target_ids = [str(rec.get("target_id", "") or "") for rec in launch_records]
                launch_target_ids = [tid for tid in launch_target_ids if tid]
                launch_target_id = launch_target_ids[0] if launch_target_ids else ""
                lock_target_id = ""
                lock_timer_frames = 0
                if launch_records:
                    lock_target_id = str(launch_records[0].get("lock_target_id_at_launch", "") or "")
                    lock_timer_frames = int(launch_records[0].get("lock_timer_frames_at_launch", 0) or 0)
                if not lock_target_id:
                    lock_target_id = str(getattr(self, "_lock_target", {}).get(rid, "") or "")
                    lock_timer_frames = int(getattr(self, "_lock_timer", {}).get(rid, 0) or 0)
                track_logs["reward_target_matches_lock"] = float(bool(target_id and lock_target_id == target_id))
                track_logs["reward_target_matches_launch"] = float(bool(target_id and target_id in launch_target_ids))
                _eff_min = getattr(self, "_missile_launch_min_range_m_effective",
                                    getattr(self, "MISSILE_LAUNCH_MIN_RANGE", 500.0))
                _eff_max = getattr(self, "_missile_launch_range_m_effective",
                                   getattr(self, "MISSILE_LAUNCH_RANGE_THRESH", 10000.0))
                _reward_target_valid = 1.0 if target is not None else 0.0
                _target_dist = float(geom.get("target_distance_m", 0.0))
                if target is None:
                    launch_range_ok = 0.0
                    below_min_launch_range = 0.0
                else:
                    launch_range_ok = 1.0 if float(_eff_min) <= _target_dist <= float(_eff_max) else 0.0
                    below_min_launch_range = 1.0 if _target_dist < float(_eff_min) else 0.0
                vals.update(geom)
                vals.update(speed_logs)
                vals.update(dist_logs)
                vals.update(track_logs)
                vals.update(dodge_logs)
                vals.update(event_logs)
                vals.update({
                    "tam_speed_weighted": speed_w * float(speed_logs["tam_speed_raw"]),
                    "tam_angle_weighted": angle_w * float(geom["tam_angle_raw"]),
                    "tam_distance_weighted": dist_w * float(dist_logs["tam_distance_raw"]),
                    "launch_range_ok": launch_range_ok,
                    "below_min_launch_range": below_min_launch_range,
                    "reward_target_distance_m": _target_dist,
                    "reward_target_switch_count": float(switch_count),
                    "reward_target_valid": _reward_target_valid,
                    "effective_launch_min_range_m": float(_eff_min),
                    "effective_launch_max_range_m": float(_eff_max),
                })
                total = flight + vals["tam_speed_weighted"] + vals["tam_angle_weighted"] + vals["tam_distance_weighted"] + event
                vals["uav_total"] = float(total)
                visibility = self._mav_shared_track_state(rid, target_id) if target_id else {
                    "track_source": "unobserved",
                }
                action_source = "scripted_evasion" if dodge_logs["evasion_override_active"] > 0.5 else "policy"
                self._reward_target_diagnostic_records.append({
                    "agent_id": rid,
                    "reward_target_id": target_id or "",
                    "reward_target_distance_m": float(geom.get("target_distance_m", 0.0)),
                    "reward_target_observed": track_logs["reward_target_observed"],
                    "reward_target_direct_visible": track_logs["reward_target_direct_visible"],
                    "reward_target_mav_shared_visible": track_logs["reward_target_mav_shared_visible"],
                    "reward_target_unavailable": track_logs["reward_target_unavailable"],
                    "reward_target_track_source": str(visibility["track_source"]),
                    "lock_target_id": lock_target_id,
                    "lock_timer_frames": lock_timer_frames,
                    "launch_target_id": launch_target_id,
                    "launch_target_ids": "|".join(launch_target_ids),
                    "launch_count_this_step": len(launch_target_ids),
                    "reward_target_matches_lock": track_logs["reward_target_matches_lock"],
                    "reward_target_matches_launch": track_logs["reward_target_matches_launch"],
                    "reward_target_switch_count": switch_count,
                    "script_selected_missile_id": missile_id,
                    "tam_dodge_geometry_valid": dodge_logs["tam_dodge_geometry_valid"],
                    "tam_dodge_missing_reason": dodge_logs["tam_dodge_missing_reason"],
                    "evasion_override_active": dodge_logs["evasion_override_active"],
                    "death_reason": self._brma_tam_death_reason(rid),
                    "action_source": action_source,
                })
            vals["brma_tam_scripted_composite_total"] = float(total)
            vals["total"] = float(total)
            vals["evasion_override_agent_steps"] = float(vals.get("evasion_override_active", 0.0))
            vals["above_altitude_max_agent_steps"] = float(vals.get("above_altitude_max_steps", 0.0))
            if role != "mav" and float(vals.get("evasion_override_active", 0.0)) > 0.5:
                _any_evasion_override = 1.0
            if float(vals.get("above_altitude_max_steps", 0.0)) > 0.5:
                _any_above_altitude_max = 1.0
            comp.update(vals)
            base_rewards[rid] = float(total)
            components[rid] = comp
        _mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"), None)
        _env_flag_agent = _mav_id if _mav_id is not None else (self.red_ids[0] if self.red_ids else None)
        for rid in self.red_ids:
            _is_env_flag_agent = (rid == _env_flag_agent)
            components[rid]["evasion_override_env_steps"] = _any_evasion_override if _is_env_flag_agent else 0.0
            components[rid]["above_altitude_max_env_steps"] = _any_above_altitude_max if _is_env_flag_agent else 0.0
        return base_rewards, components

    def _compute_tam_brma_paper_aligned_v1(self, base_rewards: dict, components: dict):
        cfg = self.tam_brma_paper_aligned_v1_config
        scale = float(cfg.get("mav_reward_scale", 0.05))
        include_uav_death = bool((cfg.get("uav", {}) or {}).get("include_r_death", False))
        mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"),
                      self.red_ids[0] if self.red_ids else None)
        for rid in self.red_ids:
            comp = components.setdefault(rid, {})
            flight = sum(float(comp.get(k, 0.0)) for k in ("r_pitch", "r_roll", "r_alt", "r_bound", "r_vel"))
            if self.agent_roles.get(rid) != "mav":
                adv = float(comp.get("r_adv", 0.0))
                end = float(comp.get("r_end", 0.0))
                death_log = float(comp.get("r_death", 0.0))
                total = flight + adv + end + (death_log if include_uav_death else 0.0)
                comp.update({
                    "paper_v1_uav_flight": float(flight),
                    "paper_v1_uav_adv": adv,
                    "paper_v1_uav_end": end,
                    "paper_v1_uav_r_death_log": death_log,
                    "paper_v1_uav_total": float(total),
                    "total": float(total),
                })
                base_rewards[rid] = float(total)
                components[rid] = comp
                continue

            mav = self.red_planes.get(rid)
            removed_adv = float(comp.get("r_adv", 0.0))
            removed_end = float(comp.get("r_end", 0.0))
            death_log = float(comp.get("r_death", 0.0))
            comp["r_adv"] = 0.0
            comp["r_end"] = 0.0
            safety, safety_logs = self._paper_v1_mav_safety(mav, cfg)
            support, support_logs = self._paper_v1_mav_support(mav, cfg)
            event_raw, event_logs = self._paper_v1_mav_event(rid, mav, cfg)
            scaled_tam = scale * (safety + support + event_raw)
            total = float(flight + scaled_tam)
            comp.update(safety_logs)
            comp.update(support_logs)
            comp.update(event_logs)
            comp.update(self._paper_v1_shared_track_logs(mav_id or rid))
            comp.update({
                "paper_v1_mav_flight": float(flight),
                "paper_v1_mav_removed_r_adv": removed_adv,
                "paper_v1_mav_removed_r_end": removed_end,
                "paper_v1_mav_r_death_log": death_log,
                "paper_v1_mav_scaled_tam": float(scaled_tam),
                "paper_v1_mav_total": total,
                "total": total,
            })
            for old_key in ("mav_survival", "mav_support", "mav_attack", "mav_dodge", "death_penalty"):
                comp.pop(old_key, None)
            base_rewards[rid] = total
            components[rid] = comp
        return base_rewards, components

    def _compute_happo_ref_v1_mav_support(self, base_rewards: dict, components: dict):
        """HAPPO reference v1: keep v0 UAV rewards, replace MAV reward.

        MAV total is rebuilt from BRMA flight base plus TAM-style safety,
        support, and event terms.  This method intentionally does not call the
        v7/TAM helpers, so their death/team-credit state is not mutated.
        """
        self._ensure_happo_ref_v0_reward_component_keys(components)
        cfg = self.happo_ref_v1_mav_support_config
        scale = float(cfg.get("scale", 1.0))
        mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"),
                      self.red_ids[0] if self.red_ids else None)

        # UAV side: copied from happo_ref_v0 UAV logic. MAV is excluded.
        for rid in self.red_ids:
            if self.agent_roles.get(rid, "") == "mav":
                continue
            sim = self.red_planes.get(rid)
            comp = components.setdefault(rid, {})
            if sim is None:
                continue
            safety = 0.0
            if sim.is_alive:
                obs = self._last_step_obs.get(rid, {})
                altitude = float(np.asarray(obs.get("altitude", [0.0])).reshape(-1)[0]) if obs else 0.0
                velocity = np.asarray(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
                speed = float(np.linalg.norm(velocity)) if velocity.size else 0.0
                safety += 0.002 if 2500.0 <= altitude <= 12000.0 else -0.003
                safety += 0.002 if 120.0 <= speed <= 420.0 else -0.003
            comp["safety"] = float(np.clip(safety, -0.01, 0.01))
            base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["safety"]

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
            comp["uav_attack_window"] = float(np.clip(window, 0.0, 0.04))
            base_rewards[rid] = base_rewards.get(rid, 0.0) + comp["uav_attack_window"]

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

        if not mav_id or mav_id not in components:
            return base_rewards, components

        mav = self.red_planes.get(mav_id)
        comp = components.setdefault(mav_id, {})
        removed_adv = float(comp.get("r_adv", 0.0))
        removed_end = float(comp.get("r_end", 0.0))
        flight_base = sum(float(comp.get(k, 0.0)) for k in ("r_pitch", "r_roll", "r_alt", "r_bound", "r_vel"))
        comp["r_adv"] = 0.0
        comp["r_end"] = 0.0

        safety, safety_logs = self._happo_v1_mav_safety(mav, cfg)
        support, support_logs = self._happo_v1_mav_support(mav_id, mav, cfg)
        event, event_logs = self._happo_v1_mav_event(mav_id, mav, cfg)
        total_pre_clip = flight_base + scale * (safety + support + event)
        total = float(total_pre_clip)

        comp.update(safety_logs)
        comp.update(support_logs)
        comp.update(event_logs)
        comp.update(self._happo_v1_episode_support_logs(mav_id, mav))
        comp.update({
            "v1_mav_removed_r_adv": removed_adv,
            "v1_mav_removed_r_end": removed_end,
            "v1_mav_removed_v0_overlay": 1.0,
            "v1_mav_flight_base": float(flight_base),
            "v1_mav_total_pre_clip": float(total_pre_clip),
            "v1_mav_total": total,
            "mav_reward_safety_sum": safety,
            "mav_reward_support_sum": support,
            "mav_reward_event_sum": event,
            "mav_reward_total_sum": total,
            "mav_removed_r_adv_sum": removed_adv,
            "mav_removed_r_end_sum": removed_end,
            "total": total,
        })
        for old_key in ("mav_survival", "mav_support", "mav_attack", "mav_dodge", "death_penalty"):
            comp.pop(old_key, None)
        base_rewards[mav_id] = total
        components[mav_id] = comp
        return base_rewards, components

    def _happo_v1_mav_safety(self, mav, cfg: dict) -> tuple[float, dict]:
        scfg = cfg.get("mav_safety", {})
        d_danger = float(scfg.get("d_danger_m", 8000.0))
        d_safe = float(scfg.get("d_safe_m", 15000.0))
        alive_blue = [s for s in self.blue_planes.values() if getattr(s, "is_alive", False)]
        if mav is None or not getattr(mav, "is_alive", False):
            logs = {
                "v1_mav_safety_dist": 0.0,
                "v1_mav_safety_threat": 0.0,
                "v1_mav_safety_aspect": 0.0,
                "v1_mav_safety_danger_m": d_danger,
                "v1_mav_safety_safe_m": d_safe,
            }
            logs["v1_mav_safety"] = 0.0
            return 0.0, logs
        mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
        if alive_blue:
            d_min = min(float(np.linalg.norm(mav_pos - np.asarray(b.get_position(), dtype=np.float64)))
                        for b in alive_blue)
        else:
            d_min = d_safe
        near_d = float(d_min)
        if near_d <= d_danger:
            r_dist = -(1.0 - near_d / max(d_danger, 1e-6))
        elif near_d < d_safe:
            r_dist = -0.5 * (1.0 - (near_d - d_danger) / max(d_safe - d_danger, 1e-6))
        else:
            r_dist = 0.2
        r_threat = -1.0 if mav.check_missile_warning() is not None else 0.0
        blue_launch_window_on_mav = 0.0
        r_aspect = 0.0
        try:
            mav_feat = self._tam_v2_feature(mav)
        except Exception:
            mav_feat = None
        for blue in alive_blue:
            try:
                m = self._missile_candidate_metrics(blue, mav)
            except Exception:
                m = {}
            if mav_feat is not None:
                try:
                    blue_feat = self._tam_v2_feature(blue)
                    _ao, ta, _r = get2d_AO_TA_R(mav_feat, blue_feat)
                    ta = float(ta)
                    if ta < np.pi / 4.0:
                        r_aspect -= (1.0 - ta / (np.pi / 4.0))
                except Exception:
                    pass
            if m.get("range_ok") and m.get("ao_ok") and m.get("ta_ok"):
                blue_launch_window_on_mav += 1.0
        r_safety = (
            float(scfg.get("dist_weight", 0.5)) * r_dist
            + float(scfg.get("threat_weight", 0.3)) * r_threat
            + float(scfg.get("aspect_weight", 0.2)) * r_aspect
        )
        logs = {
            "v1_mav_safety": float(r_safety),
            "v1_mav_safety_dist": float(r_dist),
            "v1_mav_safety_threat": float(r_threat),
            "v1_mav_safety_aspect": float(r_aspect),
            "v1_mav_safety_danger_m": d_danger,
            "v1_mav_safety_safe_m": d_safe,
            "v1_mav_blue_launch_window_on_mav_log": float(blue_launch_window_on_mav),
        }
        return float(r_safety), logs

    def _happo_v1_mav_support(self, mav_id: str, mav, cfg: dict) -> tuple[float, dict]:
        sp = cfg.get("mav_support", {})
        obs = self._last_step_obs.get(mav_id, {})
        observed = np.asarray(obs.get("enemy_observed_mask", []), dtype=np.float32).reshape(-1)
        observed_count = 0
        aware_raw = 0.0
        if mav is not None and getattr(mav, "is_alive", False):
            for idx, bid in enumerate(self.blue_ids):
                if idx >= observed.size or observed[idx] <= 0.5:
                    continue
                blue = self.blue_planes.get(bid)
                if blue is None or not blue.is_alive:
                    continue
                observed_count += 1
                try:
                    m = self._missile_candidate_metrics(mav, blue)
                    ao = float(m.get("AO_rad", np.pi))
                except Exception:
                    ao = np.pi
                if ao < np.pi / 2.0:
                    aware_raw += 0.3 * (1.0 - ao / (np.pi / 2.0))
        pos = 0.0
        pos_active = bool(sp.get("pos_active", False))
        if pos_active:
            raise ValueError(
                "R_pos is not implemented/paper-grounded for "
                "happo_ref_v1_mav_support."
            )
        support = (
            float(sp.get("pos_weight", 0.6)) * pos * float(pos_active)
            + float(sp.get("aware_weight", 0.4)) * aware_raw
        )
        logs = {
            "v1_mav_support": float(support),
            "v1_mav_support_pos": float(pos),
            "v1_mav_support_pos_active": 1.0 if pos_active else 0.0,
            "v1_mav_support_aware": float(aware_raw),
            "v1_mav_support_observed_count": float(observed_count),
            "v1_mav_support_aware_raw": float(aware_raw),
        }
        return float(support), logs

    def _happo_v1_mav_event(self, mav_id: str, mav, cfg: dict) -> tuple[float, dict]:
        ecfg = cfg.get("mav_event", {})
        death = 0.0
        if mav is not None and (not mav.is_alive) and not self._happo_v1_mav_death_penalized:
            death = float(ecfg.get("death_penalty", -4.0))
            self._happo_v1_mav_death_penalized = True
        mav_alive = bool(mav is not None and mav.is_alive)
        kills = 0
        for rid in self.red_ids:
            if rid == mav_id or self.agent_roles.get(rid) != "attack_uav":
                continue
            kills += int(self._step_kill_count.get(rid, 0))
        cap = float(ecfg.get("team_credit_cap", 1.0))
        if mav_alive and kills > 0:
            available = max(0.0, cap - self._happo_v1_mav_team_credit_used)
            credit = min(float(ecfg.get("team_credit_per_kill", 0.5)) * kills, available)
            self._happo_v1_mav_team_credit_used += credit
        else:
            credit = 0.0
        event = death + credit
        logs = {
            "v1_mav_event": float(event),
            "v1_mav_event_death": float(death),
            "v1_mav_event_team_credit_delta": float(credit),
            "v1_mav_event_team_credit_used": float(self._happo_v1_mav_team_credit_used),
            "v1_mav_event_team_credit_cap": cap,
        }
        return float(event), logs

    def _happo_v1_episode_support_logs(self, mav_id: str, mav) -> dict:
        mav_obs = self._last_step_obs.get(mav_id, {})
        observed = np.asarray(mav_obs.get("enemy_observed_mask", []), dtype=np.float32).reshape(-1)
        alive_blue_count = 0
        observed_alive_count = 0
        for idx, bid in enumerate(self.blue_ids):
            blue = self.blue_planes.get(bid)
            if blue is None or not blue.is_alive:
                continue
            alive_blue_count += 1
            if idx < observed.size and observed[idx] > 0.5:
                observed_alive_count += 1
        mav_observed_ratio = observed_alive_count / max(alive_blue_count, 1)

        shared = 0.0
        slots = 0.0
        for rid in self.red_ids:
            if rid == mav_id or self.agent_roles.get(rid) != "attack_uav":
                continue
            src = np.asarray(self._last_step_obs.get(rid, {}).get("enemy_track_source", []), dtype=np.float32)
            if src.ndim == 2 and src.shape[1] >= 2:
                slots += float(src.shape[0])
                shared += float(np.sum(src[:, 1] > 0.5))
        mav_shared_track_ratio = shared / max(slots, 1.0)

        launches = [
            r for r in (getattr(self, "_launch_quality_step_records", None) or [])
            if str(r.get("shooter_id", "")).startswith("red_")
            and self.agent_roles.get(str(r.get("shooter_id", ""))) == "attack_uav"
        ]
        hits = [
            r for r in (getattr(self, "_launch_quality_done_step_records", None) or [])
            if str(r.get("shooter_id", "")).startswith("red_")
            and self.agent_roles.get(str(r.get("shooter_id", ""))) == "attack_uav"
            and str(r.get("raw_termination_reason", "")) == "hit"
        ]
        shared_launches = sum(1 for r in launches if str(r.get("launch_track_source", "")) == "mav_shared")
        shared_hits = sum(1 for r in hits if str(r.get("launch_track_source", "")) == "mav_shared")
        mav_alive = bool(mav is not None and mav.is_alive)
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids
                         if rid != mav_id and self.agent_roles.get(rid) == "attack_uav")
        uav_alive_steps = sum(1 for rid in self.red_ids
                              if rid != mav_id and self.agent_roles.get(rid) == "attack_uav"
                              and self.red_planes.get(rid) and self.red_planes[rid].is_alive)
        red_launch_before = float(len(launches) if mav_alive else 0)
        red_launch_after = float(0 if mav_alive else len(launches))
        red_uav_alive_before = float(uav_alive_steps if mav_alive else 0)
        red_uav_alive_after = float(0 if mav_alive else uav_alive_steps)
        return {
            "mav_observed_ratio": float(mav_observed_ratio),
            "mav_shared_track_ratio": float(mav_shared_track_ratio),
            "red_launch_with_mav_shared_track": float(shared_launches),
            "red_hit_with_mav_shared_track": float(shared_hits),
            "team_kill_while_mav_alive": float(team_kills if mav_alive else 0),
            "team_kill_after_mav_death": float(0 if mav_alive else team_kills),
            "red_launch_before_mav_death": red_launch_before,
            "red_launch_after_mav_death": red_launch_after,
            "red_uav_alive_steps_before_mav_death": red_uav_alive_before,
            "red_uav_alive_steps_after_mav_death": red_uav_alive_after,
            "red_launch_rate_before_mav_death": red_launch_before / max(red_uav_alive_before, 1.0),
            "red_launch_rate_after_mav_death": red_launch_after / max(red_uav_alive_after, 1.0),
        }

    def _tam_v6v3_td_env(self, distance_m: float, cfg: dict) -> float:
        scfg = cfg.get("situation", {})
        ref = scfg.get("active_distance_ref_m")
        if ref is None:
            ref = getattr(self, "_missile_launch_range_m_effective", self.MISSILE_LAUNCH_RANGE_THRESH)
        ref = max(float(ref), 1.0)
        min_range = scfg.get("min_range_m")
        if min_range is None:
            min_range = getattr(self, "MISSILE_LAUNCH_MIN_RANGE", 500.0)
        min_range = float(min_range)
        optimal_min = scfg.get("optimal_min_m")
        if optimal_min is None:
            optimal_min = 0.5 * ref
        optimal_max = scfg.get("optimal_max_m")
        if optimal_max is None:
            optimal_max = ref
        optimal_min = max(float(optimal_min), min_range + 1e-6)
        optimal_max = max(float(optimal_max), optimal_min)
        d = float(distance_m)
        if not np.isfinite(d) or d < min_range:
            return 0.0
        if d < optimal_min:
            value = (d - min_range) / max(optimal_min - min_range, 1e-6)
        elif d <= optimal_max:
            value = 1.0
        else:
            value = np.exp(1.0 - d / ref)
        return float(np.clip(value, 0.0, 1.0))

    def _tam_v6v3_uav_reward(self, aid: str, sim, alive_blue: list, cfg: dict,
                              base_components: dict) -> tuple[float, dict]:
        prefix = "tam_v6v3_uav"
        vals: dict[str, float] = {}
        if sim.is_alive:
            vals.update(self._tam_v6v3_flight_components(prefix, sim, cfg))
            raw, own, threat, logs = self._tam_v6v3_situation_reward(sim, cfg)
            vals[f"{prefix}_situation_raw_td10_rear"] = raw
            vals[f"{prefix}_situation"] = float(cfg["situation"].get("weight", 0.15)) * raw
            vals[f"{prefix}_own_adv_td10_rear_log"] = own
            vals[f"{prefix}_enemy_threat_td10_rear_log"] = threat
            vals[f"{prefix}_own_adv_td10_no_rear_log"] = logs["own_adv_td10_no_rear_log"]
            vals[f"{prefix}_own_adv_td15_log"] = logs["own_adv_td15_log"]
            vals[f"{prefix}_rear_factor_mean_log"] = logs["rear_factor_mean_log"]
            d_raw, d_angle, d_speed = self._tam_v2_dodge_reward(
                sim, 1000.0, getattr(self, "_tam_v2_missile_speed_cache", {}))
            vals[f"{prefix}_dodge_raw_log"] = d_raw
            vals[f"{prefix}_dodge_angle_log"] = d_angle
            vals[f"{prefix}_dodge_speed_log"] = d_speed
        else:
            for key in ("pitch", "roll", "altitude", "speed", "boundary", "flight",
                        "situation_raw_td10_rear", "situation"):
                vals[f"{prefix}_{key}"] = 0.0
            for key in ("own_adv_td10_rear_log", "enemy_threat_td10_rear_log",
                        "own_adv_td10_no_rear_log", "own_adv_td15_log",
                        "rear_factor_mean_log", "dodge_raw_log", "dodge_angle_log",
                        "dodge_speed_log"):
                vals[f"{prefix}_{key}"] = 0.0

        ev = cfg["uav"]["event"]
        kills = int(self._step_kill_count.get(aid, 0))
        vals[f"{prefix}_kill"] = kills * float(ev.get("kill_enemy", 200.0))
        if (not sim.is_alive) and aid not in self._tam_v6v3_uav_death_penalized:
            vals[f"{prefix}_death"] = float(ev.get("death", -200.0))
            self._tam_v6v3_uav_death_penalized.add(aid)
        else:
            vals[f"{prefix}_death"] = 0.0
        vals[f"{prefix}_event"] = vals[f"{prefix}_kill"] + vals[f"{prefix}_death"]
        vals[f"{prefix}_terminal"] = 0.0
        total = (
            vals[f"{prefix}_flight"]
            + vals[f"{prefix}_situation"]
            + vals[f"{prefix}_event"]
            + vals[f"{prefix}_terminal"]
        ) * float(cfg.get("global_scale", 1.0))
        vals[f"{prefix}_total"] = total
        return total, vals

    def _tam_v6v3_mav_reward(self, mav_id: str, mav, alive_blue: list, cfg: dict,
                              base_components: dict,
                              mav_log_track: float = 0.0,
                              mav_log_fire: float = 0.0,
                              mav_log_hit: float = 0.0) -> tuple[float, dict]:
        prefix = "tam_v6v3_mav"
        vals: dict[str, float] = {}
        if mav.is_alive:
            vals.update(self._tam_v6v3_flight_components(prefix, mav, cfg))
            mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
            if alive_blue:
                distances = [compute_3d_range(mav_pos, b.get_position()) for b in alive_blue]
                near_d = min(distances)
                d_danger = float(cfg["mav"].get("d_danger_m", 5000.0))
                d_safe = float(cfg["mav"].get("d_safe_m", 14000.0))
                if near_d <= d_danger:
                    r_dist = -(1.0 - near_d / max(d_danger, 1e-6))
                elif near_d < d_safe:
                    r_dist = -0.5 * (1.0 - (near_d - d_danger) / max(d_safe - d_danger, 1e-6))
                else:
                    r_dist = 0.2
                r_threat = -1.0 if any(getattr(m, "is_alive", False) for m in getattr(mav, "under_missiles", []) or []) else 0.0
                aspect_threats = [self._tam_v6v3_mav_aspect_threat(mav, b) for b in alive_blue]
                r_aspect = -max(aspect_threats) if aspect_threats else 0.0
                blue_centroid = np.mean([b.get_position() for b in alive_blue], axis=0)
                d_b = compute_3d_range(mav_pos, blue_centroid)
                d_opt = float(cfg["mav"].get("d_opt_m", 8000.0))
                d_max = float(cfg["mav"].get("d_max_m", 25000.0))
                if d_b < d_opt:
                    r_pos = d_b / max(d_opt, 1e-6) - 1.0
                elif d_b < d_max:
                    r_pos = 1.0 - (d_b - d_opt) / max(d_max - d_opt, 1e-6)
                else:
                    r_pos = -0.5
                aware_vals = []
                obs_range = float(getattr(self, "mav_observation_range_m", 80000.0))
                for b in alive_blue:
                    d = compute_3d_range(mav_pos, b.get_position())
                    ao = compute_body_x_q_los(mav.get_position(), mav.get_rpy(), b.get_position())
                    aware_vals.append(0.3 * (1.0 - ao / (np.pi / 2)) if d <= obs_range and ao < np.pi / 2 else 0.0)
                r_aware = float(np.mean(aware_vals)) if aware_vals else 0.0
            else:
                r_dist = r_threat = r_aspect = r_pos = r_aware = 0.0
            vals[f"{prefix}_dist"] = r_dist
            vals[f"{prefix}_threat"] = r_threat
            vals[f"{prefix}_aspect"] = r_aspect
            sw = cfg["mav"]["safety_weights"]
            safety_raw = sw["dist"] * r_dist + sw["threat"] * r_threat + sw["aspect"] * r_aspect
            vals[f"{prefix}_safety_raw"] = float(np.clip(safety_raw, -1.0, 0.2))
            scale_cfg = cfg["mav"].get("safety_scale", {})
            vals[f"{prefix}_safety"] = vals[f"{prefix}_safety_raw"] * (
                float(scale_cfg.get("positive", 0.05))
                if vals[f"{prefix}_safety_raw"] > 0.0 else float(scale_cfg.get("negative", 0.20))
            )
            vals[f"{prefix}_pos"] = r_pos
            vals[f"{prefix}_aware"] = r_aware
            sup_w = cfg["mav"]["support_weights"]
            support_raw = sup_w["pos"] * r_pos + sup_w["aware"] * r_aware
            vals[f"{prefix}_support_raw"] = float(np.clip(support_raw, -0.6, 0.72))
            vals[f"{prefix}_support"] = vals[f"{prefix}_support_raw"] * float(cfg["mav"].get("support_scale", 0.10))
            vals[f"{prefix}_observed_blue_count_log"] = float(sum(
                1 for b in alive_blue
                if compute_3d_range(mav.get_position(), b.get_position()) <= getattr(self, "mav_observation_range_m", 80000.0)
            ))
        else:
            for key in ("pitch", "roll", "altitude", "speed", "boundary", "flight",
                        "dist", "threat", "aspect", "safety_raw", "safety",
                        "pos", "aware", "support_raw", "support",
                        "observed_blue_count_log"):
                vals[f"{prefix}_{key}"] = 0.0

        ev = cfg["mav"]["event"]
        if (not mav.is_alive) and not self._tam_v6v3_mav_death_penalized:
            vals[f"{prefix}_death"] = float(ev.get("death_penalty", -300.0))
            self._tam_v6v3_mav_death_penalized = True
        else:
            vals[f"{prefix}_death"] = 0.0
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
        cap = float(ev.get("team_kill_credit_cap_episode", 100.0))
        per_kill = float(ev.get("team_kill_credit_per_kill", 40.0))
        if mav.is_alive and team_kills > 0:
            available = max(0.0, cap - self._tam_v6v3_mav_team_credit_used)
            credit = min(available, team_kills * per_kill)
            self._tam_v6v3_mav_team_credit_used += credit
        else:
            credit = 0.0
        vals[f"{prefix}_team_credit_delta"] = credit
        vals[f"{prefix}_team_credit_used"] = self._tam_v6v3_mav_team_credit_used
        vals[f"{prefix}_event"] = vals[f"{prefix}_death"] + credit
        vals[f"{prefix}_terminal"] = 0.0
        vals[f"{prefix}_shared_track_usage_log"] = mav_log_track
        vals[f"{prefix}_red_fire_with_mav_track_log"] = mav_log_fire
        vals[f"{prefix}_red_hit_with_mav_track_log"] = mav_log_hit
        total = (
            vals[f"{prefix}_flight"]
            + vals[f"{prefix}_safety"]
            + vals[f"{prefix}_support"]
            + vals[f"{prefix}_event"]
            + vals[f"{prefix}_terminal"]
        ) * float(cfg.get("global_scale", 1.0))
        vals[f"{prefix}_total"] = total
        return total, vals

    def _tam_v6v3_terminal_outcome(self, cfg: dict) -> float:
        red_roles = getattr(self, "agent_roles", {})
        mav_id = next((rid for rid in self.red_ids if red_roles.get(rid) == "mav"), self.red_ids[0])
        uav_ids = [rid for rid in self.red_ids if rid != mav_id]
        n_blue_initial = max(len(self.blue_ids), 1)
        n_uav_initial = max(len(uav_ids), 1)
        n_blue_alive = sum(1 for bid in self.blue_ids if self.blue_planes.get(bid) and self.blue_planes[bid].is_alive)
        n_uav_dead = sum(1 for rid in uav_ids if self.red_planes.get(rid) and not self.red_planes[rid].is_alive)
        mav_dead = bool(self.red_planes.get(mav_id) and not self.red_planes[mav_id].is_alive)
        blue_loss_frac = (n_blue_initial - n_blue_alive) / n_blue_initial
        tcfg = cfg.get("terminal", {})
        if tcfg.get("mav_loss_weight_mode", "match_uav_count") == "match_uav_count":
            w_mav = float(n_uav_initial)
        else:
            w_mav = float(tcfg.get("mav_loss_weight", n_uav_initial))
        red_loss_weighted = (w_mav * float(mav_dead) + float(n_uav_dead)) / max(w_mav + n_uav_initial, 1e-6)
        return float(tcfg.get("coef_per_agent", 30.0)) * (blue_loss_frac - red_loss_weighted)

    def _compute_tam_paper_reward_v6_jsbsim_aligned_v3(self, base_rewards: dict, components: dict):
        cfg = self.tam_paper_reward_v6_jsbsim_aligned_v3_config
        alive_blue = self._tam_v2_alive_blue()
        mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"),
                      self.red_ids[0] if self.red_ids else None)
        n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
        n_red_alive = sum(1 for s in self.red_planes.values() if s.is_alive)
        round_over = n_blue_alive == 0 or n_red_alive == 0 or self.current_step >= self.max_steps
        terminal = 0.0
        if round_over and not self._tam_v6v3_terminal_applied:
            terminal = self._tam_v6v3_terminal_outcome(cfg)
            self._tam_v6v3_terminal_applied = True
        # ── MAV log-only fields: shared track usage ──
        mav_shared_track_usage = 0.0
        mav_red_fire_with_mav_track = 0.0
        mav_red_hit_with_mav_track = 0.0
        obs = getattr(self, "_last_step_obs", {})
        if obs:
            red_uav_track_count = 0
            for uid in self.red_ids:
                if self.agent_roles.get(uid) == "mav":
                    continue
                uav_obs = obs.get(uid, {})
                ets = uav_obs.get("enemy_track_source", None)
                if ets is not None:
                    for ts_row in ets:
                        if len(ts_row) > 1 and int(ts_row[1]) == 1:
                            red_uav_track_count += 1
            if red_uav_track_count > 0:
                mav_shared_track_usage = float(red_uav_track_count) / max(len(self.red_ids) - 1, 1)
        launch_step = getattr(self, "_launch_quality_step_records", None)
        if launch_step:
            for rec in launch_step:
                if isinstance(rec, dict) and str(rec.get("launch_track_source", "")) == "mav_shared":
                    mav_red_fire_with_mav_track += 1.0
        launch_done = getattr(self, "_launch_quality_done_step_records", None)
        if launch_done:
            for rec in launch_done:
                if (isinstance(rec, dict)
                        and str(rec.get("launch_track_source", "")) == "mav_shared"
                        and str(rec.get("raw_termination_reason", "")) == "hit"):
                    mav_red_hit_with_mav_track += 1.0
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            if rid == mav_id:
                _, comp = self._tam_v6v3_mav_reward(rid, sim, alive_blue, cfg, components,
                                                     mav_shared_track_usage,
                                                     mav_red_fire_with_mav_track,
                                                     mav_red_hit_with_mav_track)
                event_key = "tam_v6v3_mav_event"
                term_key = "tam_v6v3_mav_terminal"
                total_key = "tam_v6v3_mav_total"
            else:
                _, comp = self._tam_v6v3_uav_reward(rid, sim, alive_blue, cfg, components)
                event_key = "tam_v6v3_uav_event"
                term_key = "tam_v6v3_uav_terminal"
                total_key = "tam_v6v3_uav_total"
            comp[term_key] = terminal
            comp[total_key] += terminal * float(cfg.get("global_scale", 1.0))
            comp["tam_v6v3_terminal"] = terminal
            comp["tam_v6v3_terminal_per_agent"] = terminal
            comp["tam_v6v3_terminal_team_mean_log"] = terminal
            comp["tam_v6v3_total"] = comp[total_key]
            base_rewards[rid] = comp[total_key]
            components[rid] = comp
        return base_rewards, components

    @staticmethod
    def _tam_v6v3_td_brma15_log(distance_m: float) -> float:
        return float(td_distance_advantage(float(distance_m)))

    def _tam_v6v3_launch_rear_factor(self, ta_rad: float, cfg: dict) -> float:
        scfg = cfg.get("situation", {})
        floor = float(scfg.get("rear_floor", 0.2))
        start = np.deg2rad(float(scfg.get("rear_start_deg", 90.0)))
        full = np.deg2rad(float(scfg.get("rear_full_deg", 150.0)))
        ta = float(ta_rad)
        if not np.isfinite(ta) or ta < start:
            return floor
        if ta >= full:
            return 1.0
        return float(floor + (1.0 - floor) * (ta - start) / max(full - start, 1e-6))

    def _tam_v6v3_altitude_penalty(self, altitude_m: float, cfg: dict) -> float:
        floor = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        ceiling = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        alt = float(altitude_m)
        if not np.isfinite(alt) or alt < floor:
            return -1.0
        if alt < floor + 500.0:
            return -1.0 + (alt - floor) / 500.0
        if alt <= ceiling - 500.0:
            return 0.0
        if alt <= ceiling:
            return -0.5 * (alt - (ceiling - 500.0)) / 500.0
        return -1.0

    @staticmethod
    def _tam_v6v3_speed_penalty(speed_mps: float, cfg: dict) -> float:
        mach = float(speed_mps) / 340.0
        if not np.isfinite(mach) or mach < 0.2:
            return -1.0
        if mach < 0.3:
            return -(0.3 - mach) / 0.1
        return 0.0

    def _tam_v6v3_boundary_penalty(self, sim, cfg: dict) -> float:
        pos = np.asarray(sim.get_position(), dtype=np.float64)
        alt = float(sim.get_geodetic()[2])
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        alt_min = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        alt_max = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        raw = float(cfg.get("flight_status", {}).get("boundary_raw_penalty", -10.0))
        if abs(float(pos[0])) > half or abs(float(pos[1])) > half or alt < alt_min or alt > alt_max:
            return raw
        return 0.0

    def _tam_v6v3_flight_components(self, prefix: str, sim, cfg: dict) -> dict:
        fs = cfg.get("flight_status", {})
        speed = float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))
        alt = float(sim.get_geodetic()[2])
        vals = {
            f"{prefix}_pitch": float(fs.get("pitch_weight", 0.01)) * self._pitch_penalty(sim),
            f"{prefix}_roll": float(fs.get("roll_weight", 0.002)) * self._roll_penalty(sim),
            f"{prefix}_altitude": float(fs.get("altitude_weight", 0.04)) * self._tam_v6v3_altitude_penalty(alt, cfg),
            f"{prefix}_speed": float(fs.get("speed_weight", 0.02)) * self._tam_v6v3_speed_penalty(speed, cfg),
            f"{prefix}_boundary": float(fs.get("boundary_weight", 0.04)) * self._tam_v6v3_boundary_penalty(sim, cfg),
        }
        vals[f"{prefix}_flight"] = sum(vals.values())
        return vals

    def _tam_v6v3_situation_reward(self, sim, cfg: dict) -> tuple[float, float, float, dict]:
        scfg = cfg.get("situation", {})
        alive_blue = self._tam_v2_alive_blue()
        if not alive_blue:
            return 0.0, 0.0, 0.0, {
                "own_adv_td10_rear_log": 0.0,
                "enemy_threat_td10_rear_log": 0.0,
                "own_adv_td10_no_rear_log": 0.0,
                "own_adv_td15_log": 0.0,
                "rear_factor_mean_log": 0.0,
            }
        own_adv = enemy_threat = 0.0
        own_no_rear = own_td15 = rear_sum = 0.0; rear_error_count = 0
        ego_pos = sim.get_position()
        ego_rpy = sim.get_rpy()
        for blue in alive_blue:
            b_pos = blue.get_position()
            b_rpy = blue.get_rpy()
            d_3d = compute_3d_range(ego_pos, b_pos)
            q_ij = compute_body_x_q_los(ego_pos, ego_rpy, b_pos)
            q_ji = compute_body_x_q_los(b_pos, b_rpy, ego_pos)
            ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
            ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))
            td_env = self._tam_v6v3_td_env(d_3d, cfg)
            td15 = self._tam_v6v3_td_brma15_log(d_3d)
            rear_err_i = 0
            try:
                rear_ij = self._tam_v6v3_launch_rear_factor(
                    self._build_launch_geometry_3d(sim, blue)["TA_rad"], cfg)
                rear_ji = self._tam_v6v3_launch_rear_factor(
                    self._build_launch_geometry_3d(blue, sim)["TA_rad"], cfg)
            except Exception:
                rear_floor = float(cfg.get("situation", {}).get("rear_floor", 0.2))
                rear_ij = rear_ji = rear_floor
                rear_err_i = 1
            own_adv += ta_ij * td_env * rear_ij
            enemy_threat += ta_ji * td_env * rear_ji
            own_no_rear += ta_ij * td_env
            own_td15 += ta_ij * td15
            rear_sum += rear_ij
            rear_error_count += rear_err_i
        if bool(scfg.get("normalize_by_alive_blue", True)):
            denom = max(len(alive_blue), 1)
            own_adv /= denom
            enemy_threat /= denom
            own_no_rear /= denom
            own_td15 /= denom
            rear_sum /= denom
            rear_error_count /= denom
        raw = own_adv - float(scfg.get("enemy_threat_weight", 0.8)) * enemy_threat
        return raw, own_adv, enemy_threat, {
            "own_adv_td10_rear_log": own_adv,
            "enemy_threat_td10_rear_log": enemy_threat,
            "own_adv_td10_no_rear_log": own_no_rear,
            "own_adv_td15_log": own_td15,
            "rear_factor_mean_log": rear_sum,
            "rear_error_count_log": float(rear_error_count),
        }

    @staticmethod
    def _tam_v6v3_mav_aspect_threat(mav, blue) -> float:
        angle = float(compute_body_x_q_los(blue.get_position(), blue.get_rpy(), mav.get_position()))
        limit = np.deg2rad(45.0)
        if not np.isfinite(angle) or angle >= limit:
            return 0.0
        return float(np.clip(1.0 - angle / limit, 0.0, 1.0))

    # -- TAM Paper Reward v7 role-aligned -----------------------------------

    def _tam_v7_reset_episode_state(self) -> None:
        self._tam_v7_mav_death_penalized = False
        self._tam_v7_mav_team_credit_used = 0.0
        self._tam_v7_uav_death_penalized: set[str] = set()
        self._tam_v7_uav_out_of_zone_penalized: set[str] = set()
        self._tam_v7_terminal_applied = False

    def _tam_v7_distance_advantage(self, distance_m: float, cfg: dict) -> float:
        scfg = cfg.get("situation", {})
        if bool(scfg.get("use_effective_launch_range", True)):
            ref = getattr(self, "_missile_launch_range_m_effective", self.MISSILE_LAUNCH_RANGE_THRESH)
        else:
            ref = scfg.get("range_ref_m", self.MISSILE_LAUNCH_RANGE_THRESH)
        ref = max(float(ref), 1.0)
        min_range = float(getattr(self, "MISSILE_LAUNCH_MIN_RANGE", 500.0))
        d = float(distance_m)
        if not np.isfinite(d) or d < min_range:
            return 0.0
        if d <= ref:
            return float(np.clip((d - min_range) / max(ref - min_range, 1e-6), 0.0, 1.0))
        return float(np.clip(np.exp(1.0 - d / ref), 0.0, 1.0))

    def _tam_v7_altitude_raw(self, altitude_m: float) -> float:
        floor = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        ceiling = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        alt = float(altitude_m)
        if not np.isfinite(alt) or alt < floor or alt > ceiling:
            return -1.0
        return 0.0

    def _tam_v7_speed_raw(self, speed_mps: float) -> float:
        vmin = float(getattr(self, "VELOCITY_MIN", 102.0))
        vmax = float(getattr(self, "VELOCITY_MAX", 408.0))
        speed = float(speed_mps)
        if not np.isfinite(speed):
            return -1.0
        if speed < vmin:
            return -1.0
        if speed > vmax:
            return -1.0
        return 0.0

    def _tam_v7_out_of_zone(self, sim) -> bool:
        pos = np.asarray(sim.get_position(), dtype=np.float64)
        alt = float(sim.get_geodetic()[2])
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        alt_min = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        alt_max = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        return bool(
            abs(float(pos[0])) > half
            or abs(float(pos[1])) > half
            or alt < alt_min
            or alt > alt_max
        )

    def _tam_v7_boundary_raw(self, sim, cfg: dict) -> float:
        if self._tam_v7_out_of_zone(sim):
            return float(cfg.get("flight", {}).get("boundary_raw_penalty", -10.0))
        return 0.0

    def _tam_v7_flight_components(self, prefix: str, sim, cfg: dict) -> dict:
        fc = cfg.get("flight", {})
        speed = float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))
        alt = float(sim.get_geodetic()[2])
        vals = {
            f"{prefix}_pitch": float(fc.get("pitch_weight", 0.01)) * self._pitch_penalty(sim),
            f"{prefix}_roll": float(fc.get("roll_weight", 0.002)) * self._roll_penalty(sim),
            f"{prefix}_altitude": float(fc.get("altitude_weight", 0.04)) * self._tam_v7_altitude_raw(alt),
            f"{prefix}_speed": float(fc.get("speed_weight", 0.02)) * self._tam_v7_speed_raw(speed),
            f"{prefix}_boundary": float(fc.get("boundary_weight", 0.04)) * self._tam_v7_boundary_raw(sim, cfg),
        }
        vals[f"{prefix}_flight"] = sum(vals.values())
        return vals

    def _tam_v7_situation_reward(self, sim, cfg: dict) -> tuple[float, float, float, float]:
        alive_blue = self._tam_v2_alive_blue()
        if not alive_blue:
            return 0.0, 0.0, 0.0, float(getattr(self, "_missile_launch_range_m_effective", self.MISSILE_LAUNCH_RANGE_THRESH))
        own_adv = 0.0
        enemy_threat = 0.0
        ego_pos = sim.get_position()
        ego_rpy = sim.get_rpy()
        ref = float(getattr(self, "_missile_launch_range_m_effective", self.MISSILE_LAUNCH_RANGE_THRESH))
        for blue in alive_blue:
            blue_pos = blue.get_position()
            blue_rpy = blue.get_rpy()
            distance = compute_3d_range(ego_pos, blue_pos)
            dist_adv = self._tam_v7_distance_advantage(distance, cfg)
            q_red_blue = compute_body_x_q_los(ego_pos, ego_rpy, blue_pos)
            q_blue_red = compute_body_x_q_los(blue_pos, blue_rpy, ego_pos)
            own_adv += ta_angle_advantage_fixed(np.rad2deg(q_red_blue)) * dist_adv
            enemy_threat += ta_angle_advantage_fixed(np.rad2deg(q_blue_red)) * dist_adv
        denom = max(len(alive_blue), 1)
        own_adv /= denom
        enemy_threat /= denom
        raw = own_adv - float(cfg.get("situation", {}).get("enemy_threat_weight", 0.8)) * enemy_threat
        return float(raw), float(own_adv), float(enemy_threat), ref

    def _tam_v7_uav_reward(self, aid: str, sim, alive_blue: list, cfg: dict) -> tuple[float, dict]:
        prefix = "tam_v7_uav"
        vals: dict[str, float] = {}
        if sim.is_alive:
            vals.update(self._tam_v7_flight_components(prefix, sim, cfg))
            raw, own, threat, ref = self._tam_v7_situation_reward(sim, cfg)
            vals[f"{prefix}_own_adv_mean"] = own
            vals[f"{prefix}_enemy_threat_mean"] = threat
            vals[f"{prefix}_distance_ref_m"] = ref
            vals[f"{prefix}_situation_raw"] = raw
            vals[f"{prefix}_situation"] = float(cfg.get("situation", {}).get("weight", 0.15)) * raw
        else:
            for key in (
                "pitch", "roll", "altitude", "speed", "boundary", "flight",
                "own_adv_mean", "enemy_threat_mean", "distance_ref_m",
                "situation_raw", "situation",
            ):
                vals[f"{prefix}_{key}"] = 0.0

        ev = cfg.get("uav_event", {})
        kills = int(self._step_kill_count.get(aid, 0))
        vals[f"{prefix}_kill"] = kills * float(ev.get("kill_enemy", 200.0))
        if (not sim.is_alive) and aid not in self._tam_v7_uav_death_penalized:
            vals[f"{prefix}_death"] = float(ev.get("death", -200.0))
            self._tam_v7_uav_death_penalized.add(aid)
        else:
            vals[f"{prefix}_death"] = 0.0
        if self._tam_v7_out_of_zone(sim) and aid not in self._tam_v7_uav_out_of_zone_penalized:
            vals[f"{prefix}_first_out_of_zone"] = float(ev.get("first_out_of_zone", -100.0))
            self._tam_v7_uav_out_of_zone_penalized.add(aid)
        else:
            vals[f"{prefix}_first_out_of_zone"] = 0.0
        vals[f"{prefix}_event"] = (
            vals[f"{prefix}_kill"]
            + vals[f"{prefix}_death"]
            + vals[f"{prefix}_first_out_of_zone"]
        )
        vals[f"{prefix}_terminal"] = 0.0
        total = (
            vals[f"{prefix}_flight"]
            + vals[f"{prefix}_situation"]
            + vals[f"{prefix}_event"]
            + vals[f"{prefix}_terminal"]
        ) * float(cfg.get("global_scale", 1.0))
        vals[f"{prefix}_total"] = total
        return total, vals

    def _tam_v7_mav_reward(self, mav_id: str, mav, alive_blue: list, cfg: dict,
                           mav_log_track: float = 0.0,
                           mav_log_fire: float = 0.0,
                           mav_log_hit: float = 0.0) -> tuple[float, dict]:
        prefix = "tam_v7_mav"
        vals: dict[str, float] = {}
        if mav.is_alive:
            vals.update(self._tam_v7_flight_components(prefix, mav, cfg))
            mav_pos = np.asarray(mav.get_position(), dtype=np.float64)
            if alive_blue:
                distances = [compute_3d_range(mav_pos, b.get_position()) for b in alive_blue]
                near_d = min(distances)
                d_danger = float(cfg.get("mav_safety", {}).get("d_danger_m", 5000.0))
                d_safe = float(cfg.get("mav_safety", {}).get("d_safe_m", 14000.0))
                if near_d <= d_danger:
                    r_dist = -(1.0 - near_d / max(d_danger, 1e-6))
                elif near_d < d_safe:
                    r_dist = -0.5 * (1.0 - (near_d - d_danger) / max(d_safe - d_danger, 1e-6))
                else:
                    r_dist = 0.2
                r_threat = -1.0 if any(getattr(m, "is_alive", False) for m in getattr(mav, "under_missiles", []) or []) else 0.0
                aspect_threats = [self._tam_v6v3_mav_aspect_threat(mav, blue) for blue in alive_blue]
                r_aspect = -max(aspect_threats) if aspect_threats else 0.0
                blue_centroid = np.mean([b.get_position() for b in alive_blue], axis=0)
                d_blue = compute_3d_range(mav_pos, blue_centroid)
                d_opt = float(cfg.get("mav_support", {}).get("d_opt_m", 8000.0))
                d_max = float(cfg.get("mav_support", {}).get("d_max_m", 25000.0))
                if d_blue < d_opt:
                    r_pos = d_blue / max(d_opt, 1e-6) - 1.0
                elif d_blue < d_max:
                    r_pos = 1.0 - (d_blue - d_opt) / max(d_max - d_opt, 1e-6)
                else:
                    r_pos = -0.5
                obs_range = float(getattr(self, "mav_observation_range_m", 80000.0))
                aware_vals = []
                for blue in alive_blue:
                    d = compute_3d_range(mav_pos, blue.get_position())
                    ao = compute_body_x_q_los(mav.get_position(), mav.get_rpy(), blue.get_position())
                    aware_vals.append(1.0 if d <= obs_range and ao < np.pi / 2 else 0.0)
                r_aware = float(np.mean(aware_vals)) if aware_vals else 0.0
            else:
                r_dist = r_threat = r_aspect = r_pos = r_aware = 0.0
            sc = cfg.get("mav_safety", {})
            safety_raw = (
                float(sc.get("dist_weight", 0.5)) * r_dist
                + float(sc.get("threat_weight", 0.3)) * r_threat
                + float(sc.get("aspect_weight", 0.2)) * r_aspect
            )
            vals[f"{prefix}_safety_raw"] = float(safety_raw)
            vals[f"{prefix}_safety_dist"] = float(r_dist)
            vals[f"{prefix}_safety_threat"] = float(r_threat)
            vals[f"{prefix}_safety_aspect"] = float(r_aspect)
            vals[f"{prefix}_safety"] = (
                float(sc.get("negative_scale", 0.20)) * min(safety_raw, 0.0)
                + float(sc.get("positive_scale", 0.05)) * max(safety_raw, 0.0)
            )
            sp = cfg.get("mav_support", {})
            support_raw = float(sp.get("pos_weight", 0.6)) * r_pos + float(sp.get("aware_weight", 0.4)) * r_aware
            vals[f"{prefix}_support_raw"] = float(support_raw)
            vals[f"{prefix}_support_pos"] = float(r_pos)
            vals[f"{prefix}_support_aware"] = float(r_aware)
            vals[f"{prefix}_support"] = float(sp.get("scale", 0.10)) * float(np.clip(support_raw, -1.0, 1.0))
        else:
            for key in (
                "pitch", "roll", "altitude", "speed", "boundary", "flight",
                "safety_raw", "safety_dist", "safety_threat", "safety_aspect",
                "safety", "support_raw", "support_pos", "support_aware", "support",
            ):
                vals[f"{prefix}_{key}"] = 0.0

        ev = cfg.get("mav_event", {})
        if (not mav.is_alive) and not self._tam_v7_mav_death_penalized:
            vals[f"{prefix}_death"] = float(ev.get("death", -300.0))
            self._tam_v7_mav_death_penalized = True
        else:
            vals[f"{prefix}_death"] = 0.0
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
        cap = float(ev.get("team_credit_cap", 100.0))
        per_kill = float(ev.get("team_credit_per_uav_kill", 40.0))
        credit = 0.0
        if mav.is_alive and team_kills > 0:
            available = max(0.0, cap - self._tam_v7_mav_team_credit_used)
            credit = min(available, team_kills * per_kill)
            self._tam_v7_mav_team_credit_used += credit
        vals[f"{prefix}_team_credit_delta"] = credit
        vals[f"{prefix}_team_credit_used"] = self._tam_v7_mav_team_credit_used
        vals[f"{prefix}_event"] = vals[f"{prefix}_death"] + credit
        vals[f"{prefix}_terminal"] = 0.0
        vals["tam_v7_shared_track_usage_log"] = mav_log_track
        vals["tam_v7_red_fire_with_mav_track_log"] = mav_log_fire
        vals["tam_v7_red_hit_with_mav_track_log"] = mav_log_hit
        total = (
            vals[f"{prefix}_flight"]
            + vals[f"{prefix}_safety"]
            + vals[f"{prefix}_support"]
            + vals[f"{prefix}_event"]
            + vals[f"{prefix}_terminal"]
        ) * float(cfg.get("global_scale", 1.0))
        vals[f"{prefix}_total"] = total
        return total, vals

    def _tam_v7_terminal_outcome(self, cfg: dict) -> tuple[float, dict]:
        red_roles = getattr(self, "agent_roles", {})
        mav_id = next((rid for rid in self.red_ids if red_roles.get(rid) == "mav"), self.red_ids[0])
        uav_ids = [rid for rid in self.red_ids if rid != mav_id]
        n_blue_initial = max(len(self.blue_ids), 1)
        n_uav_initial = max(len(uav_ids), 1)
        n_blue_alive = sum(1 for bid in self.blue_ids if self.blue_planes.get(bid) and self.blue_planes[bid].is_alive)
        n_uav_dead = sum(1 for rid in uav_ids if self.red_planes.get(rid) and not self.red_planes[rid].is_alive)
        mav_dead = bool(self.red_planes.get(mav_id) and not self.red_planes[mav_id].is_alive)
        blue_loss_frac = (n_blue_initial - n_blue_alive) / n_blue_initial
        tcfg = cfg.get("terminal", {})
        if tcfg.get("mav_weight_mode", "match_uav_count") == "match_uav_count":
            w_mav = float(n_uav_initial)
        else:
            w_mav = float(tcfg.get("mav_weight", n_uav_initial))
        red_loss_weighted = (w_mav * float(mav_dead) + float(n_uav_dead)) / max(w_mav + n_uav_initial, 1e-6)
        value = float(tcfg.get("coef_per_agent", 30.0)) * (blue_loss_frac - red_loss_weighted)
        return value, {
            "tam_v7_blue_loss_frac": float(blue_loss_frac),
            "tam_v7_red_loss_weighted": float(red_loss_weighted),
        }

    def _tam_v7_log_only_track_fields(self) -> tuple[float, float, float]:
        shared_usage = 0.0
        red_fire_with_mav_track = 0.0
        red_hit_with_mav_track = 0.0
        obs = getattr(self, "_last_step_obs", {})
        if obs:
            red_uav_track_count = 0
            for uid in self.red_ids:
                if self.agent_roles.get(uid) == "mav":
                    continue
                uav_obs = obs.get(uid, {})
                ets = uav_obs.get("enemy_track_source", None)
                if ets is not None:
                    for ts_row in ets:
                        if len(ts_row) > 1 and int(ts_row[1]) == 1:
                            red_uav_track_count += 1
            if red_uav_track_count > 0:
                shared_usage = float(red_uav_track_count) / max(len(self.red_ids) - 1, 1)
        for rec in getattr(self, "_launch_quality_step_records", []) or []:
            if isinstance(rec, dict) and str(rec.get("launch_track_source", "")) == "mav_shared":
                red_fire_with_mav_track += 1.0
        for rec in getattr(self, "_launch_quality_done_step_records", []) or []:
            if (
                isinstance(rec, dict)
                and str(rec.get("launch_track_source", "")) == "mav_shared"
                and str(rec.get("raw_termination_reason", "")) == "hit"
            ):
                red_hit_with_mav_track += 1.0
        return shared_usage, red_fire_with_mav_track, red_hit_with_mav_track

    def _compute_tam_paper_reward_v7_role_aligned(self, base_rewards: dict, components: dict):
        cfg = self.tam_paper_reward_v7_role_aligned_config
        alive_blue = self._tam_v2_alive_blue()
        mav_id = next((rid for rid in self.red_ids if self.agent_roles.get(rid) == "mav"),
                      self.red_ids[0] if self.red_ids else None)
        n_blue_alive = sum(1 for sim in self.blue_planes.values() if sim.is_alive)
        n_red_alive = sum(1 for sim in self.red_planes.values() if sim.is_alive)
        round_over = n_blue_alive == 0 or n_red_alive == 0 or self.current_step >= self.max_steps
        terminal = 0.0
        terminal_logs = {"tam_v7_blue_loss_frac": 0.0, "tam_v7_red_loss_weighted": 0.0}
        if round_over and not self._tam_v7_terminal_applied:
            terminal, terminal_logs = self._tam_v7_terminal_outcome(cfg)
            self._tam_v7_terminal_applied = True
        shared_usage, fire_log, hit_log = self._tam_v7_log_only_track_fields()
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            if rid == mav_id:
                _, comp = self._tam_v7_mav_reward(rid, sim, alive_blue, cfg, shared_usage, fire_log, hit_log)
                role_prefix = "tam_v7_mav"
            else:
                _, comp = self._tam_v7_uav_reward(rid, sim, alive_blue, cfg)
                role_prefix = "tam_v7_uav"
            comp[f"{role_prefix}_terminal"] = terminal
            comp[f"{role_prefix}_total"] += terminal * float(cfg.get("global_scale", 1.0))
            comp["tam_v7_flight"] = comp[f"{role_prefix}_flight"]
            comp["tam_v7_event"] = comp[f"{role_prefix}_event"]
            comp["tam_v7_terminal"] = terminal
            comp["tam_v7_terminal_per_agent"] = terminal
            comp.update(terminal_logs)
            comp["tam_v7_total"] = comp[f"{role_prefix}_total"]
            base_rewards[rid] = comp[f"{role_prefix}_total"]
            components[rid] = comp
        return base_rewards, components

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

    def _needs_last_step_obs_cache(self) -> bool:
        return self.hetero_reward_mode in {
            "minimal_v1", "role_v1", "happo_ref_v0", "paper_role_reward_v1",
            "tam_paper_reward_v2", "tam_paper_reward_v3", "tam_paper_reward_v4",
            "tam_paper_reward_v6_jsbsim_aligned_v3", "tam_paper_reward_v7_role_aligned",
            "tam_brma_scripted_reward_v1", "brma_paper_homogeneous_v1",
            "brma_role_no_missile_reward_v8", "happo_ref_v1_mav_support",
            "tam_brma_paper_aligned_v1", "tam_happo_table1_v1",
            "brma_tam_scripted_composite_v1", "brma_tam_scale_aligned_v1",
        }

    def step(self, actions: dict):
        if self.hetero_reward_mode in {"brma_tam_scripted_composite_v1", "brma_tam_scale_aligned_v1"}:
            self._brma_tam_alive_before_step = {
                aid: bool(getattr((self.red_planes.get(aid) or self.blue_planes.get(aid)), "is_alive", False))
                for aid in self.agent_ids
            }
            self._reward_target_diagnostic_records = []
        if self.hetero_reward_mode == "brma_tam_scale_aligned_v1":
            self._scale_v1_alive_before_step = dict(self._brma_tam_alive_before_step)
        trimmed = self._apply_action_trim(actions)
        obs, rewards, terminated, truncated, info = super().step(trimmed)
        if self._needs_last_step_obs_cache():
            self._last_step_obs = obs
        return obs, rewards, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._last_step_obs = {}
        self._mav_death_penalized = False
        self._uav_death_penalized = set()
        self._paper_reset_reward_state()
        self._paper_terminal_applied = False
        self._tam_v2_out_of_zone_penalized = set()
        self._tam_v2_missile_speed_cache = {}
        self._tam_v3_out_of_zone_active: set[str] = set()
        self._tam_v4_terminal_applied: bool = False
        self._brma_homo_reset_episode_state()
        self._happo_ref_v1_reset_episode_state()
        self._tam_brma_paper_v1_reset_episode_state()
        self._tam_happo_table1_v1_reset_episode_state()
        self._brma_tam_scripted_reset_episode_state()
        self._brma_tam_scale_v1_reset_episode_state()
        self._tam_v6v3_reset_episode_state()
        self._tam_v7_reset_episode_state()
        self._tam_brma_scripted_terminal_applied: bool = False
        self._tam_brma_scripted_mav_death_penalized: bool = False
        self._tam_brma_scripted_uav_death_penalized: set[str] = set()
        obs, info = super().reset(*args, **kwargs)
        if self._needs_last_step_obs_cache():
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

    def _add_mav_shared_geo_spaces(self, spaces: dict, max_allies: int, max_enemies: int) -> None:
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
        spaces["enemy_relative_pos_xyz"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_enemies, 3), dtype=np.float32)
        spaces["enemy_relative_vel_xyz"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_enemies, 3), dtype=np.float32)
        spaces["enemy_bearing_elevation"] = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(max_enemies, 2), dtype=np.float32)
        spaces["enemy_speed_heading"] = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_enemies, 2), dtype=np.float32)
        spaces["enemy_full_geo_valid_mask"] = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(max_enemies,), dtype=np.float32)

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
        enemy_relative_pos_xyz = np.zeros((max_enemies, 3), dtype=np.float32)
        enemy_relative_vel_xyz = np.zeros((max_enemies, 3), dtype=np.float32)
        enemy_bearing_elevation = np.zeros((max_enemies, 2), dtype=np.float32)
        enemy_speed_heading = np.zeros((max_enemies, 2), dtype=np.float32)
        enemy_full_geo_valid_mask = np.zeros(max_enemies, dtype=np.float32)

        if not ego_alive:
            out = {
                "ego_geo_state": ego_geo_state,
                "ally_geo_states": ally_geo_states,
                "enemy_geo_states": enemy_geo_states,
                "ally_alive_mask": ally_alive_mask,
                "enemy_alive_mask": enemy_alive_mask,
                "enemy_observed_mask": enemy_observed_mask,
                "enemy_track_source": enemy_track_source,
            }
            out.update({
                "enemy_relative_pos_xyz": enemy_relative_pos_xyz,
                "enemy_relative_vel_xyz": enemy_relative_vel_xyz,
                "enemy_bearing_elevation": enemy_bearing_elevation,
                "enemy_speed_heading": enemy_speed_heading,
                "enemy_full_geo_valid_mask": enemy_full_geo_valid_mask,
            })
            return out

        ego_geo_state = self._ego_geo_state(ego_sim)

        for i, ally_id in enumerate(ally_ids):
            ally_sim = self._get_sim(ally_id)
            if ally_sim is not None and ally_sim.is_alive:
                ally_alive_mask[i] = 1.0
                ally_geo_states[i] = self._relative_geo_state(ego_sim, ally_sim)

        for i, enemy_id in enumerate(enemy_ids):
            enemy_sim = self._get_sim(enemy_id)
            if enemy_sim is None or not enemy_sim.is_alive:
                continue
            enemy_alive_mask[i] = 1.0
            visibility = self._mav_shared_track_state(agent_id, enemy_id)
            own_direct = visibility["direct_visible"]
            mav_shared = visibility["mav_shared_visible"]

            if own_direct or mav_shared:
                enemy_geo_states[i] = self._relative_geo_state(ego_sim, enemy_sim)
                enemy_observed_mask[i] = 1.0
                enemy_track_source[i] = np.array(
                    [1.0 if own_direct else 0.0, 1.0 if mav_shared else 0.0],
                    dtype=np.float32,
                )
                self._fill_enemy_full_geo(
                    ego_sim, enemy_sim, i, enemy_relative_pos_xyz,
                    enemy_relative_vel_xyz, enemy_bearing_elevation,
                    enemy_speed_heading, enemy_full_geo_valid_mask)

        out = {
            "ego_geo_state": ego_geo_state,
            "ally_geo_states": ally_geo_states,
            "enemy_geo_states": enemy_geo_states,
            "ally_alive_mask": ally_alive_mask,
            "enemy_alive_mask": enemy_alive_mask,
            "enemy_observed_mask": enemy_observed_mask,
            "enemy_track_source": enemy_track_source,
        }
        out.update({
            "enemy_relative_pos_xyz": enemy_relative_pos_xyz,
            "enemy_relative_vel_xyz": enemy_relative_vel_xyz,
            "enemy_bearing_elevation": enemy_bearing_elevation,
            "enemy_speed_heading": enemy_speed_heading,
            "enemy_full_geo_valid_mask": enemy_full_geo_valid_mask,
        })
        return out

    @staticmethod
    def _fill_enemy_full_geo(
        ego_sim,
        enemy_sim,
        index: int,
        enemy_relative_pos_xyz: np.ndarray,
        enemy_relative_vel_xyz: np.ndarray,
        enemy_bearing_elevation: np.ndarray,
        enemy_speed_heading: np.ndarray,
        enemy_full_geo_valid_mask: np.ndarray,
    ) -> None:
        rel_pos = np.asarray(enemy_sim.get_position(), dtype=np.float64) - np.asarray(ego_sim.get_position(), dtype=np.float64)
        rel_vel = np.asarray(enemy_sim.get_velocity(), dtype=np.float64) - np.asarray(ego_sim.get_velocity(), dtype=np.float64)
        enemy_vel = np.asarray(enemy_sim.get_velocity(), dtype=np.float64)
        enemy_rpy = np.asarray(enemy_sim.get_rpy(), dtype=np.float64)
        horizontal = float(np.linalg.norm(rel_pos[:2]))
        bearing = math.atan2(float(rel_pos[1]), float(rel_pos[0])) / np.pi
        elevation = math.atan2(float(rel_pos[2]), max(horizontal, 1e-6)) / np.pi
        enemy_relative_pos_xyz[index] = (rel_pos / 40000.0).astype(np.float32)
        enemy_relative_vel_xyz[index] = (rel_vel / 600.0).astype(np.float32)
        enemy_bearing_elevation[index] = np.array([bearing, elevation], dtype=np.float32)
        enemy_speed_heading[index] = np.array([
            float(np.linalg.norm(enemy_vel)) / 600.0,
            float(enemy_rpy[2]) / np.pi,
        ], dtype=np.float32)
        enemy_full_geo_valid_mask[index] = 1.0

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

                comp["tam_mav_dense_reward"] = r_safe + r_sup + r_alive

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
            # Count current-step fire launches from launch quality records
            if self.agent_roles.get(rid, "") != "mav":
                step_launches = [
                    r for r in (getattr(self, "_launch_quality_step_records", None) or [])
                    if str(r.get("shooter_id", "")) == str(rid)]
                comp["uav_fire_direct_count"] = sum(
                    1 for r in step_launches if not bool(r.get("mav_guided_at_launch", False)))
                comp["uav_fire_mav_guided_count"] = sum(
                    1 for r in step_launches if bool(r.get("mav_guided_at_launch", False)))
            comp["event_total"] = sum(
                comp.get(k, 0.0) for k in ("event_uav_kill", "event_team_kill",
                    "event_uav_death", "event_uav_crash",
                    "event_mav_death", "event_mav_loss_team", "event_out_zone"))

        # ── Final clipping [-10, 10] for red agents ──
        for rid in self.red_ids:
            pre = base_rewards.get(rid, 0.0)
            base_rewards[rid] = float(np.clip(pre, -10.0, 10.0))
            components.setdefault(rid, {})["reward_pre_clip"] = pre
            components.setdefault(rid, {})["reward_clip_delta"] = base_rewards[rid] - pre

        return base_rewards, components

    # ── TAM Paper Reward v2 ───────────────────────────────────────────

    @staticmethod
    def _tam_v2_feature(sim) -> np.ndarray:
        """Absolute feature vector for 2D AO/TA geometry (aligned with tam_uav)."""
        position = np.asarray(sim.get_position(), dtype=np.float64)
        velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
        return np.array([
            position[0], position[1], -position[2],
            velocity[0], velocity[1], -velocity[2],
        ], dtype=np.float64)

    @staticmethod
    def _tam_v3_uav_distance_reward(distance_m: float) -> float:
        """Launch-window-aligned: peak 3-7km, positive 0.5-10km, soft decay beyond."""
        d_km = distance_m / 1000.0
        if d_km <= 0.5:
            return -0.5  # too close, sub-min-launch-range
        if d_km <= 3.0:
            return 0.5 + 0.5 * (d_km - 0.5) / 2.5  # 0.5 → 1.0
        if d_km <= 7.0:
            return 1.0  # optimal engagement
        if d_km <= 10.0:
            return 1.0 - 0.8 * (d_km - 7.0) / 3.0  # 1.0 → 0.2
        if d_km <= 15.0:
            return 0.2 - 0.7 * (d_km - 10.0) / 5.0  # 0.2 → -0.5
        return -1.0  # disengaged

    @staticmethod
    def _tam_v2_uav_distance_reward(distance_m: float) -> float:
        d_km = distance_m / 1000.0
        if d_km <= 5.0:
            return 1.0
        if d_km < 10.0:
            return np.exp(-0.921 * (d_km - 5.0))
        return -1.0

    @staticmethod
    def _tam_v2_speed_reward(red_speed: float, blue_speed: float) -> float:
        red_speed = max(float(red_speed), 1e-8)
        blue_speed = float(blue_speed)
        if blue_speed < 0.5 * red_speed:
            return 1.0
        if blue_speed <= 1.5 * red_speed:
            return 2.0 - 2.0 * blue_speed / red_speed
        return -1.0

    def _tam_v2_height_reward(self, altitude_m: float, cfg: dict) -> float:
        g = cfg["geometry"]
        effective_min = max(float(g.get("min_altitude_m", 750.0)),
                           float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0)))
        optimum = float(g.get("optimal_altitude_m", 6000.0))
        maximum = float(g.get("max_altitude_m", 12000.0))
        optimum = float(np.clip(optimum, effective_min, maximum))
        if altitude_m < effective_min:
            return -1.0
        if altitude_m > maximum:
            return -0.5
        value = 1.0 - abs(float(altitude_m) - optimum) / (maximum - effective_min)
        return float(np.clip(value, 0.0, 1.0))

    def _tam_v2_alive_blue(self) -> list:
        return [sim for bid in self.blue_ids if (sim := self.blue_planes.get(bid)) and sim.is_alive]

    def _tam_v2_dodge_reward(self, sim, v_norm_mps: float, cache: dict) -> tuple:
        threat = getattr(sim, "under_missiles", None)
        if not threat:
            return 0.0, 0.0, 0.0
        candidates = []
        for missile in list(threat):
            if not getattr(missile, "is_alive", False):
                continue
            uid = str(getattr(missile, "uid", getattr(missile, "_uid", id(missile))))
            mv = np.array(missile.get_velocity(), dtype=np.float64)
            sp = np.linalg.norm(mv)
            los = np.array(sim.get_position(), dtype=np.float64) - np.array(missile.get_position(), dtype=np.float64)
            los_norm = np.linalg.norm(los)
            if los_norm < 1e-6:
                continue
            cos_angle = float(np.dot(mv, los) / (sp * los_norm) if sp > 1e-6 else 0.0)
            r_angle = -float(np.clip(cos_angle, -1.0, 1.0))
            prev_sp = cache.get(uid)
            r_speed = 0.0 if prev_sp is None else (prev_sp - sp) / v_norm_mps
            cache[uid] = sp
            candidates.append((r_angle + r_speed, r_angle, r_speed))
        if not candidates:
            return 0.0, 0.0, 0.0
        return max(candidates, key=lambda item: item[0])

    def _tam_v2_mav_reward(self, mav_id: str, mav, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        vals: dict[str, float] = {}
        mav_pos = np.array(mav.get_position(), dtype=np.float64)
        mav_vel = np.array(mav.get_velocity(), dtype=np.float64)

        # ── Safety ──
        sw = cfg["mav"]["safety_weights"]
        d_danger = float(cfg["mav"]["d_danger_m"])
        d_safe = float(cfg["mav"]["d_safe_m"])
        r_dist = 0.0; r_threat = 0.0; r_aspect = 0.0; r_aware = 0.0; r_pos = 0.0
        if mav.is_alive:
            if alive_blue:
                distances = [float(np.linalg.norm(b.get_position() - mav_pos)) for b in alive_blue]
                near_d = min(distances)
                if near_d <= d_danger:
                    r_dist = -(1.0 - near_d / d_danger)
                elif near_d < d_safe:
                    r_dist = -0.5 * (1.0 - (near_d - d_danger) / (d_safe - d_danger))
                else:
                    r_dist = 0.2
                vals["tam_v2_mav_dist"] = r_dist

                # threat: -1.0 if any live incoming missile under MAV
                threat_missiles = getattr(mav, "under_missiles", None)
                r_threat = -1.0 if (threat_missiles and any(getattr(m, "is_alive", False) for m in threat_missiles)) else 0.0
                vals["tam_v2_mav_threat"] = r_threat

                # aspect: for each blue heading toward MAV (TA < pi/4), penalize
                mav_feat = HeteroUavCombatEnv._tam_v2_feature(mav)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    if ta < np.pi / 4:
                        r_aspect -= (1.0 - ta / (np.pi / 4))
                vals["tam_v2_mav_aspect"] = r_aspect

                # aware: for each blue within MAV obs range and visible
                mav_obs_range = getattr(self, "mav_observation_range_m", 80000.0)
                r_aware = 0.0
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, _ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    d = float(np.linalg.norm(b.get_position() - mav_pos))
                    if d < mav_obs_range and ao < np.pi / 2:
                        r_aware += 0.3 * (1.0 - ao / (np.pi / 2))
                vals["tam_v2_mav_aware"] = r_aware
            else:
                for k in ("tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect", "tam_v2_mav_aware"):
                    vals[k] = 0.0

            vals["tam_v2_mav_safety"] = sw["dist"] * vals["tam_v2_mav_dist"] + sw["threat"] * vals["tam_v2_mav_threat"] + sw["aspect"] * vals["tam_v2_mav_aspect"]

            # ── Support ──
            sup_w = cfg["mav"]["support_weights"]
            d_opt = float(cfg["mav"]["d_opt_m"])
            d_max_mav = float(cfg["mav"]["d_max_m"])
            if alive_blue:
                blue_centroid = np.mean([b.get_position() for b in alive_blue], axis=0)
                d_b = float(np.linalg.norm(blue_centroid - mav_pos))
                if d_b <= d_opt:
                    r_pos = d_b / d_opt - 1.0
                elif d_b < d_max_mav:
                    r_pos = 1.0 - (d_b - d_opt) / (d_max_mav - d_opt)
                else:
                    r_pos = -0.5
            else:
                r_pos = 0.0
            vals["tam_v2_mav_pos"] = r_pos
            vals["tam_v2_mav_support"] = sup_w["pos"] * r_pos + sup_w["aware"] * vals["tam_v2_mav_aware"]
        else:
            for k in ("tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect", "tam_v2_mav_aware",
                      "tam_v2_mav_safety", "tam_v2_mav_pos", "tam_v2_mav_support"):
                vals[k] = 0.0

        # ── Event ──
        r_event = 0.0
        # MAV death (one-shot)
        if (not mav.is_alive) and (not self._mav_death_penalized):
            r_event -= float(cfg["mav"]["death_penalty"])
            self._mav_death_penalized = True
            vals["tam_v2_mav_death"] = -float(cfg["mav"]["death_penalty"])
        else:
            vals["tam_v2_mav_death"] = 0.0
        # Team kill bonus
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
        per_kill = min(float(cfg["mav"]["team_kill_bonus"]), float(cfg["mav"].get("team_kill_bonus_cap", 200.0)))
        team_bonus = team_kills * per_kill
        vals["tam_v2_mav_team_bonus"] = team_bonus
        r_event += team_bonus
        vals["tam_v2_mav_event"] = r_event

        # ── Log-only BRMA fields ──
        orig_brma = base_components.get(mav_id, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v2_mav_shared_log"] = 0.0
        vals["tam_v2_mav_assist_log"] = 0.0
        vals["tam_v2_geometry_feature_semantics"] = "absolute"
        vals["tam_v2_height_formula_source"] = "tam_uav_paper_approx_not_exact_formula"

        gs = float(cfg["global_scale"])
        total = (vals["tam_v2_mav_safety"] + vals["tam_v2_mav_support"] + vals["tam_v2_mav_event"]) * gs
        vals["tam_v2_total"] = total
        return total, vals

    def _tam_v2_uav_reward(self, aid: str, sim, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        vals: dict[str, float] = {}
        w = cfg["uav"]["reward_weights"]
        v_norm = float(cfg["uav"].get("v_norm_mps", 1000.0))
        geo = cfg["geometry"]
        missile_range = float(geo.get("missile_range_m", 14000.0))
        zone_radius = float(geo.get("combat_zone_radius_m", 50000.0))
        sim_pos = np.array(sim.get_position(), dtype=np.float64)
        sim_vel = np.array(sim.get_velocity(), dtype=np.float64)
        sim_sp = float(np.linalg.norm(sim_vel))
        alt = float(sim.get_geodetic()[2])

        if sim.is_alive:
            # ── Height ──
            vals["tam_v2_uav_height"] = w["height"] * self._tam_v2_height_reward(alt, cfg)

            # ── Speed ──
            if alive_blue:
                blue_speeds = [float(np.linalg.norm(b.get_velocity())) for b in alive_blue]
                best_speed = max(self._tam_v2_speed_reward(sim_sp, bs) for bs in blue_speeds)
            else:
                best_speed = 0.0
            vals["tam_v2_uav_speed"] = w["speed"] * best_speed

            # ── Angle ──
            if alive_blue:
                best_angle_raw = -1.0
                red_feat = HeteroUavCombatEnv._tam_v2_feature(sim)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, ta, _r = get2d_AO_TA_R(red_feat, b_feat)
                    aa = np.pi - ta
                    angle_val = 1.0 - (ao + aa) / np.pi
                    if angle_val > best_angle_raw:
                        best_angle_raw = angle_val
                vals["tam_v2_uav_angle_raw"] = best_angle_raw
                vals["tam_v2_uav_angle"] = w["angle"] * max(best_angle_raw, -1.0)
            else:
                vals["tam_v2_uav_angle_raw"] = 0.0
                vals["tam_v2_uav_angle"] = 0.0

            # ── Distance ──
            if alive_blue:
                dists = [float(np.linalg.norm(b.get_position() - sim_pos)) for b in alive_blue]
                best_dist = max(self._tam_v2_uav_distance_reward(d) for d in dists)
            else:
                best_dist = 0.0
            vals["tam_v2_uav_distance"] = w["distance"] * best_dist

            # ── Dodge ──
            d_total, d_angle, d_speed = self._tam_v2_dodge_reward(sim, v_norm, self._tam_v2_missile_speed_cache)
            vals["tam_v2_uav_dodge"] = w["dodge"] * d_total
            vals["tam_v2_uav_dodge_angle"] = d_angle
            vals["tam_v2_uav_dodge_speed"] = d_speed
        else:
            for k in ("tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
                      "tam_v2_uav_angle_raw", "tam_v2_uav_distance",
                      "tam_v2_uav_dodge", "tam_v2_uav_dodge_angle", "tam_v2_uav_dodge_speed"):
                vals[k] = 0.0

        # ── Event ──
        ev = cfg["uav"]["event"]
        r_event = 0.0
        kills = int(self._step_kill_count.get(aid, 0))
        r_event += kills * float(ev["kill_enemy"])
        vals["tam_v2_uav_kill"] = kills * float(ev["kill_enemy"])
        if (not sim.is_alive) and aid not in self._uav_death_penalized:
            r_event += float(ev["death"])
            self._uav_death_penalized.add(aid)
            vals["tam_v2_uav_death"] = float(ev["death"])
        else:
            vals["tam_v2_uav_death"] = 0.0
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        alt_max = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        alt_min = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        out = (abs(float(sim_pos[0])) > half or abs(float(sim_pos[1])) > half
               or alt > alt_max or alt < alt_min)
        if out and aid not in self._tam_v2_out_of_zone_penalized:
            r_event += float(ev["out_of_zone"])
            self._tam_v2_out_of_zone_penalized.add(aid)
            vals["tam_v2_uav_out_of_zone"] = float(ev["out_of_zone"])
        else:
            vals["tam_v2_uav_out_of_zone"] = 0.0
        vals["tam_v2_uav_event"] = r_event

        # ── Log-only fields ──
        orig_brma = base_components.get(aid, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v2_uav_fire_log"] = 0.0
        vals["tam_v2_uav_mav_shared_track_log"] = 0.0
        vals["tam_v2_geometry_feature_semantics"] = "absolute"
        vals["tam_v2_dodge_los_semantics"] = "missile_to_aircraft"
        vals["tam_v2_height_formula_source"] = "tam_uav_paper_approx_not_exact_formula"

        gs = float(cfg["global_scale"])
        dense_event = (
            vals["tam_v2_uav_height"] + vals["tam_v2_uav_speed"] + vals["tam_v2_uav_angle"]
            + vals["tam_v2_uav_distance"] + vals["tam_v2_uav_dodge"] + vals["tam_v2_uav_event"]
        )
        total = dense_event * gs
        vals["tam_v2_total"] = total
        return total, vals

    def _compute_tam_paper_reward_v2(self, base_rewards: dict, components: dict):
        cfg = self.tam_paper_reward_v2_config
        alive_blue = self._tam_v2_alive_blue()
        mav_id = next((
            aid for aid in self.red_ids if self.agent_roles.get(aid) == "mav"
        ), self.red_ids[0] if self.red_ids else None)
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            if rid == mav_id:
                reward, comp = self._tam_v2_mav_reward(rid, sim, alive_blue, cfg, components)
            else:
                reward, comp = self._tam_v2_uav_reward(rid, sim, alive_blue, cfg, components)
            base_rewards[rid] = reward
            components[rid] = comp
        return base_rewards, components

    # ── end TAM paper reward v2 ────────────────────────────────────────

    # ── TAM Paper Reward v3 (env-consistent) ───────────────────────────
    # Same TAM-HAPPO categories as v2, but formulas adapted to current
    # JSBSim 3v2 environment boundaries (BATTLEFIELD_ALTITUDE_MIN=2500,
    # BATTLEFIELD_ALTITUDE_MAX=10000, BATTLEFIELD_HALF_SIZE=40000).

    def _tam_v3_height_reward(self, altitude_m: float, cfg: dict) -> float:
        """Env-consistent: ceiling at BATTLEFIELD_ALTITUDE_MAX=10000m."""
        g = cfg["geometry"]
        env_min = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        env_max = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        eff_min = max(float(g.get("min_altitude_m", 750.0)), env_min)
        eff_max = env_max  # NOT config max — env boundary
        optimum = float(np.clip(float(g.get("optimal_altitude_m", 6000.0)), eff_min, eff_max))
        if altitude_m < eff_min:
            return -1.0
        if altitude_m > eff_max:
            return -1.0  # strongly negative above env ceiling
        val = 1.0 - abs(float(altitude_m) - optimum) / (eff_max - eff_min)
        return float(np.clip(val, 0.0, 1.0))

    @staticmethod
    def _tam_v3_speed_reward(red_speed: float, blue_speed: float) -> float:
        """Env-consistent: penalise near-stall speeds (<100 m/s)."""
        red_speed = max(float(red_speed), 1e-8)
        if red_speed < 100.0:
            return -1.0  # near stall — cannot manoeuvre
        blue_speed = float(blue_speed)
        if blue_speed < 0.5 * red_speed:
            return 1.0
        if blue_speed <= 1.5 * red_speed:
            return 2.0 - 2.0 * blue_speed / red_speed
        return -1.0

    def _tam_v3_out_of_zone_penalty(self, sim, aid: str, cfg: dict) -> float:
        """Env-consistent continuous boundary penalty, configurable per step."""
        half = float(getattr(self, "BATTLEFIELD_HALF_SIZE", 40000.0))
        alt_max = float(getattr(self, "BATTLEFIELD_ALTITUDE_MAX", 10000.0))
        alt_min = float(getattr(self, "BATTLEFIELD_ALTITUDE_MIN", 2500.0))
        pos = np.asarray(sim.get_position(), dtype=np.float64)
        alt = float(sim.get_geodetic()[2])
        if abs(float(pos[0])) > half or abs(float(pos[1])) > half or alt > alt_max or alt < alt_min:
            if aid not in self._tam_v3_out_of_zone_active:
                self._tam_v3_out_of_zone_active.add(aid)
            return float(cfg["uav"]["event"].get("out_of_zone_per_step", -2.0))
        return 0.0

    def _tam_v3_mav_reward(self, mav_id: str, mav, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        """MAV reward — v2 structure + v3 continuous boundary penalty + continuous r_pos."""
        vals: dict[str, float] = {}
        mav_pos = np.array(mav.get_position(), dtype=np.float64)
        mav_vel = np.array(mav.get_velocity(), dtype=np.float64)
        sw = cfg["mav"]["safety_weights"]
        d_danger = float(cfg["mav"]["d_danger_m"])
        d_safe = float(cfg["mav"]["d_safe_m"])
        r_dist = 0.0; r_threat = 0.0; r_aspect = 0.0; r_aware = 0.0; r_pos = 0.0
        if mav.is_alive:
            if alive_blue:
                distances = [float(np.linalg.norm(b.get_position() - mav_pos)) for b in alive_blue]
                near_d = min(distances)
                if near_d <= d_danger:
                    r_dist = -(1.0 - near_d / d_danger)
                elif near_d < d_safe:
                    r_dist = -0.5 * (1.0 - (near_d - d_danger) / (d_safe - d_danger))
                else:
                    r_dist = 0.2
                vals["tam_v2_mav_dist"] = r_dist
                threat_missiles = getattr(mav, "under_missiles", None)
                r_threat = -1.0 if (threat_missiles and any(getattr(m, "is_alive", False) for m in threat_missiles)) else 0.0
                vals["tam_v2_mav_threat"] = r_threat
                mav_feat = HeteroUavCombatEnv._tam_v2_feature(mav)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    if ta < np.pi / 4:
                        r_aspect -= (1.0 - ta / (np.pi / 4))
                vals["tam_v2_mav_aspect"] = r_aspect
                mav_obs_range = getattr(self, "mav_observation_range_m", 80000.0)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, _ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    d = float(np.linalg.norm(b.get_position() - mav_pos))
                    if d < mav_obs_range and ao < np.pi / 2:
                        r_aware += 0.3 * (1.0 - ao / (np.pi / 2))
                vals["tam_v2_mav_aware"] = r_aware

                # r_pos: tam_uav-aligned linear formula
                sup_w = cfg["mav"]["support_weights"]
                d_opt = float(cfg["mav"]["d_opt_m"])
                d_max_mav = float(cfg["mav"]["d_max_m"])
                blue_centroid = np.mean([b.get_position() for b in alive_blue], axis=0)
                d_b = float(np.linalg.norm(blue_centroid - mav_pos))
                if d_b < d_opt:
                    r_pos = d_b / d_opt - 1.0  # negative when closer than d_opt
                elif d_b < d_max_mav:
                    r_pos = 1.0 - (d_b - d_opt) / (d_max_mav - d_opt)
                else:
                    r_pos = -0.5
                vals["tam_v2_mav_pos"] = r_pos
                vals["tam_v2_mav_support"] = sup_w["pos"] * r_pos + sup_w["aware"] * r_aware
            else:
                for k in ("tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect", "tam_v2_mav_aware",
                          "tam_v2_mav_pos", "tam_v2_mav_safety", "tam_v2_mav_support"):
                    vals[k] = 0.0
            vals["tam_v2_mav_safety"] = sw["dist"] * r_dist + sw["threat"] * r_threat + sw["aspect"] * r_aspect
        else:
            for k in ("tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect", "tam_v2_mav_aware",
                      "tam_v2_mav_safety", "tam_v2_mav_pos", "tam_v2_mav_support"):
                vals[k] = 0.0

        # ── Event (v3: continuous out-of-zone applies to MAV too) ──
        r_event = 0.0
        if (not mav.is_alive) and (not self._mav_death_penalized):
            r_event -= float(cfg["mav"]["death_penalty"])
            self._mav_death_penalized = True
            vals["tam_v2_mav_death"] = -float(cfg["mav"]["death_penalty"])
        else:
            vals["tam_v2_mav_death"] = 0.0
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
        per_kill = min(float(cfg["mav"]["team_kill_bonus"]), float(cfg["mav"].get("team_kill_bonus_cap", 200.0)))
        vals["tam_v2_mav_team_bonus"] = team_kills * per_kill
        r_event += team_kills * per_kill
        if mav.is_alive:
            oz_penalty = self._tam_v3_out_of_zone_penalty(mav, mav_id, cfg)
        else:
            oz_penalty = 0.0
        r_event += oz_penalty
        vals["tam_v2_mav_event"] = r_event

        # Log-only
        orig_brma = base_components.get(mav_id, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v2_mav_shared_log"] = 0.0
        vals["tam_v2_mav_assist_log"] = 0.0
        vals["tam_v2_geometry_feature_semantics"] = "absolute"
        vals["tam_v2_height_formula_source"] = "tam_paper_v3_env_consistent"

        gs = float(cfg["global_scale"])
        total = (vals["tam_v2_mav_safety"] + vals["tam_v2_mav_support"] + vals["tam_v2_mav_event"]) * gs
        vals["tam_v2_total"] = total
        return total, vals

    def _tam_v3_uav_reward(self, aid: str, sim, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        """UAV reward — v2 structure with v3 height/speed/out-of-zone + per-target consistency."""
        vals: dict[str, float] = {}
        w = cfg["uav"]["reward_weights"]
        v_norm = float(cfg["uav"].get("v_norm_mps", 1000.0))
        sim_pos = np.array(sim.get_position(), dtype=np.float64)
        sim_vel = np.array(sim.get_velocity(), dtype=np.float64)
        sim_sp = float(np.linalg.norm(sim_vel))
        alt = float(sim.get_geodetic()[2])

        if sim.is_alive:
            vals["tam_v2_uav_height"] = w["height"] * self._tam_v3_height_reward(alt, cfg)
            best_target_idx = -1
            if alive_blue:
                red_feat = HeteroUavCombatEnv._tam_v2_feature(sim)
                best_combined = -1e9
                candidates = []
                for idx, b in enumerate(alive_blue):
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, ta, dist_m = get2d_AO_TA_R(red_feat, b_feat)
                    aa = np.pi - ta
                    speed_raw = self._tam_v3_speed_reward(sim_sp, float(np.linalg.norm(b.get_velocity())))
                    angle_raw = 1.0 - (ao + aa) / np.pi
                    dist_raw = self._tam_v3_uav_distance_reward(dist_m)
                    # Normalised weights for combined score
                    w_norm = float(w["speed"]) + float(w["angle"]) + float(w["distance"])
                    combined = (float(w["speed"]) * speed_raw + float(w["angle"]) * angle_raw
                                + float(w["distance"]) * dist_raw) / max(w_norm, 1e-8)
                    candidates.append((combined, idx, speed_raw, angle_raw, dist_raw))
                best = max(candidates, key=lambda x: x[0])
                _, best_target_idx, best_speed_raw, best_angle_raw, best_dist_raw = best
                vals["tam_v2_uav_speed"] = w["speed"] * best_speed_raw
                vals["tam_v2_uav_angle_raw"] = best_angle_raw
                vals["tam_v2_uav_angle"] = w["angle"] * max(best_angle_raw, -1.0)
                vals["tam_v2_uav_distance"] = w["distance"] * best_dist_raw
            else:
                for k in ("tam_v2_uav_speed", "tam_v2_uav_angle", "tam_v2_uav_angle_raw", "tam_v2_uav_distance"):
                    vals[k] = 0.0
            vals["tam_v3_uav_shaping_target"] = float(best_target_idx)
            d_total, d_angle, d_speed = self._tam_v2_dodge_reward(sim, v_norm, self._tam_v2_missile_speed_cache)
            vals["tam_v2_uav_dodge"] = w["dodge"] * d_total
            vals["tam_v2_uav_dodge_angle"] = d_angle
            vals["tam_v2_uav_dodge_speed"] = d_speed
        else:
            for k in ("tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
                      "tam_v2_uav_angle_raw", "tam_v2_uav_distance",
                      "tam_v2_uav_dodge", "tam_v2_uav_dodge_angle", "tam_v2_uav_dodge_speed"):
                vals[k] = 0.0
            vals["tam_v3_uav_shaping_target"] = -1.0

        # ── Event (v3: continuous out-of-zone) ──
        ev = cfg["uav"]["event"]
        r_event = 0.0
        kills = int(self._step_kill_count.get(aid, 0))
        r_event += kills * float(ev["kill_enemy"])
        vals["tam_v2_uav_kill"] = kills * float(ev["kill_enemy"])
        if (not sim.is_alive) and aid not in self._uav_death_penalized:
            r_event += float(ev["death"])
            self._uav_death_penalized.add(aid)
            vals["tam_v2_uav_death"] = float(ev["death"])
        else:
            vals["tam_v2_uav_death"] = 0.0
        if sim.is_alive:
            oz_penalty = self._tam_v3_out_of_zone_penalty(sim, aid, cfg)
        else:
            oz_penalty = 0.0
        r_event += oz_penalty
        vals["tam_v2_uav_out_of_zone"] = oz_penalty
        vals["tam_v2_uav_event"] = r_event

        # Log-only
        orig_brma = base_components.get(aid, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v2_uav_fire_log"] = 0.0
        vals["tam_v2_uav_mav_shared_track_log"] = 0.0
        vals["tam_v2_geometry_feature_semantics"] = "absolute"
        vals["tam_v2_dodge_los_semantics"] = "missile_to_aircraft"
        vals["tam_v2_height_formula_source"] = "tam_paper_v3_env_consistent"

        gs = float(cfg["global_scale"])
        dense_event = (
            vals["tam_v2_uav_height"] + vals["tam_v2_uav_speed"] + vals["tam_v2_uav_angle"]
            + vals["tam_v2_uav_distance"] + vals["tam_v2_uav_dodge"] + vals["tam_v2_uav_event"]
        )
        total = dense_event * gs
        vals["tam_v2_total"] = total
        return total, vals

    def _compute_tam_paper_reward_v3(self, base_rewards: dict, components: dict):
        cfg = self.tam_paper_reward_v3_config
        alive_blue = self._tam_v2_alive_blue()
        mav_id = next((
            aid for aid in self.red_ids if self.agent_roles.get(aid) == "mav"
        ), self.red_ids[0] if self.red_ids else None)
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            if rid == mav_id:
                reward, comp = self._tam_v3_mav_reward(rid, sim, alive_blue, cfg, components)
            else:
                reward, comp = self._tam_v3_uav_reward(rid, sim, alive_blue, cfg, components)
            base_rewards[rid] = reward
            components[rid] = comp
        return base_rewards, components

    # ── end TAM paper reward v3 ────────────────────────────────────────

    # ── TAM Paper Reward v4 (BRMA flight status + situation + outcome) ─

    def _tam_v4_terminal_outcome(self, cfg: dict) -> float:
        """Team-level terminal outcome: +200 red win, -200 blue win, 0 draw."""
        n_red = sum(1 for s in self.red_planes.values() if s.is_alive)
        n_blue = sum(1 for s in self.blue_planes.values() if s.is_alive)
        if n_blue == 0 and n_red > 0:
            return float(cfg["uav"]["event"].get("team_win", 200.0))
        if n_red == 0 and n_blue > 0:
            return float(cfg["uav"]["event"].get("team_loss", -200.0))
        return float(cfg["uav"]["event"].get("team_draw", 0.0))

    def _tam_v4_situation_reward(self, sim, cfg: dict) -> tuple[float, float, float]:
        """BRMA-style angle×distance coupling: own_adv − enemy_threat_weight×enemy_threat.

        Returns (raw_total, own_adv_raw, enemy_threat_raw) — raw values, before weight.
        """
        e_w = float(cfg["situation"].get("enemy_threat_weight", 0.8))
        normalize = bool(cfg["situation"].get("normalize_by_alive_blue", True))
        alive_blue = [s for bid in self.blue_ids if (s := self.blue_planes.get(bid)) and s.is_alive]
        if not alive_blue:
            return 0.0, 0.0, 0.0
        n_b = len(alive_blue)
        ego_pos = sim.get_position()
        ego_rpy = sim.get_rpy()
        own_adv = 0.0
        enemy_threat = 0.0
        for b in alive_blue:
            b_pos = b.get_position()
            b_rpy = b.get_rpy()
            q_ij = compute_body_x_q_los(ego_pos, ego_rpy, b_pos)
            q_ji = compute_body_x_q_los(b_pos, b_rpy, ego_pos)
            d_3d = compute_3d_range(ego_pos, b_pos)
            Ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
            Td_ij = td_distance_advantage(d_3d)
            Ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))
            own_adv += Ta_ij * Td_ij
            enemy_threat += Ta_ji * Td_ij
        if normalize:
            own_adv /= n_b
            enemy_threat /= n_b
        return own_adv - e_w * enemy_threat, own_adv, enemy_threat

    def _tam_v4_mav_reward(self, mav_id: str, mav, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        """MAV reward — v3 structure + BRMA pitch/roll + terminal outcome."""
        vals: dict[str, float] = {}
        mav_pos = np.array(mav.get_position(), dtype=np.float64)
        sw = cfg["mav"]["safety_weights"]
        d_danger = float(cfg["mav"]["d_danger_m"])
        d_safe = float(cfg["mav"]["d_safe_m"])
        r_dist = 0.0; r_threat = 0.0; r_aspect = 0.0; r_aware = 0.0; r_pos = 0.0
        if mav.is_alive:
            if alive_blue:
                distances = [float(np.linalg.norm(b.get_position() - mav_pos)) for b in alive_blue]
                near_d = min(distances)
                if near_d <= d_danger:
                    r_dist = -(1.0 - near_d / d_danger)
                elif near_d < d_safe:
                    r_dist = -0.5 * (1.0 - (near_d - d_danger) / (d_safe - d_danger))
                else:
                    r_dist = 0.2
                vals["tam_v4_mav_dist"] = r_dist
                threat_missiles = getattr(mav, "under_missiles", None)
                r_threat = -1.0 if (threat_missiles and any(getattr(m, "is_alive", False) for m in threat_missiles)) else 0.0
                vals["tam_v4_mav_threat"] = r_threat
                mav_feat = HeteroUavCombatEnv._tam_v2_feature(mav)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    if ta < np.pi / 4:
                        r_aspect -= (1.0 - ta / (np.pi / 4))
                vals["tam_v4_mav_aspect"] = r_aspect
                mav_obs_range = getattr(self, "mav_observation_range_m", 80000.0)
                for b in alive_blue:
                    b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                    ao, _ta, _r = get2d_AO_TA_R(mav_feat, b_feat)
                    d = float(np.linalg.norm(b.get_position() - mav_pos))
                    if d < mav_obs_range and ao < np.pi / 2:
                        r_aware += 0.3 * (1.0 - ao / (np.pi / 2))
                vals["tam_v4_mav_aware"] = r_aware
                sup_w = cfg["mav"]["support_weights"]
                d_opt = float(cfg["mav"]["d_opt_m"])
                d_max_mav = float(cfg["mav"]["d_max_m"])
                blue_centroid = np.mean([b.get_position() for b in alive_blue], axis=0)
                d_b = float(np.linalg.norm(blue_centroid - mav_pos))
                if d_b < d_opt:
                    r_pos = d_b / d_opt - 1.0
                elif d_b < d_max_mav:
                    r_pos = 1.0 - (d_b - d_opt) / (d_max_mav - d_opt)
                else:
                    r_pos = -0.5
                vals["tam_v4_mav_pos"] = r_pos
                vals["tam_v4_mav_support"] = sup_w["pos"] * r_pos + sup_w["aware"] * r_aware
            else:
                for k in ("tam_v4_mav_dist", "tam_v4_mav_threat", "tam_v4_mav_aspect", "tam_v4_mav_aware",
                          "tam_v4_mav_pos", "tam_v4_mav_safety", "tam_v4_mav_support"):
                    vals[k] = 0.0
            vals["tam_v4_mav_safety"] = sw["dist"] * r_dist + sw["threat"] * r_threat + sw["aspect"] * r_aspect
            # BRMA flight status
            fs = cfg.get("flight_status", {})
            if "mav" in fs.get("apply_to_roles", []):
                r_pitch_raw = self._pitch_penalty(mav) if hasattr(self, "_pitch_penalty") else 0.0
                r_roll_raw = self._roll_penalty(mav) if hasattr(self, "_roll_penalty") else 0.0
                vals["tam_v4_mav_pitch"] = float(fs.get("pitch_weight", 0.01)) * r_pitch_raw
                vals["tam_v4_mav_roll"] = float(fs.get("roll_weight", 0.002)) * r_roll_raw
                vals["tam_v4_mav_flight_status"] = vals["tam_v4_mav_pitch"] + vals["tam_v4_mav_roll"]
            else:
                for k in ("tam_v4_mav_pitch", "tam_v4_mav_roll", "tam_v4_mav_flight_status"):
                    vals[k] = 0.0
        else:
            for k in ("tam_v4_mav_dist", "tam_v4_mav_threat", "tam_v4_mav_aspect", "tam_v4_mav_aware",
                      "tam_v4_mav_safety", "tam_v4_mav_pos", "tam_v4_mav_support",
                      "tam_v4_mav_pitch", "tam_v4_mav_roll", "tam_v4_mav_flight_status"):
                vals[k] = 0.0

        # Event
        r_event = 0.0
        if (not mav.is_alive) and (not self._mav_death_penalized):
            r_event -= float(cfg["mav"]["death_penalty"])
            self._mav_death_penalized = True
            vals["tam_v4_mav_death"] = -float(cfg["mav"]["death_penalty"])
        else:
            vals["tam_v4_mav_death"] = 0.0
        team_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids if rid != mav_id)
        per_kill = min(float(cfg["mav"]["team_kill_bonus"]), float(cfg["mav"].get("team_kill_bonus_cap", 200.0)))
        vals["tam_v4_mav_team_bonus"] = team_kills * per_kill
        r_event += team_kills * per_kill
        if mav.is_alive:
            vals["tam_v4_mav_out_of_zone"] = self._tam_v3_out_of_zone_penalty(mav, mav_id, cfg)
        else:
            vals["tam_v4_mav_out_of_zone"] = 0.0
        r_event += vals["tam_v4_mav_out_of_zone"]
        # Terminal team outcome
        vals["tam_v4_team_outcome"] = 0.0
        vals["tam_v4_mav_event"] = r_event

        # Log-only
        orig_brma = base_components.get(mav_id, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v4_height_formula_source"] = "tam_paper_v4"

        gs = float(cfg["global_scale"])
        total = (vals["tam_v4_mav_safety"] + vals["tam_v4_mav_support"]
                 + vals.get("tam_v4_mav_flight_status", 0.0)
                 + vals["tam_v4_mav_event"]) * gs
        vals["tam_v4_total"] = total
        return total, vals

    def _tam_v4_uav_reward(self, aid: str, sim, alive_blue: list, cfg: dict,
                            base_components: dict) -> tuple[float, dict]:
        """UAV reward — v3 core + BRMA situation + flight status + terminal outcome."""
        vals: dict[str, float] = {}
        w = cfg["uav"]["reward_weights"]
        v_norm = float(cfg["uav"].get("v_norm_mps", 1000.0))
        sim_sp = float(np.linalg.norm(sim.get_velocity()))
        alt = float(sim.get_geodetic()[2])

        if sim.is_alive:
            vals["tam_v4_uav_height"] = w["height"] * self._tam_v3_height_reward(alt, cfg)
            vals["tam_v4_uav_speed"] = w["speed"] * max(
                self._tam_v3_speed_reward(sim_sp, float(np.linalg.norm(b.get_velocity())))
                for b in alive_blue) if alive_blue else 0.0
            # BRMA situation reward
            sit_raw, own_adv_raw, enemy_threat_raw = self._tam_v4_situation_reward(sim, cfg)
            vals["tam_v4_uav_situation_raw"] = sit_raw
            vals["tam_v4_uav_situation"] = w["situation"] * sit_raw
            vals["tam_v4_uav_own_adv_log"] = own_adv_raw
            vals["tam_v4_uav_enemy_threat_log"] = -enemy_threat_raw
            # Dodge
            d_total, d_angle, d_speed = self._tam_v2_dodge_reward(sim, v_norm, self._tam_v2_missile_speed_cache)
            vals["tam_v4_uav_dodge"] = w["dodge"] * d_total
            vals["tam_v4_uav_dodge_angle"] = d_angle
            vals["tam_v4_uav_dodge_speed"] = d_speed
            # BRMA flight status
            fs = cfg.get("flight_status", {})
            if "attack_uav" in fs.get("apply_to_roles", []):
                r_pitch_raw = self._pitch_penalty(sim) if hasattr(self, "_pitch_penalty") else 0.0
                r_roll_raw = self._roll_penalty(sim) if hasattr(self, "_roll_penalty") else 0.0
                vals["tam_v4_uav_pitch"] = float(fs.get("pitch_weight", 0.01)) * r_pitch_raw
                vals["tam_v4_uav_roll"] = float(fs.get("roll_weight", 0.002)) * r_roll_raw
                vals["tam_v4_uav_flight_status"] = vals["tam_v4_uav_pitch"] + vals["tam_v4_uav_roll"]
            else:
                for k in ("tam_v4_uav_pitch", "tam_v4_uav_roll", "tam_v4_uav_flight_status"):
                    vals[k] = 0.0
            # Diagnostic angle/distance (not active)
            best_angle_raw = -1.0
            best_dist_raw = -1.0
            red_feat = HeteroUavCombatEnv._tam_v2_feature(sim)
            for b in alive_blue:
                b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                ao, ta, _r = get2d_AO_TA_R(red_feat, b_feat)
                aa = np.pi - ta
                av = 1.0 - (ao + aa) / np.pi
                dv = self._tam_v3_uav_distance_reward(float(np.linalg.norm(b.get_position() - sim.get_position())))
                if av > best_angle_raw: best_angle_raw = av
                if dv > best_dist_raw: best_dist_raw = dv
            vals["tam_v2_uav_angle_diag"] = w.get("angle", 15.0) * max(best_angle_raw, -1.0) if alive_blue else 0.0
            vals["tam_v2_uav_distance_diag"] = w.get("distance", 10.0) * best_dist_raw if alive_blue else 0.0
        else:
            for k in ("tam_v4_uav_height", "tam_v4_uav_speed", "tam_v4_uav_situation_raw",
                      "tam_v4_uav_situation", "tam_v4_uav_dodge",
                      "tam_v4_uav_dodge_angle", "tam_v4_uav_dodge_speed",
                      "tam_v4_uav_pitch", "tam_v4_uav_roll", "tam_v4_uav_flight_status",
                      "tam_v2_uav_angle_diag", "tam_v2_uav_distance_diag"):
                vals[k] = 0.0

        # Event
        ev = cfg["uav"]["event"]
        r_event = 0.0
        kills = int(self._step_kill_count.get(aid, 0))
        r_event += kills * float(ev["kill_enemy"])
        vals["tam_v4_uav_kill"] = kills * float(ev["kill_enemy"])
        if (not sim.is_alive) and aid not in self._uav_death_penalized:
            r_event += float(ev["death"])
            self._uav_death_penalized.add(aid)
            vals["tam_v4_uav_death"] = float(ev["death"])
        else:
            vals["tam_v4_uav_death"] = 0.0
        if sim.is_alive:
            vals["tam_v4_uav_out_of_zone"] = self._tam_v3_out_of_zone_penalty(sim, aid, cfg)
        else:
            vals["tam_v4_uav_out_of_zone"] = 0.0
        r_event += vals["tam_v4_uav_out_of_zone"]
        vals["tam_v4_uav_event"] = r_event
        vals["tam_v4_team_outcome"] = 0.0

        # Log-only
        orig_brma = base_components.get(aid, {})
        vals["brma_r_adv_log"] = orig_brma.get("r_adv", 0.0)
        vals["brma_r_pitch_log"] = orig_brma.get("r_pitch", 0.0)
        vals["brma_r_roll_log"] = orig_brma.get("r_roll", 0.0)
        vals["brma_r_alt_log"] = orig_brma.get("r_alt", 0.0)
        vals["brma_r_bound_log"] = orig_brma.get("r_bound", 0.0)
        vals["brma_r_vel_log"] = orig_brma.get("r_vel", 0.0)
        vals["tam_v4_height_formula_source"] = "tam_paper_v4"

        gs = float(cfg["global_scale"])
        dense_event = (
            vals["tam_v4_uav_height"] + vals["tam_v4_uav_speed"]
            + vals["tam_v4_uav_situation"]
            + vals.get("tam_v4_uav_flight_status", 0.0)
            + vals["tam_v4_uav_dodge"] + vals["tam_v4_uav_event"]
        )
        total = dense_event * gs
        vals["tam_v4_total"] = total
        return total, vals

    def _compute_tam_paper_reward_v4(self, base_rewards: dict, components: dict):
        cfg = self.tam_paper_reward_v4_config
        alive_blue = self._tam_v2_alive_blue()
        mav_id = next((
            aid for aid in self.red_ids if self.agent_roles.get(aid) == "mav"
        ), self.red_ids[0] if self.red_ids else None)
        # Terminal outcome — once per episode end, shared across red agents
        team_outcome = 0.0
        n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
        n_red_alive = sum(1 for s in self.red_planes.values() if s.is_alive)
        round_over = (n_blue_alive == 0 or n_red_alive == 0
                      or self.current_step >= self.max_steps)
        if round_over and not self._tam_v4_terminal_applied:
            team_outcome = self._tam_v4_terminal_outcome(cfg)
            self._tam_v4_terminal_applied = True
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            if rid == mav_id:
                reward, comp = self._tam_v4_mav_reward(rid, sim, alive_blue, cfg, components)
            else:
                reward, comp = self._tam_v4_uav_reward(rid, sim, alive_blue, cfg, components)
            # Inject terminal team outcome into event and total
            comp["tam_v4_team_outcome"] = team_outcome
            comp["tam_v4_mav_event" if rid == mav_id else "tam_v4_uav_event"] += team_outcome
            total_key = "tam_v4_total"
            gs = float(cfg["global_scale"])
            comp[total_key] += team_outcome * gs
            base_rewards[rid] = comp[total_key]
            components[rid] = comp
        return base_rewards, components

    # ── end TAM paper reward v4 ────────────────────────────────────────

    # ── TAM-BRMA Scripted Reward v1 ────────────────────────────────────

    @staticmethod
    def _tam_brma_v1_d_gate(d: float, cfg: dict) -> float:
        g = cfg["gate"]
        r_min = float(g.get("min_range_m", 500.0))
        r_opt = float(g.get("opt_range_m", 5000.0))
        r_launch = float(g.get("launch_range_m", 10000.0))
        if d < r_min:
            return -1.0
        if d < r_opt:
            return (d - r_min) / (r_opt - r_min)
        if d <= r_launch:
            return 1.0
        return np.exp(1.0 - d / r_launch)

    @staticmethod
    def _tam_brma_v1_a_own(ao_rad: float, cfg: dict) -> float:
        thresh = np.deg2rad(float(cfg["gate"].get("ao_thresh_deg", 45.0)))
        return float(np.clip(1.0 - ao_rad / thresh, 0.0, 1.0))

    @staticmethod
    def _tam_brma_v1_t_rear(ta_rad: float, cfg: dict) -> float:
        thresh = np.deg2rad(float(cfg["gate"].get("ta_thresh_deg", 90.0)))
        if ta_rad <= thresh:
            return 0.0
        return float(np.clip((ta_rad - thresh) / (np.pi - thresh), 0.0, 1.0))

    def _tam_brma_v1_terminal_outcome(self, cfg: dict) -> float:
        n_red = sum(1 for s in self.red_planes.values() if s.is_alive)
        n_blue = sum(1 for s in self.blue_planes.values() if s.is_alive)
        mav_alive = bool(self.red_planes.get("red_0") and self.red_planes["red_0"].is_alive)
        initial_blue = self.max_num_blue
        initial_uav = max(self.max_num_red - 1, 0)
        n_uav_alive = sum(1 for rid in self.red_ids
                          if rid != "red_0" and self.red_planes.get(rid) and self.red_planes[rid].is_alive)
        t = cfg.get("terminal", {})
        if n_blue == 0 and mav_alive:
            return float(t.get("full_win", 300.0))
        if n_blue == 0 and not mav_alive:
            return float(t.get("costly_win", 50.0))
        if n_red == 0:
            return float(t.get("loss", -300.0))
        if self.current_step >= self.max_steps:
            return (float(t.get("timeout_blue_kill_scale", 150.0))
                    * (initial_blue - n_blue) / max(initial_blue, 1)
                    + float(t.get("timeout_mav_dead_penalty", -100.0)) * (0.0 if mav_alive else 1.0)
                    + float(t.get("timeout_uav_loss_scale", -60.0))
                    * (initial_uav - n_uav_alive) / max(initial_uav, 1))
        return 0.0

    def _compute_tam_brma_scripted_reward_v1(self, base_rewards: dict, components: dict):
        cfg = self.tam_brma_scripted_reward_v1_config
        flight_scale = float(cfg.get("flight_scale", 5.0))
        alive_blue = [s for bid in self.blue_ids if (s := self.blue_planes.get(bid)) and s.is_alive]
        n_b = max(len(alive_blue), 1)

        # ── Terminal outcome (one-shot per episode end) ──
        n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
        n_red_alive = sum(1 for s in self.red_planes.values() if s.is_alive)
        round_over = (n_blue_alive == 0 or n_red_alive == 0
                      or self.current_step >= self.max_steps)
        team_outcome = 0.0
        if round_over and not self._tam_brma_scripted_terminal_applied:
            team_outcome = self._tam_brma_v1_terminal_outcome(cfg)
            self._tam_brma_scripted_terminal_applied = True

        # ── Team-wide event accumulators ──
        total_red_kills = sum(int(self._step_kill_count.get(rid, 0)) for rid in self.red_ids)

        # ── Team-wide shared events ──
        # Count UAV first-deaths this step → team_uav_loss_shared
        num_uav_first_deaths = 0
        mav_loss_to_uav = 0.0
        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            role = self.agent_roles.get(rid, "")
            if role == "mav":
                if not sim.is_alive and not self._tam_brma_scripted_mav_death_penalized:
                    mav_loss_to_uav = float(cfg["uav"]["event"].get("mav_loss_to_uav", -160.0))
            else:
                if not sim.is_alive and rid not in self._tam_brma_scripted_uav_death_penalized:
                    num_uav_first_deaths += 1
        team_uav_loss = num_uav_first_deaths * float(cfg["uav"]["event"].get("team_uav_loss_shared", -30.0))
        team_kill_shared_per_agent = float(cfg["uav"]["event"].get("team_kill_shared", 30.0)) * total_red_kills

        for rid in self.red_ids:
            sim = self.red_planes.get(rid)
            if sim is None:
                continue
            role = self.agent_roles.get(rid, "")
            orig_comp = components.get(rid, {})
            vals: dict[str, float] = {}

            # ── BRMA flight base (from base reward components) ──
            f_pitch = float(orig_comp.get("r_pitch", 0.0))
            f_roll = float(orig_comp.get("r_roll", 0.0))
            f_alt = float(orig_comp.get("r_alt", 0.0))
            f_bound = float(orig_comp.get("r_bound", 0.0))
            f_vel = float(orig_comp.get("r_vel", 0.0))
            f_brma = f_pitch + f_roll + f_alt + f_bound + f_vel
            vals["tam_brma_v1_flight"] = flight_scale * f_brma

            if role == "mav":
                self._tam_brma_v1_mav_reward(vals, rid, sim, alive_blue, n_b, cfg)
            else:
                self._tam_brma_v1_uav_reward(vals, rid, sim, alive_blue, cfg)

            # ── Event ──
            self._tam_brma_v1_events(vals, rid, sim, role, total_red_kills, cfg)

            # Inject team shared events (applies to ALL red agents)
            if role != "mav":
                vals["tam_brma_v1_uav_event"] = vals.get("tam_brma_v1_uav_event", 0.0) + team_kill_shared_per_agent
                if team_uav_loss != 0.0:
                    vals["tam_brma_v1_uav_event"] = vals.get("tam_brma_v1_uav_event", 0.0) + team_uav_loss
                if mav_loss_to_uav != 0.0:
                    vals["tam_brma_v1_uav_event"] = vals.get("tam_brma_v1_uav_event", 0.0) + mav_loss_to_uav
            else:
                vals["tam_brma_v1_mav_event"] = vals.get("tam_brma_v1_mav_event", 0.0) + team_kill_shared_per_agent
                vals["tam_brma_v1_mav_team_kill_shared"] = team_kill_shared_per_agent
                if team_uav_loss != 0.0:
                    vals["tam_brma_v1_mav_event"] = vals.get("tam_brma_v1_mav_event", 0.0) + team_uav_loss

            # ── Terminal ──
            vals["tam_brma_v1_team_terminal"] = team_outcome

            # ── Total (only active components, NOT diagnostics) ──
            if role == "mav":
                total = (vals.get("tam_brma_v1_flight", 0.0)
                         + vals.get("tam_brma_v1_mav_safe", 0.0)
                         + vals.get("tam_brma_v1_mav_support", 0.0)
                         + vals.get("tam_brma_v1_mav_aware", 0.0)
                         + vals.get("tam_brma_v1_mav_event", 0.0)
                         + vals.get("tam_brma_v1_team_terminal", 0.0))
            else:
                total = (vals.get("tam_brma_v1_flight", 0.0)
                         + vals.get("tam_brma_v1_uav_gate_sit", 0.0)
                         + vals.get("tam_brma_v1_uav_event", 0.0)
                         + vals.get("tam_brma_v1_team_terminal", 0.0))
            vals["tam_brma_v1_total"] = total
            base_rewards[rid] = total
            components[rid] = vals

        return base_rewards, components

    def _tam_brma_v1_uav_reward(self, vals: dict, aid: str, sim, alive_blue: list, cfg: dict):
        if not sim.is_alive:
            vals["tam_brma_v1_uav_gate_sit"] = 0.0
            return
        uav_cfg = cfg["uav"]
        gate_cfg = cfg["gate"]
        e_w = float(gate_cfg.get("enemy_threat_weight", 0.8))
        g_weight = float(uav_cfg.get("gate_sit_weight", 0.8))
        red_feat = HeteroUavCombatEnv._tam_v2_feature(sim)
        sim_pos = np.array(sim.get_position(), dtype=np.float64)

        # Score each blue target
        best_score = -1e9
        best_idx = -1
        best_vals = {}
        for idx, b in enumerate(alive_blue):
            b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
            ao, ta, dist_m = get2d_AO_TA_R(red_feat, b_feat)
            if dist_m is None or np.isnan(dist_m):
                dist_m = float(np.linalg.norm(b.get_position() - sim_pos))
            a_own = self._tam_brma_v1_a_own(ao, cfg)
            t_rear = self._tam_brma_v1_t_rear(ta, cfg)
            d_gate = self._tam_brma_v1_d_gate(dist_m, cfg)
            score = 0.40 * a_own + 0.35 * t_rear + 0.25 * max(d_gate, 0.0)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_vals = {"a_own": a_own, "t_rear": t_rear, "d_gate": d_gate,
                             "ao": ao, "ta": ta, "dist_m": dist_m}

        if not best_vals:
            vals["tam_brma_v1_uav_gate_sit"] = 0.0
            vals["tam_brma_v1_uav_target_idx"] = -1.0
            return

        # Compute G_own and G_enemy using best target
        a_own_b = best_vals["a_own"]
        t_rear_b = best_vals["t_rear"]
        d_gate_b = best_vals["d_gate"]
        g_own = a_own_b * t_rear_b * d_gate_b

        # Enemy threat: for the best blue target, compute its AO/TA toward red
        b_target = alive_blue[best_idx]
        b_feat = HeteroUavCombatEnv._tam_v2_feature(b_target)
        ao_enemy, ta_enemy, dist_enemy = get2d_AO_TA_R(b_feat, red_feat)
        if dist_enemy is None or np.isnan(dist_enemy):
            dist_enemy = float(np.linalg.norm(b_target.get_position() - sim_pos))
        a_enemy = self._tam_brma_v1_a_own(ao_enemy, cfg)
        t_enemy_rear = self._tam_brma_v1_t_rear(ta_enemy, cfg)
        d_gate_enemy = self._tam_brma_v1_d_gate(dist_enemy, cfg)
        g_enemy = a_enemy * t_enemy_rear * max(d_gate_enemy, 0.0)

        r_gate_sit = float(np.clip(g_own - e_w * g_enemy, -1.0, 1.0))
        vals["tam_brma_v1_uav_gate_sit"] = g_weight * r_gate_sit
        vals["tam_brma_v1_uav_g_own"] = g_own
        vals["tam_brma_v1_uav_g_enemy"] = g_enemy
        vals["tam_brma_v1_uav_a_own"] = a_own_b
        vals["tam_brma_v1_uav_t_rear"] = t_rear_b
        vals["tam_brma_v1_uav_d_gate"] = d_gate_b
        vals["tam_brma_v1_uav_target_idx"] = float(best_idx)

    def _tam_brma_v1_mav_reward(self, vals: dict, rid: str, mav, alive_blue: list,
                                  n_b: int, cfg: dict):
        if not mav.is_alive:
            for k in ("tam_brma_v1_mav_safe", "tam_brma_v1_mav_dist",
                       "tam_brma_v1_mav_missile_threat", "tam_brma_v1_mav_aspect",
                       "tam_brma_v1_mav_support", "tam_brma_v1_mav_link",
                       "tam_brma_v1_mav_rear", "tam_brma_v1_mav_aware",
                       "tam_brma_v1_mav_event"):
                vals[k] = 0.0
            return
        mav_cfg = cfg["mav"]
        mav_pos = np.array(mav.get_position(), dtype=np.float64)
        red_uavs = [self.red_planes.get(aid) for aid in self.red_ids
                    if aid != rid and self.red_planes.get(aid) and self.red_planes[aid].is_alive]
        gate_cfg = cfg["gate"]

        # ── Safety ──
        d_danger = float(mav_cfg.get("d_danger_m", 5000.0))
        d_safe = float(mav_cfg.get("d_safe_m", 14000.0))
        d_far = float(mav_cfg.get("d_far_m", 35000.0))
        r_dist = 0.0
        if alive_blue:
            near_d = min(float(np.linalg.norm(b.get_position() - mav_pos)) for b in alive_blue)
            if near_d < d_danger:
                r_dist = -1.0
            elif near_d < d_safe:
                r_dist = -0.5 * (1.0 - (near_d - d_danger) / (d_safe - d_danger))
            elif near_d < d_far:
                r_dist = 0.2
        vals["tam_brma_v1_mav_dist"] = r_dist

        threat_msl = getattr(mav, "under_missiles", None)
        r_missile = -1.0 if (threat_msl and any(getattr(m, "is_alive", False) for m in threat_msl)) else 0.0
        vals["tam_brma_v1_mav_missile_threat"] = r_missile

        # Aspect: -max_j G_enemy(j, MAV)
        r_aspect = 0.0
        if alive_blue:
            mav_feat = HeteroUavCombatEnv._tam_v2_feature(mav)
            for b in alive_blue:
                b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                ao_e, ta_e, dist_e = get2d_AO_TA_R(b_feat, mav_feat)
                if dist_e is None or np.isnan(dist_e):
                    dist_e = float(np.linalg.norm(b.get_position() - mav_pos))
                a_e = self._tam_brma_v1_a_own(ao_e, cfg)
                t_e = self._tam_brma_v1_t_rear(ta_e, cfg)
                d_e = self._tam_brma_v1_d_gate(dist_e, cfg)
                g_e = a_e * t_e * max(d_e, 0.0)
                if g_e > r_aspect:
                    r_aspect = g_e
            r_aspect = -r_aspect
        vals["tam_brma_v1_mav_aspect"] = r_aspect

        vals["tam_brma_v1_mav_safe"] = float(mav_cfg.get("safety_weight", 0.7)) * (
            0.5 * r_dist + 0.3 * r_missile + 0.2 * r_aspect)

        # ── Support ──
        r_link = 0.0
        r_rear = 0.0
        if red_uavs and alive_blue:
            c_u = np.mean([np.array(u.get_position(), dtype=np.float64) for u in red_uavs], axis=0)
            c_b = np.mean([np.array(b.get_position(), dtype=np.float64) for b in alive_blue], axis=0)
            link_d = float(mav_cfg.get("link_distance_m", 12000.0))
            d_mav_cu = float(np.linalg.norm(mav_pos - c_u))
            r_link = np.exp(-abs(d_mav_cu - link_d) / link_d)
            e_bu = (c_u - c_b) / max(np.linalg.norm(c_u - c_b), 1e-8)
            rear_d = float(mav_cfg.get("rear_distance_m", 15000.0))
            r_rear = float(np.clip(np.dot(mav_pos - c_u, e_bu) / rear_d, -1.0, 1.0))
        vals["tam_brma_v1_mav_link"] = r_link
        vals["tam_brma_v1_mav_rear"] = r_rear
        vals["tam_brma_v1_mav_support"] = float(mav_cfg.get("support_weight", 0.4)) * (
            0.6 * r_link + 0.4 * r_rear)

        # ── Aware ──
        r_aware = 0.0
        if alive_blue:
            mav_feat = HeteroUavCombatEnv._tam_v2_feature(mav)
            obs_range = getattr(self, "mav_observation_range_m", 80000.0)
            for b in alive_blue:
                b_feat = HeteroUavCombatEnv._tam_v2_feature(b)
                ao_m, _ta_m, _r = get2d_AO_TA_R(mav_feat, b_feat)
                d_m = float(np.linalg.norm(b.get_position() - mav_pos))
                if d_m < obs_range and ao_m < np.pi / 2:
                    r_aware += (1.0 - ao_m / (np.pi / 2))
            r_aware /= n_b
        vals["tam_brma_v1_mav_aware"] = float(mav_cfg.get("aware_weight", 0.3)) * r_aware

    def _tam_brma_v1_read_death_reason(self, aid: str) -> str:
        """Read death reason from env state — checked before info is populated."""
        dr = getattr(self, "_death_reasons", {})
        reason = dr.get(aid, "")
        if reason:
            return reason
        events = getattr(self, "_death_events_step", [])
        for ev in (events or []):
            if isinstance(ev, dict) and ev.get("agent_id") == aid:
                return str(ev.get("death_reason", ""))
        return ""

    def _tam_brma_v1_events(self, vals: dict, aid: str, sim, role: str,
                             total_red_kills: int, cfg: dict,
                             mav_id: str = None):
        uav_ev = cfg["uav"]["event"]
        mav_ev = cfg["mav"]["event"]
        r_event = 0.0
        if role == "mav":
            team_kill_credit = float(mav_ev.get("team_kill_credit", 40.0))
            r_event += team_kill_credit * total_red_kills
            if not sim.is_alive and not self._tam_brma_scripted_mav_death_penalized:
                reason = self._tam_brma_v1_read_death_reason(aid)
                is_noncombat = any(kw in reason.lower()
                                   for kw in ("crash", "lowalt", "overg", "extreme", "boundary", "out_of_zone"))
                penalty = float(mav_ev.get("noncombat_loss", -300.0) if is_noncombat
                                else mav_ev.get("death", -300.0))
                r_event += penalty
                self._tam_brma_scripted_mav_death_penalized = True
                vals["tam_brma_v1_mav_death"] = penalty
            else:
                vals["tam_brma_v1_mav_death"] = 0.0
        else:
            kills = int(self._step_kill_count.get(aid, 0))
            r_event += kills * float(uav_ev.get("kill_enemy", 160.0))
            vals["tam_brma_v1_uav_kill"] = kills * float(uav_ev.get("kill_enemy", 160.0))
            if not sim.is_alive and aid not in self._tam_brma_scripted_uav_death_penalized:
                self._tam_brma_scripted_uav_death_penalized.add(aid)
                reason = self._tam_brma_v1_read_death_reason(aid)
                is_noncombat = any(kw in reason.lower()
                                   for kw in ("crash", "lowalt", "overg", "extreme", "boundary", "out_of_zone"))
                penalty = float(uav_ev.get("noncombat_loss", -180.0) if is_noncombat
                                else uav_ev.get("death", -160.0))
                r_event += penalty
                vals["tam_brma_v1_uav_death"] = penalty
            else:
                vals["tam_brma_v1_uav_death"] = 0.0
            # team_kill_shared is added at dispatch level (applies to ALL red agents)
        vals["tam_brma_v1_uav_event" if role != "mav" else "tam_brma_v1_mav_event"] = r_event

    # ── end TAM-BRMA Scripted Reward v1 ────────────────────────────────

    def _compute_rewards(self) -> tuple[dict, dict]:
        """Override to add minimal hetero role-aware overlay."""
        base_rewards, components = super()._compute_rewards()

        if self.hetero_reward_mode not in {"minimal_v1", "role_v1", "happo_ref_v0", "happo_ref_v1_mav_support", "paper_role_reward_v1"}:
            if self.hetero_reward_mode == "tam_paper_reward_v2":
                return self._compute_tam_paper_reward_v2(base_rewards, components)
            if self.hetero_reward_mode == "tam_paper_reward_v3":
                return self._compute_tam_paper_reward_v3(base_rewards, components)
            if self.hetero_reward_mode == "tam_paper_reward_v4":
                return self._compute_tam_paper_reward_v4(base_rewards, components)
            if self.hetero_reward_mode == "tam_paper_reward_v6_jsbsim_aligned_v3":
                return self._compute_tam_paper_reward_v6_jsbsim_aligned_v3(base_rewards, components)
            if self.hetero_reward_mode == "tam_paper_reward_v7_role_aligned":
                return self._compute_tam_paper_reward_v7_role_aligned(base_rewards, components)
            if self.hetero_reward_mode == "tam_brma_scripted_reward_v1":
                return self._compute_tam_brma_scripted_reward_v1(base_rewards, components)
            if self.hetero_reward_mode == "brma_paper_homogeneous_v1":
                return self._compute_brma_paper_homogeneous_v1(base_rewards, components)
            if self.hetero_reward_mode == "brma_role_no_missile_reward_v8":
                return self._compute_brma_role_no_missile_reward_v8(base_rewards, components)
            if self.hetero_reward_mode == "tam_brma_paper_aligned_v1":
                return self._compute_tam_brma_paper_aligned_v1(base_rewards, components)
            if self.hetero_reward_mode == "tam_happo_table1_v1":
                return self._compute_tam_happo_table1_v1(base_rewards, components)
            if self.hetero_reward_mode == "brma_tam_scripted_composite_v1":
                return self._compute_brma_tam_scripted_composite_v1(base_rewards, components)
            if self.hetero_reward_mode == "brma_tam_scale_aligned_v1":
                return self._compute_brma_tam_scale_aligned_v1(base_rewards, components)
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

        if self.hetero_reward_mode == "happo_ref_v1_mav_support":
            return self._compute_happo_ref_v1_mav_support(base_rewards, components)

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
        if self.hetero_reward_mode == "brma_tam_scripted_composite_v1":
            info["__reward_target_diagnostics__"] = [
                dict(row) for row in getattr(self, "_reward_target_diagnostic_records", []) or []
            ]
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
