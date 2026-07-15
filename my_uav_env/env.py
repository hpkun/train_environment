"""
UavCombatEnv: Multi-agent UAV combat environment with Dict observation spaces
for zero-shot scale generalization. Uses JSBSim for flight dynamics and PID
controllers to convert high-level tactical commands to control-surface inputs.
"""
from __future__ import annotations

import copy
import logging
import numpy as np
import gymnasium

from configs.brma_mappo_paper_spec import (
    DEFAULT_PAPER_ENVIRONMENT_CONFIG,
    PaperEnvironmentConfig,
    environment_config_snapshot,
    paper_value,
)
from configs.paper_minimal_3v3_spec import (
    MINIMAL_EXTREME_LOAD_INVALID_THRESHOLD_G,
    MINIMAL_PAPER_ENVIRONMENT_CONFIG,
    PAPER_MINIMAL_ENVIRONMENT_PROFILE,
    REFERENCE_ENVIRONMENT_PROFILE,
    minimal_environment_snapshot,
)
from my_uav_env.sensors import SensorTrack, radar_diagnostic
from my_uav_env.fire_control import FireControlState
from my_uav_env.blue_policy_profiles import (
    BluePolicyController,
    validate_blue_policy_profile,
)

from my_uav_env.alignment.los_geometry import (
    compute_3d_range,
    compute_body_x_q_los,
    compute_velocity_q_los,
)
from my_uav_env.alignment.launch_quality import (
    LAUNCH_QUALITY_FIELDS,
    make_launch_quality_record,
    nan_float as _nan_float,
)
from my_uav_env.alignment.reward_utils import (
    AltitudeRewardConfig,
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
    altitude_reward_pairwise_sum_eq17,
    altitude_reward_pairwise_mean_eq17,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)
from my_uav_env.alignment.state_extractor import (
    extract_relative_state,
    extract_self_state_with_meta,
    ordered_entity_slots,
    slot_aligned_alive_mask,
)

from .simulator import AircraftSimulator, MissileSimulator
from .pid_controller import PIDController
from .utils import get2d_AO_TA_R, get2d_heading_AO_TA_R
from .render_tacview import TacviewLogger

logger = logging.getLogger(__name__)


def _minimal_missile_rng(seed: int | None, parent_uid: str,
                         sequence: int) -> np.random.Generator:
    """Independent deterministic stream for one minimal-profile launch."""
    pair_index = int(parent_uid.split("_", 1)[1])
    team_id = 0 if parent_uid.startswith("red_") else 1
    return np.random.default_rng(np.random.SeedSequence([
        int(seed or 0), team_id, pair_index, int(sequence)]))

LAUNCH_DIAG_TEAMS = ("red", "blue")
LAUNCH_DIAG_KEYS = (
    "scan_frames",
    "alive_shooters",
    "alive_enemy_pairs",
    "unengaged_enemy_pairs",
    "range_ok_pairs",
    "ao_ok_pairs",
    "ta_ok_pairs",
    "geometry_ok_pairs",
    "lock_started",
    "lock_continued",
    "lock_lost",
    "lock_mature_pairs",
    "cooldown_blocked",
    "kill_cooldown_blocked",
    "engaged_blocked",
    "launches",
)
from my_uav_env.alignment.state_extractor import body_angles_from_neu_vector


def make_empty_launch_diag() -> dict:
    """Return a fresh per-step missile launch diagnostics counter."""

    return {team: {key: 0 for key in LAUNCH_DIAG_KEYS}
            for team in LAUNCH_DIAG_TEAMS}


def _make_entity_vec(ego_pos, ego_vel, tgt_pos, tgt_vel, tgt_rpy, alive: bool):
    """Build an 11-dim entity feature vector for *tgt* as seen from *ego*.

    Coordinates should be in ego's BODY frame (paper Table 2):
      x=forward, y=right, z=down.

    [Δx, Δy, Δz, AO_signed, TA, R, V_tgt,
     sin(roll_tgt), cos(roll_tgt), sin(pitch_tgt), cos(pitch_tgt)]

    AO_signed ∈ [−π, π]: body-frame signed Angle-Off — cross(ego_vel, LOS)
    in body x-y plane tells whether the target is to the left (−) or right (+).
    TA ∈ [0, π]: unsigned Target Aspect.
    V_tgt = ||tgt_vel|| — target speed magnitude (m/s).
    Returns zeros if the target is dead.
    """
    if not alive:
        return np.zeros(11, dtype=np.float32)

    dn = tgt_pos[0] - ego_pos[0]
    de = tgt_pos[1] - ego_pos[1]
    du = tgt_pos[2] - ego_pos[2]

    # Build feature arrays for 2D AO/TA computation (north, east, down, vn, ve, vd)
    ego_feat = np.array([ego_pos[0], ego_pos[1], -ego_pos[2],
                         ego_vel[0], ego_vel[1], -ego_vel[2]], dtype=np.float64)
    enm_feat = np.array([tgt_pos[0], tgt_pos[1], -tgt_pos[2],
                         tgt_vel[0], tgt_vel[1], -tgt_vel[2]], dtype=np.float64)
    AO_unsigned, TA, R, side_flag = get2d_AO_TA_R(ego_feat, enm_feat,
                                                   return_side=True)
    AO_signed = _signed_ao_from_unsigned_and_side(AO_unsigned, side_flag)

    V_tgt = float(np.linalg.norm(tgt_vel))

    return np.array([
        dn, de, du, AO_signed, TA, R, V_tgt,
        np.sin(tgt_rpy[0]), np.cos(tgt_rpy[0]),
        np.sin(tgt_rpy[1]), np.cos(tgt_rpy[1]),
    ], dtype=np.float32)


def _signed_ao_from_unsigned_and_side(ao_unsigned: float, side_flag: float) -> float:
    """Return signed AO while preserving front/back collinear cases.

    ``get2d_AO_TA_R(return_side=True)`` returns ``side_flag = sign(cross(v_ego_xy, los_xy))``.
    When the velocity and LOS are exactly collinear (target directly ahead or behind),
    the cross product is zero and ``side_flag == 0``.  Multiplying by zero collapses
    the unsigned AO to 0 for *both* cases, making behind indistinguishable from ahead
    in the 11-dim entity observation vector.

    This helper preserves the full unsigned AO when side_flag == 0:

    - side_flag > 0: target on right → +AO_unsigned
    - side_flag < 0: target on left  → −AO_unsigned
    - side_flag == 0: collinear → +AO_unsigned (≈ 0 ahead, ≈ π behind)
    """
    if side_flag > 0:
        return float(ao_unsigned)
    if side_flag < 0:
        return float(-ao_unsigned)
    return float(ao_unsigned)


class UavCombatEnv(gymnasium.Env):
    """
    Multi-agent UAV combat environment (paper BRMA-MAPPO baseline).

    Action space (per agent): Box(3,) → paper §2.4 ABSOLUTE targets
      - target_pitch:    ±90° (act[0] → θ ∈ (−π/2, π/2])
      - target_heading:  ±180° absolute (act[1] → ψ ∈ (−π, π])
      - target_velocity: 0.3–1.2 Mach ≈ 102–408 m/s (act[2] → V)

    Observation space (per agent): Dict with keys. obs_mode="paper_strict"
    uses 10-dim paper Table 1/Table 2 ego/ally/enemy entities; obs_mode=
    "engineering" uses the legacy normalized 11-dim entity layout.
      - "ego_state"     (entity_dim,)       self state
      - "ally_states"   (max_allies-1, entity_dim)  allied aircraft, excluding self
      - "enemy_states"  (max_enemies, entity_dim)    enemy aircraft
      - "alive_mask"    (max_allies+max_enemies,)  1=valid/alive, 0=invalid/dead;
        slots are ordered ego, allies excluding ego, then enemies
      - "death_mask"    deprecated alias with the same values as alive_mask
    """

    # ---- Action scale constants -------------------------------------------------
    # Paper §2.4: action space uses ABSOLUTE target values (not deltas).
    #
    #   θ ∈ (−π/2, π/2]       pitch   act[0] ∈ [-1, 1] → ±90°
    #   ψ ∈ (−π, π]           heading act[1] ∈ [-1, 1] → ±180° (absolute)
    #   V ∈ [0.3, 1.2] Mach   velocity act[2] ∈ [-1, 1] → [102, 408] m/s
    #
    # Both teams share identical action authority per paper specification.
    # Blue-only GCAS is retained only as an explicit engineering debug option.
    #
    # Velocity:  F-16 F100-PW-229 MilThrust ≈ 17 800 lbf; jet can sustain M0.8–1.0
    #            in level flight at 10 kft.  Mach reference: a ≈ 340 m/s at sea level,
    #            ≈ 328 m/s at 10 kft ISA.
    PITCH_DEG = 90.0             # paper §2.4: full longitudinal authority (±90°)
    VELOCITY_MIN = paper_value("action_speed_mach")[0] * paper_value("mach_reference_mps")
    VELOCITY_MAX = paper_value("action_speed_mach")[1] * paper_value("mach_reference_mps")

    MISSILE_COOLDOWN_STEPS = 30        # default 0.5 s at 60 Hz; __init__ scales with sim_freq
    MISSILE_LOCK_DELAY_FRAMES = 15     # default 0.25 s at 60 Hz; __init__ scales with sim_freq
    KILL_COOLDOWN_STEPS = 3            # legacy engineering guard; disabled by default
    MISSILE_LAUNCH_AO_THRESH = np.deg2rad(45)
    MISSILE_LAUNCH_RANGE_THRESH = paper_value("electro_optical_range_m")
    MISSILE_LAUNCH_MIN_RANGE = paper_value("minimum_launch_range_m")
    MISSILE_LAUNCH_TA_THRESH = np.pi / 2   # 90° — must be in enemy rear hemisphere (3-9 line)

    # ---- Airborne radar (paper: ±60° azimuth, [-10°, +32°] elevation) ----
    RADAR_AZIMUTH_HALF = np.deg2rad(60)       # ±60° horizontal FOV
    RADAR_ELEVATION_MIN = np.deg2rad(-10)     # look-down limit
    RADAR_ELEVATION_MAX = np.deg2rad(32)      # look-up limit
    RADAR_K = paper_value("radar_range_constant")
    RCS_FRONTAL = paper_value("radar_rcs_frontal_m2")
    RCS_SIDE = paper_value("radar_rcs_side_m2")

    # ---- Battlefield boundaries ----
    # Paper eq.18 uses |x|,|y| > 4e4. Table 4 describes 100 km x 100 km x
    # 10 km, so the two statements are not fully identical; reward/boundary
    # logic follows eq.18 here.
    BATTLEFIELD_HALF_SIZE = paper_value("boundary_reward_half_width_m")
    BATTLEFIELD_ALTITUDE_MAX = 10000.0  # m — ceiling
    BATTLEFIELD_ALTITUDE_MIN = 2500.0   # m — floor (crash)
    OVERLOAD_G_LIMIT = paper_value("maximum_aircraft_load_g")
    OVERLOAD_TIME_LIMIT = 10.0         # s after which >9G triggers termination
    MAX_SPEED = paper_value("maximum_aircraft_speed_mps")

    # ---- GCAS (Ground Collision Avoidance System) ----
    GCAS_ALTITUDE_THRESH = 3000.0       # m — 静态触发阈值 (低下降率时)
    GCAS_RECOVERY_THRESH = 3500.0       # m — 静态恢复解除阈值 (低下降率时)
    GCAS_MAX_PITCH_DEG = 25.0           # deg — 紧急恢复俯仰角 (比常规 ±15° 更激进)
    GCAS_DESCENT_TIME_BUDGET = 15.0     # s — 保留 15 秒下降时间作为恢复余量
    # 动态触发公式: trigger_alt = 2500 + abs(v_up) * GCAS_DESCENT_TIME_BUDGET
    #   v_up = −20 m/s → trigger =  2800 → clamped to 3000
    #   v_up = −33 m/s → trigger =  2995 → clamped to 3000
    #   v_up = −60 m/s → trigger =  3400
    #   v_up = −90 m/s → trigger =  3850
    #   v_up =−120 m/s → trigger =  4300

    metadata = {"render_modes": []}

    def __init__(self, max_num_blue=2, max_num_red=2, num_missiles_per_plane=999,
                 sim_freq=60, agent_interaction_steps=12, max_steps=1400,
                 enable_gcas_for_blue: bool = False,
                 enable_kill_cooldown_gate: bool = False,
                 enable_single_kill_per_step_gate: bool = False,
                 missile_detection_half_angle_deg: float = 45.0,
                 missile_min_launch_range_m: float = 500.0,
                 pid_profile: str = "paper",
                 pid_throttle_base: float = 0.0,
                 reward_mode: str = "paper_joint",
                 missile_guidance_mode: str = "paper_eq9",
                 altitude_reward_config=None,
                 obs_mode: str = "paper_strict",
                 blue_policy_profile: str = "paper_pursuit",
                 environment_profile: str = REFERENCE_ENVIRONMENT_PROFILE,
                 suppress_jsbsim_output: bool = True,
                 environment_config: PaperEnvironmentConfig | None = None,
                 render_mode=None):
        super().__init__()
        if environment_profile not in (
                REFERENCE_ENVIRONMENT_PROFILE, PAPER_MINIMAL_ENVIRONMENT_PROFILE):
            raise ValueError(
                "environment_profile must be 'brma_paper_profile_v1' or "
                "'paper_minimal_3v3_v1'")
        self.environment_profile = str(environment_profile)
        self.is_paper_minimal = (
            self.environment_profile == PAPER_MINIMAL_ENVIRONMENT_PROFILE)
        if self.is_paper_minimal:
            if environment_config not in (None, MINIMAL_PAPER_ENVIRONMENT_CONFIG):
                raise ValueError(
                    "paper_minimal_3v3_v1 does not accept a different "
                    "environment_config")
            self.environment_config = MINIMAL_PAPER_ENVIRONMENT_CONFIG
            obs_mode = "paper_strict"
            reward_mode = "paper_minimal_joint_v1"
            missile_guidance_mode = "paper_minimal_point_mass_v1"
            pid_profile = "paper_minimal_shared_v1"
            enable_gcas_for_blue = False
            enable_kill_cooldown_gate = False
            enable_single_kill_per_step_gate = False
            if blue_policy_profile == "paper_pursuit":
                blue_policy_profile = "paper_minimal_fixed_pair_v1"
        else:
            self.environment_config = environment_config or DEFAULT_PAPER_ENVIRONMENT_CONFIG
        scenario_cfg = self.environment_config.scenario
        self.scenario_config = scenario_cfg
        self.arena_half_width_m = float(scenario_cfg.arena_half_width_m.value)
        self.arena_altitude_min_m = float(scenario_cfg.arena_altitude_min_m.value)
        self.arena_altitude_max_m = float(scenario_cfg.arena_altitude_max_m.value)
        self.reward_boundary_half_width_m = float(
            scenario_cfg.reward_boundary_half_width_m.value)
        self.max_num_blue = max_num_blue
        self.max_num_red = max_num_red
        self.num_missiles_per_plane = int(num_missiles_per_plane)
        self.enable_gcas_for_blue = enable_gcas_for_blue
        self.enable_kill_cooldown_gate = enable_kill_cooldown_gate
        self.enable_single_kill_per_step_gate = enable_single_kill_per_step_gate
        # The paper requires a detection cone and 10 km maximum range but does
        # not publish the cone half-angle or a minimum launch range.
        self.missile_launch_ao_thresh = float(
            self.environment_config.electro_optical.half_angle_rad.value)
        self.missile_launch_min_range = float(
            self.environment_config.electro_optical.minimum_launch_range_m.value)
        if pid_profile not in ("paper", "engineering_safe", "paper_minimal_shared_v1"):
            raise ValueError(
                "pid_profile must be 'paper', 'engineering_safe', or "
                "'paper_minimal_shared_v1'")
        self.pid_profile = pid_profile
        if not 0.0 <= float(pid_throttle_base) <= 1.0:
            raise ValueError("pid_throttle_base must be in [0, 1]")
        self.pid_throttle_base = float(pid_throttle_base)
        if reward_mode not in (
                "paper_joint", "engineering_local", "paper_minimal_joint_v1"):
            raise ValueError(
                "reward_mode must be 'paper_joint', 'engineering_local', or "
                "'paper_minimal_joint_v1'")
        self.reward_mode = reward_mode
        self._reward_summary_step: dict = {}
        if missile_guidance_mode not in (
                "paper_eq9", "legacy_simplified", "paper_minimal_point_mass_v1"):
            raise ValueError(
                "missile_guidance_mode must be 'paper_eq9', 'legacy_simplified', "
                "or 'paper_minimal_point_mass_v1'")
        self.missile_guidance_mode = missile_guidance_mode
        if self.is_paper_minimal:
            self.altitude_reward_config = AltitudeRewardConfig(
                version="eq17_minimal_finite_tail_v1",
                h_min_m=0.0, h_att_m=2000.0, h_adv_m=5000.0,
                h_max_m=10000.0, d_att_max_m=10000.000001,
                high_altitude_tail=0.0)
        else:
            self.altitude_reward_config = (
                DEFAULT_ALTITUDE_REWARD_CONFIG
                if altitude_reward_config is None else altitude_reward_config)
        self.sim_freq = sim_freq
        self.agent_interaction_steps = agent_interaction_steps
        self.max_steps = max_steps
        self.suppress_jsbsim_output = suppress_jsbsim_output
        if obs_mode not in ("engineering", "paper_strict"):
            raise ValueError("obs_mode must be 'engineering' or 'paper_strict'")
        self.obs_mode = obs_mode
        self.blue_policy_profile = validate_blue_policy_profile(
            blue_policy_profile)
        self.blue_policy_controller = BluePolicyController(
            self.blue_policy_profile)
        self.entity_dim = 10 if obs_mode == "paper_strict" else 11
        self.physics_dt = 1.0 / sim_freq
        self.env_dt = agent_interaction_steps * self.physics_dt
        self.missile_cooldown_frames = int(round(
            self.environment_config.fire_control.launch_interval_s.value * self.sim_freq))
        self.missile_lock_delay_frames = int(round(
            self.environment_config.fire_control.lock_time_s.value * self.sim_freq))
        self.np_random = np.random.default_rng()
        self._seed: int | None = None
        self._environment_config_snapshot: dict = {}

        # Agent ID lists (fixed order for observation construction)
        self.blue_ids = [f"blue_{i}" for i in range(max_num_blue)]
        self.red_ids = [f"red_{i}" for i in range(max_num_red)]
        self.agent_ids = self.blue_ids + self.red_ids

        # ---- Action space (Dict) ----
        self.action_space = gymnasium.spaces.Dict({
            aid: gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for aid in self.agent_ids
        })

        # ---- Observation space (Dict) ----
        obs_spaces = {}
        for i, aid in enumerate(self.blue_ids):
            obs_spaces[aid] = gymnasium.spaces.Dict({
                "ego_state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.entity_dim,), dtype=np.float32),
                "ally_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_blue - 1, self.entity_dim), dtype=np.float32),
                "enemy_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_red, self.entity_dim), dtype=np.float32),
                "death_mask": gymnasium.spaces.Box(
                    low=0, high=1,
                    shape=(max_num_blue + max_num_red,), dtype=np.int64),
                "alive_mask": gymnasium.spaces.Box(
                    low=0, high=1,
                    shape=(max_num_blue + max_num_red,), dtype=np.int64),
                "missile_warning": gymnasium.spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32),
                "altitude": gymnasium.spaces.Box(
                    low=0, high=20000, shape=(1,), dtype=np.float32),
                "velocity": gymnasium.spaces.Box(
                    low=-1000, high=1000, shape=(3,), dtype=np.float32),
            })
        for i, aid in enumerate(self.red_ids):
            obs_spaces[aid] = gymnasium.spaces.Dict({
                "ego_state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.entity_dim,), dtype=np.float32),
                "ally_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_red - 1, self.entity_dim), dtype=np.float32),
                "enemy_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_blue, self.entity_dim), dtype=np.float32),
                "death_mask": gymnasium.spaces.Box(
                    low=0, high=1,
                    shape=(max_num_blue + max_num_red,), dtype=np.int64),
                "alive_mask": gymnasium.spaces.Box(
                    low=0, high=1,
                    shape=(max_num_blue + max_num_red,), dtype=np.int64),
                "missile_warning": gymnasium.spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32),
                "altitude": gymnasium.spaces.Box(
                    low=0, high=20000, shape=(1,), dtype=np.float32),
                "velocity": gymnasium.spaces.Box(
                    low=-1000, high=1000, shape=(3,), dtype=np.float32),
            })
        self.observation_space = gymnasium.spaces.Dict(obs_spaces)

        # ---- Internal state (populated in reset) ----
        self.blue_planes: dict[str, AircraftSimulator] = {}
        self.red_planes: dict[str, AircraftSimulator] = {}
        self.pid_controllers: dict[str, PIDController] = {}
        self.current_step = 0

        # Missile tracking
        self._missile_cooldown: dict[str, int] = {}
        self._missiles_in_flight: dict[str, MissileSimulator] = {}
        self._missile_id_counter = 0
        # Lock-delay: paper requires 0.25s continuous sensor track before launch
        self._lock_timer: dict[str, int] = {}     # physics frames continuously locked
        self._lock_target: dict[str, str | None] = {}  # uid of currently tracked enemy
        self._fire_control_states: dict[str, FireControlState] = {}

        # Overload tracking
        self._overload_timers: dict[str, float] = {}
        self._aircraft_diagnostics: dict[str, dict] = {}
        self._sensor_tracks: dict[tuple[str, str], SensorTrack] = {}
        self._sensor_diagnostics_step: list[dict] = []
        self._episode_stats: dict = {}
        self._terminal_cleanup_done = False
        self._invalid_numerical_episode = False
        self._invalid_numerical_reasons: list[str] = []
        self._mws_enabled_by_team = {"red": True, "blue": True}

        # Missile launch counters (per-episode, for debugging)
        self._missile_launch_counts: dict[str, int] = {}
        self._minimal_launch_sequence: dict[str, int] = {}
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_records: dict[str, dict] = {}
        self._launch_quality_step_records: list[dict] = []
        self._launch_quality_done_step_records: list[dict] = []
        self._physics_frame = 0
        # Missile termination reason counters: {"red": Counter(), "blue": Counter()}
        self._missile_term_reasons: dict[str, dict[str, int]] = {
            "red": {}, "blue": {},
        }

        # Death reason tracking (set on the step the agent dies, cleared on reset)
        self._death_reasons: dict[str, str | None] = {}

        # Kill cooldown: prevent "machine gun" multi-kill bursts (paper: 0.5 s between kills)
        self._last_kill_step: dict[str, int] = {}      # agent_id → env step of last kill
        self._step_kill_count: dict[str, int] = {}      # kills per agent this env step
        self._agents_deny_kill: set[str] = set()         # agents blocked from scoring kills this step

        # Engaged-targets set: hot-updated across agents within the same
        # physics frame to prevent same-frame double-launch (paper §2.1.3).
        # Populated at the start of each env step from in-flight missiles;
        # mutated in-place by _check_missile_launch + blue_coordinated_actions.
        self._engaged_targets: set[str] = set()
        self._crashed_this_step: set[str] = set()

        # TacView rendering
        self._tacview_recorder: TacviewLogger | None = None
        self._sim_time = 0.0
        self._acmi_filepath: str | None = None

        # ACMI numeric ID mapping (for TacView format compliance)
        self._agent_acmi_id: dict[str, int] = {}
        for i in range(max_num_blue):
            self._agent_acmi_id[f"blue_{i}"] = 101 + i
        for i in range(max_num_red):
            self._agent_acmi_id[f"red_{i}"] = 201 + i
        self._missile_acmi_id: dict[str, int] = {}
        self._next_missile_acmi_id = 1001

    # ------------------------------------------------------------------
    #  RL Environment API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.np_random = np.random.default_rng(self._seed)
        elif self._seed is None:
            # Gymnasium created a generator above; retain an explicit traceable seed.
            self._seed = int(self.np_random.integers(0, 2**32 - 1))
            self.np_random = np.random.default_rng(self._seed)
        if self.is_paper_minimal:
            self._environment_config_snapshot = minimal_environment_snapshot(
                num_red=self.max_num_red, num_blue=self.max_num_blue,
                sim_freq=self.sim_freq,
                agent_interaction_steps=self.agent_interaction_steps,
                max_episode_length=self.max_steps,
                seed=self._seed,
                blue_policy_profile=self.blue_policy_profile)
        else:
            self._environment_config_snapshot = environment_config_snapshot(
                self.environment_config, num_red=self.max_num_red,
                num_blue=self.max_num_blue, sim_freq=self.sim_freq,
                agent_interaction_steps=self.agent_interaction_steps, seed=self._seed,
                blue_policy_profile=self.blue_policy_profile)
        self.current_step = 0
        self._physics_frame = 0
        self._sim_time = 0.0
        self._missile_id_counter = 0
        self._missiles_in_flight.clear()
        self._launch_quality_records.clear()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        self._reward_summary_step = {}
        self._missile_acmi_id.clear()
        self._missile_term_reasons = {"red": {}, "blue": {}}
        self._sensor_tracks = {}
        self._sensor_diagnostics_step = []
        self._episode_stats = {
            "EpisodeRedJointReturn": 0.0, "EpisodeBlueJointReturn": 0.0,
            "EpisodeRedLocalRewardSum": 0.0, "EpisodeBlueLocalRewardSum": 0.0,
            "EpisodeRedTerminalReward": 0.0, "EpisodeBlueTerminalReward": 0.0,
            "EpisodeLength": 0,
            "maximum_live_missiles_observed": 0,
        }
        self._terminal_cleanup_done = False
        self._invalid_numerical_episode = False
        self._invalid_numerical_reasons = []
        self._next_missile_acmi_id = 1001
        if self._tacview_recorder is not None:
            self._tacview_recorder.reset()

        # Create or reload blue aircraft (reuse to avoid JSBSim C++ memory leak)
        first_reset = len(self.blue_planes) == 0
        for i in range(self.max_num_blue):
            aid = self.blue_ids[i]
            init_state = self._make_init_state("Blue", i)
            if first_reset:
                sim = AircraftSimulator(
                    uid=aid, color="Blue", model=self.scenario_config.aircraft_model.value,
                    sim_freq=self.sim_freq, num_missiles=self.num_missiles_per_plane,
                    init_state=init_state,
                    suppress_jsbsim_output=self.suppress_jsbsim_output,
                )
                self.blue_planes[aid] = sim
            else:
                self.blue_planes[aid].reload(new_state=init_state)

        # Create or reload red aircraft
        for i in range(self.max_num_red):
            aid = self.red_ids[i]
            init_state = self._make_init_state("Red", i)
            if first_reset:
                sim = AircraftSimulator(
                    uid=aid, color="Red", model=self.scenario_config.aircraft_model.value,
                    sim_freq=self.sim_freq, num_missiles=self.num_missiles_per_plane,
                    init_state=init_state,
                    suppress_jsbsim_output=self.suppress_jsbsim_output,
                )
                self.red_planes[aid] = sim
            else:
                self.red_planes[aid].reload(new_state=init_state)

        # Link partners and enemies
        blue_list = list(self.blue_planes.values())
        red_list = list(self.red_planes.values())
        for sim in blue_list:
            sim.partners = [s for s in blue_list if s.uid != sim.uid]
            sim.enemies = red_list.copy()
        for sim in red_list:
            sim.partners = [s for s in red_list if s.uid != sim.uid]
            sim.enemies = blue_list.copy()

        self.blue_policy_controller.reset(
            self.blue_ids, self.red_ids,
            {aid: float(sim.get_rpy()[2]) for aid, sim in self.blue_planes.items()},
            {aid: float(sim.get_geodetic()[2]) for aid, sim in self.blue_planes.items()},
        )

        # Create or reset PID controllers
        if first_reset:
            for aid in self.agent_ids:
                self.pid_controllers[aid] = PIDController(
                    self.physics_dt, profile=self.pid_profile,
                    throttle_base=self.pid_throttle_base,
                    config=self.environment_config.pid)
        else:
            for pid in self.pid_controllers.values():
                pid.reset()

        # Reset missile cooldowns
        self._missile_cooldown = {aid: 0 for aid in self.agent_ids}

        # Reset lock-delay timers
        self._lock_timer = {aid: 0 for aid in self.agent_ids}
        self._lock_target = {aid: None for aid in self.agent_ids}
        self._fire_control_states = {
            aid: FireControlState() for aid in self.agent_ids}
        self._evasion_diagnostics = {
            aid: {"activations": 0, "active_frames": 0, "active": False}
            for aid in self.agent_ids}

        # Reset overload timers
        self._overload_timers = {aid: 0.0 for aid in self.agent_ids}
        self._aircraft_diagnostics = {
            aid: {"maximum_speed_mps_seen": 0.0, "overspeed_frames": 0,
                  "maximum_speed_before_limit_mps": 0.0,
                  "maximum_speed_after_limit_mps": 0.0,
                  "speed_limiter_activations": 0,
                  "maximum_load_g_seen": 0.0, "over_g_frames": 0,
                  "load_limiter_activations": 0}
            for aid in self.agent_ids}

        # Reset missile launch counters
        self._missile_launch_counts = {aid: 0 for aid in self.agent_ids}
        self._minimal_launch_sequence = {aid: 0 for aid in self.agent_ids}
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []

        # Reset death reasons
        self._death_reasons = {}

        # Reset kill cooldown tracking
        self._last_kill_step = {}
        self._step_kill_count = {aid: 0 for aid in self.agent_ids}
        self._agents_deny_kill = set()
        self._engaged_targets = set()
        self._crashed_this_step: set[str] = set()

        # Record initial frame at time 0.00 for TacView
        if self._tacview_recorder is not None:
            self._render_frame()

        return self._get_obs(), self._get_info()

    def refresh_engaged_targets(self) -> set[str]:
        """Rebuild and return the live engaged-targets set from in-flight missiles.

        Call this once per env step, **before** calling rule-based agents
        (e.g. ``blue_coordinated_actions``) and **before** ``step()``.

        The returned set is a *live* reference to ``self._engaged_targets``.
        Callers may mutate it in-place to add flight-assigned targets.
        ``_check_missile_launch`` reads this same set and hot-updates it
        after every launch, guaranteeing same-frame deconfliction across
        all agents within the physics loop.
        """
        self._engaged_targets = set()
        for m in self._missiles_in_flight.values():
            if m.is_alive:
                self._engaged_targets.add(m._target_id)
        return self._engaged_targets

    def get_blue_own_positions(self) -> dict[str, np.ndarray]:
        """Return current blue ownship positions for cruise boundary patrol.

        This is not part of the learning observation and does not expose enemy
        state. It is only used by the hand-coded blue policy to avoid no-target
        cruise flying indefinitely out of the battlefield.
        """

        result: dict[str, np.ndarray] = {}
        for bid, sim in self.blue_planes.items():
            if sim is not None and sim.is_alive:
                result[bid] = np.asarray(sim.get_position(), dtype=np.float32)
        return result

    def get_blue_own_kinematics(self) -> dict[str, dict]:
        """Return blue ownship position and heading for rule-based policy.

        This is not part of learning observation and does not expose enemy
        state. It is only used by the hand-coded blue policy for boundary
        patrol/safety.
        """

        result: dict[str, dict] = {}
        for bid, sim in self.blue_planes.items():
            if sim is not None and sim.is_alive:
                result[bid] = {
                    "position": np.asarray(sim.get_position(), dtype=np.float32),
                    "heading": float(sim.get_rpy()[2]),
                }
        return result

    def blue_policy_actions(self, blue_obs: dict[str, dict]) -> dict[str, np.ndarray]:
        """Generate actions with this environment's isolated blue controller."""
        engaged = self.refresh_engaged_targets()
        kinematics = self.get_blue_own_kinematics()
        selected_missiles: dict[str, str | None] = {}
        mws_detected: dict[str, bool] = {}
        own_alive = {
            blue_id: bool(sim.is_alive)
            for blue_id, sim in self.blue_planes.items()
        }
        for blue_id, sim in self.blue_planes.items():
            missile = (sim.check_missile_warning()
                       if sim.is_alive and self._mws_enabled_for_agent(blue_id)
                       else None)
            selected_missiles[blue_id] = getattr(missile, "uid", None)
            mws_detected[blue_id] = missile is not None
        return self.blue_policy_controller.act(
            blue_obs, self.max_num_blue, self.max_num_red, engaged,
            {aid: data["position"] for aid, data in kinematics.items()},
            {aid: data["heading"] for aid, data in kinematics.items()},
            self.current_step, selected_missiles, mws_detected,
            own_alive=own_alive)

    def set_team_mws_enabled(self, team: str, enabled: bool) -> None:
        """Set an audit-only team MWS gate without changing profile defaults."""
        normalized = str(team).lower()
        if normalized not in ("red", "blue"):
            raise ValueError("team must be 'red' or 'blue'")
        self._mws_enabled_by_team[normalized] = bool(enabled)

    def _mws_enabled_for_agent(self, agent_id: str) -> bool:
        team = "blue" if agent_id.startswith("blue") else "red"
        gates = getattr(self, "_mws_enabled_by_team", {"red": True, "blue": True})
        if not gates.get(team, True):
            return False
        if team == "blue":
            fallback = getattr(self, "blue_policy_profile", "paper_pursuit") not in (
                "fixed_pair_no_mws_v1", "frozen_route_blue_v1",
                "paper_minimal_straight_patrol_v1")
            return bool(getattr(
                self.blue_policy_controller, "blue_mws_override_enabled", fallback))
        return True

    def step(self, actions: dict):
        self.current_step += 1
        self._crashed_this_step.clear()
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        self._reward_summary_step = {}
        self._sensor_diagnostics_step = []
        self.refresh_engaged_targets()

        # 0. Optional legacy anti-burst guard (off for paper_strict baseline).
        #    The paper defines launch interval/deconfliction, not post-hit
        #    kill denial.  Keep this available only for explicit debugging.
        self._agents_deny_kill = set()
        if self.enable_kill_cooldown_gate:
            for aid in self.agent_ids:
                last_kill = self._last_kill_step.get(aid, -999)
                if self.current_step - last_kill < self.KILL_COOLDOWN_STEPS:
                    self._agents_deny_kill.add(aid)
        self._step_kill_count = {aid: 0 for aid in self.agent_ids}

        # 1. Parse actions and compute PID control targets
        targets = self._parse_actions(actions)

        # 2. Run physics for agent_interaction_steps frames
        for _ in range(self.agent_interaction_steps):
            self._apply_pid_controls(targets)
            self._run_one_physics_frame()
            self._physics_frame += 1
            self._check_missile_launch()
            self._update_missiles()
            self._update_overload_timers()
            self._cleanup_missiles()

        # 3. Check terminations
        self._check_crash_terminations()

        # 4. Compute rewards
        rewards, reward_components = self._compute_rewards()

        # 5. Advance sim time (one env step = agent_interaction_steps × physics_dt)
        self._sim_time += self.env_dt

        # 6. Render before missile cleanup so explosions are captured
        if self._tacview_recorder is not None:
            self._render_frame()

        # 7. Clean up done missiles (after rendering to capture explosion logs)
        self._cleanup_missiles()

        # 8. Build observations, terminations, truncations
        obs = self._get_obs()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info(reward_components)
        if all(terminated[aid] or truncated[aid] for aid in self.agent_ids):
            final_info = self._freeze_terminal_info(info)
            cleanup = self._clear_terminal_combat_state()
            final_info["__terminal_cleanup__"].update(cleanup)
            info = final_info

        return obs, rewards, terminated, truncated, info

    def _freeze_terminal_info(self, info: dict) -> dict:
        """Deep-copy final combat statistics before releasing episode objects."""
        frozen = copy.deepcopy(info)
        fire_snapshot = {
            aid: state.snapshot() for aid, state in self._fire_control_states.items()
        }
        locks_snapshot = {
            aid: {"target_id": self._lock_target.get(aid),
                  "continuous_detection_frames": int(self._lock_timer.get(aid, 0))}
            for aid in self.agent_ids
        }
        censored = []
        for mid, missile in self._missiles_in_flight.items():
            if missile.is_alive:
                record = copy.deepcopy(self._launch_quality_records.get(mid, {}))
                record.update({
                    "missile_id": mid,
                    "censored": True,
                    "censor_reason": "episode_end_cleanup",
                    "flight_time_sec": float(getattr(missile, "_t", 0.0)),
                })
                censored.append(record)
        active_fc = sum(int(
            state.current_target_id is not None
            or state.continuous_detection_frames > 0
            or state.lock_mature
            or state.cooldown_frames_remaining > 0)
            for state in self._fire_control_states.values())
        active_locks = sum(int(
            self._lock_target.get(aid) is not None
            or self._lock_timer.get(aid, 0) > 0) for aid in self.agent_ids)
        frozen["__terminal_cleanup__"] = {
            "missiles_in_flight_at_episode_end": sum(
                int(m.is_alive) for m in self._missiles_in_flight.values()),
            "fire_control_active_at_episode_end": active_fc,
            "locks_active_at_episode_end": active_locks,
            "engaged_targets_at_episode_end": sorted(self._engaged_targets),
            "final_fire_control_snapshot": fire_snapshot,
            "final_lock_snapshot": locks_snapshot,
            "completed_launch_records": copy.deepcopy(
                self._launch_quality_done_step_records),
            "censored_launch_records": censored,
        }
        return frozen

    def _clear_terminal_combat_state(self) -> dict:
        """Idempotently release objects without creating physical terminations."""
        if self._terminal_cleanup_done:
            return {
                "red_episode_end_cleanup_count": 0,
                "blue_episode_end_cleanup_count": 0,
                "missiles_remaining_after_cleanup": len(self._missiles_in_flight),
                "fire_control_remaining_after_cleanup": 0,
                "locks_remaining_after_cleanup": 0,
                "engaged_targets_remaining_after_cleanup": len(self._engaged_targets),
            }
        cleanup_counts = {"red": 0, "blue": 0}
        for missile in list(self._missiles_in_flight.values()):
            if missile.is_alive:
                team = "red" if missile._parent_id.startswith("red") else "blue"
                cleanup_counts[team] += 1
            missile.detach_references()
            missile.close()
        self._missiles_in_flight.clear()
        self._engaged_targets.clear()
        for aid in self.agent_ids:
            self._lock_timer[aid] = 0
            self._lock_target[aid] = None
            self._fire_control_states[aid] = FireControlState(
                transition_reason="episode_end")
        self._launch_quality_records.clear()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        self._terminal_cleanup_done = True
        return {
            "red_episode_end_cleanup_count": cleanup_counts["red"],
            "blue_episode_end_cleanup_count": cleanup_counts["blue"],
            "missiles_remaining_after_cleanup": len(self._missiles_in_flight),
            "fire_control_remaining_after_cleanup": sum(int(
                state.current_target_id is not None
                or state.continuous_detection_frames > 0
                or state.lock_mature
                or state.cooldown_frames_remaining > 0)
                for state in self._fire_control_states.values()),
            "locks_remaining_after_cleanup": sum(int(
                self._lock_target.get(aid) is not None
                or self._lock_timer.get(aid, 0) > 0) for aid in self.agent_ids),
            "engaged_targets_remaining_after_cleanup": len(self._engaged_targets),
        }

    # ------------------------------------------------------------------
    #  Action parsing
    # ------------------------------------------------------------------

    def _parse_actions(self, actions: dict) -> dict:
        """Convert normalised actor outputs ∈ [-1, 1] to physical setpoints.

        Control-flow priority (team-aware):
          Layer 1 — Missile evasion:     BOTH teams  (paper §2.1.3, scripted)
          Layer 2 — GCAS safety net:     BLUE only   (hard-coded baseline)
          Layer 3 — Agent action:        BOTH teams  (identical §2.4 mapping)

        Paper §2.4 mapping — IDENTICAL for both teams (ABSOLUTE targets):
          act[0] ∈ [-1, 1]  →  target_pitch   ∈ [-π/2, +π/2]     [rad]  (±90°)
          act[1] ∈ [-1, 1]  →  target_heading ∈ [-π,   +π]       [rad]  (±180° absolute)
          act[2] ∈ [-1, 1]  →  target_velocity ∈ [102, 408]      [m/s]  (M0.3–M1.2)

        All quantities are in SI / radian units consumed by the PID controller.
        """
        targets = {}
        for aid, act in actions.items():
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                targets[aid] = None
                continue

            is_blue = aid.startswith("blue")
            rpy = sim.get_rpy()
            current_heading = float(rpy[2])  # ψ ∈ [−π, π]

            # =================================================================
            #  Layer 1 — Missile Evasion Script (paper §2.1.3)
            #
            #  BOTH teams.  When MWS detects an incoming missile, the script
            #  overrides all other control.  The paper explicitly lists missile
            #  evasion as scripted behaviour that is NOT learned.
            # =================================================================
            mws_enabled = self._mws_enabled_for_agent(aid)
            incoming = sim.check_missile_warning() if mws_enabled else None
            if incoming is not None and mws_enabled:
                evasion = self._evasion_diagnostics[aid]
                if not evasion["active"]:
                    evasion["activations"] += 1
                evasion["active"] = True
                evasion["active_frames"] += 1
                alt_m = sim.get_geodetic()[2]

                # Determine turn direction from missile bearing (+right, −left)
                ego_pos = sim.get_position()
                ego_vel = sim.get_velocity()
                msl_pos = incoming.get_position()
                dn = msl_pos[0] - ego_pos[0]
                de = msl_pos[1] - ego_pos[1]
                vn, ve = float(ego_vel[0]), float(ego_vel[1])
                vh = np.hypot(vn, ve) + 1e-8
                rh = np.hypot(dn, de) + 1e-8
                ao = np.arctan2((vn * de - ve * dn) / (vh * rh),
                                (vn * dn + ve * de) / (vh * rh))
                if getattr(self, "is_paper_minimal", False):
                    turn_dir = -1.0 if ao > 0 else 1.0
                    target_heading = current_heading + turn_dir * np.deg2rad(60.0)
                    targets[aid] = (0.0, target_heading, 300.0)
                    if is_blue:
                        self.blue_policy_controller.record_executed_heading(
                            aid, target_heading, "mws_override")
                    continue
                turn_dir = 1.0 if ao > 0 else -1.0

                if alt_m > 8500.0:
                    # Preserve the break turn but do not climb through the
                    # physical ceiling while carrying positive vertical speed.
                    target_pitch = np.deg2rad(-5.0 if sim.get_velocity()[2] > 5.0 else 0.0)
                    target_heading = current_heading + turn_dir * np.deg2rad(60.0)
                elif alt_m > 5000.0:
                    # High altitude: break turn with ~60° bank.
                    # Pull 25° pitch while executing a ~60° heading break.
                    target_pitch = np.deg2rad(25.0)
                    target_heading = current_heading + turn_dir * np.deg2rad(60.0)
                else:
                    # Low altitude (< 5000 m): wings-level zoom climb.
                    # Pull 30° pitch, maintain current heading (roll out first).
                    target_pitch = np.deg2rad(15.0 if alt_m < 1200.0 else 25.0)
                    ego_roll = float(rpy[0])
                    if abs(ego_roll) > np.deg2rad(5):
                        target_heading = current_heading - np.sign(ego_roll) * np.deg2rad(15.0)
                    else:
                        target_heading = current_heading

                targets[aid] = (target_pitch, target_heading, self.VELOCITY_MAX)
                if is_blue:
                    self.blue_policy_controller.record_executed_heading(
                        aid, target_heading, "mws_override")
                continue
            self._evasion_diagnostics[aid]["active"] = False

            # =================================================================
            #  Layer 2 — GCAS Safety Net (BLUE ONLY)
            #
            #  Blue is the hard-coded rule-based baseline.  It receives full
            #  altitude protection to establish a credible reference opponent.
            #
            #  Red team does NOT go through here — see §2.5.1 rationale above.
            # =================================================================
            if is_blue and self.enable_gcas_for_blue:
                alt_m = sim.get_geodetic()[2]
                vel = sim.get_velocity()
                v_up = float(vel[2])  # positive = climbing

                # Dynamic trigger: faster descent → earlier intervention
                if v_up >= 0:
                    trigger_alt = self.GCAS_ALTITUDE_THRESH
                else:
                    trigger_alt = max(self.GCAS_ALTITUDE_THRESH,
                                      2500.0 + abs(v_up) * self.GCAS_DESCENT_TIME_BUDGET)
                recovery_alt = trigger_alt + 500.0

                if alt_m < trigger_alt or alt_m < recovery_alt:
                    ego_roll = float(rpy[0])
                    # Roll wings level, pull hard up
                    if abs(ego_roll) > np.deg2rad(5):
                        target_heading = current_heading - np.sign(ego_roll) * np.deg2rad(15.0)
                    else:
                        target_heading = current_heading
                    targets[aid] = (np.deg2rad(self.GCAS_MAX_PITCH_DEG),
                                    target_heading, self.VELOCITY_MAX)
                    self.blue_policy_controller.record_executed_heading(
                        aid, target_heading, "gcas_override")
                    continue

            # =================================================================
            #  Layer 3 — Agent Action (paper §2.4 — both teams identical)
            #
            #    target_pitch   = act[0] * 90°             ∈ [−90°, +90°]
            #    target_heading = act[1] * 180°            ∈ [−180°, +180°]  (absolute)
            #    target_velocity ∈ [102, 408] m/s
            # =================================================================
            target_velocity = self.VELOCITY_MIN + (float(act[2]) + 1.0) / 2.0 * (
                self.VELOCITY_MAX - self.VELOCITY_MIN)
            target_pitch = float(act[0]) * np.deg2rad(self.PITCH_DEG)
            target_heading = float(act[1]) * np.pi

            targets[aid] = (target_pitch, target_heading, target_velocity)
            if is_blue:
                self.blue_policy_controller.record_executed_heading(
                    aid, target_heading, "base_policy")
        return targets

    # ------------------------------------------------------------------
    #  PID control application (per physics frame)
    # ------------------------------------------------------------------

    def _apply_pid_controls(self, targets: dict):
        """Read current flight state, compute BTT PID, write to JSBSim."""
        for aid, target in targets.items():
            if target is None:
                continue
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue

            target_pitch, target_heading, target_velocity = target
            rpy = sim.get_rpy()                          # (φ, θ, ψ) — rad
            vel = sim.get_velocity()
            current_speed = float(np.linalg.norm(vel))   # scalar m/s

            pid = self.pid_controllers[aid]
            # Convert velocity from (vn,ve,vu) up-positive to (vn,ve,vd) NED down-positive
            # for gimbal-safe R_BI construction (Fix 3).
            vel_ned = np.array([vel[0], vel[1], -vel[2]], dtype=np.float64)
            aileron, elevator, rudder, throttle = pid.compute_control(
                rpy, current_speed,
                target_pitch, target_heading, target_velocity,
                ned_velocity=vel_ned,  # true NED (z=down)
            )

            diag = self._aircraft_diagnostics[aid]
            max_speed = float(self.environment_config.aircraft.maximum_speed_mps.value)
            if current_speed > max_speed:
                throttle = min(throttle, float(
                    self.environment_config.aircraft.overspeed_throttle_limit.value))
                diag["overspeed_frames"] += 1
            diag["maximum_speed_mps_seen"] = max(
                diag["maximum_speed_mps_seen"], current_speed)
            try:
                g_load = float(np.linalg.norm([
                    sim.get_property_value("accelerations/n-pilot-x-norm"),
                    sim.get_property_value("accelerations/n-pilot-y-norm"),
                    sim.get_property_value("accelerations/n-pilot-z-norm")]))
            except Exception:
                g_load = float("nan")
            if not np.isfinite(g_load):
                sim.crash()
                if self.is_paper_minimal:
                    self._mark_invalid_numerical(aid, "NonFiniteLoad")
                else:
                    self._death_reasons.setdefault(aid, "Crash_NumericalLoad")
                self._crashed_this_step.add(aid)
                continue
            if (self.is_paper_minimal
                    and g_load > MINIMAL_EXTREME_LOAD_INVALID_THRESHOLD_G):
                sim.crash()
                self._mark_invalid_numerical(aid, "ExtremeFiniteLoad")
                self._crashed_this_step.add(aid)
                continue
            if g_load > 100.0 and not self.is_paper_minimal:
                sim.crash()
                self._death_reasons.setdefault(aid, "Crash_NumericalLoad")
                self._crashed_this_step.add(aid)
                continue
            g_limit = float(self.environment_config.aircraft.maximum_load_g.value)
            if self.pid_profile in ("paper", "paper_minimal_shared_v1") and g_load > g_limit:
                minimum_scale = 0.0 if self.is_paper_minimal else 0.1
                scale = float(np.clip(
                    g_limit / max(g_load, 1e-9), minimum_scale, 1.0))
                aileron *= scale
                elevator *= scale
                if self.is_paper_minimal:
                    rudder *= scale
                diag["load_limiter_activations"] += 1

            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            sim.set_property_value("fcs/throttle-cmd-norm", throttle)

    # ------------------------------------------------------------------
    #  Physics stepping
    # ------------------------------------------------------------------

    def _run_one_physics_frame(self):
        """Advance every alive aircraft by one JSBSim frame."""
        for aid, sim in self._all_sims_with_ids():
            if sim.is_alive:
                sim.run()
                self._enforce_aircraft_constraints(aid, sim)

    def _enforce_aircraft_constraints(self, aid: str, sim: AircraftSimulator):
        """Project speed onto the paper envelope and reject non-finite state."""
        velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
        position = np.asarray(sim.get_position(), dtype=np.float64)
        rpy = np.asarray(sim.get_rpy(), dtype=np.float64)
        if not np.all(np.isfinite(np.concatenate([velocity, position, rpy]))):
            sim.crash()
            if self.is_paper_minimal:
                self._mark_invalid_numerical(aid, "NonFiniteState")
            else:
                self._death_reasons.setdefault(aid, "Crash_NonFinite")
            self._crashed_this_step.add(aid)
            return
        speed = float(np.linalg.norm(velocity))
        maximum = float(self.environment_config.aircraft.maximum_speed_mps.value)
        diag = self._aircraft_diagnostics[aid]
        diag["maximum_speed_mps_seen"] = max(
            diag["maximum_speed_mps_seen"], speed)
        if self.is_paper_minimal and speed > maximum:
            diag["maximum_speed_before_limit_mps"] = max(
                diag["maximum_speed_before_limit_mps"], speed)
            projected_speed = sim.project_velocity_magnitude(maximum)
            diag["maximum_speed_after_limit_mps"] = max(
                diag["maximum_speed_after_limit_mps"], projected_speed)
            diag["speed_limiter_activations"] += 1
            if (not np.isfinite(projected_speed)
                    or projected_speed > maximum + 1e-3):
                sim.crash()
                self._mark_invalid_numerical(aid, "SpeedProjectionFailure")
                self._crashed_this_step.add(aid)
            return
        diag["maximum_speed_after_limit_mps"] = max(
            diag["maximum_speed_after_limit_mps"], speed)
        if speed > 10.0 * maximum:
            sim.crash()
            if self.is_paper_minimal:
                self._mark_invalid_numerical(aid, "NumericalEnvelope")
            else:
                self._death_reasons.setdefault(aid, "Crash_NumericalEnvelope")
            self._crashed_this_step.add(aid)

    def _mark_invalid_numerical(self, aid: str, reason: str) -> None:
        self._invalid_numerical_episode = True
        label = f"{aid}:{reason}"
        if label not in self._invalid_numerical_reasons:
            self._invalid_numerical_reasons.append(label)
        self._death_reasons.setdefault(aid, "Invalid_Numerical")

    def _check_missile_launch(self):
        """Rule-based missile launch with lock-delay + hot-update deconfliction.

        For each armed agent, finds the closest **unengaged** enemy within the
        sensor cone (AO < 45°, R ∈ [0.5, 10] km, TA > 90° rear-hemisphere).
        The target must be continuously tracked for 0.25 s before the weapon
        is released. Launch cooldown is 0.5 s. Both are stored as physics-frame
        counts derived from ``sim_freq``.

        **Hot-update engaged-targets gate (paper §2.1.3):**
        Uses a single shared ``self._engaged_targets`` set (pre-populated by
        ``refresh_engaged_targets()`` from in-flight missiles, and optionally
        extended by the caller with flight-assigned targets).  When an agent
        launches, the target is **immediately** added to this set so that
        subsequent agents in the same physics frame see it and skip that
        target — preventing same-frame double-launch.
        """
        for aid in self.agent_ids:
            team = "red" if aid.startswith("red") else "blue"
            diag = self._launch_diag_step[team]
            diag["scan_frames"] += 1
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None
                continue
            if sim.num_left_missiles <= 0:
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None
                continue
            diag["alive_shooters"] += 1
            # Decrement cooldown every physics frame
            if self._missile_cooldown[aid] > 0:
                self._missile_cooldown[aid] -= 1

            # ---- Shared engaged-targets set (hot-updated across agents) ----
            # Uses self._engaged_targets directly — no per-agent recomputation.
            # The set contains enemy UIDs that have an in-flight friendly
            # missile tracking them AND targets flight-assigned by the
            # coordinated-actions allocator.

            # ---- Find the closest UNENGAGED enemy in the launch cone ----
            enemies = self.red_planes if sim.color == "Blue" else self.blue_planes
            if self.is_paper_minimal:
                try:
                    paired_index = int(aid.split("_", 1)[1])
                except (ValueError, IndexError):
                    paired_index = -1
                target_prefix = "red" if sim.color == "Blue" else "blue"
                paired = enemies.get(f"{target_prefix}_{paired_index}")
                candidate_enemies = [] if paired is None else [paired]
            else:
                candidate_enemies = list(enemies.values())
            best_enemy = None
            best_distance = float("inf")

            for enemy_sim in candidate_enemies:
                if not enemy_sim.is_alive:
                    continue
                diag["alive_enemy_pairs"] += 1
                # --- Target-deconfliction: skip enemies already engaged ---
                if enemy_sim.uid in self._engaged_targets:
                    diag["engaged_blocked"] += 1
                    continue
                diag["unengaged_enemy_pairs"] += 1

                ego_pos = sim.get_position()
                enm_pos = enemy_sim.get_position()
                AO = compute_body_x_q_los(ego_pos, sim.get_rpy(), enm_pos)
                _, TA, _ = get2d_heading_AO_TA_R(
                    ego_pos, sim.get_rpy()[2], enm_pos, enemy_sim.get_rpy()[2])
                R = compute_3d_range(ego_pos, enm_pos)
                eo_visible = self._is_detected_by_electro_optical(sim, enemy_sim)
                range_ok = self.missile_launch_min_range < R and eo_visible
                ao_ok = (True if self.is_paper_minimal
                         else AO < self.missile_launch_ao_thresh)
                ta_ok = TA > self.environment_config.fire_control.rear_hemisphere_ta_rad.value
                if range_ok:
                    diag["range_ok_pairs"] += 1
                if ao_ok:
                    diag["ao_ok_pairs"] += 1
                if ta_ok:
                    diag["ta_ok_pairs"] += 1

                in_cone = (ao_ok and range_ok and ta_ok)
                if in_cone:
                    diag["geometry_ok_pairs"] += 1

                if in_cone and R < best_distance:
                    best_distance = R
                    best_enemy = enemy_sim

            # ---- Lock-delay state machine ----
            # If the currently locked target becomes engaged, abandon the
            # lock immediately so the agent can start building a new lock
            # on the next-best unengaged target.
            if (best_enemy is not None
                    and self._lock_target.get(aid) is not None
                    and self._lock_target[aid] in self._engaged_targets):
                # Previously locked target is now engaged — force reset
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None

            if best_enemy is not None:
                if self._lock_target.get(aid) == best_enemy.uid:
                    # Same target — accumulate lock
                    self._lock_timer[aid] += 1
                    diag["lock_continued"] += 1
                else:
                    # Target switched — reset lock
                    self._lock_target[aid] = best_enemy.uid
                    self._lock_timer[aid] = 1
                    diag["lock_started"] += 1
            else:
                # No eligible unengaged enemy — lose lock immediately
                if self._lock_target.get(aid) is not None:
                    diag["lock_lost"] += 1
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None

            # ---- Launch when lock mature and weapon ready ----
            # (best_enemy is already guaranteed unengaged by the filter above)
            on_kill_cooldown = self.enable_kill_cooldown_gate and aid in self._agents_deny_kill
            lock_mature = (best_enemy is not None
                           and self._lock_timer[aid] >= self.missile_lock_delay_frames)
            if lock_mature:
                diag["lock_mature_pairs"] += 1
                if self._missile_cooldown[aid] != 0:
                    diag["cooldown_blocked"] += 1
                if on_kill_cooldown:
                    diag["kill_cooldown_blocked"] += 1
            if (best_enemy is not None
                    and self._lock_timer[aid] >= self.missile_lock_delay_frames
                    and self._missile_cooldown[aid] == 0
                    and not on_kill_cooldown):
                launch_quality = self._build_launch_quality_record(
                    sim, best_enemy, best_distance)
                self._launch_missile(sim, best_enemy, launch_quality)
                diag["launches"] += 1
                # ---- HOT-UPDATE: immediately mark target as engaged ----
                # Subsequent agents in the same physics frame will see this
                # and skip the target, preventing same-frame double-launch.
                self._engaged_targets.add(best_enemy.uid)
                # Reset lock after launch (must re-acquire)
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None
                # Cooldown is set inside _launch_missile

            state = self._fire_control_states.setdefault(aid, FireControlState())
            state.current_target_id = self._lock_target.get(aid)
            state.continuous_detection_frames = int(self._lock_timer.get(aid, 0))
            state.lock_mature = bool(lock_mature)
            state.cooldown_frames_remaining = int(self._missile_cooldown.get(aid, 0))
            if state.current_target_id is None and not lock_mature:
                state.detection_state = "no_target"
            elif lock_mature and state.cooldown_frames_remaining > 0:
                state.detection_state = "cooldown"
                state.blocked_reason = "cooldown"
            elif lock_mature:
                state.detection_state = "ready_to_launch"
            else:
                state.detection_state = "tracking"

    def _build_launch_quality_record(
        self,
        shooter: AircraftSimulator,
        target: AircraftSimulator,
        range_m: float | None = None,
    ) -> dict:
        """Build a launch-quality snapshot without affecting launch decisions."""

        team = "red" if shooter.uid.startswith("red") else "blue"
        try:
            shooter_pos = shooter.get_position()
            target_pos = target.get_position()
            shooter_vel = shooter.get_velocity()
            target_vel = target.get_velocity()
            ao = compute_body_x_q_los(
                shooter_pos, shooter.get_rpy(), target_pos)
            _, ta, _ = get2d_heading_AO_TA_R(
                shooter_pos, shooter.get_rpy()[2],
                target_pos, target.get_rpy()[2])
            r = compute_3d_range(shooter_pos, target_pos)
        except Exception:
            shooter_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            shooter_vel = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            target_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            target_vel = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            ao, ta = _nan_float(), _nan_float()
            r = _nan_float() if range_m is None else float(range_m)

        if range_m is not None:
            r = float(range_m)

        return make_launch_quality_record(
            team=team,
            shooter_id=shooter.uid,
            target_id=target.uid,
            current_step=self.current_step,
            physics_frame=self._physics_frame,
            range_m=r,
            AO_rad=ao,
            TA_rad=ta,
            shooter_pos=shooter_pos,
            shooter_vel=shooter_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            target_alive_at_launch=bool(target.is_alive),
        )

    def _launch_missile(
        self,
        parent: AircraftSimulator,
        target: AircraftSimulator,
        launch_quality: dict | None = None,
    ):
        missile_rng = self.np_random
        launch_speed_mps = None
        overshoot_window_s = None
        if self.is_paper_minimal:
            from configs.paper_minimal_3v3_spec import (
                MINIMAL_MISSILE_LAUNCH_SPEED_MPS,
                MINIMAL_MISSILE_OVERSHOOT_WINDOW_S,
            )
            sequence = self._minimal_launch_sequence[parent.uid]
            missile_rng = _minimal_missile_rng(
                self._seed, parent.uid, sequence)
            self._minimal_launch_sequence[parent.uid] = sequence + 1
            launch_speed_mps = MINIMAL_MISSILE_LAUNCH_SPEED_MPS
            overshoot_window_s = MINIMAL_MISSILE_OVERSHOOT_WINDOW_S
        missile = MissileSimulator.create(
            parent, target, f"m{self._missile_id_counter}",
            guidance_mode=self.missile_guidance_mode,
            config=self.environment_config.missile,
            rng=missile_rng,
            launch_speed_mps=launch_speed_mps,
            overshoot_window_s=overshoot_window_s)
        self._missile_id_counter += 1
        self._missiles_in_flight[missile.uid] = missile
        self._episode_stats["maximum_live_missiles_observed"] = max(
            int(self._episode_stats["maximum_live_missiles_observed"]),
            len(self._missiles_in_flight))
        if launch_quality is not None:
            launch_quality["missile_id"] = missile.uid
            self._launch_quality_records[missile.uid] = launch_quality
            self._launch_quality_step_records.append(dict(launch_quality))
        self._missile_acmi_id[missile.uid] = self._next_missile_acmi_id
        self._next_missile_acmi_id += 1
        self._missile_cooldown[parent.uid] = self.missile_cooldown_frames
        state = self._fire_control_states.setdefault(parent.uid, FireControlState())
        state.detection_state = "launched"
        state.last_launch_frame = int(self._physics_frame)
        state.transition_reason = "automatic_launch"
        parent.num_left_missiles = max(0, parent.num_left_missiles - 1)  # fire-for-effect tracking (capacity 999)
        self._missile_launch_counts[parent.uid] += 1

    def _finalize_launch_quality_record(self, missile: MissileSimulator) -> None:
        """Attach missile termination diagnostics to its launch snapshot."""

        record = self._launch_quality_records.get(missile.uid)
        if record is None or record.get("termination_reason"):
            return
        raw_reason = missile._termination_reason or ("hit" if missile.is_success else "unknown")
        if raw_reason in ("hit", "timeout", "kill_cooldown_blocked", "multi_kill_blocked"):
            reason = raw_reason
        elif raw_reason in ("p_hit_fail", "low_speed", "overshoot", "target_dead"):
            reason = "miss"
        else:
            reason = "unknown"
        target_alive = ""
        if missile.target_aircraft is not None:
            target_alive = bool(missile.target_aircraft.is_alive)
        launch_step = record.get("launch_step", record.get("current_step", self.current_step))
        try:
            step_delta = int(self.current_step) - int(launch_step)
        except Exception:
            step_delta = ""
        record.update({
            "raw_termination_reason": raw_reason,
            "termination_reason": reason,
            "is_success": bool(missile.is_success),
            "flight_time_sec": float(getattr(missile, "_t", _nan_float())),
            "termination_step": int(self.current_step),
            "step_delta": step_delta,
            "target_alive_at_termination": target_alive,
        })
        self._launch_quality_done_step_records.append(dict(record))

    def _update_missiles(self):
        """Advance all in-flight missiles and process hit/miss events.

        Optional legacy anti-burst gates can block post-hit kills for debugging,
        but both are disabled by default because the paper specifies launch
        interval/deconfliction rather than a post-hit kill cooldown.
        """
        for mid, missile in list(self._missiles_in_flight.items()):
            was_done_before = missile.is_done
            if not missile.is_done:
                missile.run()
            # Record termination reason on first frame the missile becomes done
            if missile.is_done and not was_done_before:
                team = "red" if missile._parent_id.startswith("red") else "blue"
                reason = missile._termination_reason or "unknown"
                self._missile_term_reasons[team][reason] = \
                    self._missile_term_reasons[team].get(reason, 0) + 1
            if missile.is_success and not missile._kill_rewarded:
                shooter_id = missile._parent_id

                # ---- Kill-cooldown gate ----
                # Shooter has scored a kill too recently → override to MISS.
                if self.enable_kill_cooldown_gate and shooter_id in self._agents_deny_kill:
                    missile._status = MissileSimulator.MISS
                    missile._termination_reason = "kill_cooldown_blocked"
                    # Reverse the shotdown that missile.run() applied
                    if missile.target_aircraft is not None:
                        missile.target_aircraft._status = AircraftSimulator.ALIVE
                    self._finalize_launch_quality_record(missile)
                    continue

                # ---- Single-target gate (AOE prevention) ----
                # An agent may score at most 1 kill per env step.  If the same
                # shooter already killed a different target this step, block
                # any further kills.
                if (self.enable_single_kill_per_step_gate
                        and self._step_kill_count.get(shooter_id, 0) >= 1):
                    missile._status = MissileSimulator.MISS
                    missile._termination_reason = "multi_kill_blocked"
                    if missile.target_aircraft is not None:
                        missile.target_aircraft._status = AircraftSimulator.ALIVE
                    self._finalize_launch_quality_record(missile)
                    continue

                # ---- Kill accepted ----
                missile._kill_rewarded = True
                self._last_kill_step[shooter_id] = self.current_step
                self._step_kill_count[shooter_id] = 1
                # Record death reason (only first death sticks)
                target_id = missile._target_id
                if target_id not in self._death_reasons:
                    self._death_reasons[target_id] = "Missile_Kill"
            if missile.is_done and not was_done_before:
                self._finalize_launch_quality_record(missile)

    def _update_overload_timers(self):
        """Track how long each aircraft has been above the G-limit."""
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue
            try:
                nx = abs(sim.get_property_value("accelerations/n-pilot-x-norm"))
                ny = abs(sim.get_property_value("accelerations/n-pilot-y-norm"))
                nz = abs(sim.get_property_value("accelerations/n-pilot-z-norm"))
                g_load = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
            except Exception:
                g_load = 0.0

            if g_load > self.OVERLOAD_G_LIMIT:
                self._overload_timers[aid] += self.physics_dt
                self._aircraft_diagnostics[aid]["over_g_frames"] += 1
            else:
                self._overload_timers[aid] = max(0.0, self._overload_timers[aid] - self.physics_dt)
            self._aircraft_diagnostics[aid]["maximum_load_g_seen"] = max(
                self._aircraft_diagnostics[aid]["maximum_load_g_seen"], float(g_load))

    # ------------------------------------------------------------------
    #  Termination checks
    # ------------------------------------------------------------------

    def _check_crash_terminations(self):
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue

            crashed = False
            reason = None

            alt = float(sim.get_geodetic()[2])
            pos = np.asarray(sim.get_position(), dtype=np.float64)
            if alt < self.arena_altitude_min_m:
                sim.crash()
                crashed = True
                reason = "Crash_LowAlt"
            elif self.is_paper_minimal and alt > self.arena_altitude_max_m:
                sim.crash()
                crashed = True
                reason = "Crash_HighAlt"
            elif self.is_paper_minimal and (
                    abs(pos[0]) > self.arena_half_width_m
                    or abs(pos[1]) > self.arena_half_width_m):
                sim.crash()
                crashed = True
                reason = "Crash_BattleVolume"
            elif self.is_paper_minimal:
                # Other tactical death categories are disabled in this profile.
                continue
            elif alt > self.arena_altitude_max_m:
                sim.crash()
                crashed = True
                reason = "Crash_HighAlt"
            elif abs(pos[0]) > self.arena_half_width_m or abs(pos[1]) > self.arena_half_width_m:
                sim.crash()
                crashed = True
                reason = "Crash_BattleVolume"
            elif (self.pid_profile != "paper"
                  and self._overload_timers[aid] > self.OVERLOAD_TIME_LIMIT):
                sim.crash()
                crashed = True
                reason = "Crash_OverG"
            else:
                try:
                    extreme = sim.get_property_value("detect/extreme-state")
                    if extreme:
                        sim.crash()
                        crashed = True
                        reason = "Crash_Extreme"
                except Exception:
                    pass

            if crashed:
                self._crashed_this_step.add(aid)
                if aid not in self._death_reasons:
                    self._death_reasons[aid] = reason
                # Crash reduces N_red or N_blue → penalised via r_end = 30×(ΔN)
                # in the step that the round ends.  No separate crash penalty needed.

    def _get_terminated(self) -> dict:
        blue_all_dead = all(not s.is_alive for s in self.blue_planes.values())
        red_all_dead = all(not s.is_alive for s in self.red_planes.values())
        round_over = blue_all_dead or red_all_dead

        terminated = {}
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            agent_dead = sim is not None and not sim.is_alive
            terminated[aid] = agent_dead or round_over
        return terminated

    def _get_truncated(self) -> dict:
        truncated = self.current_step >= self.max_steps or (
            self.is_paper_minimal and self._invalid_numerical_episode)
        return {aid: truncated for aid in self.agent_ids}

    # ------------------------------------------------------------------
    #  Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> tuple[dict, dict]:
        """Per-agent reward (paper §2.5, eq 15–23).

        r_i = ω_θ·r_θ + ω_φ·r_φ + ω_V·r_V + ω_h·r_h + ω_b·r_b + ω_adv·r_adv + r_end

        Weights (paper Table 4):
          ω_θ=0.01  ω_φ=0.002  ω_h=0.04  ω_b=0.04  ω_V=0.02  ω_adv=0.15

        Terminal (eq 23):  r_end = 30×(N_team − N_enemy) if round over, else 0.
        r_end is a GLOBAL team reward (paper eq 23 + joint reward r_R = Σ r_i + r_end).
        It MUST be divided equally among all teammates so that sum(r_end across team)
        equals the raw team-level value — NOT N_team × the raw value.
        Crash penalty:     r_death = −10 injected on the frame of LowAlt / OverG
                           death, so PPO can causally link the fatal action to death.
        """
        if self.reward_mode in ("paper_joint", "paper_minimal_joint_v1"):
            return self._compute_paper_joint_rewards()

        n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
        n_red_alive = sum(1 for s in self.red_planes.values() if s.is_alive)
        round_over = (n_blue_alive == 0 or n_red_alive == 0
                      or self.current_step >= self.max_steps)

        # Paper eq.23 defines a team-level terminal reward. This environment
        # returns per-agent rewards, so the team-level value is shared across
        # teammates and sums back to the paper's rend. This avoids multiplying
        # terminal reward by the number of agents when team size changes.
        raw_r_end_red  = 30.0 * (n_red_alive - n_blue_alive)
        raw_r_end_blue = 30.0 * (n_blue_alive - n_red_alive)

        rewards = {}
        components = {}
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                components[aid] = {}
                r_death = -10.0 if aid in self._crashed_this_step else 0.0
                if round_over:
                    if aid.startswith("blue"):
                        r_end = raw_r_end_blue / self.max_num_blue
                    else:
                        r_end = raw_r_end_red / self.max_num_red
                    rewards[aid] = r_end + r_death
                    components[aid]["r_end"] = float(r_end)
                    if r_death != 0.0:
                        components[aid]["r_death"] = float(r_death)
                else:
                    rewards[aid] = r_death
                    if r_death != 0.0:
                        components[aid]["r_death"] = float(r_death)
                continue

            # A. Flight status penalties (raw, before weight)
            r_theta  = self._pitch_penalty(sim)
            r_phi    = self._roll_penalty(sim)
            r_V      = self._speed_penalty(sim)
            r_alt    = self._altitude_reward(sim)
            r_bound  = self._boundary_penalty(sim)
            # B. Situation coupling reward (raw)
            r_adv = self._situation_reward(sim)

            # C. Win-lose reward (terminal only) — team-level, per-agent share
            if round_over:
                if aid.startswith("blue"):
                    r_end = raw_r_end_blue / self.max_num_blue
                else:
                    r_end = raw_r_end_red / self.max_num_red
            else:
                r_end = 0.0

            # D. Weighted components (paper Table 4)
            w_pitch = 0.01 * r_theta
            w_roll  = 0.002 * r_phi
            w_vel   = 0.02 * r_V
            w_alt   = 0.04 * r_alt
            w_bound = 0.04 * r_bound
            w_adv   = 0.15 * r_adv

            rewards[aid] = (w_pitch + w_roll + w_vel + w_alt + w_bound
                          + w_adv + r_end)

            components[aid] = {
                "r_pitch": float(w_pitch),
                "r_roll":  float(w_roll),
                "r_alt":   float(w_alt),
                "r_bound": float(w_bound),
                "r_vel":   float(w_vel),
                "r_adv":   float(w_adv),

                "r_end":   float(r_end),
                "r_death": 0.0,
            }
        return rewards, components

    def _compute_paper_joint_rewards(self) -> tuple[dict, dict]:
        """Return the paper team-joint reward identically to every teammate."""
        if self.is_paper_minimal and self._invalid_numerical_episode:
            rewards = {aid: 0.0 for aid in self.agent_ids}
            components = {
                aid: {
                    "r_pitch": 0.0, "r_roll": 0.0, "r_alt": 0.0,
                    "r_bound": 0.0, "r_vel": 0.0, "r_adv": 0.0,
                    "r_end": 0.0, "joint_reward": 0.0,
                    "invalid_numerical_episode": True,
                }
                for aid in self.agent_ids
            }
            self._reward_summary_step = {
                "red_local_reward_sum": 0.0,
                "blue_local_reward_sum": 0.0,
                "red_team_terminal_reward": 0.0,
                "blue_team_terminal_reward": 0.0,
                "red_joint_reward": 0.0,
                "blue_joint_reward": 0.0,
                "reward_version": "paper_literal_minimal_unspecified_v1",
                "reward_mode": self.reward_mode,
                "invalid_numerical_episode": True,
            }
            return rewards, components
        n_blue_alive = sum(1 for s in self.blue_planes.values() if s.is_alive)
        n_red_alive = sum(1 for s in self.red_planes.values() if s.is_alive)
        round_over = (n_blue_alive == 0 or n_red_alive == 0
                      or self.current_step >= self.max_steps)

        terminal_coefficient = float(
            self.environment_config.reward.terminal_coefficient.value)
        terminal_red = terminal_coefficient * (
            n_red_alive - n_blue_alive) if round_over else 0.0
        terminal_blue = -terminal_red
        components: dict[str, dict] = {}
        local_rewards: dict[str, float] = {}

        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                raw_pitch = raw_roll = raw_vel = raw_alt = raw_bound = raw_adv = 0.0
            else:
                raw_pitch = self._pitch_penalty(sim)
                raw_roll = self._roll_penalty(sim)
                raw_vel = self._speed_penalty(sim)
                raw_alt = self._altitude_reward(sim)
                raw_bound = self._boundary_penalty(sim)
                raw_adv = self._situation_reward(sim)

            weights = self.environment_config.reward.weights.value
            weighted = {
                "r_pitch": weights["pitch"] * raw_pitch,
                "r_roll": weights["roll"] * raw_roll,
                "r_alt": weights["altitude"] * raw_alt,
                "r_bound": weights["boundary"] * raw_bound,
                "r_vel": weights["speed"] * raw_vel,
                "r_adv": weights["advantage"] * raw_adv,
            }
            local_reward = float(sum(weighted.values()))
            local_rewards[aid] = local_reward
            components[aid] = {
                "raw_r_pitch": float(raw_pitch),
                "raw_r_roll": float(raw_roll),
                "raw_r_alt": float(raw_alt),
                "raw_r_bound": float(raw_bound),
                "raw_r_vel": float(raw_vel),
                "raw_r_adv": float(raw_adv),
                **{key: float(value) for key, value in weighted.items()},
                "local_reward": local_reward,
                "r_death": 0.0,
            }

        red_local_sum = float(sum(local_rewards[aid] for aid in self.red_ids))
        blue_local_sum = float(sum(local_rewards[aid] for aid in self.blue_ids))
        red_joint = red_local_sum + terminal_red
        blue_joint = blue_local_sum + terminal_blue
        rewards = {
            aid: float(blue_joint if aid.startswith("blue") else red_joint)
            for aid in self.agent_ids
        }

        self._reward_summary_step = {
            "red_local_reward_sum": red_local_sum,
            "blue_local_reward_sum": blue_local_sum,
            "red_team_terminal_reward": float(terminal_red),
            "blue_team_terminal_reward": float(terminal_blue),
            "red_joint_reward": float(red_joint),
            "blue_joint_reward": float(blue_joint),
            "reward_version": (
                "paper_literal_minimal_unspecified_v1"
                if self.is_paper_minimal else REWARD_VERSION),
            "reward_mode": self.reward_mode,
        }
        self._episode_stats["EpisodeRedJointReturn"] += red_joint
        self._episode_stats["EpisodeBlueJointReturn"] += blue_joint
        self._episode_stats["EpisodeRedLocalRewardSum"] += red_local_sum
        self._episode_stats["EpisodeBlueLocalRewardSum"] += blue_local_sum
        self._episode_stats["EpisodeRedTerminalReward"] += terminal_red
        self._episode_stats["EpisodeBlueTerminalReward"] += terminal_blue
        self._episode_stats["EpisodeLength"] = int(self.current_step)
        for aid in self.agent_ids:
            is_blue = aid.startswith("blue")
            components[aid].update({
                "r_end": float(terminal_blue if is_blue else terminal_red),
                "joint_reward": float(blue_joint if is_blue else red_joint),
                **self._reward_summary_step,
            })
        return rewards, components

    # ------------------------------------------------------------------
    #  Flight status penalties (paper formulas)
    # ------------------------------------------------------------------

    def _pitch_penalty(self, sim: AircraftSimulator) -> float:
        """r_θ: penalty for |pitch| > π/4, severe at > π/3."""
        theta = abs(sim.get_rpy()[1])
        if theta > np.pi / 3:
            return -1.0
        if theta > np.pi / 4:
            return -(theta / np.pi - 0.25) / 12.0
        return 0.0

    def _roll_penalty(self, sim: AircraftSimulator) -> float:
        """r_phi: paper eq.16 dual condition for excessive roll and pitch."""
        rpy = sim.get_rpy()
        phi = abs(rpy[0])
        theta = abs(rpy[1])
        if phi > np.pi / 4 and theta > np.pi / 4:
            return -(phi / np.pi - 0.25) * (4.0 / 3.0)
        return 0.0

    def _speed_penalty(self, sim: AircraftSimulator) -> float:
        """r_V: paper eq (19) — penalty for low speed (Mach < 0.3)."""
        v = np.linalg.norm(sim.get_velocity())
        mach = v / float(self.environment_config.reward.mach_reference_mps.value)
        if mach < 0.2:
            return -1.0
        if mach < 0.3:
            return -(0.3 - mach) / 0.1
        return 0.0

    # ------------------------------------------------------------------
    #  Situation coupling reward (paper Formula B)
    # ------------------------------------------------------------------

    def _situation_reward(self, ego_sim: AircraftSimulator) -> float:
        """r_adv^i = Σ_j (1.0 × Ta_i^j × Td_i^j - 0.8 × Ta_j^i × Td_j^i).

        q_ij and q_ji use each observer's 3D velocity-to-LOS angle.  Launch
        AO/TA remains a separate fire-control geometry diagnostic.
        """
        ego_pos = ego_sim.get_position()
        ego_vel = ego_sim.get_velocity()

        enemies = self.red_planes if ego_sim.color == "Blue" else self.blue_planes
        total = 0.0
        for enemy_sim in enemies.values():
            if not enemy_sim.is_alive:
                continue
            enemy_pos = enemy_sim.get_position()
            enemy_vel = enemy_sim.get_velocity()

            q_ij = compute_velocity_q_los(ego_pos, ego_vel, enemy_pos)
            q_ji = compute_velocity_q_los(enemy_pos, enemy_vel, ego_pos)
            d_ij = compute_3d_range(ego_pos, enemy_pos)
            d_ji = compute_3d_range(enemy_pos, ego_pos)

            Ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
            Td_ij = td_distance_advantage(d_ij)
            Ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))
            Td_ji = td_distance_advantage(d_ji)

            total += 1.0 * Ta_ij * Td_ij - 0.8 * Ta_ji * Td_ji

        return total

    def _altitude_reward(self, sim: AircraftSimulator) -> float:
        """Paper eq.17-style pairwise relative altitude reward."""
        alt_ego = sim.get_geodetic()[2]
        enemies = self.red_planes if sim.color == "Blue" else self.blue_planes
        enemy_alts = [s.get_geodetic()[2] for s in enemies.values() if s.is_alive]
        if not enemy_alts:
            return 0.0

        if self.is_paper_minimal:
            return altitude_reward_pairwise_mean_eq17(
                alt_ego, enemy_alts, config=self.altitude_reward_config)
        return altitude_reward_pairwise_sum_eq17(
            alt_ego, enemy_alts, config=self.altitude_reward_config)

    def _boundary_penalty(self, sim: AircraftSimulator) -> float:
        """Horizontal battlefield boundary penalty.

        Paper eq.18: return a fixed -10 if either |x| or |y| exceeds 4e4.
        The penalty is not accumulated per axis.
        """
        pos = sim.get_position()
        x, y = pos[0], pos[1]
        if (abs(x) > self.reward_boundary_half_width_m
                or abs(y) > self.reward_boundary_half_width_m):
            return -10.0
        return 0.0

    # ------------------------------------------------------------------
    #  Observation normalisation
    # ------------------------------------------------------------------

    def _normalize_obs_vec(self, raw: np.ndarray) -> np.ndarray:
        """Scale an 11-dim entity vector to roughly [-1, 1] for NN training.

        Raw layout (body-frame, paper Table 2):
          [Δx_body, Δy_body, Δz_body, AO_body, TA_body, R, V_tgt,
           sin(φ), cos(φ), sin(θ), cos(θ)]
        idx:    0        1        2        3        4     5    6      7      8      9     10

        AO_body ∈ [−π, π]  (+ right, − left)    — body-frame signed angle-off
        TA_body ∈ [0, π]    (unsigned)            — body-frame target aspect

        Returns zeros unchanged (dead / non-existent entity).
        """
        if not np.any(raw):
            return raw

        out = raw.copy()
        # Position deltas — horizontal / vertical
        out[0] = raw[0] / self.BATTLEFIELD_HALF_SIZE       # Δn  ∈ [−1, 1]
        out[1] = raw[1] / self.BATTLEFIELD_HALF_SIZE       # Δe  ∈ [−1, 1]
        out[2] = raw[2] / self.BATTLEFIELD_ALTITUDE_MAX    # Δu  ∈ [−1, 1]
        # AO_signed — radians → [−1, 1]  (sign tells turn direction)
        out[3] = raw[3] / np.pi                            # AO  ∈ [−1, 1]
        out[4] = raw[4] / np.pi                            # TA  ∈ [0, 1]
        # Range — metres → [0, ~1]
        out[5] = raw[5] / (self.BATTLEFIELD_HALF_SIZE * 2.0)  # R  ∈ [0, ~1]
        # Target speed — m/s → [0, 1]
        out[6] = raw[6] / self.MAX_SPEED                   # V_tgt ∈ [0, 1]
        # idx 7-10: sin/cos already in [-1, 1] — no scaling needed
        return out

    # ------------------------------------------------------------------
    #  Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        obs = {}
        for aid in self.agent_ids:
            obs[aid] = self._get_agent_obs(aid)
        return obs

    def _get_agent_obs(self, agent_id: str) -> dict:
        if self.obs_mode == "paper_strict":
            return self._get_agent_obs_paper_strict(agent_id)

        sim = self._get_sim(agent_id)
        alive = sim is not None and sim.is_alive
        color = "Blue" if agent_id.startswith("blue") else "Red"

        _ego_slots, ally_slots, enemy_slots = ordered_entity_slots(self, agent_id)
        ally_sims = [slot[1] for slot in ally_slots]
        enemy_sims = [slot[1] for slot in enemy_slots]

        # ---- ego_state (self-observation: delta=0, frame-independent) ----
        if alive:
            ego_pos = sim.get_position()          # (north, east, up) — m
            ego_vel = sim.get_velocity()          # (vn, ve, vu)     — m/s
            ego_rpy = sim.get_rpy()               # (φ, θ, ψ)        — rad
            raw_ego = _make_entity_vec(ego_pos, ego_vel, ego_pos, ego_vel, ego_rpy, True)
            ego_state = self._normalize_obs_vec(raw_ego)

            # Pre-compute body-frame rotation matrix and ego body-frame velocity
            R_BI = PIDController.ned_to_body_matrix(
                float(ego_rpy[0]), float(ego_rpy[1]), float(ego_rpy[2]))
            ego_vel_ned = np.array([ego_vel[0], ego_vel[1], -ego_vel[2]], dtype=np.float64)
            ego_vel_body = R_BI @ ego_vel_ned
            # Pseudo-NED for _make_entity_vec: body x→north, body y→east, −body z→up
            ego_pos_bf = np.zeros(3, dtype=np.float64)
            ego_vel_bf = np.array([ego_vel_body[0], ego_vel_body[1], -ego_vel_body[2]],
                                  dtype=np.float64)
        else:
            ego_state = np.zeros(11, dtype=np.float32)

        # ---- ally_states ----
        max_allies = len(ally_sims)

        ally_vecs = np.zeros((max_allies, 11), dtype=np.float32)
        if alive:
            for j, ally in enumerate(ally_sims):
                if not ally.is_alive:
                    continue
                raw_ally = self._build_body_frame_entity(
                    ego_pos, ego_pos_bf, ego_vel_bf, R_BI,
                    ally.get_position(), ally.get_velocity(), ally.get_rpy(),
                    ally.is_alive,
                )
                ally_vecs[j] = self._normalize_obs_vec(raw_ally)

        # ---- enemy_states (partial observability per paper) ----
        max_enemies = len(enemy_sims)

        enemy_vecs = np.zeros((max_enemies, 11), dtype=np.float32)
        if alive:
            for j, enemy in enumerate(enemy_sims):
                if not enemy.is_alive:
                    continue

                if self._is_detected_by_radar(sim, enemy):
                    # ---- Full track (within FOV + detection range) ----
                    raw_enemy = self._build_body_frame_entity(
                        ego_pos, ego_pos_bf, ego_vel_bf, R_BI,
                        enemy.get_position(), enemy.get_velocity(), enemy.get_rpy(),
                        True,
                    )
                    enemy_vecs[j] = self._normalize_obs_vec(raw_enemy)
                else:
                    # ---- Blind zone: AWACS gives coarse body-frame position ----
                    enm_pos = enemy.get_position()
                    dn_ned = enm_pos[0] - ego_pos[0]
                    de_ned = enm_pos[1] - ego_pos[1]
                    dd_ned = -enm_pos[2] - (-ego_pos[2])
                    delta_ned = np.array([dn_ned, de_ned, dd_ned], dtype=np.float64)
                    delta_body = R_BI @ delta_ned
                    dx, dy, dz = float(delta_body[0]), float(delta_body[1]), float(delta_body[2])
                    R_b = float(np.linalg.norm([dx, dy, dz]))

                    # Body-frame signed AO: arctan2(dy, dx)
                    ao_body = float(np.arctan2(dy, dx + 1e-12))
                    ao_norm = float(ao_body / np.pi)

                    enemy_vecs[j] = np.array([
                        dx / self.BATTLEFIELD_HALF_SIZE,              # Δx_body
                        dy / self.BATTLEFIELD_HALF_SIZE,              # Δy_body
                        (-dz) / self.BATTLEFIELD_ALTITUDE_MAX,         # Δup_body
                        ao_norm, 0.0,                                  # AO_body, TA=0
                        R_b / (self.BATTLEFIELD_HALF_SIZE * 2.0),     # R norm
                        0.0,                                           # V_tgt=0
                        0.0, 0.0, 0.0, 0.0,                            # attitude masked
                    ], dtype=np.float32)

        # Canonical mask: 1=alive/valid, 0=dead/invalid.
        alive_mask = slot_aligned_alive_mask(self, agent_id)

        # ---- missile_warning ----
        mw = 0.0
        if alive and sim.check_missile_warning() is not None:
            mw = 1.0
        missile_warning = np.array([mw], dtype=np.float32)

        # ---- altitude / velocity (raw NED, for rule-based safety checks) ----
        alt_m = sim.get_geodetic()[2] if alive else 0.0
        altitude = np.array([alt_m], dtype=np.float32)
        vel = sim.get_velocity() if alive else np.zeros(3)
        velocity = np.array([vel[0], vel[1], vel[2]], dtype=np.float32)

        return {
            "ego_state": ego_state,
            "ally_states": ally_vecs,
            "enemy_states": enemy_vecs,
            "alive_mask": alive_mask,
            "death_mask": alive_mask.copy(),
            "missile_warning": missile_warning,
            "altitude": altitude,
            "velocity": velocity,
        }

    def _get_agent_obs_paper_strict(self, agent_id: str) -> dict:
        """Build Table 1 / Table 2 10-dim observations for reset/step."""
        sim = self._get_sim(agent_id)
        alive = sim is not None and sim.is_alive
        _ego_slots, ally_slots, enemy_slots = ordered_entity_slots(self, agent_id)
        ally_sims = [slot[1] for slot in ally_slots]
        enemy_sims = [slot[1] for slot in enemy_slots]

        if alive:
            ego_state, _meta = extract_self_state_with_meta(sim)
        else:
            ego_state = np.zeros(10, dtype=np.float32)

        max_allies = len(ally_sims)
        max_enemies = len(enemy_sims)

        ally_vecs = np.zeros((max_allies, 10), dtype=np.float32)
        enemy_vecs = np.zeros((max_enemies, 10), dtype=np.float32)
        if alive:
            for j, ally in enumerate(ally_sims):
                if ally.is_alive:
                    ally_vecs[j] = extract_relative_state(
                        sim, ally, radar_detected=True)
            for j, enemy in enumerate(enemy_sims):
                if enemy.is_alive:
                    track = self._get_sensor_track(sim, enemy)
                    enemy_vecs[j] = extract_relative_state(
                        sim, enemy,
                        radar_detected=(track.source == "radar_full"),
                        target_position_override=track.position_estimate)

        alive_mask = slot_aligned_alive_mask(self, agent_id)
        mw = 1.0 if alive and sim.check_missile_warning() is not None else 0.0
        missile_warning = np.array([mw], dtype=np.float32)
        alt_m = sim.get_geodetic()[2] if alive else 0.0
        altitude = np.array([alt_m], dtype=np.float32)
        vel = sim.get_velocity() if alive else np.zeros(3)
        velocity = np.array([vel[0], vel[1], vel[2]], dtype=np.float32)

        return {
            "ego_state": ego_state.astype(np.float32),
            "ally_states": ally_vecs,
            "enemy_states": enemy_vecs,
            "alive_mask": alive_mask,
            "death_mask": alive_mask.copy(),
            "missile_warning": missile_warning,
            "altitude": altitude,
            "velocity": velocity,
        }

    @staticmethod
    def _build_body_frame_entity(ego_pos_ned, ego_pos_bf, ego_vel_bf, R_BI,
                                  tgt_pos_ned, tgt_vel_ned, tgt_rpy, alive):
        """Build 11-dim entity vector with relative coordinates in ego's body frame.

        Rotates the NED-frame delta into body frame, then expresses the result
        in a pseudo-NED system where body x→north, body y→east, −body z→up.
        This allows ``_make_entity_vec`` (which calls ``get2d_AO_TA_R``) to
        compute AO/TA in the body x-y plane — exactly what paper Table 2 requires.
        """
        if not alive:
            return np.zeros(11, dtype=np.float32)

        # NED delta (north, east, down)
        dn = tgt_pos_ned[0] - ego_pos_ned[0]
        de = tgt_pos_ned[1] - ego_pos_ned[1]
        dd = -tgt_pos_ned[2] - (-ego_pos_ned[2])
        delta_ned = np.array([dn, de, dd], dtype=np.float64)

        # Rotate to body frame: body x=forward, y=right, z=down
        delta_body = R_BI @ delta_ned

        # Target velocity in body frame
        tgt_vn, tgt_ve, tgt_vu = tgt_vel_ned
        tgt_vel_ned_vec = np.array([tgt_vn, tgt_ve, -tgt_vu], dtype=np.float64)
        tgt_vel_body = R_BI @ tgt_vel_ned_vec

        # Express in pseudo-NED: body x→north, body y→east, −body z→up
        tgt_pos_bf = np.array([delta_body[0], delta_body[1], -delta_body[2]],
                              dtype=np.float64)
        tgt_vel_bf = np.array([tgt_vel_body[0], tgt_vel_body[1], -tgt_vel_body[2]],
                              dtype=np.float64)

        return _make_entity_vec(ego_pos_bf, ego_vel_bf,
                                tgt_pos_bf, tgt_vel_bf, tgt_rpy, True)

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = {}
        death_categories = {
            "Missile_Kill": "missile_hit",
            "Crash_LowAlt": "low_altitude",
            "Crash_HighAlt": "high_altitude",
            "Crash_BattleVolume": "horizontal_out_of_bounds",
            "Crash_NumericalEnvelope": "overspeed_instability",
            "Crash_NumericalLoad": "over_g_instability",
            "Crash_NonFinite": "invalid_jsbsim_state",
            "Crash_Extreme": "numerical_instability",
            "Crash_OverG": "over_g_instability",
        }
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            # Return per-step delta and reset counter so callers can safely
            # accumulate without double-counting across env steps.
            delta = self._missile_launch_counts.get(aid, 0)
            self._missile_launch_counts[aid] = 0
            info[aid] = {
                "alive": sim is not None and sim.is_alive,
                "step": self.current_step,
                "missiles_fired_this_step": delta,
                "missiles_left": sim.num_left_missiles if sim is not None else 0,
                "death_reason": self._death_reasons.get(aid, None),
                "death_category": death_categories.get(
                    self._death_reasons.get(aid),
                    "unknown" if self._death_reasons.get(aid) else ""),
                **dict(self._aircraft_diagnostics.get(aid, {})),
                "fire_control": self._fire_control_states[aid].snapshot(),
                "evasion": dict(self._evasion_diagnostics.get(aid, {})),
            }
            if sim is not None and sim.is_alive:
                _missile, mws_diag = sim.get_missile_warning_diagnostic()
                info[aid]["mws_threat"] = mws_diag
            # Merge weighted reward-component breakdown for diagnostics
            if reward_components and aid in reward_components:
                info[aid].update(reward_components[aid])
        # Attach accumulated missile termination stats (read-only snapshot)
        info["__missile_term__"] = {
            team: dict(reasons) for team, reasons in self._missile_term_reasons.items()
        }
        info["__launch_diag__"] = {
            team: dict(vals) for team, vals in self._launch_diag_step.items()
        }
        info["__launch_quality_step__"] = [
            dict(record) for record in self._launch_quality_step_records
        ]
        info["__launch_quality_done__"] = [
            dict(record) for record in self._launch_quality_done_step_records
        ]
        info["__environment_config__"] = dict(self._environment_config_snapshot)
        info["__sensor_diagnostics__"] = [
            dict(row) for row in self._sensor_diagnostics_step]
        info["__reward_summary__"] = dict(self._reward_summary_step)
        info["__blue_policy_diag__"] = (
            self.blue_policy_controller.snapshot_episode_diagnostics())
        n_red_alive = sum(int(s.is_alive) for s in self.red_planes.values())
        n_blue_alive = sum(int(s.is_alive) for s in self.blue_planes.values())
        timeout = self.current_step >= self.max_steps
        invalid = bool(self.is_paper_minimal and self._invalid_numerical_episode)
        ended = timeout or invalid or n_red_alive == 0 or n_blue_alive == 0
        winner = ("red" if n_red_alive > n_blue_alive else
                  "blue" if n_blue_alive > n_red_alive else "draw")
        if not ended:
            winner = ""
        if invalid:
            winner = ""
        end_reason = ("invalid_numerical_episode" if invalid else
                      "timeout" if timeout else "red_eliminated" if n_red_alive == 0
                      else "blue_eliminated" if n_blue_alive == 0 else "")
        info["__episode__"] = {
            **dict(self._episode_stats),
            "PerStepRedJointRewardMean": (
                self._episode_stats.get("EpisodeRedJointReturn", 0.0)
                / max(self.current_step, 1)),
            "PerStepBlueJointRewardMean": (
                self._episode_stats.get("EpisodeBlueJointReturn", 0.0)
                / max(self.current_step, 1)),
            "episode_end_reason": end_reason,
            "winner": winner,
            "red_alive": n_red_alive,
            "blue_alive": n_blue_alive,
            "timeout": bool(timeout),
            "invalid_numerical_episode": invalid,
            "invalid_numerical_reasons": list(self._invalid_numerical_reasons),
            "battle_volume_violation": any(
                reason in ("Crash_BattleVolume", "Crash_HighAlt", "Crash_LowAlt")
                for reason in self._death_reasons.values()),
        }
        return info

    # ------------------------------------------------------------------
    #  Radar / Sensor model (paper: partial observability)
    # ------------------------------------------------------------------

    def _compute_radar_max_range(self, TA: float) -> float:
        """Compatibility helper for the paper fourth-root range equation."""
        from my_uav_env.sensors import bilinear_rcs_m2
        rcs = bilinear_rcs_m2(
            float(TA), 0.0, self.environment_config.rcs.azimuth_grid_deg.value,
            self.environment_config.rcs.elevation_grid_deg.value,
            self.environment_config.rcs.table_m2.value)
        return float(self.environment_config.rcs.range_constant.value
                     * np.power(max(rcs, 0.0), 0.25))

    def _is_detected_by_electro_optical(
            self, observer_sim: AircraftSimulator,
            target_sim: AircraftSimulator) -> bool:
        """Minimal EO is deterministic and depends only on 3D range."""
        distance = compute_3d_range(
            observer_sim.get_position(), target_sim.get_position())
        if self.is_paper_minimal:
            return bool(distance <= self.environment_config.electro_optical.maximum_range_m.value)
        q_los = compute_body_x_q_los(
            observer_sim.get_position(), observer_sim.get_rpy(),
            target_sim.get_position())
        return bool(
            distance < self.environment_config.electro_optical.maximum_range_m.value
            and q_los < self.missile_launch_ao_thresh)

    def _is_detected_by_radar(self, ego_sim: AircraftSimulator,
                              enemy_sim: AircraftSimulator) -> bool:
        """True if *enemy_sim* is within ego's radar FOV AND detection range.

        Radar FOV (paper):
          - Azimuth: ±60°  (120° forward sector)
          - Elevation: [-10°, +32°] in the aircraft body frame

        Detection range is RCS-dependent (see ``_compute_radar_max_range``).

        Radar CANNOT detect missiles — only aircraft.
        """
        if isinstance(enemy_sim, MissileSimulator):
            return False
        diag = radar_diagnostic(
            ego_sim, enemy_sim, self.environment_config.radar,
            self.environment_config.rcs)
        self._sensor_diagnostics_step.append({
            "observer_id": ego_sim.uid, "target_id": enemy_sim.uid, **diag})
        return bool(diag["radar_detected"])

    def _get_sensor_track(self, observer_sim, target_sim) -> SensorTrack:
        """Return radar-full or seeded AWACS-coarse track for Table 2."""
        now = self._physics_frame * self.physics_dt
        if self._is_detected_by_radar(observer_sim, target_sim):
            return SensorTrack("radar_full", target_sim.uid,
                               np.asarray(target_sim.get_position(), dtype=np.float64),
                               now, 0.0, 1.0, True, True, True)
        if self.is_paper_minimal:
            # Deterministic per-decision AWACS/RWS fallback: no noise, delay,
            # stale-track memory, hold, or random loss.
            rws_detected = self._is_detected_by_radar(target_sim, observer_sim)
            return SensorTrack(
                "rws_awacs_fused" if rws_detected else "awacs_coarse",
                target_sim.uid,
                np.asarray(target_sim.get_position(), dtype=np.float64),
                now, 0.0, 0.65 if rws_detected else 0.5,
                False, False, True)
        key = (observer_sim.uid, target_sim.uid)
        previous = self._sensor_tracks.get(key)
        rws_detected = self._is_detected_by_radar(target_sim, observer_sim)
        period = float(self.environment_config.awacs.update_period_s.value)
        hold = float(self.environment_config.awacs.track_hold_s.value)
        if previous is None or now - previous.timestamp >= period:
            true_pos = np.asarray(target_sim.get_position(), dtype=np.float64)
            error = np.array([
                self.np_random.normal(0.0, self.environment_config.awacs.horizontal_error_std_m.value),
                self.np_random.normal(0.0, self.environment_config.awacs.horizontal_error_std_m.value),
                self.np_random.normal(0.0, self.environment_config.awacs.vertical_error_std_m.value),
            ])
            estimate = true_pos + error
            source = "awacs_coarse"
            confidence = 0.5
            if rws_detected:
                relative = estimate - np.asarray(observer_sim.get_position())
                horizontal_range = float(np.hypot(relative[0], relative[1]))
                bearing = float(np.arctan2(relative[1], relative[0]))
                bearing += float(self.np_random.normal(
                    0.0, self.environment_config.rws.bearing_error_std_rad.value))
                estimate[0] = observer_sim.get_position()[0] + horizontal_range * np.cos(bearing)
                estimate[1] = observer_sim.get_position()[1] + horizontal_range * np.sin(bearing)
                source = "rws_awacs_fused"
                confidence = 0.65
            previous = SensorTrack(source, target_sim.uid,
                                   estimate, now, 0.0, confidence,
                                   False, False, True)
            self._sensor_tracks[key] = previous
        age = now - previous.timestamp
        if age > hold:
            return SensorTrack("no_track", target_sim.uid, None,
                               previous.timestamp, age, 0.0, False, False, False)
        return SensorTrack(previous.source, previous.target_id,
                           previous.position_estimate.copy(), previous.timestamp,
                           age, previous.confidence, False, False, True)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_sim(self, agent_id: str):
        if agent_id.startswith("blue"):
            return self.blue_planes.get(agent_id)
        return self.red_planes.get(agent_id)

    def _all_sims(self):
        for sim in self.blue_planes.values():
            yield sim
        for sim in self.red_planes.values():
            yield sim

    def _all_sims_with_ids(self):
        for aid, sim in self.blue_planes.items():
            yield aid, sim
        for aid, sim in self.red_planes.items():
            yield aid, sim

    # ------------------------------------------------------------------
    #  Strict paper observation API (optional, does not affect reset/step)
    # ------------------------------------------------------------------

    def get_strict_entity_observation(self, agent_id: str):
        """Return strict 10-dim entity observation for one agent.

        This is an optional paper-aligned observation API.  It does not affect
        ``reset()`` / ``step()`` outputs or ``observation_space``.

        Returns:
            entities: np.ndarray, shape (N_entities, 10)
            mask:     np.ndarray, shape (N_entities,)
            meta:     dict
        """
        from my_uav_env.alignment.state_extractor import \
            build_strict_paper_entity_observation
        return build_strict_paper_entity_observation(self, agent_id)

    def get_strict_team_observations(self, team: str = "red") -> dict:
        """Return strict 10-dim observations for every agent on a team.

        Args:
            team: ``"red"`` or ``"blue"``.

        Returns:
            dict mapping agent_id → (entities, mask, meta).
        """
        if team not in ("red", "blue"):
            raise ValueError(f"team must be 'red' or 'blue', got {team!r}")
        agent_ids = self.red_ids if team == "red" else self.blue_ids
        result = {}
        for aid in agent_ids:
            result[aid] = self.get_strict_entity_observation(aid)
        return result

    def _make_init_state(self, color: str, index: int) -> dict:
        """Strict paper baseline (Table 4): head-on at exactly 10 km, altitude 20 000 ft.

        No randomization — headings, distance, and altitude are locked to the
        paper specification so the RL agent learns from a reproducible initial
        condition distribution.
        """
        cfg = self.scenario_config
        N = self.max_num_red if color == "Red" else self.max_num_blue
        lon_centre = float(cfg.reference_longitude_deg.value)
        lat_centre = float(cfg.reference_latitude_deg.value)
        formation_spacing_m = float(cfg.formation_spacing_m.value)
        half_distance_m = float(cfg.initial_head_on_range_m.value) / 2.0
        if self.is_paper_minimal:
            centre_lat_rad = np.deg2rad(lat_centre)
            meridional_radius = 6_378_137.0 * (1.0 - 0.00669437999014) / (
                1.0 - 0.00669437999014 * np.sin(centre_lat_rad) ** 2) ** 1.5
            metres_per_lat_degree = (
                meridional_radius + float(cfg.initial_altitude_m.value)) * np.pi / 180.0
            lat_offset_deg = (
                index - (N - 1) / 2.0) * formation_spacing_m / metres_per_lat_degree
            lane_latitude = lat_centre + lat_offset_deg
            lat_rad = np.deg2rad(lane_latitude)
        else:
            lat_offset_deg = (
                index - (N - 1) / 2.0) * formation_spacing_m / 111320.0
            lane_latitude = lat_centre + lat_offset_deg
            lat_rad = np.deg2rad(lat_centre)
        prime_vertical_radius = 6_378_137.0 / np.sqrt(
            1.0 - 0.00669437999014 * np.sin(lat_rad) ** 2)
        radius_at_altitude = (prime_vertical_radius + (
            float(cfg.initial_altitude_m.value) if self.is_paper_minimal else 0.0))
        metres_per_lon_degree = radius_at_altitude * np.cos(lat_rad) * np.pi / 180.0
        half_distance_deg_lon = half_distance_m / metres_per_lon_degree

        if color == "Blue":
            heading = 90.0   # fly east
            lon = lon_centre - half_distance_deg_lon
        else:
            heading = -90.0  # fly west
            lon = lon_centre + half_distance_deg_lon

        return {
            "ic/long-gc-deg": lon,
            "ic/lat-geod-deg": lane_latitude,
            "ic/h-sl-ft": float(cfg.initial_altitude_m.value) / 0.3048,
            "ic/psi-true-deg": heading,
            "ic/u-fps": float(cfg.initial_speed_mps.value) / 0.3048,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }

    def _cleanup_missiles(self):
        done = [mid for mid, m in self._missiles_in_flight.items() if m.is_done]
        for mid in done:
            missile = self._missiles_in_flight.pop(mid)
            missile.detach_references()
            self._engaged_targets.discard(missile._target_id)

    # ------------------------------------------------------------------
    #  Rendering (TacView .acmi export)
    # ------------------------------------------------------------------

    def render(self, filepath: str | None = None):
        """Enable TacView recording for the current episode.

        Call once before ``reset()`` to start recording.  Frames are
        recorded automatically on every ``step()``.  Call ``save_acmi()``
        after the episode to write the .acmi file.

        Args:
            filepath: optional output path; can also be passed to ``save_acmi()``.
        """
        if self._tacview_recorder is None:
            self._tacview_recorder = TacviewLogger()
        if filepath is not None:
            self._acmi_filepath = filepath

    def _render_frame(self):
        """Collect ACMI log lines from all aircraft and missiles."""
        entries: list[dict] = []
        explosions: list[dict] = []

        # Aircraft entries
        for _aid, sim in self._all_sims_with_ids():
            aid = sim.uid
            acmi_id = self._agent_acmi_id[aid]
            lon, lat, alt = sim.get_geodetic()
            roll, pitch, yaw = sim.get_rpy() * (180.0 / np.pi)
            entries.append({
                "acmi_id": acmi_id,
                "lon": lon, "lat": lat, "alt": alt,
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "name": sim.model.upper(),
                "color": sim.color,
                "alive": sim.is_alive,
            })

        # Missile entries
        for mid, missile in self._missiles_in_flight.items():
            acmi_id = self._missile_acmi_id[mid]
            if missile.is_alive:
                lon, lat, alt = missile.get_geodetic()
                roll, pitch, yaw = missile.get_rpy() * (180.0 / np.pi)
                entries.append({
                    "acmi_id": acmi_id,
                    "lon": lon, "lat": lat, "alt": alt,
                    "roll": roll, "pitch": pitch, "yaw": yaw,
                    "name": missile.model.upper(),
                    "color": missile.color,
                    "alive": True,
                })
            elif missile.is_done and not missile.render_explosion:
                missile.render_explosion = True
                if missile.is_success:
                    # True hit — yellow explosion at missile position
                    lon, lat, alt = missile.get_geodetic()
                    explosions.append({
                        "acmi_id": acmi_id,
                        "lon": lon, "lat": lat, "alt": alt,
                        "color": "Yellow",
                        "radius": missile._Rc,
                    })
                # MISS (target dead / timeout / lost lock): no explosion.
                # The missile simply disappears — do not render a misleading
                # 300 m fireball far from the target.

        self._tacview_recorder.record_frame(self._sim_time, entries, explosions)

    def save_acmi(self, filepath: str | None = None):
        """Write recorded frames to an .acmi file and reset the recorder.

        Returns the number of frames written, or 0 if no recorder was active.
        """
        path = filepath or self._acmi_filepath
        if self._tacview_recorder is None or path is None:
            return 0
        n = self._tacview_recorder.frame_count
        self._tacview_recorder.write(path)
        self._tacview_recorder = None
        return n

    def close(self):
        self.blue_policy_controller.clear()
        for sim in self._all_sims():
            sim.close()
        self.blue_planes.clear()
        self.red_planes.clear()
