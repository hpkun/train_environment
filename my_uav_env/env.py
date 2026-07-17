"""
UavCombatEnv: Multi-agent UAV combat environment with Dict observation spaces
for zero-shot scale generalization. Uses JSBSim for flight dynamics and PID
controllers to convert high-level tactical commands to control-surface inputs.
"""
from __future__ import annotations

import copy
import logging
from collections import deque
import numpy as np
import gymnasium

from configs.brma_mappo_paper_spec import PaperEnvironmentConfig, paper_value
from configs.paper_3v3_spec import (
    AIRCRAFT_ENVELOPE_FRAMES,
    CATASTROPHIC_G,
    COARSE_ALTITUDE_GRID_M,
    COARSE_HORIZONTAL_GRID_M,
    MISSILE_LAUNCH_SPEED_MPS,
    MISSILE_OVERSHOOT_DISTANCE_HYSTERESIS_M,
    MISSILE_OVERSHOOT_WINDOW_S,
    MISSILE_POSITIVE_CLOSING_THRESHOLD_MPS,
    PAPER_BLUE_POLICY_PROFILE,
    PAPER_ENVIRONMENT_CONFIG,
    PAPER_ENVIRONMENT_PROFILE,
    PAPER_MISSILE_GUIDANCE_MODE,
    PAPER_PID_PROFILE,
    PAPER_REWARD_MODE,
    PERSISTENT_EXTREME_FRAMES,
    PERSISTENT_EXTREME_G,
    PID_THROTTLE_BASE,
    paper_environment_snapshot,
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
from .utils import get2d_heading_AO_TA_R
from .render_tacview import TacviewLogger

logger = logging.getLogger(__name__)

LOAD_DIAGNOSTIC_START_G = 9.0


def _is_adapted_profile(env) -> bool:
    return bool(getattr(env, "is_paper_3v3", False))


def _paper_missile_rng(seed: int | None, parent_uid: str,
                       sequence: int) -> np.random.Generator:
    """Independent deterministic stream for one formal-profile launch."""
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
    "engaged_blocked",
    "range_low_blocked",
    "range_high_blocked",
    "launches",
)
from my_uav_env.alignment.state_extractor import body_angles_from_neu_vector


def make_empty_launch_diag() -> dict:
    """Return a fresh per-step missile launch diagnostics counter."""

    return {team: {key: 0 for key in LAUNCH_DIAG_KEYS}
            for team in LAUNCH_DIAG_TEAMS}


class UavCombatEnv(gymnasium.Env):
    """
    Multi-agent UAV combat environment (paper BRMA-MAPPO baseline).

    Action space (per agent): Box(3,) → paper §2.4 ABSOLUTE targets
      - target_pitch:    ±90° (act[0] → θ ∈ (−π/2, π/2])
      - target_heading:  ±180° absolute (act[1] → ψ ∈ (−π, π])
      - target_velocity: 0.3–1.2 Mach ≈ 102–408 m/s (act[2] → V)

    Observation space (per agent): paper Table 1/Table 2 entities plus mask.
      - "ego_state"     (entity_dim,)       self state
      - "ally_states"   (max_allies-1, entity_dim)  allied aircraft, excluding self
      - "enemy_states"  (max_enemies, entity_dim)    enemy aircraft
      - "entity_mask"   (6,) 0=valid/alive, 1=invalid/dead
    """

    # ---- Action scale constants -------------------------------------------------
    # Paper §2.4: action space uses ABSOLUTE target values (not deltas).
    #
    #   θ ∈ (−π/2, π/2]       pitch   act[0] ∈ [-1, 1] → ±90°
    #   ψ ∈ (−π, π]           heading act[1] ∈ [-1, 1] → ±180° (absolute)
    #   V ∈ [0.3, 1.2] Mach   velocity act[2] ∈ [-1, 1] → [102, 408] m/s
    #
    # Both teams share identical action authority per paper specification.
    #
    # Velocity:  F-16 F100-PW-229 MilThrust ≈ 17 800 lbf; jet can sustain M0.8–1.0
    #            in level flight at 10 kft.  Mach reference: a ≈ 340 m/s at sea level,
    #            ≈ 328 m/s at 10 kft ISA.
    PITCH_DEG = 90.0             # paper §2.4: full longitudinal authority (±90°)
    VELOCITY_MIN = paper_value("action_speed_mach")[0] * paper_value("mach_reference_mps")
    VELOCITY_MAX = paper_value("action_speed_mach")[1] * paper_value("mach_reference_mps")

    MISSILE_COOLDOWN_STEPS = 30        # default 0.5 s at 60 Hz; __init__ scales with sim_freq
    MISSILE_LOCK_DELAY_FRAMES = 15     # default 0.25 s at 60 Hz; __init__ scales with sim_freq
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

    metadata = {"render_modes": []}

    def __init__(self, max_num_blue=3, max_num_red=3, num_missiles_per_plane=2,
                 sim_freq=60, agent_interaction_steps=12, max_steps=1400,
                 missile_detection_half_angle_deg: float = 45.0,
                 missile_min_launch_range_m: float = 500.0,
                 pid_profile: str = PAPER_PID_PROFILE,
                 pid_throttle_base: float = PID_THROTTLE_BASE,
                 reward_mode: str = PAPER_REWARD_MODE,
                 missile_guidance_mode: str = PAPER_MISSILE_GUIDANCE_MODE,
                 altitude_reward_config=None,
                 obs_mode: str = "paper_strict",
                 blue_policy_profile: str = PAPER_BLUE_POLICY_PROFILE,
                 environment_profile: str = PAPER_ENVIRONMENT_PROFILE,
                 initial_condition_randomization_mode: str = "deterministic_v1",
                 suppress_jsbsim_output: bool = True,
                 environment_config: PaperEnvironmentConfig | None = None,
                 render_mode=None):
        super().__init__()
        fixed = (max_num_red, max_num_blue, sim_freq,
                 agent_interaction_steps, max_steps)
        if fixed != (3, 3, 60, 12, 1400):
            raise ValueError(
                "paper_3v3_v1 requires 3V3, 60 Hz, 12 physics frames per "
                "decision, and 1400 decision steps")
        if environment_profile != PAPER_ENVIRONMENT_PROFILE:
            raise ValueError("only paper_3v3_v1 is supported")
        if environment_config not in (None, PAPER_ENVIRONMENT_CONFIG):
            raise ValueError("paper_3v3_v1 does not accept another config")
        if initial_condition_randomization_mode != "deterministic_v1":
            raise ValueError("paper_3v3_v1 uses deterministic_v1 initialization")
        if obs_mode != "paper_strict":
            raise ValueError("paper_3v3_v1 only supports paper_strict observations")
        if pid_profile != PAPER_PID_PROFILE:
            raise ValueError(f"pid_profile must be {PAPER_PID_PROFILE!r}")
        if reward_mode != PAPER_REWARD_MODE:
            raise ValueError(f"reward_mode must be {PAPER_REWARD_MODE!r}")
        if missile_guidance_mode != PAPER_MISSILE_GUIDANCE_MODE:
            raise ValueError(
                f"missile_guidance_mode must be {PAPER_MISSILE_GUIDANCE_MODE!r}")
        if blue_policy_profile != PAPER_BLUE_POLICY_PROFILE:
            raise ValueError(
                f"blue_policy_profile must be {PAPER_BLUE_POLICY_PROFILE!r}")
        self.environment_profile = PAPER_ENVIRONMENT_PROFILE
        self.is_paper_3v3 = True
        self.is_paper_adapted = True
        self.initial_condition_randomization_mode = "deterministic_v1"
        self.environment_config = PAPER_ENVIRONMENT_CONFIG
        num_missiles_per_plane = 2
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
        # The paper requires a detection cone and 10 km maximum range but does
        # not publish the cone half-angle or a minimum launch range.
        self.missile_launch_ao_thresh = float(
            self.environment_config.electro_optical.half_angle_rad.value)
        self.missile_launch_min_range = float(
            self.environment_config.electro_optical.minimum_launch_range_m.value)
        self.missile_launch_max_range = (
            float(self.environment_config.electro_optical.maximum_range_m.value))
        self.pid_profile = pid_profile
        if not 0.0 <= float(pid_throttle_base) <= 1.0:
            raise ValueError("pid_throttle_base must be in [0, 1]")
        self.pid_throttle_base = float(pid_throttle_base)
        self.reward_mode = reward_mode
        self._reward_summary_step: dict = {}
        self.missile_guidance_mode = missile_guidance_mode
        self.altitude_reward_config = AltitudeRewardConfig(
            version="eq17_finite_engineering_mean_v1",
            h_min_m=0.0, h_att_m=2000.0, h_adv_m=5000.0,
            h_max_m=10000.0, d_att_max_m=10000.000001,
            high_altitude_tail=0.0)
        self.sim_freq = sim_freq
        self.agent_interaction_steps = agent_interaction_steps
        self.max_steps = max_steps
        self.suppress_jsbsim_output = suppress_jsbsim_output
        self.obs_mode = "paper_strict"
        self.blue_policy_profile = validate_blue_policy_profile(
            blue_policy_profile)
        self.blue_policy_controller = BluePolicyController(
            self.blue_policy_profile)
        self.entity_dim = 10
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
        for aid in self.agent_ids:
            own_count = self.max_num_blue if aid.startswith("blue_") else self.max_num_red
            enemy_count = self.max_num_red if aid.startswith("blue_") else self.max_num_blue
            obs_spaces[aid] = gymnasium.spaces.Dict({
                "ego_state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.entity_dim,), dtype=np.float32),
                "ally_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(own_count - 1, self.entity_dim), dtype=np.float32),
                "enemy_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(enemy_count, self.entity_dim), dtype=np.float32),
                "entity_mask": gymnasium.spaces.Box(
                    low=0, high=1,
                    shape=(max_num_blue + max_num_red,), dtype=np.int64),
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
        self._mws_enabled_by_team = self._profile_mws_defaults()
        self._learnable_mws_state: dict[str, dict] = {}
        self._learnable_setpoint_state: dict[str, tuple[float, float, float]] = {}
        self._learnable_requested_setpoints: dict[str, tuple[float, float, float]] = {}
        self._learnable_previous_setpoints: dict[str, tuple[float, float, float]] = {}
        self._learnable_command_sources: dict[str, str] = {}
        self._learnable_selected_mws_diagnostics: dict[str, dict | None] = {}
        self._load_frame_history: dict[str, deque] = {}
        self._retained_extreme_load_traces: list[dict] = []
        self._last_load_trace_level: dict[str, int] = {}

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
        self._missile_trajectory_sink = None

        # Death reason tracking (set on the step the agent dies, cleared on reset)
        self._death_reasons: dict[str, str | None] = {}

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
        self._mws_enabled_by_team = self._profile_mws_defaults()
        if seed is not None:
            self._seed = int(seed)
            self.np_random = np.random.default_rng(self._seed)
        elif self._seed is None:
            # Gymnasium created a generator above; retain an explicit traceable seed.
            self._seed = int(self.np_random.integers(0, 2**32 - 1))
            self.np_random = np.random.default_rng(self._seed)
        self._initial_jitter_by_index = {}
        self._environment_config_snapshot = paper_environment_snapshot(seed=self._seed)
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
            "red_first_launch_step": None,
            "blue_first_launch_step": None,
            "red_first_hit_step": None,
            "blue_first_hit_step": None,
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
                  "consecutive_above_600mps_frames": 0,
                  "maximum_consecutive_above_600mps_frames": 0,
                  "speed_envelope_violation": False,
                  "maximum_load_g_seen": 0.0, "over_g_frames": 0,
                  "maximum_abs_pilot_z_load_seen": 0.0,
                  "load_limiter_activations": 0,
                  "frames_above_9g": 0,
                  "consecutive_above_9g_frames": 0,
                  "maximum_consecutive_above_9g_frames": 0,
                  "episode_ever_exceeded_9g": False,
                  "overload_envelope_violation": False,
                  "consecutive_above_30g_frames": 0,
                  "maximum_consecutive_above_30g_frames": 0,
                  "transient_above_30g_events": 0,
                  "load_protection_active_frames": 0,
                  "load_protection_minimum_scale": 1.0,
                  "setpoint_rate_limit_activations": 0,
                  "heading_rate_limit_activations": 0,
                  "pitch_rate_limit_activations": 0,
                  "velocity_rate_limit_activations": 0,
                  "requested_heading_jump_max_rad": 0.0,
                  "applied_heading_jump_max_rad": 0.0,
                  "requested_pitch_jump_max_rad": 0.0,
                  "applied_pitch_jump_max_rad": 0.0,
                  "requested_velocity_jump_max_mps": 0.0,
                  "applied_velocity_jump_max_mps": 0.0,
                  "maximum_absolute_e_phi": 0.0,
                  "maximum_absolute_e_theta": 0.0,
                  "maximum_absolute_derivative_term": 0.0,
                  "pid_output_saturation_frames": 0,
                  "degenerate_arctan_ratio_count": 0}
            for aid in self.agent_ids}
        self._learnable_mws_state = {
            aid: self._empty_mws_state() for aid in self.agent_ids}
        self._learnable_setpoint_state = {}
        self._learnable_requested_setpoints = {}
        self._learnable_previous_setpoints = {}
        self._learnable_command_sources = {}
        self._learnable_selected_mws_diagnostics = {
            aid: None for aid in self.agent_ids}
        self._load_frame_history = {
            aid: deque(maxlen=12) for aid in self.agent_ids}
        self._retained_extreme_load_traces = []
        self._last_load_trace_level = {aid: 0 for aid in self.agent_ids}

        # Reset missile launch counters
        self._missile_launch_counts = {aid: 0 for aid in self.agent_ids}
        self._minimal_launch_sequence = {aid: 0 for aid in self.agent_ids}
        self._mws_decision_diagnostics = {
            "red_detected_agent_decisions": 0,
            "red_override_agent_decisions": 0,
            "blue_detected_agent_decisions": 0,
            "blue_override_agent_decisions": 0,
            "red_warning_generations": 0,
            "blue_warning_generations": 0,
            "red_direction_changes_within_same_missile": 0,
            "blue_direction_changes_within_same_missile": 0,
            "red_suppressed_direction_flip_attempts": 0,
            "blue_suppressed_direction_flip_attempts": 0,
            "red_maximum_continuous_decisions": 0,
            "blue_maximum_continuous_decisions": 0,
            "red_target_heading_delta_max_deg": 0.0,
            "blue_target_heading_delta_max_deg": 0.0,
        }
        self._missile_first_warning_frame = {}
        self._warning_to_terminal_s = {"red": [], "blue": []}
        self._warning_to_hit_s = {"red": [], "blue": []}
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []

        # Reset death reasons
        self._death_reasons = {}

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
        enemy_positions = {
            red_id: np.asarray(sim.get_position(), dtype=np.float64)
            for red_id, sim in self.red_planes.items()
        }
        enemy_alive = {
            red_id: bool(sim.is_alive) for red_id, sim in self.red_planes.items()
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
            own_alive=own_alive,
            enemy_positions=enemy_positions,
            enemy_alive=enemy_alive,
            assigned_targets=None)

    def set_missile_trajectory_sink(self, sink) -> None:
        """Install an audit-only per-physics-frame missile diagnostic sink."""
        self._missile_trajectory_sink = sink

    def _profile_mws_defaults(self) -> dict[str, bool]:
        return {"red": True, "blue": True}

    def _mws_enabled_for_agent(self, agent_id: str) -> bool:
        team = "blue" if agent_id.startswith("blue") else "red"
        gates = getattr(self, "_mws_enabled_by_team", {"red": True, "blue": True})
        if not gates.get(team, True):
            return False
        return bool(gates.get(team, True))

    def step(self, actions: dict):
        self.current_step += 1
        self._crashed_this_step.clear()
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        self._reward_summary_step = {}
        self._sensor_diagnostics_step = []
        self.refresh_engaged_targets()

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
            if self.is_paper_3v3:
                self._clear_control_state(aid)
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

    @staticmethod
    def _wrap_angle_rad(angle: float) -> float:
        return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _empty_mws_state() -> dict:
        return {
            "active_missile_uid": None,
            "break_direction": 0.0,
            "break_reference_heading": None,
            "break_target_heading": None,
            "warning_start_physics_frame": None,
            "warning_start_decision_step": None,
            "previous_warning_active": False,
            "warning_generation": 0,
            "continuous_decisions": 0,
        }

    def _clear_control_state(self, aid: str) -> None:
        self._learnable_mws_state[aid] = self._empty_mws_state()
        self._learnable_setpoint_state.pop(aid, None)
        self._learnable_requested_setpoints.pop(aid, None)
        self._learnable_previous_setpoints.pop(aid, None)
        self._learnable_command_sources.pop(aid, None)
        self._learnable_selected_mws_diagnostics.pop(aid, None)

    def _deactivate_mws_state(self, aid: str) -> None:
        generation = int(self._learnable_mws_state.get(
            aid, {}).get("warning_generation", 0))
        self._learnable_mws_state[aid] = self._empty_mws_state()
        self._learnable_mws_state[aid]["warning_generation"] = generation

    def _mws_evasion_target(
            self, aid: str, incoming, current_heading: float,
            turn_direction: float) -> tuple[float, float, float]:
        state = self._learnable_mws_state.setdefault(
            aid, self._empty_mws_state())
        team = "blue" if aid.startswith("blue_") else "red"
        missile_uid = str(incoming.uid)
        if state["active_missile_uid"] != missile_uid:
            generation = int(state.get("warning_generation", 0)) + 1
            target_heading = self._wrap_angle_rad(
                current_heading + turn_direction * np.deg2rad(60.0))
            state.update({
                "active_missile_uid": missile_uid,
                "break_direction": float(turn_direction),
                "break_reference_heading": float(current_heading),
                "break_target_heading": target_heading,
                "warning_start_physics_frame": int(self._physics_frame),
                "warning_start_decision_step": int(self.current_step),
                "previous_warning_active": True,
                "warning_generation": generation,
                "continuous_decisions": 0,
            })
            self._mws_decision_diagnostics[f"{team}_warning_generations"] += 1
        elif float(state["break_direction"]) != float(turn_direction):
            # The stored direction is authoritative for an existing missile.
            self._mws_decision_diagnostics[
                f"{team}_suppressed_direction_flip_attempts"] += 1
        state["previous_warning_active"] = True
        state["continuous_decisions"] += 1
        continuous_key = f"{team}_maximum_continuous_decisions"
        self._mws_decision_diagnostics[continuous_key] = max(
            self._mws_decision_diagnostics[continuous_key],
            int(state["continuous_decisions"]))
        delta_deg = abs(np.rad2deg(self._wrap_angle_rad(
            state["break_target_heading"]
            - state["break_reference_heading"])))
        delta_key = f"{team}_target_heading_delta_max_deg"
        self._mws_decision_diagnostics[delta_key] = max(
            self._mws_decision_diagnostics[delta_key],
            float(delta_deg))
        return 0.0, float(state["break_target_heading"]), 300.0

    def _finalize_target(
            self, aid: str, requested: tuple[float, float, float],
            source: str) -> tuple[float, float, float]:
        """Record target jumps and pass paper Eq.12 inputs through unchanged."""
        if not self.is_paper_3v3:
            if aid.startswith("blue"):
                self.blue_policy_controller.record_executed_heading(
                    aid, requested[1], source)
            return requested
        sim = self._get_sim(aid)
        current_rpy = sim.get_rpy()
        current_speed = float(np.linalg.norm(sim.get_velocity()))
        previous = self._learnable_setpoint_state.get(
            aid, (float(current_rpy[1]), float(current_rpy[2]), current_speed))
        req_pitch, req_heading, req_velocity = map(float, requested)
        prev_pitch, prev_heading, prev_velocity = previous
        heading_delta = self._wrap_angle_rad(req_heading - prev_heading)
        pitch_delta = req_pitch - prev_pitch
        velocity_delta = req_velocity - prev_velocity
        applied = (req_pitch, req_heading, req_velocity)
        diag = self._aircraft_diagnostics[aid]
        diag["requested_heading_jump_max_rad"] = max(
            diag["requested_heading_jump_max_rad"], abs(heading_delta))
        diag["applied_heading_jump_max_rad"] = max(
            diag["applied_heading_jump_max_rad"],
            abs(self._wrap_angle_rad(applied[1] - prev_heading)))
        diag["requested_pitch_jump_max_rad"] = max(
            diag["requested_pitch_jump_max_rad"], abs(pitch_delta))
        diag["applied_pitch_jump_max_rad"] = max(
            diag["applied_pitch_jump_max_rad"], abs(applied[0] - prev_pitch))
        diag["requested_velocity_jump_max_mps"] = max(
            diag["requested_velocity_jump_max_mps"], abs(velocity_delta))
        diag["applied_velocity_jump_max_mps"] = max(
            diag["applied_velocity_jump_max_mps"], abs(applied[2] - prev_velocity))
        self._learnable_requested_setpoints[aid] = (
            req_pitch, req_heading, req_velocity)
        self._learnable_previous_setpoints[aid] = previous
        self._learnable_command_sources[aid] = str(source)
        self._learnable_setpoint_state[aid] = applied
        if aid.startswith("blue"):
            self.blue_policy_controller.record_executed_heading(
                aid, applied[1], source)
        return applied

    def _parse_actions(self, actions: dict) -> dict:
        """Convert normalised actor outputs ∈ [-1, 1] to physical setpoints.

        Control-flow priority (team-aware):
          Layer 1 — Missile evasion:     BOTH teams  (paper §2.1.3, scripted)
          Layer 2 — Agent action:        BOTH teams  (identical §2.4 mapping)

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
                if self.is_paper_3v3:
                    self._clear_control_state(aid)
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
            if mws_enabled:
                if hasattr(sim, "get_missile_warning_diagnostic"):
                    incoming, incoming_diag = (
                        sim.get_missile_warning_diagnostic())
                else:
                    incoming = sim.check_missile_warning()
                    incoming_diag = None
            else:
                incoming, incoming_diag = None, None
            if self.is_paper_3v3:
                self._learnable_selected_mws_diagnostics[aid] = (
                    dict(incoming_diag)
                    if isinstance(incoming_diag, dict) else None)
            if incoming is not None and mws_enabled:
                team = "blue" if is_blue else "red"
                mws_counts = getattr(self, "_mws_decision_diagnostics", None)
                if mws_counts is None:
                    mws_counts = {
                        "red_detected_agent_decisions": 0,
                        "red_override_agent_decisions": 0,
                        "blue_detected_agent_decisions": 0,
                        "blue_override_agent_decisions": 0,
                    }
                    self._mws_decision_diagnostics = mws_counts
                mws_counts[
                    f"{team}_detected_agent_decisions"] += 1
                mws_counts[
                    f"{team}_override_agent_decisions"] += 1
                warning_frames = getattr(
                    self, "_missile_first_warning_frame", None)
                if warning_frames is None:
                    warning_frames = {}
                    self._missile_first_warning_frame = warning_frames
                warning_frames.setdefault(
                    incoming.uid, int(getattr(self, "_physics_frame", 0)))
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
                if getattr(self, "is_paper_adapted", False):
                    turn_dir = -1.0 if ao > 0 else 1.0
                    requested = self._mws_evasion_target(
                        aid, incoming, current_heading, turn_dir)
                    targets[aid] = self._finalize_target(
                        aid, requested, "mws_override")
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
            if self.is_paper_3v3:
                self._deactivate_mws_state(aid)

            # =================================================================
            #  Layer 2 — Agent Action (paper §2.4 — both teams identical)
            #
            #    target_pitch   = act[0] * 90°             ∈ [−90°, +90°]
            #    target_heading = act[1] * 180°            ∈ [−180°, +180°]  (absolute)
            #    target_velocity ∈ [102, 408] m/s
            # =================================================================
            target_velocity = self.VELOCITY_MIN + (float(act[2]) + 1.0) / 2.0 * (
                self.VELOCITY_MAX - self.VELOCITY_MIN)
            target_pitch = float(act[0]) * np.deg2rad(self.PITCH_DEG)
            target_heading = float(act[1]) * np.pi

            targets[aid] = self._finalize_target(
                aid, (target_pitch, target_heading, target_velocity),
                "base_policy")
        return targets

    # ------------------------------------------------------------------
    #  PID control application (per physics frame)
    # ------------------------------------------------------------------

    def _retain_load_frame(
            self, aid: str, sim, g_components: tuple[float, float, float],
            g_load: float, requested, applied, commands, scale: float) -> None:
        rpy = tuple(float(value) for value in sim.get_rpy())
        velocity = tuple(float(value) for value in sim.get_velocity())
        position = tuple(float(value) for value in sim.get_position())
        history = self._load_frame_history.setdefault(aid, deque(maxlen=12))
        previous_g = (float(history[-1]["g_load_total"])
                      if history else float("nan"))
        previous_setpoints = self._learnable_previous_setpoints.get(aid)
        requested_heading_delta = (
            self._wrap_angle_rad(requested[1] - previous_setpoints[1])
            if requested is not None and previous_setpoints is not None else 0.0)
        requested_pitch_delta = (
            float(requested[0] - previous_setpoints[0])
            if requested is not None and previous_setpoints is not None else 0.0)
        mws_state = copy.deepcopy(self._learnable_mws_state.get(aid, {}))
        selected_mws_diag = self._learnable_selected_mws_diagnostics.get(aid)
        pid_diag = copy.deepcopy(getattr(
            self.pid_controllers.get(aid), "_last_diagnostic", {}))
        def _property(name: str) -> float:
            try:
                return float(sim.get_property_value(name))
            except Exception:
                return float("nan")
        row = {
            "physics_frame": int(self._physics_frame),
            "decision_step": int(self.current_step),
            "agent_id": aid,
            "team": "blue" if aid.startswith("blue") else "red",
            "position_m": position,
            "velocity_mps": velocity,
            "speed_mps": float(np.linalg.norm(velocity)),
            "vertical_velocity_mps": float(velocity[2]),
            "roll_pitch_heading_rad": rpy,
            "body_angular_rates_rad_s": (
                _property("velocities/p-rad_sec"),
                _property("velocities/q-rad_sec"),
                _property("velocities/r-rad_sec")),
            "alpha_rad": _property("aero/alpha-rad"),
            "beta_rad": _property("aero/beta-rad"),
            "g_components": tuple(float(value) for value in g_components),
            "g_load_total": float(g_load),
            "previous_g_load_total": previous_g,
            "requested_setpoints": requested,
            "applied_setpoints": applied,
            "previous_applied_setpoints": previous_setpoints,
            "requested_heading_delta_rad": requested_heading_delta,
            "requested_pitch_delta_rad": requested_pitch_delta,
            "command_source": self._learnable_command_sources.get(aid, ""),
            "pid_commands_before_load_protection": tuple(
                float(value) for value in commands),
            "pid_internal": pid_diag,
            "d_B_des": pid_diag.get("d_B_des"),
            "paper_e_phi": pid_diag.get("e_phi"),
            "paper_e_theta": pid_diag.get("e_theta"),
            "pid_unsaturated_commands": pid_diag.get("unsaturated_commands"),
            "pid_saturated_commands": pid_diag.get("saturated_commands"),
            "pid_saturation": tuple(
                (abs(float(value)) >= 1.0 - 1e-9 if index < 3
                 else float(value) <= 1e-9 or float(value) >= 1.0 - 1e-9)
                for index, value in enumerate(commands)),
            "load_protection_scale": float(scale),
            "mws_override": self._learnable_command_sources.get(aid) == "mws_override",
            "incoming_missile_uid": mws_state.get("active_missile_uid"),
            "mws_break_direction": mws_state.get("break_direction", 0.0),
            "mws_reference_heading": mws_state.get("break_reference_heading"),
            "mws_state": mws_state,
            "mws_just_entered": (
                mws_state.get("warning_start_decision_step") == self.current_step),
            "mws_just_exited": False,
            "action_update": int(self.current_step),
            "setpoint_changed_this_frame": bool(
                pid_diag.get("setpoint_changed_this_frame", False)),
            "pitch_setpoint_delta_rad": float(
                pid_diag.get("pitch_setpoint_delta_rad", 0.0)),
            "heading_setpoint_delta_rad": float(
                pid_diag.get("heading_setpoint_delta_rad", 0.0)),
            "velocity_setpoint_delta_mps": float(
                pid_diag.get("velocity_setpoint_delta_mps", 0.0)),
            "derivative_suppressed_for_setpoint_change": bool(
                pid_diag.get(
                    "derivative_suppressed_for_setpoint_change", False)),
            "selected_missile_closing_speed_mps": (
                selected_mws_diag.get("closing_speed_mps")
                if isinstance(selected_mws_diag, dict) else None),
            "selected_missile_ttc_s": (
                selected_mws_diag.get("time_to_closest_approach_s")
                if isinstance(selected_mws_diag, dict) else None),
            "candidate_is_approaching": bool(
                selected_mws_diag.get("candidate_is_approaching", False))
                if isinstance(selected_mws_diag, dict) else False,
        }
        history.append(row)
        level = (4 if not np.isfinite(g_load) or g_load > CATASTROPHIC_G
                 else 3 if g_load > PERSISTENT_EXTREME_G
                 else 2 if g_load > 20.0
                 else 1 if g_load > LOAD_DIAGNOSTIC_START_G
                 else 0)
        previous_level = self._last_load_trace_level.get(aid, 0)
        persistent_trigger = int(self._aircraft_diagnostics[aid].get(
            "consecutive_above_30g_frames", 0)) == PERSISTENT_EXTREME_FRAMES
        if level > previous_level or persistent_trigger:
            self._retained_extreme_load_traces.append({
                "trigger_agent_id": aid,
                "trigger_g": float(g_load),
                "trigger_level": int(level),
                "frames": copy.deepcopy(list(history)),
            })
        self._last_load_trace_level[aid] = level

    def _update_load_diagnostics(
            self, aid: str, g_load: float) -> None:
        """Observe the paper 9g envelope without modifying flight controls."""
        diag = self._aircraft_diagnostics[aid]
        diag["maximum_load_g_seen"] = max(
            diag["maximum_load_g_seen"], float(g_load))
        if g_load > LOAD_DIAGNOSTIC_START_G:
            diag["frames_above_9g"] += 1
            diag["consecutive_above_9g_frames"] += 1
            diag["maximum_consecutive_above_9g_frames"] = max(
                diag["maximum_consecutive_above_9g_frames"],
                diag["consecutive_above_9g_frames"])
            diag["episode_ever_exceeded_9g"] = True
        else:
            diag["consecutive_above_9g_frames"] = 0
        previous_above_30 = int(diag["consecutive_above_30g_frames"])
        if g_load > PERSISTENT_EXTREME_G:
            diag["consecutive_above_30g_frames"] += 1
            diag["maximum_consecutive_above_30g_frames"] = max(
                diag["maximum_consecutive_above_30g_frames"],
                diag["consecutive_above_30g_frames"])
        else:
            if 0 < previous_above_30 < PERSISTENT_EXTREME_FRAMES:
                diag["transient_above_30g_events"] += 1
            diag["consecutive_above_30g_frames"] = 0

    @staticmethod
    def _load_invalid_reason(
            g_load: float, consecutive_above_30g_frames: int) -> str | None:
        if not np.isfinite(g_load):
            return "NonFiniteLoad"
        if g_load > CATASTROPHIC_G:
            return "CatastrophicFiniteLoad"
        if consecutive_above_30g_frames >= PERSISTENT_EXTREME_FRAMES:
            return "PersistentExtremeFiniteLoad"
        return None

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
            pid_diag = dict(getattr(pid, "_last_diagnostic", {}))
            if pid_diag:
                diag["maximum_absolute_e_phi"] = max(
                    diag.get("maximum_absolute_e_phi", 0.0),
                    abs(float(pid_diag.get("e_phi", 0.0))))
                diag["maximum_absolute_e_theta"] = max(
                    diag.get("maximum_absolute_e_theta", 0.0),
                    abs(float(pid_diag.get("e_theta", 0.0))))
                derivative_terms = [abs(float(pid_diag.get(loop, {}).get("d", 0.0)))
                                    for loop in ("roll_pid", "pitch_pid", "velocity_pid")]
                diag["maximum_absolute_derivative_term"] = max(
                    diag.get("maximum_absolute_derivative_term", 0.0),
                    max(derivative_terms, default=0.0))
                saturated = pid_diag.get("saturated_commands", ())
                unsaturated = pid_diag.get("unsaturated_commands", ())
                if len(saturated) == len(unsaturated) and any(
                        abs(float(before) - float(after)) > 1e-12
                        for before, after in zip(unsaturated, saturated)):
                    diag["pid_output_saturation_frames"] = int(
                        diag.get("pid_output_saturation_frames", 0)) + 1
                diag["degenerate_arctan_ratio_count"] = int(
                    diag.get("degenerate_arctan_ratio_count", 0)) + int(
                        pid_diag.get("degenerate_arctan_ratio", False))
            diag["maximum_speed_mps_seen"] = max(
                diag["maximum_speed_mps_seen"], current_speed)
            try:
                g_components = tuple(float(value) for value in (
                    sim.get_property_value("accelerations/n-pilot-x-norm"),
                    sim.get_property_value("accelerations/n-pilot-y-norm"),
                    sim.get_property_value("accelerations/n-pilot-z-norm")))
                g_load = float(np.linalg.norm(g_components))
            except Exception:
                g_components = (float("nan"),) * 3
                g_load = float("nan")
            pilot_z_abs = abs(float(g_components[2]))
            if np.isfinite(pilot_z_abs):
                diag["maximum_abs_pilot_z_load_seen"] = max(
                    diag["maximum_abs_pilot_z_load_seen"], pilot_z_abs)
            else:
                diag["maximum_abs_pilot_z_load_seen"] = float("inf")
            if not np.isfinite(g_load):
                diag["maximum_load_g_seen"] = float("inf")
                diag["nonfinite_load_frames"] = int(
                    diag.get("nonfinite_load_frames", 0)) + 1
                if self.is_paper_3v3:
                    self._retain_load_frame(
                        aid, sim, g_components, g_load,
                        self._learnable_requested_setpoints.get(aid), target,
                        (aileron, elevator, rudder, throttle), 1.0)
                sim.crash()
                if _is_adapted_profile(self):
                    self._mark_invalid_numerical(aid, "NonFiniteLoad")
                else:
                    self._death_reasons.setdefault(aid, "Crash_NumericalLoad")
                self._crashed_this_step.add(aid)
                continue
            if g_load > 100.0 and not _is_adapted_profile(self):
                sim.crash()
                self._death_reasons.setdefault(aid, "Crash_NumericalLoad")
                self._crashed_this_step.add(aid)
                continue
            if self.is_paper_3v3:
                self._update_load_diagnostics(aid, g_load)
                self._retain_load_frame(
                    aid, sim, g_components, g_load,
                    self._learnable_requested_setpoints.get(aid), target,
                    (aileron, elevator, rudder, throttle), 1.0)
                invalid_reason = self._load_invalid_reason(
                    g_load, self._aircraft_diagnostics[aid][
                        "consecutive_above_30g_frames"])
                if invalid_reason in (
                        "CatastrophicFiniteLoad",
                        "PersistentExtremeFiniteLoad"):
                    sim.crash()
                    self._mark_invalid_numerical(aid, invalid_reason)
                    self._crashed_this_step.add(aid)
                    continue
                if (self._aircraft_diagnostics[aid][
                        "consecutive_above_9g_frames"]
                        >= AIRCRAFT_ENVELOPE_FRAMES):
                    self._aircraft_diagnostics[aid][
                        "overload_envelope_violation"] = True

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
        """Apply numerical guards and audit the paper 600 m/s envelope."""
        velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
        position = np.asarray(sim.get_position(), dtype=np.float64)
        rpy = np.asarray(sim.get_rpy(), dtype=np.float64)
        if not np.all(np.isfinite(np.concatenate([velocity, position, rpy]))):
            sim.crash()
            if _is_adapted_profile(self):
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
        diag["maximum_speed_after_limit_mps"] = max(
            diag["maximum_speed_after_limit_mps"], speed)
        if speed > maximum:
            diag["overspeed_frames"] += 1
            diag["consecutive_above_600mps_frames"] += 1
            diag["maximum_consecutive_above_600mps_frames"] = max(
                diag["maximum_consecutive_above_600mps_frames"],
                diag["consecutive_above_600mps_frames"])
        else:
            diag["consecutive_above_600mps_frames"] = 0
        if diag["consecutive_above_600mps_frames"] >= AIRCRAFT_ENVELOPE_FRAMES:
            diag["speed_envelope_violation"] = True

    def _mark_invalid_numerical(self, aid: str, reason: str) -> None:
        self._invalid_numerical_episode = True
        label = f"{aid}:{reason}"
        if label not in self._invalid_numerical_reasons:
            self._invalid_numerical_reasons.append(label)
        if self.is_paper_3v3:
            history = list(self._load_frame_history.get(aid, ()))
            last_trace = (self._retained_extreme_load_traces[-1]
                          if self._retained_extreme_load_traces else None)
            if (last_trace is not None
                    and last_trace.get("trigger_agent_id") == aid):
                last_trace["invalid_reason"] = reason
            elif history:
                self._retained_extreme_load_traces.append({
                    "trigger_agent_id": aid,
                    "trigger_g": history[-1].get("g_load_total"),
                    "trigger_level": "numerical_invalid",
                    "invalid_reason": reason,
                    "frames": copy.deepcopy(history),
                })
        self._death_reasons.setdefault(aid, "Invalid_Numerical")

    def _check_missile_launch(self):
        """Advance the single automatic EO-track and launch state machine."""
        for aid in self.agent_ids:
            team = "red" if aid.startswith("red") else "blue"
            diag = self._launch_diag_step[team]
            diag["scan_frames"] += 1
            sim = self._get_sim(aid)
            state = self._fire_control_states.setdefault(aid, FireControlState())
            if self._missile_cooldown.get(aid, 0) > 0:
                self._missile_cooldown[aid] -= 1
            if sim is None or not sim.is_alive:
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None
                state.current_target_id = None
                state.continuous_detection_frames = 0
                state.lock_mature = False
                state.cooldown_frames_remaining = int(
                    self._missile_cooldown.get(aid, 0))
                state.detection_state = "shooter_dead"
                state.blocked_reason = "shooter_dead"
                continue
            diag["alive_shooters"] += 1

            # ---- Shared engaged-targets set (hot-updated across agents) ----
            # Uses self._engaged_targets directly — no per-agent recomputation.
            # The set contains enemy UIDs that have an in-flight friendly
            # missile tracking them AND targets flight-assigned by the
            # coordinated-actions allocator.

            # Select one target for continuous EO tracking. TA and live-missile
            # deconfliction are checked only at the launch instant.
            enemies = self.red_planes if sim.color == "Blue" else self.blue_planes
            candidate_enemies = list(enemies.values())
            best_enemy = None
            best_distance = float("inf")
            best_ta = float("nan")

            for enemy_sim in candidate_enemies:
                if not enemy_sim.is_alive:
                    continue
                diag["alive_enemy_pairs"] += 1
                if enemy_sim.uid not in self._engaged_targets:
                    diag["unengaged_enemy_pairs"] += 1

                ego_pos = sim.get_position()
                enm_pos = enemy_sim.get_position()
                AO = compute_body_x_q_los(ego_pos, sim.get_rpy(), enm_pos)
                _, TA, _ = get2d_heading_AO_TA_R(
                    ego_pos, sim.get_rpy()[2], enm_pos, enemy_sim.get_rpy()[2])
                R = compute_3d_range(ego_pos, enm_pos)
                eo_visible = self._is_detected_by_electro_optical(sim, enemy_sim)
                if not np.isfinite(R) or R <= 0.0:
                    diag["range_low_blocked"] += 1
                if R > self.missile_launch_max_range:
                    diag["range_high_blocked"] += 1
                range_ok = bool(
                    np.isfinite(R) and 0.0 < R <= self.missile_launch_max_range)
                ao_ok = AO <= self.missile_launch_ao_thresh
                ta_ok = TA > self.environment_config.fire_control.rear_hemisphere_ta_rad.value
                if range_ok:
                    diag["range_ok_pairs"] += 1
                if ao_ok:
                    diag["ao_ok_pairs"] += 1
                if ta_ok:
                    diag["ta_ok_pairs"] += 1

                continuous_eo_ok = bool(ao_ok and range_ok and eo_visible)
                if continuous_eo_ok and ta_ok:
                    diag["geometry_ok_pairs"] += 1

                if continuous_eo_ok and R < best_distance:
                    best_distance = R
                    best_enemy = enemy_sim
                    best_ta = TA

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

            lock_mature = (best_enemy is not None
                           and self._lock_timer[aid] >= self.missile_lock_delay_frames)
            ta_ok_at_launch = bool(
                best_enemy is not None
                and best_ta > self.environment_config.fire_control.rear_hemisphere_ta_rad.value)
            engaged_blocked = bool(
                best_enemy is not None
                and best_enemy.uid in self._engaged_targets)
            if lock_mature:
                diag["lock_mature_pairs"] += 1
                if self._missile_cooldown[aid] != 0:
                    diag["cooldown_blocked"] += 1
                if engaged_blocked:
                    diag["engaged_blocked"] += 1
            if (best_enemy is not None
                    and lock_mature
                    and ta_ok_at_launch
                    and self._missile_cooldown[aid] == 0
                    and not engaged_blocked
                    and sim.num_left_missiles > 0):
                launch_quality = self._build_launch_quality_record(
                    sim, best_enemy, best_distance)
                self._launch_missile(sim, best_enemy, launch_quality)
                diag["launches"] += 1
                # ---- HOT-UPDATE: immediately mark target as engaged ----
                # Subsequent agents in the same physics frame will see this
                # and skip the target, preventing same-frame double-launch.
                self._engaged_targets.add(best_enemy.uid)

            state.current_target_id = self._lock_target.get(aid)
            state.continuous_detection_frames = int(self._lock_timer.get(aid, 0))
            state.lock_mature = bool(lock_mature)
            state.cooldown_frames_remaining = int(self._missile_cooldown.get(aid, 0))
            if state.current_target_id is None and not lock_mature:
                state.detection_state = "no_target"
            elif lock_mature and state.cooldown_frames_remaining > 0:
                state.detection_state = "cooldown"
                state.blocked_reason = "cooldown"
            elif lock_mature and not ta_ok_at_launch:
                state.detection_state = "locked_waiting_rear_hemisphere"
                state.blocked_reason = "ta_not_rear_hemisphere"
            elif lock_mature and engaged_blocked:
                state.detection_state = "locked_deconflicted"
                state.blocked_reason = "live_missile_on_target"
            elif lock_mature:
                state.detection_state = "ready_to_launch"
                state.blocked_reason = ""
            else:
                state.detection_state = "tracking"
                state.blocked_reason = ""

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
        sequence = self._minimal_launch_sequence[parent.uid]
        missile_rng = _paper_missile_rng(self._seed, parent.uid, sequence)
        self._minimal_launch_sequence[parent.uid] = sequence + 1
        missile = MissileSimulator.create(
            parent, target, f"m{self._missile_id_counter}",
            guidance_mode=self.missile_guidance_mode,
            config=self.environment_config.missile,
            rng=missile_rng,
            launch_speed_mps=MISSILE_LAUNCH_SPEED_MPS,
            overshoot_window_s=MISSILE_OVERSHOOT_WINDOW_S,
            overshoot_distance_hysteresis_m=(
                MISSILE_OVERSHOOT_DISTANCE_HYSTERESIS_M),
            positive_closing_threshold_mps=(
                MISSILE_POSITIVE_CLOSING_THRESHOLD_MPS))
        missile._trajectory_sink = self._missile_trajectory_sink
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
        parent.num_left_missiles = max(0, parent.num_left_missiles - 1)
        self._missile_launch_counts[parent.uid] += 1
        team = "red" if parent.uid.startswith("red") else "blue"
        first_launch_key = f"{team}_first_launch_step"
        if self._episode_stats.get(first_launch_key) is None:
            self._episode_stats[first_launch_key] = int(self.current_step)

    def _finalize_launch_quality_record(self, missile: MissileSimulator) -> None:
        """Attach missile termination diagnostics to its launch snapshot."""

        record = self._launch_quality_records.get(missile.uid)
        if record is None or record.get("termination_reason"):
            return
        raw_reason = missile._termination_reason or ("hit" if missile.is_success else "unknown")
        if raw_reason in ("hit", "timeout"):
            reason = raw_reason
        elif raw_reason in ("p_hit_fail", "low_speed", "overshoot", "target_dead"):
            reason = "miss"
        elif raw_reason == "numerical_invalid":
            reason = "numerical_invalid"
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
            "pn_guidance_frames": int(getattr(missile, "_pn_guidance_frames", 0)),
            "pn_nonzero_command_frames": int(getattr(
                missile, "_pn_nonzero_command_frames", 0)),
            "maximum_command_g": float(getattr(
                missile, "_maximum_command_g", 0.0)),
        })
        warning_frame = self._missile_first_warning_frame.get(missile.uid)
        if warning_frame is not None:
            target_team = "red" if missile._target_id.startswith("red") else "blue"
            duration = max(
                0.0, (int(self._physics_frame) - int(warning_frame))
                / float(self.sim_freq))
            self._warning_to_terminal_s[target_team].append(duration)
            if raw_reason == "hit":
                self._warning_to_hit_s[target_team].append(duration)
        self._launch_quality_done_step_records.append(dict(record))

    def _update_missiles(self):
        """Advance all in-flight missiles and process hit/miss events."""
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
                if reason == "numerical_invalid":
                    self._invalid_numerical_episode = True
                    label = f"{mid}:MissileNumericalInvalid"
                    if label not in self._invalid_numerical_reasons:
                        self._invalid_numerical_reasons.append(label)
            if missile.is_success and not missile._kill_rewarded:
                shooter_id = missile._parent_id

                # ---- Kill-cooldown gate ----
                # Shooter has scored a kill too recently → override to MISS.

                # ---- Single-target gate (AOE prevention) ----
                # An agent may score at most 1 kill per env step.  If the same
                # shooter already killed a different target this step, block
                # any further kills.

                # ---- Kill accepted ----
                missile._kill_rewarded = True
                # Record death reason (only first death sticks)
                target_id = missile._target_id
                if target_id not in self._death_reasons:
                    self._death_reasons[target_id] = "Missile_Kill"
                team = "red" if shooter_id.startswith("red") else "blue"
                first_hit_key = f"{team}_first_hit_step"
                if self._episode_stats.get(first_hit_key) is None:
                    self._episode_stats[first_hit_key] = int(self.current_step)
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
            elif _is_adapted_profile(self) and alt > self.arena_altitude_max_m:
                sim.crash()
                crashed = True
                reason = "Crash_HighAlt"
            elif _is_adapted_profile(self) and (
                    abs(pos[0]) > self.arena_half_width_m
                    or abs(pos[1]) > self.arena_half_width_m):
                sim.crash()
                crashed = True
                reason = "Crash_BattleVolume"
            elif _is_adapted_profile(self):
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
            _is_adapted_profile(self) and self._invalid_numerical_episode)
        return {aid: truncated for aid in self.agent_ids}

    # ------------------------------------------------------------------
    #  Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> tuple[dict, dict]:
        """Return the formal Eq.15-Eq.23 team-joint reward."""
        return self._compute_paper_joint_rewards()

    def _compute_paper_joint_rewards(self) -> tuple[dict, dict]:
        """Return the paper team-joint reward identically to every teammate."""
        if _is_adapted_profile(self) and self._invalid_numerical_episode:
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
                "reward_version": "paper_3v3_joint_eq15_23_v1",
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
            "reward_version": "paper_3v3_joint_eq15_23_v1",
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

        if _is_adapted_profile(self):
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
    #  Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        obs = {}
        for aid in self.agent_ids:
            obs[aid] = self._get_agent_obs(aid)
        return obs

    def _get_agent_obs(self, agent_id: str) -> dict:
        return self._get_agent_obs_paper_strict(agent_id)
    def _get_agent_obs_paper_strict(self, agent_id: str) -> dict:
        """Build Table 1 / Table 2 10-dim observations for reset/step."""
        sim = self._get_sim(agent_id)
        alive = sim is not None and sim.is_alive
        _ego_slots, ally_slots, enemy_slots = ordered_entity_slots(self, agent_id)
        ally_sims = [slot[1] for slot in ally_slots]
        enemy_sims = [slot[1] for slot in enemy_slots]

        if alive:
            ego_state, _meta = extract_self_state_with_meta(
                sim, require_real_alpha_beta=True)
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

        entity_mask = 1 - slot_aligned_alive_mask(self, agent_id)

        return {
            "ego_state": ego_state.astype(np.float32),
            "ally_states": ally_vecs,
            "enemy_states": enemy_vecs,
            "entity_mask": entity_mask.astype(np.int64),
        }

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
            if (self.is_paper_3v3
                    and (sim is None or not sim.is_alive)):
                self._clear_control_state(aid)
            # Return per-step delta and reset counter so callers can safely
            # accumulate without double-counting across env steps.
            delta = self._missile_launch_counts.get(aid, 0)
            self._missile_launch_counts[aid] = 0
            info[aid] = {
                "agent_id": aid,
                "team": "blue" if aid.startswith("blue") else "red",
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
            if self.is_paper_3v3:
                mws_state = self._learnable_mws_state.get(
                    aid, self._empty_mws_state())
                info[aid].update({
                    "mws_warning_generations": int(mws_state.get(
                        "warning_generation", 0)),
                    "mws_direction_changes_within_same_missile": 0,
                })
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
        mws_diag = dict(self._mws_decision_diagnostics)
        for team in ("red", "blue"):
            terminal = self._warning_to_terminal_s[team]
            hit = self._warning_to_hit_s[team]
            mws_diag[f"{team}_warning_to_terminal_mean_s"] = (
                float(np.mean(terminal)) if terminal else None)
            mws_diag[f"{team}_warning_to_terminal_p50_s"] = (
                float(np.median(terminal)) if terminal else None)
            mws_diag[f"{team}_warning_to_hit_mean_s"] = (
                float(np.mean(hit)) if hit else None)
            mws_diag[f"{team}_mws_detected_agent_decisions"] = int(
                mws_diag[f"{team}_detected_agent_decisions"])
            mws_diag[f"{team}_mws_override_agent_decisions"] = int(
                mws_diag[f"{team}_override_agent_decisions"])
            mws_diag[f"{team}_mws_warning_to_terminal_mean_s"] = mws_diag[
                f"{team}_warning_to_terminal_mean_s"]
            mws_diag[f"{team}_mws_warning_to_terminal_p50_s"] = mws_diag[
                f"{team}_warning_to_terminal_p50_s"]
            mws_diag[f"{team}_mws_warning_to_hit_mean_s"] = mws_diag[
                f"{team}_warning_to_hit_mean_s"]
        info["__mws_diag__"] = mws_diag
        if self.is_paper_3v3:
            info["__load_diag__"] = {
                "invalid_nonfinite_load_count": sum(
                    "NonFiniteLoad" in reason
                    for reason in self._invalid_numerical_reasons),
                "invalid_catastrophic_finite_load_count": sum(
                    "CatastrophicFiniteLoad" in reason
                    for reason in self._invalid_numerical_reasons),
                "invalid_persistent_extreme_finite_load_count": sum(
                    "PersistentExtremeFiniteLoad" in reason
                    for reason in self._invalid_numerical_reasons),
            }
            info["__extreme_load_traces__"] = copy.deepcopy(
                self._retained_extreme_load_traces)
        blue_policy_rows = info["__blue_policy_diag__"].get("per_blue", [])
        movement_targets = {
            str(row.get("blue_id")): row.get("assigned_target_id")
            for row in blue_policy_rows if isinstance(row, dict)
        }
        info["__target_assignment_diag__"] = {
            "movement_target_ids": movement_targets,
        }
        n_red_alive = sum(int(s.is_alive) for s in self.red_planes.values())
        n_blue_alive = sum(int(s.is_alive) for s in self.blue_planes.values())
        timeout = self.current_step >= self.max_steps
        invalid = bool(_is_adapted_profile(self) and self._invalid_numerical_episode)
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
        if self.is_paper_3v3:
            info["__episode__"].update({
                "RedMWSWarningGenerations": int(mws_diag.get(
                    "red_warning_generations", 0)),
                "RedMWSDirectionChangesWithinSameMissile": int(mws_diag.get(
                    "red_direction_changes_within_same_missile", 0)),
                "RedMWSSuppressedDirectionFlipAttempts": int(mws_diag.get(
                    "red_suppressed_direction_flip_attempts", 0)),
                "RedMWSMaximumContinuousDecisions": int(mws_diag.get(
                    "red_maximum_continuous_decisions", 0)),
                "RedMWSTargetHeadingDeltaMaxDeg": float(mws_diag.get(
                    "red_target_heading_delta_max_deg", 0.0)),
                "BlueMWSWarningGenerations": int(mws_diag.get(
                    "blue_warning_generations", 0)),
                "BlueMWSDirectionChangesWithinSameMissile": int(mws_diag.get(
                    "blue_direction_changes_within_same_missile", 0)),
                "BlueMWSSuppressedDirectionFlipAttempts": int(mws_diag.get(
                    "blue_suppressed_direction_flip_attempts", 0)),
                "BlueMWSMaximumContinuousDecisions": int(mws_diag.get(
                    "blue_maximum_continuous_decisions", 0)),
                "BlueMWSTargetHeadingDeltaMaxDeg": float(mws_diag.get(
                    "blue_target_heading_delta_max_deg", 0.0)),
            })
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
        """Return deterministic 45-degree EO visibility within 10 km."""
        distance = compute_3d_range(
            observer_sim.get_position(), target_sim.get_position())
        q_los = compute_body_x_q_los(
            observer_sim.get_position(), observer_sim.get_rpy(),
            target_sim.get_position())
        return bool(
            distance <= self.environment_config.electro_optical.maximum_range_m.value
            and q_los <= self.missile_launch_ao_thresh)

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
        if _is_adapted_profile(self):
            true_position = np.asarray(
                target_sim.get_position(), dtype=np.float64)
            estimate = true_position.copy()
            estimate[0] = np.round(
                estimate[0] / COARSE_HORIZONTAL_GRID_M
            ) * COARSE_HORIZONTAL_GRID_M
            estimate[1] = np.round(
                estimate[1] / COARSE_HORIZONTAL_GRID_M
            ) * COARSE_HORIZONTAL_GRID_M
            estimate[2] = np.round(
                estimate[2] / COARSE_ALTITUDE_GRID_M
            ) * COARSE_ALTITUDE_GRID_M
            return SensorTrack(
                "quantized_position_only", target_sim.uid, estimate,
                now, 0.0, 0.5, False, False, True)
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

    def _make_initial_jitter(self) -> dict[int, dict[str, float]]:
        count = max(self.max_num_red, self.max_num_blue)
        if (not self.is_paper_3v3
                or self.initial_condition_randomization_mode == "deterministic_v1"):
            return {
                index: {"along_m": 0.0, "lateral_m": 0.0,
                        "altitude_m": 0.0, "speed_mps": 0.0,
                        "heading_deg": 0.0}
                for index in range(count)}
        return {
            index: {
                "along_m": float(self.np_random.uniform(-250.0, 250.0)),
                "lateral_m": float(self.np_random.uniform(-100.0, 100.0)),
                "altitude_m": float(self.np_random.uniform(-100.0, 100.0)),
                "speed_mps": float(self.np_random.uniform(-10.0, 10.0)),
                "heading_deg": float(self.np_random.uniform(-2.0, 2.0)),
            }
            for index in range(count)}

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
        jitter = self._initial_jitter_by_index.get(index, {})
        formation_spacing_m = float(cfg.formation_spacing_m.value)
        lateral_jitter_m = float(jitter.get("lateral_m", 0.0))
        half_distance_m = (float(cfg.initial_head_on_range_m.value) / 2.0
                           - float(jitter.get("along_m", 0.0)))
        if _is_adapted_profile(self):
            centre_lat_rad = np.deg2rad(lat_centre)
            meridional_radius = 6_378_137.0 * (1.0 - 0.00669437999014) / (
                1.0 - 0.00669437999014 * np.sin(centre_lat_rad) ** 2) ** 1.5
            metres_per_lat_degree = (
                meridional_radius + float(cfg.initial_altitude_m.value)) * np.pi / 180.0
            lat_offset_deg = (
                ((index - (N - 1) / 2.0) * formation_spacing_m
                 + lateral_jitter_m) / metres_per_lat_degree)
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
            float(cfg.initial_altitude_m.value) if _is_adapted_profile(self) else 0.0))
        metres_per_lon_degree = radius_at_altitude * np.cos(lat_rad) * np.pi / 180.0
        half_distance_deg_lon = half_distance_m / metres_per_lon_degree

        if color == "Blue":
            heading = 90.0 + float(jitter.get("heading_deg", 0.0))
            lon = lon_centre - half_distance_deg_lon
        else:
            heading = -90.0 - float(jitter.get("heading_deg", 0.0))
            lon = lon_centre + half_distance_deg_lon

        return {
            "ic/long-gc-deg": lon,
            "ic/lat-geod-deg": lane_latitude,
            "ic/h-sl-ft": (
                float(cfg.initial_altitude_m.value)
                + float(jitter.get("altitude_m", 0.0))) / 0.3048,
            "ic/psi-true-deg": heading,
            "ic/u-fps": (
                float(cfg.initial_speed_mps.value)
                + float(jitter.get("speed_mps", 0.0))) / 0.3048,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }

    def _cleanup_missiles(self):
        done = [mid for mid, m in self._missiles_in_flight.items() if m.is_done]
        for mid in done:
            missile = self._missiles_in_flight.pop(mid)
            missile.detach_references()
        if done:
            self.refresh_engaged_targets()

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
