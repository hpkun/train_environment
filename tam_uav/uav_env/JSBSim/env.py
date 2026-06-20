"""
UavCombatEnv: Multi-agent UAV combat environment with Dict observation spaces
for zero-shot scale generalization. Uses JSBSim for flight dynamics and PID
controllers to convert high-level tactical commands to control-surface inputs.
"""
from __future__ import annotations

import logging
import numpy as np
import gymnasium

from .alignment.los_geometry import compute_3d_range, compute_body_x_q_los
from .alignment.launch_quality import (
    LAUNCH_QUALITY_FIELDS,
    make_launch_quality_record,
    nan_float as _nan_float,
)
from .alignment.reward_utils import (
    altitude_reward_pairwise_mean_eq17,
    ta_angle_advantage_fixed,
    td_distance_advantage,
)

from .simulator import AircraftSimulator, MissileSimulator
from .pid_controller import PIDController
from .utils import get2d_AO_TA_R
from .render_tacview import TacviewLogger

logger = logging.getLogger(__name__)

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
    "ammo_empty_blocked",
    "cooldown_blocked",
    "kill_cooldown_blocked",
    "engaged_blocked",
    "launches",
)


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


def collect_aircraft_state_finiteness(sim) -> tuple[bool, tuple[str, ...]]:
    """Return whether the externally consumed aircraft state is finite."""

    bad_fields: list[str] = []
    getters = (
        ("geodetic", sim.get_geodetic),
        ("position", sim.get_position),
        ("velocity", sim.get_velocity),
        ("rpy", sim.get_rpy),
    )
    for name, getter in getters:
        try:
            values = np.asarray(getter(), dtype=np.float64)
            if values.size == 0 or not np.isfinite(values).all():
                bad_fields.append(name)
        except Exception:
            bad_fields.append(name)
    return not bad_fields, tuple(bad_fields)


class UavCombatEnv(gymnasium.Env):
    """
    Multi-agent UAV combat environment (paper BRMA-MAPPO baseline).

    Two action interfaces are retained:
      - legacy_pid_3d: Box(3) absolute pitch/heading/velocity PID targets.
      - tam_direct_fcs_4d: throttle/aileron/elevator/rudder commands. The
        formal TAM configuration uses MultiDiscrete([40, 40, 40, 40]) and a
        categorical policy; it does not use PID target conversion.

    Observation space (per agent): Dict with keys
      - "ego_state"     (11,)       self state (body-frame relative)
      - "ally_states"   (max_allies-1, 11)  allied aircraft (body-frame, excluding self)
      - "enemy_states"  (max_enemies, 11)    enemy aircraft (body-frame)
      - "death_mask"    (max_allies+max_enemies,)  1=alive, 0=dead
    """

    # ---- Legacy PID action scale constants --------------------------------------
    # `legacy_pid_3d` uses absolute target values (not deltas).
    #
    #   θ ∈ (−π/2, π/2]       pitch   act[0] ∈ [-1, 1] → ±90°
    #   ψ ∈ (−π, π]           heading act[1] ∈ [-1, 1] → ±180° (absolute)
    #   V ∈ [0.3, 1.2] Mach   velocity act[2] ∈ [-1, 1] → [102, 408] m/s
    #
    # Both teams share identical action authority per paper specification.
    # GCAS for Blue is the ONLY remaining team asymmetry (hard-coded baseline
    # safety net that Red must learn through reward shaping).
    #
    # Velocity:  F-16 F100-PW-229 MilThrust ≈ 17 800 lbf; jet can sustain M0.8–1.0
    #            in level flight at 10 kft.  Mach reference: a ≈ 340 m/s at sea level,
    #            ≈ 328 m/s at 10 kft ISA.
    PITCH_DEG = 90.0             # paper §2.4: full longitudinal authority (±90°)
    VELOCITY_MIN = 102.0         # m/s TAS  (M0.30)
    VELOCITY_MAX = 408.0         # m/s TAS  (M1.20)

    MISSILE_COOLDOWN_STEPS = 30        # default 0.5 s at 60 Hz; __init__ scales with sim_freq
    MISSILE_LOCK_DELAY_FRAMES = 15     # default 0.25 s at 60 Hz; __init__ scales with sim_freq
    KILL_COOLDOWN_STEPS = 3            # env steps — same agent cannot score another kill within 3 steps
    MISSILE_LAUNCH_AO_THRESH = np.deg2rad(45)
    MISSILE_LAUNCH_RANGE_THRESH = 10000.0  # m — paper: photoelectric sensor max range
    MISSILE_LAUNCH_MIN_RANGE = 500.0      # m — minimum safe launch distance (prevents point-blank self-hit)
    MISSILE_LAUNCH_TA_THRESH = np.pi / 2   # 90° — must be in enemy rear hemisphere (3-9 line)

    # ---- Airborne radar (paper: ±60° azimuth, [-10°, +32°] elevation) ----
    RADAR_AZIMUTH_HALF = np.deg2rad(60)       # ±60° horizontal FOV
    RADAR_ELEVATION_MIN = np.deg2rad(-10)     # look-down limit
    RADAR_ELEVATION_MAX = np.deg2rad(32)      # look-up limit
    RADAR_K = 40000.0                         # range calibration constant for Rmax = K * RCS^(1/4)
    RCS_FRONTAL = 0.1                         # m² — front ±30° mean RCS
    RCS_SIDE = 2.0                            # m² — broadside RCS

    # ---- Battlefield boundaries (paper Table 4: 80×80×10 km) ----
    BATTLEFIELD_HALF_SIZE = 40000.0   # m — core area ±40 km (paper eq 18: |x|,|y| > 4×10⁴)
    BATTLEFIELD_ALTITUDE_MAX = 10000.0  # m — ceiling
    BATTLEFIELD_ALTITUDE_MIN = 2500.0   # m — floor (crash)
    OVERLOAD_G_LIMIT = 9.0             # paper Table 4: max load factor 9g
    OVERLOAD_TIME_LIMIT = 10.0         # s after which >9G triggers termination
    MAX_SPEED = 600.0                  # m/s — paper Table 4: maximum speed

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
                 sim_freq=60, agent_interaction_steps=12, max_steps=1000,
                 enable_gcas_for_blue: bool = True,
                 suppress_jsbsim_output: bool = True,
                 control_mode_by_role: dict | None = None,
                 direct_fcs_trim_by_role: dict | None = None,
                 action_interface: str = "legacy_pid_3d",
                 tam_action_distribution: str = "continuous_quantized",
                 tam_action_levels: int = 40,
                 tam_throttle_min: float = 0.4,
                 tam_throttle_max: float = 0.9,
                 scripted_evasion_red: bool = True,
                 scripted_evasion_blue: bool = False,
                 airborne_initial_state_stabilization=None,
                 render_mode=None):
        super().__init__()
        self.max_num_blue = max_num_blue
        self.max_num_red = max_num_red
        self.num_missiles_per_plane = num_missiles_per_plane
        self.enable_gcas_for_blue = enable_gcas_for_blue
        self.sim_freq = sim_freq
        self.agent_interaction_steps = agent_interaction_steps
        self.max_steps = max_steps
        self.suppress_jsbsim_output = suppress_jsbsim_output
        self.airborne_initial_state_stabilization = airborne_initial_state_stabilization or {}
        self.control_mode_by_role = dict(control_mode_by_role or {})
        self.direct_fcs_trim_by_role = dict(direct_fcs_trim_by_role or {})
        if action_interface not in {"legacy_pid_3d", "tam_direct_fcs_4d"}:
            raise ValueError(f"unknown action_interface: {action_interface}")
        if tam_action_distribution not in {
            "continuous_quantized", "multidiscrete_categorical"
        }:
            raise ValueError(f"unknown tam_action_distribution: {tam_action_distribution}")
        if tam_action_levels < 0:
            raise ValueError("tam_action_levels must be non-negative")
        if not 0.0 <= tam_throttle_min <= tam_throttle_max <= 1.0:
            raise ValueError("TAM throttle range must satisfy 0 <= min <= max <= 1")
        self.action_interface = action_interface
        self.tam_action_distribution = tam_action_distribution
        self.tam_action_levels = int(tam_action_levels)
        self.tam_throttle_min = float(tam_throttle_min)
        self.tam_throttle_max = float(tam_throttle_max)
        self.scripted_evasion_red = bool(scripted_evasion_red)
        self.scripted_evasion_blue = bool(scripted_evasion_blue)
        self._last_tam_action_commands: dict[str, dict] = {}
        self.physics_dt = 1.0 / sim_freq
        self.env_dt = agent_interaction_steps * self.physics_dt
        self.missile_cooldown_frames = int(round(0.5 * self.sim_freq))
        self.missile_lock_delay_frames = int(round(0.25 * self.sim_freq))

        # Agent ID lists (fixed order for observation construction)
        self.blue_ids = [f"blue_{i}" for i in range(max_num_blue)]
        self.red_ids = [f"red_{i}" for i in range(max_num_red)]
        self.agent_ids = self.blue_ids + self.red_ids

        # ---- Action space (Dict) ----
        if (
            self.action_interface == "tam_direct_fcs_4d"
            and self.tam_action_distribution == "multidiscrete_categorical"
        ):
            if self.tam_action_levels <= 1:
                raise ValueError("categorical TAM actions require tam_action_levels > 1")
            self.action_space = gymnasium.spaces.Dict({
                aid: gymnasium.spaces.MultiDiscrete(
                    np.full(4, self.tam_action_levels, dtype=np.int64)
                )
                for aid in self.agent_ids
            })
        else:
            action_dim = 4 if self.action_interface == "tam_direct_fcs_4d" else 3
            self.action_space = gymnasium.spaces.Dict({
                aid: gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
                )
                for aid in self.agent_ids
            })

        # ---- Observation space (Dict) ----
        obs_spaces = {}
        for i, aid in enumerate(self.blue_ids):
            obs_spaces[aid] = gymnasium.spaces.Dict({
                "ego_state": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
                "ally_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_blue - 1, 11), dtype=np.float32),
                "enemy_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_red, 11), dtype=np.float32),
                "death_mask": gymnasium.spaces.Box(
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
                    low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
                "ally_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_red - 1, 11), dtype=np.float32),
                "enemy_states": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_num_blue, 11), dtype=np.float32),
                "death_mask": gymnasium.spaces.Box(
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

        # Overload tracking
        self._overload_timers: dict[str, float] = {}

        # Missile launch counters (per-episode, for debugging)
        self._missile_launch_counts: dict[str, int] = {}
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
        self._death_events_step: list[dict] = []

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
        self.current_step = 0
        self._physics_frame = 0
        self._sim_time = 0.0
        self._missile_id_counter = 0
        self._missiles_in_flight.clear()
        self._launch_quality_records.clear()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        self._missile_acmi_id.clear()
        self._missile_term_reasons = {"red": {}, "blue": {}}
        self._next_missile_acmi_id = 1001
        if self._tacview_recorder is not None:
            self._tacview_recorder.reset()

        # Create or reload blue aircraft (reuse to avoid JSBSim C++ memory leak)
        first_reset = len(self.blue_planes) == 0
        for i in range(self.max_num_blue):
            aid = self.blue_ids[i]
            init_state = self._make_init_state("Blue", i)
            model = self._aircraft_model_for(aid, "Blue", i)
            num_missiles = self._num_missiles_for(aid)
            stab = self.airborne_initial_state_stabilization
            if first_reset:
                sim = AircraftSimulator(
                    uid=aid, color="Blue", model=model,
                    sim_freq=self.sim_freq, num_missiles=num_missiles,
                    init_state=init_state,
                    suppress_jsbsim_output=self.suppress_jsbsim_output,
                    initial_state_stabilization=stab,
                )
                self.blue_planes[aid] = sim
            else:
                self.blue_planes[aid].reload(new_state=init_state)

        # Create or reload red aircraft
        for i in range(self.max_num_red):
            aid = self.red_ids[i]
            init_state = self._make_init_state("Red", i)
            model = self._aircraft_model_for(aid, "Red", i)
            num_missiles = self._num_missiles_for(aid)
            if first_reset:
                sim = AircraftSimulator(
                    uid=aid, color="Red", model=model,
                    sim_freq=self.sim_freq, num_missiles=num_missiles,
                    init_state=init_state,
                    suppress_jsbsim_output=self.suppress_jsbsim_output,
                    initial_state_stabilization=stab,
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

        # Create or reset PID controllers
        if first_reset:
            for aid in self.agent_ids:
                self.pid_controllers[aid] = PIDController(self.physics_dt)
        else:
            for pid in self.pid_controllers.values():
                pid.reset()

        # Reset missile cooldowns
        self._missile_cooldown = {aid: 0 for aid in self.agent_ids}

        # Reset lock-delay timers
        self._lock_timer = {aid: 0 for aid in self.agent_ids}
        self._lock_target = {aid: None for aid in self.agent_ids}

        # Reset overload timers
        self._overload_timers = {aid: 0.0 for aid in self.agent_ids}

        # Reset missile launch counters
        self._missile_launch_counts = {aid: 0 for aid in self.agent_ids}
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []

        # Reset death reasons
        self._death_reasons = {}
        self._death_events_step = []

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

    def _aircraft_model_for(self, agent_id: str, color: str, index: int) -> str:
        return "f16"

    def _num_missiles_for(self, agent_id: str) -> int:
        return int(self.num_missiles_per_plane)

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

    def step(self, actions: dict):
        self.current_step += 1
        self._crashed_this_step.clear()
        self._death_events_step = []
        self._launch_diag_step = make_empty_launch_diag()
        self._launch_quality_step_records = []
        self._launch_quality_done_step_records = []
        alive_before = {
            aid: bool((sim := self._get_sim(aid)) is not None and sim.is_alive)
            for aid in self.agent_ids
        }

        # 0. Pre-compute kill-cooldown denial set (before physics loop).
        #    An agent that scored a kill within the last KILL_COOLDOWN_STEPS
        #    env steps is blocked from scoring another kill this step.
        self._agents_deny_kill = set()
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

        # 3. Check terminations
        self._check_crash_terminations()
        self._death_events_step = self._build_death_events(alive_before)

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

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    #  Action parsing
    # ------------------------------------------------------------------

    def _map_tam_direct_continuous_action(self, action) -> dict:
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != 4:
            raise ValueError(f"TAM direct-FCS action must have 4 values, got {raw.size}")
        clipped = np.clip(
            np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0
        )
        quantized = clipped.copy()
        if self.tam_action_levels > 1:
            scale = float(self.tam_action_levels - 1)
            quantized = np.round((clipped + 1.0) / 2.0 * scale) / scale * 2.0 - 1.0
        throttle = self.tam_throttle_min + (quantized[0] + 1.0) / 2.0 * (
            self.tam_throttle_max - self.tam_throttle_min
        )
        return {
            "action_distribution": "continuous_quantized",
            "raw_action": [float(value) for value in raw],
            "quantized_action": [float(value) for value in quantized],
            "throttle_cmd_norm": float(throttle),
            "aileron_cmd_norm": float(quantized[1]),
            "elevator_cmd_norm": float(quantized[2]),
            "rudder_cmd_norm": float(quantized[3]),
        }

    def _map_tam_direct_action(self, action) -> dict:
        """Compatibility alias for the legacy continuous diagnostic mapper."""
        return self._map_tam_direct_continuous_action(action)

    def _map_tam_direct_discrete_action(self, action_indices) -> dict:
        raw = np.asarray(action_indices)
        if raw.shape != (4,):
            raise ValueError(f"TAM categorical action must have shape (4,), got {raw.shape}")
        if not np.issubdtype(raw.dtype, np.number):
            raise ValueError("TAM categorical action indices must be numeric integers")
        numeric = raw.astype(np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.round(numeric)).all():
            raise ValueError("TAM categorical action indices must be losslessly convertible to int")
        indices = numeric.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= self.tam_action_levels):
            raise ValueError(
                f"TAM categorical action indices must be in [0, {self.tam_action_levels - 1}]"
            )
        levels = indices.astype(np.float64) / float(self.tam_action_levels - 1)
        return {
            "action_distribution": "multidiscrete_categorical",
            "action_indices": indices.tolist(),
            "normalized_levels": levels.tolist(),
            "throttle_cmd_norm": float(
                self.tam_throttle_min
                + levels[0] * (self.tam_throttle_max - self.tam_throttle_min)
            ),
            "aileron_cmd_norm": float(-1.0 + 2.0 * levels[1]),
            "elevator_cmd_norm": float(-1.0 + 2.0 * levels[2]),
            "rudder_cmd_norm": float(-1.0 + 2.0 * levels[3]),
        }

    def _parse_actions(self, actions: dict) -> dict:
        """Map formal direct-FCS commands or legacy PID target actions.

        The `tam_direct_fcs_4d` branch returns before the legacy layers below.
        Its categorical actions map directly to throttle, aileron, elevator,
        and rudder FCS commands.

        Legacy `legacy_pid_3d` control-flow priority (team-aware):
          Layer 1 — Missile evasion:     RED team only  (scripted)
          Layer 2 — GCAS safety net:     BLUE only   (hard-coded baseline)
          Layer 3 — Agent action:        BOTH teams  (identical §2.4 mapping)

        Legacy PID mapping, identical for both teams (absolute targets):
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

            if self.action_interface == "tam_direct_fcs_4d":
                if self.tam_action_distribution == "multidiscrete_categorical":
                    command = self._map_tam_direct_discrete_action(act)
                else:
                    command = self._map_tam_direct_continuous_action(act)
                self._last_tam_action_commands[aid] = command
                targets[aid] = command
                continue

            is_blue = aid.startswith("blue")
            rpy = sim.get_rpy()
            current_heading = float(rpy[2])  # ψ ∈ [−π, π]

            # =================================================================
            #  Layer 1 — Missile Evasion Script (paper §2.1.3)
            #
            #  RED team only.  Missile warning / scripted evasion is modeled as
            #  a red MAV/UAV formation information advantage.  The blue
            #  rule-based opponent does not use scripted missile evasion.
            # =================================================================
            incoming = None
            if not is_blue:
                incoming = sim.check_missile_warning()
            if incoming is not None and self.scripted_evasion_red:
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
                turn_dir = 1.0 if ao > 0 else -1.0

                if self._control_mode_for(aid) == "direct_fcs_3d":
                    # Direct-FCS evade: pull up + break turn + full throttle
                    elevator_sign_chosen = 1.0
                    elevator_cmd = float(np.clip(elevator_sign_chosen * 0.3, -1.0, 1.0))
                    aileron_cmd = float(np.clip(turn_dir * 0.5, -1.0, 1.0))
                    throttle_cmd = 0.9
                    targets[aid] = (elevator_cmd, aileron_cmd, throttle_cmd)
                    continue

                if alt_m > 5000.0:
                    # High altitude: break turn with ~60° bank.
                    # Pull 25° pitch while executing a ~60° heading break.
                    target_pitch = np.deg2rad(25.0)
                    target_heading = current_heading + turn_dir * np.deg2rad(60.0)
                else:
                    # Low altitude (< 5000 m): wings-level zoom climb.
                    # Pull 30° pitch, maintain current heading (roll out first).
                    target_pitch = np.deg2rad(30.0)
                    ego_roll = float(rpy[0])
                    if abs(ego_roll) > np.deg2rad(5):
                        target_heading = current_heading - np.sign(ego_roll) * np.deg2rad(15.0)
                    else:
                        target_heading = current_heading

                targets[aid] = (target_pitch, target_heading, self.VELOCITY_MAX)
                continue

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
                    continue

            # =================================================================
            #  Layer 3 — Agent Action (paper §2.4 — both teams identical)
            #
            #    target_pitch   = act[0] * 90°             ∈ [−90°, +90°]
            #    target_heading = act[1] * 180°            ∈ [−180°, +180°]  (absolute)
            #    target_velocity ∈ [102, 408] m/s
            # =================================================================
            # ---- Direct-FCS path (F22 MAV, paper-aligned) ----
            if self._control_mode_for(aid) == "direct_fcs_3d":
                elevator_cmd = float(np.clip(act[0], -1.0, 1.0))
                aileron_cmd = float(np.clip(act[1], -1.0, 1.0))
                # TAM-HAPPO throttle Ct ∈ [0.4, 0.9]
                throttle_cmd = 0.4 + (float(act[2]) + 1.0) / 2.0 * 0.5
                throttle_cmd = float(np.clip(throttle_cmd, 0.4, 0.9))
                targets[aid] = (elevator_cmd, aileron_cmd, throttle_cmd)
                continue

            target_velocity = self.VELOCITY_MIN + (float(act[2]) + 1.0) / 2.0 * (
                self.VELOCITY_MAX - self.VELOCITY_MIN)
            target_pitch = float(act[0]) * np.deg2rad(self.PITCH_DEG)
            target_heading = float(act[1]) * np.pi

            targets[aid] = (target_pitch, target_heading, target_velocity)
        return targets

    # ------------------------------------------------------------------
    #  PID control application (per physics frame)
    # ------------------------------------------------------------------

    def _control_mode_for(self, agent_id: str) -> str:
        """Return control mode for an agent from config (default: pid_target)."""
        # Resolve role: check HeteroUavCombatEnv attributes first, fallback to agent_id
        role = getattr(self, "agent_roles", {}).get(agent_id, "")
        if not role and agent_id.startswith("red_"):
            role = "mav" if agent_id == "red_0" else "attack_uav"
        return str(self.control_mode_by_role.get(role, "pid_target"))

    def _direct_fcs_trim_for(self, agent_id: str) -> dict | None:
        role = getattr(self, "agent_roles", {}).get(agent_id, "")
        if not role and agent_id.startswith("red_"):
            role = "mav" if agent_id == "red_0" else "attack_uav"
        return self.direct_fcs_trim_by_role.get(role)

    def _apply_pid_controls(self, targets: dict):
        """Read current flight state, compute BTT PID, write to JSBSim.

        For agents with control_mode = "direct_fcs_3d", the target tuple
        carries pre-mapped FCS commands directly — no PID computation.
        """
        for aid, target in targets.items():
            if target is None:
                continue
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue

            control_mode = self._control_mode_for(aid)
            if self.action_interface == "tam_direct_fcs_4d":
                sim.set_property_value("fcs/throttle-cmd-norm", target["throttle_cmd_norm"])
                sim.set_property_value("fcs/aileron-cmd-norm", target["aileron_cmd_norm"])
                sim.set_property_value("fcs/elevator-cmd-norm", target["elevator_cmd_norm"])
                sim.set_property_value("fcs/rudder-cmd-norm", target["rudder_cmd_norm"])
                continue
            if control_mode == "direct_fcs_3d":
                # target = (elevator_cmd, aileron_cmd, throttle_cmd)
                elevator, aileron, throttle_cmd = target
                rudder = 0.0
                # Apply direct-FCS trim if configured
                trim = self._direct_fcs_trim_for(aid)
                if trim is not None:
                    elevator += trim.get("elevator", 0.0)
                elevator = float(np.clip(elevator, -1.0, 1.0))
                aileron = float(np.clip(aileron, -1.0, 1.0))
                throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))
                sim.set_property_value("fcs/aileron-cmd-norm", aileron)
                sim.set_property_value("fcs/elevator-cmd-norm", elevator)
                sim.set_property_value("fcs/rudder-cmd-norm", rudder)
                sim.set_property_value("fcs/throttle-cmd-norm", throttle_cmd)
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

            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            sim.set_property_value("fcs/throttle-cmd-norm", throttle)

    # ------------------------------------------------------------------
    #  Physics stepping
    # ------------------------------------------------------------------

    def _run_one_physics_frame(self):
        """Advance every alive aircraft by one JSBSim frame."""
        for sim in self._all_sims():
            if sim.is_alive:
                sim.run()

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
            diag["alive_shooters"] += 1
            # Decrement cooldown every physics frame
            if self._missile_cooldown[aid] > 0:
                self._missile_cooldown[aid] -= 1

            if sim.num_left_missiles <= 0:
                diag["ammo_empty_blocked"] += 1
                self._lock_timer[aid] = 0
                self._lock_target[aid] = None
                continue

            # ---- Shared engaged-targets set (hot-updated across agents) ----
            # Uses self._engaged_targets directly — no per-agent recomputation.
            # The set contains enemy UIDs that have an in-flight friendly
            # missile tracking them AND targets flight-assigned by the
            # coordinated-actions allocator.

            # ---- Find the closest UNENGAGED enemy in the launch cone ----
            enemies = self.red_planes if sim.color == "Blue" else self.blue_planes
            best_enemy = None
            best_distance = float("inf")

            for enemy_sim in enemies.values():
                if not enemy_sim.is_alive:
                    continue
                diag["alive_enemy_pairs"] += 1
                # --- Target-deconfliction: skip enemies already engaged ---
                if enemy_sim.uid in self._engaged_targets:
                    diag["engaged_blocked"] += 1
                    continue
                diag["unengaged_enemy_pairs"] += 1

                ego_pos = sim.get_position()
                ego_vel = sim.get_velocity()
                enm_pos = enemy_sim.get_position()
                enm_vel = enemy_sim.get_velocity()

                ego_feat = np.array([ego_pos[0], ego_pos[1], -ego_pos[2],
                                     ego_vel[0], ego_vel[1], -ego_vel[2]])
                enm_feat = np.array([enm_pos[0], enm_pos[1], -enm_pos[2],
                                     enm_vel[0], enm_vel[1], -enm_vel[2]])
                AO, TA, R = get2d_AO_TA_R(ego_feat, enm_feat)
                range_ok = self.MISSILE_LAUNCH_MIN_RANGE < R < self.MISSILE_LAUNCH_RANGE_THRESH
                ao_ok = AO < self.MISSILE_LAUNCH_AO_THRESH
                ta_ok = TA > self.MISSILE_LAUNCH_TA_THRESH
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

            # ---- Launch when lock mature, weapon ready, and shooter
            #      is not on kill cooldown ----
            # (best_enemy is already guaranteed unengaged by the filter above)
            on_kill_cooldown = aid in self._agents_deny_kill
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

    def _build_launch_quality_record(
        self,
        shooter: AircraftSimulator,
        target: AircraftSimulator,
        range_m: float | None = None,
    ) -> dict:
        """Build a launch-quality snapshot without affecting launch decisions."""

        team = "red" if shooter.uid.startswith("red") else "blue"
        target_team = "red" if target.uid.startswith("red") else "blue"
        roles = getattr(self, "agent_roles", {})
        models = getattr(self, "agent_models", {})
        try:
            shooter_pos = shooter.get_position()
            shooter_vel = shooter.get_velocity()
            target_pos = target.get_position()
            target_vel = target.get_velocity()
            shooter_feat = np.array([shooter_pos[0], shooter_pos[1], -shooter_pos[2],
                                     shooter_vel[0], shooter_vel[1], -shooter_vel[2]])
            target_feat = np.array([target_pos[0], target_pos[1], -target_pos[2],
                                    target_vel[0], target_vel[1], -target_vel[2]])
            ao, ta, r = get2d_AO_TA_R(shooter_feat, target_feat)
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
            shooter_team=team,
            shooter_id=shooter.uid,
            shooter_role=str(roles.get(shooter.uid, "")),
            shooter_model=str(models.get(shooter.uid, getattr(shooter, "model", ""))),
            target_id=target.uid,
            target_team=target_team,
            target_role=str(roles.get(target.uid, "")),
            target_model=str(models.get(target.uid, getattr(target, "model", ""))),
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
            shooter_num_left_before_launch=int(shooter.num_left_missiles),
            shooter_num_left_after_launch="",
        )

    def _launch_missile(
        self,
        parent: AircraftSimulator,
        target: AircraftSimulator,
        launch_quality: dict | None = None,
    ):
        missile = MissileSimulator.create(parent, target, f"m{self._missile_id_counter}")
        self._missile_id_counter += 1
        self._missiles_in_flight[missile.uid] = missile
        self._missile_acmi_id[missile.uid] = self._next_missile_acmi_id
        self._next_missile_acmi_id += 1
        self._missile_cooldown[parent.uid] = self.missile_cooldown_frames
        parent.num_left_missiles = max(0, parent.num_left_missiles - 1)  # fire-for-effect tracking (capacity 999)
        self._missile_launch_counts[parent.uid] += 1
        if launch_quality is not None:
            launch_quality["missile_id"] = missile.uid
            launch_quality["shooter_num_left_after_launch"] = int(parent.num_left_missiles)
            self._launch_quality_records[missile.uid] = launch_quality
            self._launch_quality_step_records.append(dict(launch_quality))

    def _finalize_launch_quality_record(self, missile: MissileSimulator) -> None:
        """Attach missile termination diagnostics to its launch snapshot."""

        record = self._launch_quality_records.get(missile.uid)
        if record is None or record.get("termination_reason"):
            return
        raw_reason = missile._termination_reason or ("hit" if missile.is_success else "unknown")
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
            "termination_reason": raw_reason,          # no longer folded to generic "miss"
            "is_success": bool(missile.is_success),
            "flight_time_sec": float(getattr(missile, "_t", _nan_float())),
            "termination_step": int(self.current_step),
            "step_delta": step_delta,
            "target_alive_at_termination": target_alive,
        })
        self._launch_quality_done_step_records.append(dict(record))

    def _update_missiles(self):
        """Advance all in-flight missiles and process hit/miss events.

        Kill-cooldown enforcement (paper §2.1.3: 0.5 s between kills):
          - If the shooter is on kill cooldown, the hit is overridden to MISS
            and the target is revived (its shotdown is reversed).
          - Single-target lock: each agent may score at most ONE kill per env
            step, preventing "AOE" multi-target damage when several missiles
            from the same shooter arrive in the same physics window.
        """
        for mid, missile in list(self._missiles_in_flight.items()):
            was_done_before = missile.is_done
            if not missile.is_done:
                missile.run()

            if missile.is_success and not missile._kill_rewarded:
                shooter_id = missile._parent_id

                # ---- Kill-cooldown gate ----
                # Shooter has scored a kill too recently → override to MISS.
                if shooter_id in self._agents_deny_kill:
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
                if self._step_kill_count.get(shooter_id, 0) >= 1:
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
                # Record FINAL termination reason AFTER kill_cooldown / multi_kill
                # gates have potentially overridden the status.
                team = "red" if missile._parent_id.startswith("red") else "blue"
                reason = missile._termination_reason or "unknown"
                self._missile_term_reasons[team][reason] = \
                    self._missile_term_reasons[team].get(reason, 0) + 1

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
            else:
                self._overload_timers[aid] = max(0.0, self._overload_timers[aid] - self.physics_dt)

    # ------------------------------------------------------------------
    #  Termination checks
    # ------------------------------------------------------------------

    def _check_crash_terminations(self):
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            if sim is None or not sim.is_alive:
                continue

            state_finite, _bad_fields = collect_aircraft_state_finiteness(sim)
            if not state_finite:
                sim.crash()
                self._crashed_this_step.add(aid)
                if aid not in self._death_reasons:
                    self._death_reasons[aid] = "Crash_NonFiniteState"
                continue

            crashed = False
            reason = None

            alt = sim.get_geodetic()[2]
            if alt < self.BATTLEFIELD_ALTITUDE_MIN:
                sim.crash()
                crashed = True
                reason = "Crash_LowAlt"
            elif self._overload_timers[aid] > self.OVERLOAD_TIME_LIMIT:
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

    def _missile_hit_record_for_target(self, agent_id: str) -> dict | None:
        for record in self._launch_quality_done_step_records:
            if record.get("target_id") != agent_id:
                continue
            if record.get("termination_reason") != "hit":
                continue
            return record
        return None

    def _death_event_for_agent(self, agent_id: str) -> dict:
        sim = self._get_sim(agent_id)
        side = "red" if agent_id.startswith("red_") else "blue"
        reason = self._death_reasons.get(agent_id)
        missile_record = self._missile_hit_record_for_target(agent_id)
        low_altitude = None
        over_g = None
        out_of_bounds = None
        crash = None
        altitude = speed = roll_deg = pitch_deg = heading_deg = None
        if sim is not None:
            try:
                altitude = float(sim.get_geodetic()[2])
                low_altitude = bool(altitude < self.BATTLEFIELD_ALTITUDE_MIN)
            except Exception:
                pass
            try:
                velocity = np.asarray(sim.get_velocity(), dtype=np.float64)
                speed = float(np.linalg.norm(velocity))
            except Exception:
                pass
            try:
                roll, pitch, heading = sim.get_rpy()
                roll_deg = float(np.degrees(roll))
                pitch_deg = float(np.degrees(pitch))
                heading_deg = float(np.degrees(heading))
            except Exception:
                pass
            try:
                pos = sim.get_position()
                out_of_bounds = bool(
                    abs(float(pos[0])) > self.BATTLEFIELD_HALF_SIZE
                    or abs(float(pos[1])) > self.BATTLEFIELD_HALF_SIZE
                )
            except Exception:
                pass
            over_g = bool(agent_id in self._crashed_this_step and reason == "Crash_OverG")
            crash = bool(agent_id in self._crashed_this_step)

        if missile_record is not None:
            death_reason = "missile_hit"
            source = "missile_term"
        elif reason:
            death_reason = str(reason)
            source = "existing_info"
        elif low_altitude:
            death_reason = "low_altitude_or_crash"
            source = "state_heuristic"
        else:
            death_reason = "unknown_environment_death"
            source = "unknown"

        roles = getattr(self, "agent_roles", {})
        models = getattr(self, "agent_models", {})
        return {
            "agent_id": agent_id,
            "step": int(self.current_step),
            "side": side,
            "role": str(roles.get(agent_id, "")),
            "aircraft_model": str(models.get(agent_id, getattr(sim, "model", ""))) if sim is not None else None,
            "death_reason": death_reason,
            "death_reason_source": source,
            "killed_by_missile": bool(missile_record is not None),
            "missile_owner": missile_record.get("shooter_id") if missile_record else None,
            "missile_target": missile_record.get("target_id") if missile_record else None,
            "low_altitude": low_altitude,
            "over_g": over_g,
            "out_of_bounds": out_of_bounds,
            "crash": crash,
            "altitude": altitude,
            "speed": speed,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "heading_deg": heading_deg,
        }

    def _build_death_events(self, alive_before: dict[str, bool]) -> list[dict]:
        events: list[dict] = []
        for aid in self.agent_ids:
            sim = self._get_sim(aid)
            alive_now = bool(sim is not None and sim.is_alive)
            if alive_before.get(aid, False) and not alive_now:
                events.append(self._death_event_for_agent(aid))
        return events

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
        return {aid: self.current_step >= self.max_steps for aid in self.agent_ids}

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
        mach = v / 340.0
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

        Uses 3D body-x q_LOS (paper Table 2 geometry) instead of the old
        2D horizontal AO/TA.  ``q_ij`` is the angle between ego's body
        x-axis and the LOS to the enemy; ``q_ji`` is the same from the
        enemy's perspective.  Distance is 3D Euclidean.
        """
        ego_pos = ego_sim.get_position()
        ego_rpy = ego_sim.get_rpy()

        enemies = self.red_planes if ego_sim.color == "Blue" else self.blue_planes
        total = 0.0
        for enemy_sim in enemies.values():
            if not enemy_sim.is_alive:
                continue
            enemy_pos = enemy_sim.get_position()
            enemy_rpy = enemy_sim.get_rpy()

            q_ij = compute_body_x_q_los(ego_pos, ego_rpy, enemy_pos)
            q_ji = compute_body_x_q_los(enemy_pos, enemy_rpy, ego_pos)
            d_3d = compute_3d_range(ego_pos, enemy_pos)

            Ta_ij = ta_angle_advantage_fixed(np.rad2deg(q_ij))
            Td_ij = td_distance_advantage(d_3d)
            Ta_ji = ta_angle_advantage_fixed(np.rad2deg(q_ji))

            total += 1.0 * Ta_ij * Td_ij - 0.8 * Ta_ji * Td_ij

        return total

    def _altitude_reward(self, sim: AircraftSimulator) -> float:
        """Paper eq.17-style pairwise relative altitude reward."""
        alt_ego = sim.get_geodetic()[2]
        enemies = self.red_planes if sim.color == "Blue" else self.blue_planes
        enemy_alts = [s.get_geodetic()[2] for s in enemies.values() if s.is_alive]
        if not enemy_alts:
            return 0.0

        return altitude_reward_pairwise_mean_eq17(alt_ego, enemy_alts)

    def _boundary_penalty(self, sim: AircraftSimulator) -> float:
        """Horizontal battlefield boundary penalty.

        Paper eq.18: return a fixed -10 if either |x| or |y| exceeds 4e4.
        The penalty is not accumulated per axis.
        """
        pos = sim.get_position()
        x, y = pos[0], pos[1]
        if abs(x) > self.BATTLEFIELD_HALF_SIZE or abs(y) > self.BATTLEFIELD_HALF_SIZE:
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
        sim = self._get_sim(agent_id)
        alive = sim is not None and sim.is_alive
        color = "Blue" if agent_id.startswith("blue") else "Red"

        # Gather all sims sorted by ID for consistent ordering
        blue_sims = [self.blue_planes[bid] for bid in self.blue_ids]
        red_sims = [self.red_planes[rid] for rid in self.red_ids]

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
            ego_vel_body = PIDController.matvec3(R_BI, ego_vel_ned)
            # Pseudo-NED for _make_entity_vec: body x→north, body y→east, −body z→up
            ego_pos_bf = np.zeros(3, dtype=np.float64)
            ego_vel_bf = np.array([ego_vel_body[0], ego_vel_body[1], -ego_vel_body[2]],
                                  dtype=np.float64)
        else:
            ego_state = np.zeros(11, dtype=np.float32)

        # ---- ally_states ----
        if color == "Blue":
            ally_sims = [s for s in blue_sims if s.uid != agent_id]
            max_allies = self.max_num_blue - 1
        else:
            ally_sims = [s for s in red_sims if s.uid != agent_id]
            max_allies = self.max_num_red - 1

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
        enemy_sims = red_sims if color == "Blue" else blue_sims
        max_enemies = self.max_num_red if color == "Blue" else self.max_num_blue

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
                    delta_body = PIDController.matvec3(R_BI, delta_ned)
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

        # ---- death_mask ----
        all_sims_ordered = blue_sims + red_sims
        death_mask = np.array([1 if s.is_alive else 0 for s in all_sims_ordered], dtype=np.int64)

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
            "death_mask": death_mask,
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
        delta_body = PIDController.matvec3(R_BI, delta_ned)

        # Target velocity in body frame
        tgt_vn, tgt_ve, tgt_vu = tgt_vel_ned
        tgt_vel_ned_vec = np.array([tgt_vn, tgt_ve, -tgt_vu], dtype=np.float64)
        tgt_vel_body = PIDController.matvec3(R_BI, tgt_vel_ned_vec)

        # Express in pseudo-NED: body x→north, body y→east, −body z→up
        tgt_pos_bf = np.array([delta_body[0], delta_body[1], -delta_body[2]],
                              dtype=np.float64)
        tgt_vel_bf = np.array([tgt_vel_body[0], tgt_vel_body[1], -tgt_vel_body[2]],
                              dtype=np.float64)

        return _make_entity_vec(ego_pos_bf, ego_vel_bf,
                                tgt_pos_bf, tgt_vel_bf, tgt_rpy, True)

    def _get_info(self, reward_components: dict | None = None) -> dict:
        info = {}
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
            }
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
        info["death_events"] = [dict(event) for event in self._death_events_step]
        return info

    # ------------------------------------------------------------------
    #  Radar / Sensor model (paper: partial observability)
    # ------------------------------------------------------------------

    def _compute_radar_max_range(self, TA: float) -> float:
        """RCS-based radar range using paper Rmax = K * RCS^(1/4).

        The paper uses z-axis/y-axis angular RCS table interpolation, but the
        table values are not provided. This environment keeps the existing
        front-low-RCS / side-high-RCS approximation and only aligns the Rmax
        relation to the fourth root of RCS.
        """
        ta_abs_deg = np.rad2deg(TA)

        if ta_abs_deg <= 30.0:
            rcs = self.RCS_FRONTAL                              # 0.1 — front deadzone
        elif ta_abs_deg <= 90.0:
            frac = (ta_abs_deg - 30.0) / 60.0                   # 0.0 → 1.0
            rcs = self.RCS_FRONTAL + (self.RCS_SIDE - self.RCS_FRONTAL) * frac
        elif ta_abs_deg <= 150.0:
            frac = (150.0 - ta_abs_deg) / 60.0                  # 1.0 → 0.0
            rcs = self.RCS_FRONTAL + (self.RCS_SIDE - self.RCS_FRONTAL) * frac
        else:
            rcs = self.RCS_FRONTAL                              # 0.1 — rear deadzone

        return self.RADAR_K * np.power(rcs, 0.25)

    def _is_detected_by_radar(self, ego_sim: AircraftSimulator,
                              enemy_sim: AircraftSimulator) -> bool:
        """True if *enemy_sim* is within ego's radar FOV AND detection range.

        Radar FOV (paper):
          - Azimuth: ±60°  (120° forward sector)
          - Elevation: [-10°, +32°]  (body-frame, approx world-frame since
            F-16 pitch is moderate in GCAS-protected flight)

        Detection range is RCS-dependent (see ``_compute_radar_max_range``).

        Radar CANNOT detect missiles — only aircraft.
        """
        ego_pos = ego_sim.get_position()
        ego_rpy = ego_sim.get_rpy()
        enm_pos = enemy_sim.get_position()

        # ---- vector ego → target (NEU) ----
        dn = enm_pos[0] - ego_pos[0]
        de = enm_pos[1] - ego_pos[1]
        du = enm_pos[2] - ego_pos[2]

        R_h = np.hypot(dn, de)
        R_3d = np.sqrt(R_h * R_h + du * du)
        if R_3d < 1e-6:
            return True

        # ---- azimuth check (horizontal plane) ----
        los_az = np.arctan2(de, dn)
        ego_yaw = ego_rpy[2]
        az_error = (los_az - ego_yaw + np.pi) % (2.0 * np.pi) - np.pi  # → [-π, π]
        if abs(az_error) > self.RADAR_AZIMUTH_HALF:
            return False

        # ---- elevation check ----
        los_el = np.arctan2(du, R_h)
        ego_pitch = ego_rpy[1]
        el_relative = los_el - ego_pitch
        if el_relative < self.RADAR_ELEVATION_MIN or el_relative > self.RADAR_ELEVATION_MAX:
            return False

        # ---- RCS-dependent range check ----
        ego_vel = ego_sim.get_velocity()
        enm_vel = enemy_sim.get_velocity()
        ego_feat = np.array([ego_pos[0], ego_pos[1], -ego_pos[2],
                             ego_vel[0], ego_vel[1], -ego_vel[2]], dtype=np.float64)
        enm_feat = np.array([enm_pos[0], enm_pos[1], -enm_pos[2],
                             enm_vel[0], enm_vel[1], -enm_vel[2]], dtype=np.float64)
        _, TA, _ = get2d_AO_TA_R(ego_feat, enm_feat)

        R_max = self._compute_radar_max_range(TA)
        return R_3d <= R_max

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
        from .alignment.state_extractor import \
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
        N = self.max_num_red if color == "Red" else self.max_num_blue
        lon_centre = 120.0
        lat_centre = 60.0
        formation_spacing_m = 500.0
        half_distance_km = 5.0                              # ½ of 10 km
        half_distance_deg_lon = half_distance_km / 55.66    # ≈ 0.0898°

        lat_offset_deg = (index - (N - 1) / 2.0) * formation_spacing_m / 111320.0

        if color == "Blue":
            heading = 90.0   # fly east
            lon = lon_centre - half_distance_deg_lon
        else:
            heading = -90.0  # fly west
            lon = lon_centre + half_distance_deg_lon

        return {
            "ic/long-gc-deg": lon,
            "ic/lat-geod-deg": lat_centre + lat_offset_deg,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": heading,
            "ic/u-fps": 1000.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        }

    def _cleanup_missiles(self):
        done = [mid for mid, m in self._missiles_in_flight.items() if m.is_done]
        for mid in done:
            del self._missiles_in_flight[mid]

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
        for sim in self._all_sims():
            sim.close()
        self.blue_planes.clear()
        self.red_planes.clear()
