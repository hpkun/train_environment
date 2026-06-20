"""
PID flight controller implementing Bank-to-Turn (BTT) logic per paper §2.4.

Converts high-level tactical commands (target_pitch, target_heading, target_velocity)
to JSBSim control-surface commands (aileron, elevator, rudder, throttle).

Paper reference:
  - Formula (12): desired inertial direction vector d_I_des
  - Formula (13): body-frame direction d_B_des = R_BI · d_I_des
  - roll_error  e_φ = arctan(d_B_des[1] / d_B_des[2])   → arctan2(y, z)
  - pitch_error e_θ = arctan(−d_B_des[2] / d_B_des[0])   → arctan2(−z, x)

Three PID loops:
  - Roll PID:    roll_error  e_φ  → aileron_cmd   [−1, 1]
  - Pitch PID:   pitch_error e_θ  → elevator_cmd  [−1, 1]
  - Velocity PID: velocity_error → throttle_cmd   [0, 1]

Rudder is hard-locked to 0 per paper specification.
"""
import numpy as np


class PIDLoop:
    """Single PID controller with back-calculation anti-windup.

    Anti-windup logic (paper-consistent):
      - When P+D already saturates the output, the integral is frozen
        (preventing "integrator lock" during sustained large errors).
      - When P+I+D would saturate, the integral is clamped so total output
        lands exactly at the limit — the integral unwinds immediately when
        the error reverses sign.
      - An absolute safety ceiling prevents unbounded growth in edge cases.
    """

    def __init__(self, kp, ki, kd, output_min, output_max, name=""):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.name = name
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0

    def step(self, error, dt):
        # Proportional
        p = self.kp * error

        # Derivative (computed before integral clamp so d-term informs anti-windup)
        d = self.kd * (error - self._prev_error) / max(dt, 1e-8)
        self._prev_error = error

        # Accumulate integral
        self._integral += self.ki * error * dt

        # -----------------------------------------------------------------
        #  Back-calculation anti-windup
        #
        #  Case 1: P+D already saturated → freeze integral in that direction
        #  Case 2: P+D within range → clamp integral so P+I+D ∈ [min, max]
        #  Case 3: absolute safety ceiling (belt-and-suspenders)
        # -----------------------------------------------------------------
        pd = p + d
        if pd >= self.output_max:
            # Saturated HIGH — integral must be ≤ 0
            if self._integral > 0.0:
                self._integral = 0.0
        elif pd <= self.output_min:
            # Saturated LOW — integral must be ≥ 0
            if self._integral < 0.0:
                self._integral = 0.0
        else:
            # P+D within range — clamp integral so pd + i ∈ [min, max]
            i_max_allowed = self.output_max - pd
            i_min_allowed = self.output_min - pd
            if self._integral > i_max_allowed:
                self._integral = i_max_allowed
            elif self._integral < i_min_allowed:
                self._integral = i_min_allowed

        # Absolute safety ceiling (prevents runaway in extreme transients)
        i_safety = max(abs(self.output_max), abs(self.output_min)) * 3.0 / max(self.ki, 1e-8)
        self._integral = np.clip(self._integral, -i_safety, i_safety)

        i = self._integral
        output = float(np.clip(p + i + d, self.output_min, self.output_max))
        self._prev_output = output
        return output


class PIDController:
    """
    Bank-to-Turn (BTT) three-loop PID flight controller (paper §2.4).

    - Roll PID:    roll_error  → aileron_cmd   [−1, 1]
    - Pitch PID:   pitch_error → elevator_cmd  [−1, 1]
    - Velocity PID: vel_error  → throttle_cmd  [0, 1]
    - Rudder:      0.0 (hard-locked per paper)

    Gains are configurable via kwargs; defaults are tuned for F-16.
    """

    # Default F-16 PID gains (class-level for profile reuse)
    F16_DEFAULT_GAINS = {
        "roll_kp": 0.15,  "roll_ki": 0.5,  "roll_kd": 0.05,
        "pitch_kp": 2.5,  "pitch_ki": 0.5,  "pitch_kd": 0.1,
        "vel_kp": 0.04,   "vel_ki": 0.01,   "vel_kd": 0.003,
        "elevator_sign": -1,    # F-16 FCS: positive cmd → pitch DOWN
        "throttle_min": 0.0,    # no floor
        "throttle_max": 1.0,
    }

    def __init__(self, dt, debug: bool = False,
                 roll_kp=None, roll_ki=None, roll_kd=None,
                 pitch_kp=None, pitch_ki=None, pitch_kd=None,
                 vel_kp=None, vel_ki=None, vel_kd=None,
                 elevator_sign=None,
                 throttle_min=None, throttle_max=None):
        self.dt = dt
        self._debug = debug
        self._debug_step = 0          # throttled debug counter
        self._prev_target_heading = None   # for low-pass filter (Fix 2)
        self._prev_roll_error = None        # for D-term guard (clipped-error jump detection)

        # Resolve gains: explicit arg > class default
        g = self.F16_DEFAULT_GAINS
        _rkp, _rki, _rkd = (roll_kp if roll_kp is not None else g["roll_kp"]), (roll_ki if roll_ki is not None else g["roll_ki"]), (roll_kd if roll_kd is not None else g["roll_kd"])
        _pkp, _pki, _pkd = (pitch_kp if pitch_kp is not None else g["pitch_kp"]), (pitch_ki if pitch_ki is not None else g["pitch_ki"]), (pitch_kd if pitch_kd is not None else g["pitch_kd"])
        _vkp, _vki, _vkd = (vel_kp if vel_kp is not None else g["vel_kp"]), (vel_ki if vel_ki is not None else g["vel_ki"]), (vel_kd if vel_kd is not None else g["vel_kd"])

        self.elevator_sign = float(elevator_sign if elevator_sign is not None else g["elevator_sign"])
        self.throttle_min = float(throttle_min if throttle_min is not None else g["throttle_min"])
        self.throttle_max = float(throttle_max if throttle_max is not None else g["throttle_max"])

        # --- Roll PID (drives aileron) ---
        self._roll_pid = PIDLoop(
            kp=_rkp, ki=_rki, kd=_rkd,
            output_min=-1.0, output_max=1.0,
            name="roll",
        )

        # --- Pitch PID (drives elevator) ---
        self._pitch_pid = PIDLoop(
            kp=_pkp, ki=_pki, kd=_pkd,
            output_min=-1.0, output_max=1.0,
            name="pitch",
        )

        # --- Velocity PID ---
        self._velocity_pid = PIDLoop(
            kp=_vkp, ki=_vki, kd=_vkd,
            output_min=self.throttle_min, output_max=self.throttle_max,
            name="velocity",
        )

    def reset(self):
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._velocity_pid.reset()
        self._prev_target_heading = None     # clear heading LPF state
        self._prev_roll_error = None         # clear D-term guard state

    # ------------------------------------------------------------------
    #  Rotation matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _r_x(angle: float) -> np.ndarray:
        """Elementary rotation about x-axis (roll)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c,   -s],
                         [0.0, s,    c]], dtype=np.float64)

    @staticmethod
    def _r_y(angle: float) -> np.ndarray:
        """Elementary rotation about y-axis (pitch)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c,   0.0,  s],
                         [0.0, 1.0, 0.0],
                         [-s,  0.0,  c]], dtype=np.float64)

    @staticmethod
    def _r_z(angle: float) -> np.ndarray:
        """Elementary rotation about z-axis (yaw)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c,   -s, 0.0],
                         [s,    c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def matmul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Small 3x3 matrix multiply without BLAS-backed dot/matmul."""

        return np.array([
            [
                a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j]
                for j in range(3)
            ]
            for i in range(3)
        ], dtype=np.float64)

    @staticmethod
    def matvec3(a: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Small 3x3 matrix-vector multiply without BLAS-backed dot/matmul."""

        return np.array([
            a[i, 0] * v[0] + a[i, 1] * v[1] + a[i, 2] * v[2]
            for i in range(3)
        ], dtype=np.float64)

    @classmethod
    def body_to_ned_matrix(cls, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Rotation matrix from body frame to NED inertial frame.

        Aerospace Z-Y-X Euler sequence: R_IB = R_z(yaw) · R_y(pitch) · R_x(roll).

        Body axes: x=forward, y=right, z=down.
        NED axes:  x=North,  y=East,  z=Down.
        """
        return cls.matmul3(cls.matmul3(cls._r_z(yaw), cls._r_y(pitch)), cls._r_x(roll))

    @classmethod
    def ned_to_body_matrix(cls, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Rotation matrix from NED inertial frame to body frame.

        R_BI = R_IB^T = R_x(−roll) · R_y(−pitch) · R_z(−yaw).
        """
        return cls.body_to_ned_matrix(roll, pitch, yaw).T

    # ------------------------------------------------------------------
    #  BTT control computation (paper formulas 12–13)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Gimbal-safe R_BI (Fix 3) — velocity-vector construction
    # ------------------------------------------------------------------
    @staticmethod
    def _r_bi_from_velocity(ned_velocity: np.ndarray) -> np.ndarray:
        """Build body→NED rotation from velocity vector, avoiding Euler angles.

        When pitch approaches ±90° the Z-Y-X Euler sequence degenerates
        (gimbal lock).  The velocity vector is always well-defined and
        provides a stable estimate of the body x-axis direction in NED.

        Construction (Gram-Schmidt):
          x_ned = normalize(velocity)            — body forward
          y_ned = normalize([0,0,1] × x_ned)     — body right
          z_ned = x_ned × y_ned                  — body down (orthogonal)
          R_IB  = [x_ned | y_ned | z_ned]        — columns
          R_BI  = R_IB^T
        """
        speed = float(np.linalg.norm(ned_velocity))
        if speed < 1e-6:
            return np.eye(3, dtype=np.float64)

        x_ned = ned_velocity / speed                          # body x in NED
        world_down = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_ned = np.cross(world_down, x_ned)                   # body y (right)
        y_norm = float(np.linalg.norm(y_ned))
        if y_norm < 1e-9:
            # Velocity is exactly vertical — fall back to a default horizontal
            # right vector (any horizontal direction perpendicular to x_ned).
            y_ned = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_ned = np.cross(x_ned, y_ned)
            y_ned = y_ned / float(np.linalg.norm(y_ned))
        else:
            y_ned = y_ned / y_norm
        z_ned = np.cross(x_ned, y_ned)                        # body z (down)
        z_ned = z_ned / float(np.linalg.norm(z_ned))

        R_IB = np.column_stack([x_ned, y_ned, z_ned])
        return R_IB.T                                          # R_BI = R_IB^T

    def compute_control(self, current_rpy, current_velocity,
                        target_pitch, target_heading, target_velocity,
                        ned_velocity=None):
        """
        Args:
            current_rpy:       (roll φ, pitch θ, yaw ψ)  — radians
            current_velocity:  true airspeed               — m/s  (scalar)
            target_pitch:      desired absolute pitch      — radians  [−π/2, π/2]
            target_heading:    desired absolute heading    — radians  [−π, π]
            target_velocity:   desired true airspeed       — m/s
            ned_velocity:      (vn, ve, vd) in NED         — m/s  (optional, for Fix 3)

        Returns:
            (aileron, elevator, rudder, throttle) — all in [−1, 1]
        """
        roll, pitch, yaw = float(current_rpy[0]), float(current_rpy[1]), float(current_rpy[2])

        # =================================================================
        #  Fix 1 — PITCH GIMBAL PROTECTION
        #
        #  When |pitch| > 85° the Z-Y-X Euler angles degenerate (gimbal
        #  lock).  Roll and yaw become indistinguishable — the BTT arctan2
        #  errors are meaningless and the PID fights itself.
        #
        #  Strategy: FREEZE all control surfaces at neutral (0.0) and
        #  reset PID integrators.  The aircraft passively weathervanes
        #  under natural aerodynamic stability until pitch drops below
        #  the threshold, at which point normal BTT resumes.
        # =================================================================
        GIMBAL_LOCK_THRESHOLD = np.deg2rad(85.0)
        if abs(pitch) > GIMBAL_LOCK_THRESHOLD:
            # Reset all integrators and derivative memory so we start
            # clean when the aircraft exits the vertical zone.
            self._roll_pid.reset()
            self._pitch_pid.reset()
            self._velocity_pid.reset()
            if self._debug:
                self._debug_step += 1
                if self._debug_step % 300 == 0:
                    print(
                        f"[PID GIMBAL] step={self._debug_step} "
                        f"pitch={np.rad2deg(pitch):.1f}° → surfaces neutral, PIDs reset",
                        flush=True,
                    )
            return 0.0, 0.0, 0.0, 0.0

        # =================================================================
        #  Fix 2 — HEADING LOW-PASS FILTER
        #
        #  The Actor runs at 5 Hz — target_heading step-changes every
        #  0.2 s.  These step discontinuities propagate through d_I_des
        #  → d_B_des → roll_error, causing a D-kick at the 60 Hz PID
        #  rate.  A first-order low-pass with α = 0.2 smooths the step
        #  into an exponential approach, reducing D-term excitation.
        # =================================================================
        HEADING_LPF_ALPHA = 0.2
        if self._prev_target_heading is not None:
            # Circular low-pass: follow the shortest arc
            diff = (target_heading - self._prev_target_heading + np.pi) % (2 * np.pi) - np.pi
            target_heading = self._prev_target_heading + HEADING_LPF_ALPHA * diff
            # Re-normalise to [−π, π]
            target_heading = (target_heading + np.pi) % (2 * np.pi) - np.pi
        self._prev_target_heading = float(target_heading)

        # ---- Formula (12): desired direction vector in inertial (NED) frame ----
        # Paper: d_I_des = [cos(θ)cos(ψ), cos(θ)sin(ψ), sin(θ)]^T
        # NED convention (z = Down):
        #   positive pitch θ → nose UP   → d_I_des[2] = −sin(θ)
        #   zero pitch       → level      → d_I_des[2] = 0
        #   negative pitch   → nose DOWN  → d_I_des[2] = +sin(|θ|)
        c_theta = np.cos(target_pitch)
        d_I_des = np.array([
            c_theta * np.cos(target_heading),   # North
            c_theta * np.sin(target_heading),   # East
            -np.sin(target_pitch),               # Down (= −up)
        ], dtype=np.float64)

        # ---- Formula (13): body-frame desired direction ----
        # Fix 3: when |pitch| > 80° use velocity-vector construction to
        # avoid Euler-angle gimbal-lock in R_BI.
        VELOCITY_R_BI_THRESHOLD = np.deg2rad(80.0)
        if abs(pitch) > VELOCITY_R_BI_THRESHOLD and ned_velocity is not None:
            R_BI = self._r_bi_from_velocity(np.asarray(ned_velocity, dtype=np.float64))
        else:
            R_BI = self.ned_to_body_matrix(roll, pitch, yaw)
        d_B_des = self.matvec3(R_BI, d_I_des)   # desired direction expressed in body axes

        # ---- BTT tracking errors ----
        # e_φ (roll error):  arctan2(y_body, z_body)
        #   → 0 when desired direction lies in the body x-z plane
        #
        # arctan2 ∈ [−π, π].  When d_B_des crosses the body x-z plane
        # (d_B_y sign change at d_B_z<0), arctan2 wraps ≈+π↔≈−π — a 2π
        # raw jump whose circular-distance is only a few degrees.
        #
        # We DO NOT unwrap.  Raw sign flips are physically meaningful
        # (target crossed from right to left).  Soft-clip to [−90°,90°]
        # keeps bank commands within the F-16 envelope.
        #
        # D-TERM GUARD:  the soft-clip itself can cause a 180° jump
        # (+89°→+91° clips to +89°→−89°).  We detect frame-to-frame
        # jumps >30° in the *clipped* error (F-16 max roll rate ≈ 400°/s
        # = 7°/frame @60Hz) and clear derivative memory to avoid a spike.
        roll_error_raw = float(np.arctan2(d_B_des[1], d_B_des[2] + 1e-12))
        roll_error = float(np.clip(roll_error_raw, -np.pi / 2, np.pi / 2))

        if self._prev_roll_error is not None:
            clipped_delta = abs(roll_error - self._prev_roll_error)
            if clipped_delta > np.deg2rad(30):          # >1800°/s — impossible aerodynamically
                self._roll_pid._prev_error = 0.0        # suppress D-kick
        self._prev_roll_error = float(roll_error)

        # Deadband: ignore roll errors below 1° to prevent high-frequency
        # aileron fluttering during straight-line flight or mild pursuit.
        #
        # LIMIT-CYCLE FIX: when the error enters the deadband we must also
        # zero the integral AND the previous-error memory.  Otherwise:
        #   1. residual i_term slowly pushes the aircraft out of the deadband
        #   2. when roll_error re-crosses 1°, a large D-kick fires because
        #      prev_error jumped from 0 (deadband) to >1° (active), causing
        #      the aileron to jerk → overshoot → re-enter deadband → repeat.
        ROLL_DEADBAND = np.deg2rad(1.0)   # 0.0175 rad
        if abs(roll_error) < ROLL_DEADBAND:
            roll_error = 0.0
            self._roll_pid._integral = 0.0
            self._roll_pid._prev_error = 0.0

        # e_θ (pitch error): arctan2(−z_body, x_body)
        #   → 0 when body x-axis points at d_I_des
        pitch_error = float(np.arctan2(-d_B_des[2], d_B_des[0] + 1e-12))

        # ---- Anti-inversion protection ----
        # When d_B_des[0] < 0 the target is behind the aircraft.  Without
        # correction, arctan2(-z, negative) → ±π, commanding a 180° pull-up
        # that drives the aircraft into an inverted loop or flat spin.
        #
        # Fix: clamp pitch_error to [−π/2, π/2] so the nose never pulls
        # past the vertical.  If the aircraft is nearly level in roll,
        # inject a roll bias to force a lateral turn instead of a vertical
        # loop.
        if d_B_des[0] < 0.0:
            pitch_error = float(np.clip(pitch_error, -np.pi / 2, np.pi / 2))
            if abs(roll_error) < np.deg2rad(5):
                roll_error = np.deg2rad(90.0)

        # ---- Debug logging (once per 3 s @ 60 Hz) ----
        if self._debug:
            self._debug_step += 1
            if self._debug_step % 180 == 0:
                print(
                    f"[PID DEBG] step={self._debug_step} "
                    f"pitch_rad={pitch:.4f} "
                    f"d_I_NED=({d_I_des[0]:.4f},{d_I_des[1]:.4f},{d_I_des[2]:.4f}) "
                    f"d_B_Body=({d_B_des[0]:.4f},{d_B_des[1]:.4f},{d_B_des[2]:.4f}) "
                    f"R_BI=[{R_BI[0,0]:.3f},{R_BI[0,1]:.3f},{R_BI[0,2]:.3f}|"
                    f"{R_BI[1,0]:.3f},{R_BI[1,1]:.3f},{R_BI[1,2]:.3f}|"
                    f"{R_BI[2,0]:.3f},{R_BI[2,1]:.3f},{R_BI[2,2]:.3f}] "
                    f"err=(roll={np.rad2deg(roll_error):.2f}°,raw={np.rad2deg(roll_error_raw):.1f}°"
                    f",pitch={np.rad2deg(pitch_error):.2f}°) "
                    f"anti_inv={d_B_des[0] < 0.0}",
                    flush=True,
                )

        # ---- PID outputs ----
        # Roll error → aileron (roll about body x-axis)
        aileron = self._roll_pid.step(roll_error, self.dt)

        # Pitch error → elevator
        # elevator_sign is platform-specific:
        #   F-16 FCS: positive elevator-cmd-norm → pitch DOWN  → sign = −1
        #   F-22:     positive elevator-cmd-norm → pitch UP    → sign = +1
        elevator = self.elevator_sign * self._pitch_pid.step(pitch_error, self.dt)

        # Velocity error → throttle
        velocity_error = target_velocity - current_velocity
        throttle = self._velocity_pid.step(velocity_error, self.dt)

        # Rudder: hard-locked to 0 per paper specification
        rudder = 0.0

        return aileron, elevator, rudder, throttle


# ------------------------------------------------------------------
#  F22 MAV energy-aware PID profile
# ------------------------------------------------------------------

# Default F22 PID gains (conservative starting point; can be overridden via sweep)
F22_MAV_ENERGY_DEFAULT_GAINS = {
    "roll_kp": 0.12,  "roll_ki": 0.45, "roll_kd": 0.04,
    "pitch_kp": 1.2,  "pitch_ki": 0.45, "pitch_kd": 0.08,
    "vel_kp": 0.06,   "vel_ki": 0.012,  "vel_kd": 0.003,
    "elevator_sign": None,   # MUST be set via sweep before use
    "throttle_min": 0.65,     # F22 needs higher idle to sustain flight
    "throttle_max": 1.0,
}


class F22MavEnergyPIDController(PIDController):
    """F22 MAV energy-aware BTT PID controller.

    Extends the base PIDController with:
      - Energy guard: limits aggressive pitch/roll commands at low speed
      - Throttle floor: keeps engine spooled to prevent speed collapse
      - F22-specific PID gains (conservative defaults, sweep-selectable)
      - Autodetected elevator sign (NOT the F-16 hardcoded −1)

    Low-level control protection — does NOT modify reward, action dim, or
    aircraft XML.
    """

    # Energy guard thresholds
    ENERGY_LOW_SPEED = 180.0       # m/s — below this, restrict maneuvering
    ENERGY_CRITICAL_SPEED = 150.0  # m/s — below this, level/shallow climb only
    LOW_SPEED_THROTTLE_FLOOR = 0.95  # throttle floor when speed < 180 (F22 needs high idle)

    # Energy guard limits
    LOW_SPEED_PITCH_MIN_DEG = -5.0    # max nose-down at low speed
    LOW_SPEED_PITCH_MAX_DEG = 18.0    # max nose-up at low speed
    LOW_SPEED_ROLL_MAX_DEG = 30.0     # max bank at low speed
    CRITICAL_PITCH_MIN_DEG = -2.0     # max nose-down at critical speed
    CRITICAL_PITCH_MAX_DEG = 15.0     # max nose-up at critical speed
    CRITICAL_ROLL_MAX_DEG = 15.0      # max bank at critical speed

    def __init__(self, dt, debug: bool = False,
                 roll_kp=None, roll_ki=None, roll_kd=None,
                 pitch_kp=None, pitch_ki=None, pitch_kd=None,
                 vel_kp=None, vel_ki=None, vel_kd=None,
                 elevator_sign=None,
                 throttle_min=None, throttle_max=None,
                 low_speed_throttle_floor=None):
        g = F22_MAV_ENERGY_DEFAULT_GAINS
        # Validate elevator_sign BEFORE super().__init__ so the F22 profile
        # never silently falls back to the F-16 default.
        resolved_sign = elevator_sign if elevator_sign is not None else g["elevator_sign"]
        if resolved_sign is None:
            raise ValueError(
                "F22MavEnergyPIDController: elevator_sign must be set explicitly "
                "(run elevator sign sweep first)"
            )
        super().__init__(
            dt, debug=debug,
            roll_kp=roll_kp if roll_kp is not None else g["roll_kp"],
            roll_ki=roll_ki if roll_ki is not None else g["roll_ki"],
            roll_kd=roll_kd if roll_kd is not None else g["roll_kd"],
            pitch_kp=pitch_kp if pitch_kp is not None else g["pitch_kp"],
            pitch_ki=pitch_ki if pitch_ki is not None else g["pitch_ki"],
            pitch_kd=pitch_kd if pitch_kd is not None else g["pitch_kd"],
            vel_kp=vel_kp if vel_kp is not None else g["vel_kp"],
            vel_ki=vel_ki if vel_ki is not None else g["vel_ki"],
            vel_kd=vel_kd if vel_kd is not None else g["vel_kd"],
            elevator_sign=resolved_sign,
            throttle_min=throttle_min if throttle_min is not None else g["throttle_min"],
            throttle_max=throttle_max if throttle_max is not None else g["throttle_max"],
        )
        self.low_speed_throttle_floor = float(
            low_speed_throttle_floor
            if low_speed_throttle_floor is not None
            else self.LOW_SPEED_THROTTLE_FLOOR
        )

        # Per-step diagnostics (read by env for logging)
        self.last_energy_guard_active = False
        self.last_energy_guard_level = ""
        self.last_pitch_clamped = False
        self.last_roll_clamped = False
        self.last_throttle_boosted = False

    def compute_control(self, current_rpy, current_velocity,
                        target_pitch, target_heading, target_velocity,
                        ned_velocity=None):
        """Override to apply energy guard before delegating to base BTT logic.

        Energy guard modifies *target_pitch, target_velocity, target_heading*
        in place to protect low-speed F22 flight before the PID loops run.
        """
        roll, pitch, yaw = (float(current_rpy[0]), float(current_rpy[1]),
                            float(current_rpy[2]))
        current_speed = float(current_velocity)

        # Reset per-step diagnostics
        self.last_energy_guard_active = False
        self.last_energy_guard_level = ""
        self.last_pitch_clamped = False
        self.last_roll_clamped = False
        self.last_throttle_boosted = False

        # ---- Energy guard: speed-dependent protection ----
        if current_speed < self.ENERGY_CRITICAL_SPEED:
            # Critical: level or shallow climb only. No large pitch or roll.
            self.last_energy_guard_active = True
            self.last_energy_guard_level = "critical"

            target_pitch_deg = float(np.clip(
                np.rad2deg(target_pitch),
                self.CRITICAL_PITCH_MIN_DEG,
                self.CRITICAL_PITCH_MAX_DEG,
            ))
            target_pitch = np.deg2rad(target_pitch_deg)
            self.last_pitch_clamped = True

            # Minimize roll: bank not useful at very low speed
            # Override target_heading toward current heading to reduce roll demand
            target_heading = yaw
            self.last_roll_clamped = True

            # Force full throttle recovery
            target_velocity = max(target_velocity, 408.0)  # VELOCITY_MAX
            self.last_throttle_boosted = True

        elif current_speed < self.ENERGY_LOW_SPEED:
            # Low speed: restrict aggressive pitch/roll
            self.last_energy_guard_active = True
            self.last_energy_guard_level = "low"

            target_pitch_deg = float(np.clip(
                np.rad2deg(target_pitch),
                self.LOW_SPEED_PITCH_MIN_DEG,
                self.LOW_SPEED_PITCH_MAX_DEG,
            ))
            target_pitch = np.deg2rad(target_pitch_deg)
            if abs(np.rad2deg(target_pitch) - float(np.clip(
                np.rad2deg(target_pitch), -90, 90))) > 1e-6:
                pass  # already clamped above
            self.last_pitch_clamped = True

            # Boost target_velocity to ensure speed recovery
            target_velocity = max(target_velocity, current_speed + 30.0)
            self.last_throttle_boosted = True

        # ---- Throttle floor: prevent speed collapse ----
        # The velocity PID's output_min already enforces throttle_min,
        # but at low speed we further boost the target to keep the PID
        # error positive.
        if current_speed < self.ENERGY_LOW_SPEED:
            target_velocity = max(target_velocity, 250.0)  # ~M0.73

        # Delegate to base BTT logic (which handles gimbal, LPF, roll deadband, etc.)
        aileron, elevator, rudder, throttle = super().compute_control(
            current_rpy, current_velocity,
            target_pitch, target_heading, target_velocity,
            ned_velocity=ned_velocity,
        )

        # ---- Throttle floor: hard clamp at low speed ----
        if current_speed < self.ENERGY_LOW_SPEED:
            if throttle < self.low_speed_throttle_floor:
                throttle = self.low_speed_throttle_floor
                self.last_throttle_boosted = True

        return aileron, elevator, rudder, throttle
