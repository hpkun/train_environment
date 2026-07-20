from __future__ import annotations

"""Minimal three-loop bank-to-turn PID for ``paper_3v3_minimal_v2``."""

import numpy as np


PAPER_EQ13_ZERO_EPS = 1e-12
PAPER_PID_ERROR_DEFINITION = (
    "paper_quadrant_preserving_azimuth_continuous_elevation_operational_v1")
PAPER_PID_DERIVATIVE_SEMANTICS = (
    "first_sample_only_error_derivative_v2")
PAPER_PID_PROFILE = "paper_3v3_minimal_pid_v2"


class PIDLoop:
    """Bounded PID loop with first-sample derivative suppression."""

    def __init__(self, kp, ki, kd, output_min, output_max, name="",
                 integral_error_limit=float("inf"), angular_error=False):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.name = name
        self.integral_error_limit = float(integral_error_limit)
        self.angular_error = bool(angular_error)
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._prev_output = 0.0
        self._last_diagnostic = {
            "p": 0.0, "i": 0.0, "d": 0.0,
            "unsaturated": 0.0, "saturated": 0.0,
            "integral_enabled": True,
        }

    def step(self, error, dt):
        error = float(error)
        p = self.kp * error
        if self._prev_error is None:
            d = 0.0
        else:
            delta = error - self._prev_error
            if self.angular_error:
                wrapped = (delta + np.pi) % (2.0 * np.pi) - np.pi
                if np.isclose(wrapped, -np.pi) and delta > 0.0:
                    wrapped = np.pi
                delta = float(wrapped)
            d = self.kd * delta / max(float(dt), 1e-8)
        self._prev_error = error

        integral_enabled = abs(error) <= self.integral_error_limit
        if integral_enabled:
            self._integral += self.ki * error * float(dt)

        pd = p + d
        if pd >= self.output_max and self._integral > 0.0:
            self._integral = 0.0
        elif pd <= self.output_min and self._integral < 0.0:
            self._integral = 0.0
        elif self.output_min < pd < self.output_max:
            self._integral = float(np.clip(
                self._integral, self.output_min - pd, self.output_max - pd))

        safety = (max(abs(self.output_max), abs(self.output_min)) * 3.0
                  / max(self.ki, 1e-8))
        self._integral = float(np.clip(self._integral, -safety, safety))
        unsaturated = float(p + self._integral + d)
        output = float(np.clip(
            unsaturated, self.output_min, self.output_max))
        self._prev_output = output
        self._last_diagnostic = {
            "p": float(p), "i": float(self._integral), "d": float(d),
            "unsaturated": unsaturated, "saturated": output,
            "integral_enabled": bool(integral_enabled),
        }
        return output


class PIDController:
    """Paper Eq.12-Eq.14 operational BTT controller."""

    def __init__(self, dt, profile: str = PAPER_PID_PROFILE,
                 debug: bool = False,
                 integral_error_limits: tuple[float, float, float] | None = None,
                 throttle_base: float = 0.8, config=None):
        from configs.brma_mappo_paper_spec import PIDConfig

        if profile != PAPER_PID_PROFILE:
            raise ValueError(f"profile must be {PAPER_PID_PROFILE!r}")
        self.config = config or PIDConfig()
        self.dt = float(dt)
        self.profile = profile
        self._debug = bool(debug)
        if not 0.0 <= float(throttle_base) <= 1.0:
            raise ValueError("throttle_base must be in [0, 1]")
        self.throttle_base = float(throttle_base)
        self._last_diagnostic = {}
        limits = (self.config.integral_error_limits.value
                  if integral_error_limits is None else integral_error_limits)
        self.integral_error_limits = tuple(float(value) for value in limits)

        roll_gains = self.config.roll_gains.value
        pitch_gains = self.config.pitch_gains.value
        speed_gains = self.config.speed_gains.value
        self._roll_pid = PIDLoop(
            *roll_gains, -1.0, 1.0, name="roll",
            integral_error_limit=self.integral_error_limits[0],
            angular_error=True)
        self._pitch_pid = PIDLoop(
            *pitch_gains, -1.0, 1.0, name="pitch",
            integral_error_limit=self.integral_error_limits[1],
            angular_error=True)
        self._velocity_pid = PIDLoop(
            *speed_gains, -self.throttle_base, 1.0 - self.throttle_base,
            name="velocity", integral_error_limit=self.integral_error_limits[2])

    def reset(self):
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._velocity_pid.reset()
        self._last_diagnostic = {}

    @staticmethod
    def _r_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    @staticmethod
    def _r_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    @staticmethod
    def _r_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @classmethod
    def body_to_ned_matrix(cls, roll, pitch, yaw) -> np.ndarray:
        return cls._r_z(yaw) @ cls._r_y(pitch) @ cls._r_x(roll)

    @classmethod
    def ned_to_body_matrix(cls, roll, pitch, yaw) -> np.ndarray:
        return cls.body_to_ned_matrix(roll, pitch, yaw).T

    @classmethod
    def paper_direction_errors(cls, current_rpy, target_pitch, target_heading):
        roll, pitch, yaw = [float(value) for value in current_rpy]
        c_theta = np.cos(target_pitch)
        desired_neu = np.array([
            c_theta * np.cos(target_heading),
            c_theta * np.sin(target_heading),
            np.sin(target_pitch),
        ], dtype=np.float64)
        desired_ned = desired_neu * np.array([1.0, 1.0, -1.0])
        r_bi = cls.ned_to_body_matrix(roll, pitch, yaw)
        desired_body = r_bi @ desired_ned
        if desired_body[0] < 0.0 and abs(desired_body[1]) < PAPER_EQ13_ZERO_EPS:
            roll_error = float(np.pi)
        else:
            roll_error = float(np.arctan2(
                desired_body[1], desired_body[0]))
        horizontal = float(np.hypot(desired_body[0], desired_body[1]))
        pitch_error = (0.0 if abs(desired_body[2]) < PAPER_EQ13_ZERO_EPS
                       else float(np.arctan2(desired_body[2], horizontal)))
        return (roll_error, pitch_error, desired_neu, desired_ned,
                desired_body, r_bi)

    def compute_control(self, current_rpy, current_velocity,
                        target_pitch, target_heading, target_velocity,
                        ned_velocity=None):
        values = [*current_rpy, current_velocity, target_pitch,
                  target_heading, target_velocity]
        if not np.all(np.isfinite(values)):
            self.reset()
            return 0.0, 0.0, 0.0, 0.0

        roll_error, pitch_error, d_neu, d_ned, d_body, r_bi = (
            self.paper_direction_errors(
                current_rpy, target_pitch, target_heading))
        aileron = self._roll_pid.step(roll_error, self.dt)
        elevator = self._pitch_pid.step(pitch_error, self.dt)
        velocity_error = float(target_velocity - current_velocity)
        correction = self._velocity_pid.step(velocity_error, self.dt)
        throttle = float(np.clip(
            self.throttle_base + correction, 0.0, 1.0))
        unsaturated = (
            self._roll_pid._last_diagnostic["unsaturated"],
            self._pitch_pid._last_diagnostic["unsaturated"],
            0.0,
            self.throttle_base
            + self._velocity_pid._last_diagnostic["unsaturated"],
        )
        saturated = (aileron, elevator, 0.0, throttle)
        self._last_diagnostic = {
            "formula_version": PAPER_PID_ERROR_DEFINITION,
            "derivative_semantics": PAPER_PID_DERIVATIVE_SEMANTICS,
            "setpoint_changed_this_frame": False,
            "pitch_setpoint_delta_rad": 0.0,
            "heading_setpoint_delta_rad": 0.0,
            "velocity_setpoint_delta_mps": 0.0,
            "derivative_suppressed_for_setpoint_change": False,
            "d_I_des": tuple(float(value) for value in d_neu),
            "d_I_des_ned": tuple(float(value) for value in d_ned),
            "d_B_des": tuple(float(value) for value in d_body),
            "R_BI": r_bi.tolist(),
            "e_phi": float(roll_error),
            "e_theta": float(pitch_error),
            "e_velocity": velocity_error,
            "roll_pid": dict(self._roll_pid._last_diagnostic),
            "pitch_pid": dict(self._pitch_pid._last_diagnostic),
            "velocity_pid": dict(self._velocity_pid._last_diagnostic),
            "unsaturated_commands": tuple(float(value) for value in unsaturated),
            "saturated_commands": tuple(float(value) for value in saturated),
            "degenerate_arctan_ratio": False,
            "jsbsim_elevator_sign_adapter": 1.0,
        }
        return aileron, elevator, 0.0, throttle
