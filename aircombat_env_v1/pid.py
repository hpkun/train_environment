"""Small, reproducible PID loops for the paper-defined control structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import paper_direction_errors


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


class PIDLoop:
    """PID with conditional integration, output limits and derivative filtering."""

    def __init__(self, kp, ki, kd, output_min, output_max,
                 integral_error_limit=float("inf"), derivative_filter_tau=0.0):
        if output_min >= output_max:
            raise ValueError("output_min must be less than output_max")
        if integral_error_limit < 0.0 or derivative_filter_tau < 0.0:
            raise ValueError("PID limits and filter time constant must be non-negative")
        self.kp, self.ki, self.kd = map(float, (kp, ki, kd))
        self.output_min, self.output_max = float(output_min), float(output_max)
        self.integral_error_limit = float(integral_error_limit)
        self.derivative_filter_tau = float(derivative_filter_tau)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.filtered_derivative = 0.0

    def step(self, error, dt):
        error, dt = float(error), float(dt)
        if not np.isfinite(error) or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("error must be finite and dt must be positive and finite")

        raw_derivative = 0.0 if self.previous_error is None else (
            error - self.previous_error) / dt
        alpha = dt / (self.derivative_filter_tau + dt)
        self.filtered_derivative += alpha * (raw_derivative - self.filtered_derivative)
        self.previous_error = error

        candidate_integral = self.integral
        if abs(error) <= self.integral_error_limit:
            candidate_integral += error * dt

        def unconstrained(integral):
            return (self.kp * error + self.ki * integral
                    + self.kd * self.filtered_derivative)

        candidate_output = unconstrained(candidate_integral)
        saturated_high = candidate_output > self.output_max and error > 0.0
        saturated_low = candidate_output < self.output_min and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral
        return float(np.clip(unconstrained(self.integral),
                             self.output_min, self.output_max))


class PaperAutopilot:
    """Literal three-channel paper controller; rudder is fixed at zero."""

    def __init__(self, roll_gains, pitch_gains, speed_gains, throttle_base,
                 actuator_sign_elevator, integral_error_limits=(0.35, 0.25, 40.0),
                 derivative_filter_tau=0.05):
        self.throttle_base = float(throttle_base)
        self.actuator_sign_elevator = float(actuator_sign_elevator)
        if not 0.0 <= self.throttle_base <= 1.0:
            raise ValueError("throttle_base must be in [0, 1]")
        if self.actuator_sign_elevator not in (-1.0, 1.0):
            raise ValueError("actuator_sign_elevator must be -1 or +1")
        roll = PIDGains(**roll_gains) if isinstance(roll_gains, dict) else roll_gains
        pitch = PIDGains(**pitch_gains) if isinstance(pitch_gains, dict) else pitch_gains
        speed = PIDGains(**speed_gains) if isinstance(speed_gains, dict) else speed_gains
        self.roll_pid = PIDLoop(*roll.__dict__.values(), -1.0, 1.0,
                                integral_error_limits[0], derivative_filter_tau)
        self.pitch_pid = PIDLoop(*pitch.__dict__.values(), -1.0, 1.0,
                                 integral_error_limits[1], derivative_filter_tau)
        self.speed_pid = PIDLoop(*speed.__dict__.values(), -self.throttle_base,
                                 1.0 - self.throttle_base,
                                 integral_error_limits[2], derivative_filter_tau)

    @classmethod
    def from_config(cls, config):
        pid = config["pid"]
        return cls(pid["roll"], pid["pitch"], pid["speed"],
                   config["trim"]["throttle_base"],
                   config["actuator_sign_elevator"],
                   tuple(pid["integral_error_limits"]),
                   pid["derivative_filter_tau"])

    def reset(self):
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.speed_pid.reset()

    def step(self, current_roll, current_pitch, current_heading,
             current_true_airspeed, target_pitch, target_heading,
             target_true_airspeed, dt):
        values = (current_roll, current_pitch, current_heading,
                  current_true_airspeed, target_pitch, target_heading,
                  target_true_airspeed, dt)
        if not np.all(np.isfinite(values)):
            raise ValueError("autopilot inputs must be finite")
        e_roll, e_pitch = paper_direction_errors(
            current_roll, current_pitch, current_heading,
            target_pitch, target_heading)
        aileron = self.roll_pid.step(e_roll, dt)
        elevator = self.actuator_sign_elevator * self.pitch_pid.step(e_pitch, dt)
        speed_correction = self.speed_pid.step(
            target_true_airspeed - current_true_airspeed, dt)
        throttle = float(np.clip(self.throttle_base + speed_correction, 0.0, 1.0))
        return aileron, elevator, 0.0, throttle
