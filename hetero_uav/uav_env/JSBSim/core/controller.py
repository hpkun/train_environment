"""High-level action to low-level JSBSim control conversion."""

from __future__ import annotations

import numpy as np


class PID:
    def __init__(self, kp: float, ki: float, kd: float, low: float, high: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.low = low
        self.high = high
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral += error * dt
        deriv = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error
        return float(np.clip(self.kp * error + self.ki * self.integral + self.kd * deriv,
                             self.low, self.high))


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PitchHeadingSpeedController:
    """Small PID controller for the high-level [pitch, heading, speed] action."""

    def __init__(self):
        self.pitch_pid = PID(2.0, 0.05, 0.05, -1.0, 1.0)
        self.heading_pid = PID(1.5, 0.02, 0.03, -1.0, 1.0)
        self.speed_pid = PID(0.015, 0.001, 0.0, 0.0, 1.0)

    def reset(self) -> None:
        self.pitch_pid.reset()
        self.heading_pid.reset()
        self.speed_pid.reset()

    def compute(self, current_pitch: float, current_heading: float,
                current_speed: float, target_pitch: float,
                target_heading: float, target_speed: float, dt: float) -> dict[str, float]:
        pitch_error = target_pitch - current_pitch
        heading_error = wrap_pi(target_heading - current_heading)
        elevator = -self.pitch_pid.step(pitch_error, dt)
        aileron = self.heading_pid.step(heading_error, dt)
        rudder = float(np.clip(0.25 * heading_error, -1.0, 1.0))
        throttle = self.speed_pid.step(target_speed - current_speed, dt)
        return {
            "fcs/throttle-cmd-norm": throttle,
            "fcs/elevator-cmd-norm": elevator,
            "fcs/aileron-cmd-norm": aileron,
            "fcs/rudder-cmd-norm": rudder,
        }
