"""Paper-aligned point-mass PN missile used by the formal 1v1 environment."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

G = 9.81
PAPER_MISSILE_PARAMETERS = {
    "maximum_attack_range_m": 14000.0, "launch_interval_s": 25.0,
    "maximum_overload_g": 30.0, "navigation_gain_y": 3.0,
    "navigation_gain_z": 3.0,
}
PROJECT_MISSILE_ASSUMPTIONS = {
    "initial_speed_mps": 500.0, "powered_duration_s": 3.0,
    "powered_acceleration_mps2": 110.0, "hit_radius_m": 60.0,
    "effective_quadratic_drag_per_m": 0.00012,
    "lifetime_s": 56.0, "minimum_range_m": 0.0,
    "lock_angle_deg": 10.0,
}


def compute_los_angles_and_rates(relative_position, relative_velocity,
                                 epsilon=1e-8):
    """Literal reuse of the official TAM simulator LOS implementation."""
    r = np.asarray(relative_position, dtype=np.float64)
    r_dot = np.asarray(relative_velocity, dtype=np.float64)
    if r.shape != (3,) or r_dot.shape != (3,) or not np.all(
            np.isfinite(np.concatenate([r, r_dot]))):
        raise FloatingPointError("non-finite LOS input")
    rx, ry, rz = r; rdx, rdy, rdz = r_dot
    horizontal_sq = float(rx * rx + ry * ry)
    horizontal = float(np.sqrt(horizontal_sq))
    range_sq = float(np.dot(r, r))
    beta = float(np.arctan2(ry, rx))
    epsilon_angle = float(np.arctan2(rz, horizontal))
    beta_dot = float((rdy * rx - ry * rdx) / max(horizontal_sq, epsilon))
    epsilon_dot = float((horizontal_sq * rdz - rz * (rdx * rx + rdy * ry)) /
                        max(range_sq * horizontal, epsilon))
    values = np.asarray([beta, epsilon_angle, beta_dot, epsilon_dot])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite LOS output")
    return tuple(float(value) for value in values)


def compute_paper_eq9_overloads(relative_position, relative_velocity,
                                missile_velocity, navigation_constant=3.0,
                                gravity=9.81, epsilon=1e-8):
    """Literal reuse of official TAM paper Eq.9 overload commands."""
    beta, los_epsilon, beta_dot, epsilon_dot = compute_los_angles_and_rates(
        relative_position, relative_velocity, epsilon)
    velocity = np.asarray(missile_velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed <= epsilon:
        return 0.0, 0.0
    gamma_m = float(np.arcsin(np.clip(velocity[2] / speed, -1.0, 1.0)))
    cos_sum = float(np.cos(los_epsilon + beta))
    if abs(cos_sum) < epsilon:
        cos_sum = np.copysign(epsilon, cos_sum or 1.0)
    n_mc = navigation_constant * speed * np.cos(gamma_m) / gravity * (
        beta_dot + np.tan(los_epsilon) * np.tan(los_epsilon + beta) * epsilon_dot)
    n_mh = speed / gravity * navigation_constant / cos_sum * epsilon_dot
    values = np.asarray([n_mc, n_mh])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite Eq.9 command")
    return float(n_mc), float(n_mh)


@dataclass
class Missile:
    position_neu: np.ndarray
    velocity_neu: np.ndarray
    shooter: str
    target: str
    launch_time: float
    alive: bool = True
    age: float = 0.0
    distance: float = float("inf")
    overload: float = 0.0
    termination_reason: str | None = None
    decision_start_speed: float = 500.0

    def __post_init__(self):
        self.position_neu = np.asarray(self.position_neu, dtype=float).copy()
        direction = np.asarray(self.velocity_neu, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        self.velocity_neu = direction * PROJECT_MISSILE_ASSUMPTIONS["initial_speed_mps"]
        self.decision_start_speed = float(np.linalg.norm(self.velocity_neu))

    def step(self, target_position, target_velocity, target_alive, dt):
        if not self.alive:
            return self.termination_reason
        if not target_alive:
            return self.terminate("target_dead")
        relative = np.asarray(target_position) - self.position_neu
        relative_velocity = np.asarray(target_velocity) - self.velocity_neu
        self.distance = float(np.linalg.norm(relative))
        if self.distance <= PROJECT_MISSILE_ASSUMPTIONS["hit_radius_m"]:
            return self.terminate("hit")
        try:
            ny, nz = compute_paper_eq9_overloads(
                relative, relative_velocity, self.velocity_neu, 3.0)
        except FloatingPointError:
            return self.terminate("nonfinite")
        lateral = np.array([0.0, ny, nz], dtype=float) * G
        speed = max(float(np.linalg.norm(self.velocity_neu)), 1.0)
        forward = self.velocity_neu / speed
        # Build a stable velocity-aligned frame and cap total lateral overload.
        right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        accel_lateral = ny * G * right + nz * G * up
        limit = PAPER_MISSILE_PARAMETERS["maximum_overload_g"] * G
        norm = float(np.linalg.norm(accel_lateral))
        if norm > limit:
            accel_lateral *= limit / norm; norm = limit
        self.overload = norm / G
        thrust = (PROJECT_MISSILE_ASSUMPTIONS["powered_acceleration_mps2"]
                  if self.age < PROJECT_MISSILE_ASSUMPTIONS["powered_duration_s"] else 0.0)
        drag = PROJECT_MISSILE_ASSUMPTIONS["effective_quadratic_drag_per_m"] * speed ** 2
        self.velocity_neu += (accel_lateral + (thrust - drag) * forward) * dt
        self.position_neu += self.velocity_neu * dt
        self.age += dt
        if not np.isfinite(np.r_[self.position_neu, self.velocity_neu]).all():
            return self.terminate("nonfinite")
        if self.position_neu[2] <= 0.0:
            return self.terminate("ground_impact")
        if self.age >= PROJECT_MISSILE_ASSUMPTIONS["lifetime_s"]:
            return self.terminate("timeout")
        return None

    def terminate(self, reason):
        self.alive = False; self.termination_reason = reason
        return reason
