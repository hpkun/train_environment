"""AIM-9L-style missile dynamics adapted from the parent JSBSim project."""

from __future__ import annotations

from collections import deque

import numpy as np

from .geo import local_to_lla


class MissileSimulator:
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent, target, uid: str, missile_model: str = "AIM-9L", dt: float | None = None):
        missile_dt = float(dt if dt is not None else getattr(parent, "physics_dt", 1 / 60))
        missile = cls(uid=uid, color=getattr(parent, "side", "red"), model=missile_model, dt=missile_dt)
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid: str = "M0001", color: str = "red", model: str = "AIM-9L", dt: float = 1 / 60):
        self.uid = uid
        self.color = color
        self.dt = float(dt)
        self.model = model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False
        self._geodetic = np.zeros(3, dtype=np.float32)
        self._position = np.zeros(3, dtype=np.float32)
        self._posture = np.zeros(3, dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)
        self._status = MissileSimulator.INACTIVE

        self._g = 9.81
        self._t_max = 60.0
        self._t_thrust = 3.0
        self._Isp = 120.0
        self._Length = 2.87
        self._Diameter = 0.127
        self._cD = 0.4
        self._m0 = 84.0
        self._dm = 6.0
        self._K = 3.0
        self._nyz_max = 30.0
        self._Rc = 300.0
        self._v_min = 150.0

    @property
    def status(self) -> int:
        return self._status

    @property
    def status_name(self) -> str:
        return {
            self.INACTIVE: "INACTIVE",
            self.LAUNCHED: "LAUNCHED",
            self.HIT: "HIT",
            self.MISS: "MISS",
        }.get(self._status, "UNKNOWN")

    @property
    def is_alive(self) -> bool:
        return self._status == MissileSimulator.LAUNCHED

    @property
    def is_success(self) -> bool:
        return self._status == MissileSimulator.HIT

    @property
    def is_done(self) -> bool:
        return self._status in (MissileSimulator.HIT, MissileSimulator.MISS)

    @property
    def Isp(self) -> float:
        return self._Isp if self._t < self._t_thrust else 0.0

    @property
    def K(self) -> float:
        return max(self._K * (self._t_max - self._t) / self._t_max, 0.0)

    @property
    def S(self) -> float:
        s0 = np.pi * (self._Diameter / 2) ** 2
        s0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return float(s0)

    @property
    def rho(self) -> float:
        return float(1.225 * np.exp(-self._geodetic[-1] / 9300.0))

    @property
    def target_distance(self) -> float:
        return float(np.linalg.norm(self.target_aircraft.get_position() - self.get_position()))

    def get_geodetic(self) -> np.ndarray:
        return self._geodetic

    def get_position(self) -> np.ndarray:
        return self._position

    def get_rpy(self) -> np.ndarray:
        return self._posture

    def get_velocity(self) -> np.ndarray:
        return self._velocity

    def launch(self, parent) -> None:
        self.parent_aircraft = parent
        parent.launch_missiles.append(self)
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0.0
        self._geodetic[:] = parent.get_geodetic()
        if not np.any(self._geodetic):
            self._geodetic[:] = local_to_lla(
                self._position, getattr(parent, "reference_lat", 60.0),
                getattr(parent, "reference_lon", 120.0), getattr(parent, "reference_alt", 0.0)
            )
        self.reference_lat = getattr(parent, "reference_lat", 60.0)
        self.reference_lon = getattr(parent, "reference_lon", 120.0)
        self.reference_alt = getattr(parent, "reference_alt", 0.0)
        self._t = 0.0
        self._m = self._m0
        self._dtheta = 0.0
        self._dphi = 0.0
        self._status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=max(1, int(5 / self.dt)))

    def target(self, target) -> None:
        self.target_aircraft = target
        target.under_missiles.append(self)

    def run(self) -> None:
        if not self.is_alive:
            return
        self._t += self.dt
        action, distance = self._guidance()
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        if distance < self._Rc and self.target_aircraft.is_alive:
            self._status = MissileSimulator.HIT
            self.target_aircraft.shotdown()
        elif (
            self._t > self._t_max
            or np.linalg.norm(self.get_velocity()) < self._v_min
            or np.sum(self._distance_increment) >= self._distance_increment.maxlen
            or not self.target_aircraft.is_alive
        ):
            self._status = MissileSimulator.MISS
        else:
            self._state_trans(action)

    def _guidance(self):
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = max(float(np.linalg.norm([dx_m, dy_m, dz_m])), 1e-6)
        theta_m = np.arcsin(np.clip(dz_m / v_m, -1.0, 1.0))
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        rxy = max(float(np.linalg.norm([x_m - x_t, y_m - y_t])), 1e-6)
        rxyz = max(float(np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])), 1e-6)
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / rxy ** 2
        deps = ((dz_t - dz_m) * rxy ** 2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (rxyz ** 2 * rxy)
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), rxyz

    def _state_trans(self, action) -> None:
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = local_to_lla(
            self._position, self.reference_lat, self.reference_lon, self.reference_alt)
        v = max(float(np.linalg.norm(self.get_velocity())), 1e-6)
        theta, phi = self.get_rpy()[1:]
        thrust = self._g * self.Isp * self._dm
        drag = 0.5 * self._cD * self.S * self.rho * v ** 2
        nx = (thrust - drag) / (self._m * self._g)
        ny, nz = action
        dv = self._g * (nx - np.sin(theta))
        cos_theta = max(float(np.cos(theta)), 1e-3)
        self._dphi = self._g / v * (ny / cos_theta)
        self._dtheta = self._g / v * (nz - np.cos(theta))
        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta),
        ], dtype=np.float32)
        self._posture[:] = np.array([0.0, theta, phi], dtype=np.float32)
        if self._t < self._t_thrust:
            self._m -= self.dt * self._dm
