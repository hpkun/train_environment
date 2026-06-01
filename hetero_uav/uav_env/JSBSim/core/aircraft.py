"""Aircraft platform backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aircraft_types import AircraftType
from .controller import PitchHeadingSpeedController
from .geo import lla_to_local, local_to_lla
from .utils import clamp, heading_to_unit, wrap_pi


@dataclass
class SimpleKinematicAircraftPlatform:
    agent_id: str
    side: str
    aircraft_type: AircraftType
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    pitch: float = 0.0
    roll: float = 0.0
    alive: bool = True
    crashed: bool = False
    out_of_boundary: bool = False
    missile_left: int = 0
    missile_cooldown: int = 0

    def reset_runtime(self) -> None:
        self.alive = True
        self.crashed = False
        self.out_of_boundary = False
        self.missile_left = self.aircraft_type.missile_num
        self.missile_cooldown = 0

    @property
    def type_id(self) -> int:
        return self.aircraft_type.type_id

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def step(self, action: np.ndarray, dt: float, speed_range: tuple[float, float]) -> None:
        if not self.alive:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        target_pitch = float(action[0]) * (np.pi / 2.0)
        signs = self.aircraft_type.control
        target_heading = float(action[1]) * np.pi * signs.get("heading_sign", 1.0)
        speed_min, speed_max = speed_range
        speed_hi = speed_max * self.aircraft_type.max_speed_scale
        target_speed = speed_min + (float(action[2]) + 1.0) * 0.5 * (speed_hi - speed_min)

        turn_rate = np.deg2rad(18.0) * max(0.5, self.aircraft_type.max_g / 9.0)
        pitch_rate = np.deg2rad(12.0) * max(0.5, self.aircraft_type.max_g / 9.0)
        speed_rate = 45.0 * max(0.5, self.aircraft_type.max_speed_scale)

        heading_error = wrap_pi(target_heading - self.heading)
        self.heading = wrap_pi(self.heading + clamp(heading_error, -turn_rate * dt, turn_rate * dt))
        pitch_error = target_pitch - self.pitch
        self.pitch += clamp(pitch_error, -pitch_rate * dt, pitch_rate * dt)
        self.pitch = clamp(self.pitch, -np.deg2rad(35.0), np.deg2rad(35.0))
        self.roll = clamp(heading_error * 0.5, -np.deg2rad(65.0), np.deg2rad(65.0))

        current_speed = self.speed
        next_speed = current_speed + clamp(target_speed - current_speed,
                                           -speed_rate * dt, speed_rate * dt)
        next_speed = clamp(next_speed, speed_min, speed_hi)
        self.velocity = heading_to_unit(self.heading, self.pitch) * next_speed
        self.position = self.position + self.velocity * dt
        self.missile_cooldown = max(0, self.missile_cooldown - 1)

    def kill(self, reason: str = "killed") -> None:
        self.alive = False
        if reason == "crash":
            self.crashed = True
        if reason == "boundary":
            self.out_of_boundary = True


class JSBSimAircraftPlatform(SimpleKinematicAircraftPlatform):
    """JSBSim-backed aircraft with the same public attributes as the simple backend."""

    FT2M = 0.3048
    M2FT = 1.0 / 0.3048

    def __init__(self, *args, model_root: str, model_name: str,
                 reference_lat: float, reference_lon: float, reference_alt: float,
                 simulation_frequency: int = 60, suppress_output: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_root = Path(model_root)
        self.model_name = model_name
        self.reference_lat = reference_lat
        self.reference_lon = reference_lon
        self.reference_alt = reference_alt
        self.simulation_frequency = simulation_frequency
        self.physics_dt = 1.0 / float(simulation_frequency)
        self.suppress_output = suppress_output
        self.controller = PitchHeadingSpeedController()
        self.jsbsim_exec = None
        self._initial_position = np.asarray(self.position, dtype=np.float32).copy()
        self._initial_velocity = np.asarray(self.velocity, dtype=np.float32).copy()
        self._initial_heading = float(self.heading)
        self._load_jsbsim()
        self.reset_runtime()

    def _load_jsbsim(self) -> None:
        try:
            import jsbsim
        except Exception as exc:
            raise ImportError(
                "JSBSim backend requested but Python package 'jsbsim' is not installed. "
                "Install jsbsim to use dynamics_backend='jsbsim'."
            ) from exc
        self.jsbsim_exec = jsbsim.FGFDMExec(str(self.model_root))
        self.jsbsim_exec.set_debug_level(0)
        if hasattr(self.jsbsim_exec, "set_aircraft_path"):
            self.jsbsim_exec.set_aircraft_path(str(self.model_root / "aircraft"))
        if hasattr(self.jsbsim_exec, "set_engine_path"):
            self.jsbsim_exec.set_engine_path(str(self.model_root / "engine"))
        self.jsbsim_exec.set_dt(self.physics_dt)
        ok = self.jsbsim_exec.load_model(self.model_name)
        if not ok:
            raise RuntimeError(f"JSBSim failed to load model {self.model_name!r}")
        self._reset_fdm()

    def reset_runtime(self) -> None:
        super().reset_runtime()
        if getattr(self, "controller", None) is not None:
            self.controller.reset()
        if getattr(self, "jsbsim_exec", None) is not None:
            self.position = self._initial_position.copy()
            self.velocity = self._initial_velocity.copy()
            self.heading = self._initial_heading
            self.pitch = 0.0
            self.roll = 0.0
            self._reset_fdm()

    def _reset_fdm(self) -> None:
        lon, lat, alt_m = local_to_lla(
            self.position, self.reference_lat, self.reference_lon, self.reference_alt)
        speed = max(float(np.linalg.norm(self.velocity)), 1.0)
        props = {
            "ic/long-gc-deg": lon,
            "ic/lat-geod-deg": lat,
            "ic/h-sl-ft": alt_m * self.M2FT,
            "ic/psi-true-deg": np.rad2deg(self.heading),
            "ic/theta-deg": np.rad2deg(self.pitch),
            "ic/phi-deg": np.rad2deg(self.roll),
            "ic/u-fps": speed * self.M2FT,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/terrain-elevation-ft": 0.0,
        }
        for name, value in props.items():
            self.jsbsim_exec.set_property_value(name, float(value))
        if not self.jsbsim_exec.run_ic():
            raise RuntimeError(f"JSBSim run_ic failed for {self.agent_id}")
        propulsion = self.jsbsim_exec.get_propulsion()
        for i in range(propulsion.get_num_engines()):
            propulsion.get_engine(i).init_running()
        propulsion.get_steady_state()
        self._set_property("gear/gear-cmd-norm", 0.0)
        self._set_property("fcs/flap-cmd-norm", 0.0)
        self._update_from_jsbsim()

    def step(self, action: np.ndarray, dt: float, speed_range: tuple[float, float]) -> None:
        if not self.alive:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            action = np.pad(action, (0, 3 - action.size))
        signs = self.aircraft_type.control
        target_pitch = float(action[0]) * (np.pi / 2.0)
        target_heading = float(action[1]) * np.pi * signs.get("heading_sign", 1.0)
        speed_min, speed_max = speed_range
        target_speed = speed_min + (float(action[2]) + 1.0) * 0.5 * (
            speed_max * self.aircraft_type.max_speed_scale - speed_min)
        n_substeps = max(1, int(round(dt * self.simulation_frequency)))
        for _ in range(n_substeps):
            controls = self.controller.compute(
                self.pitch, self.heading, self.speed, target_pitch, target_heading,
                target_speed, self.physics_dt, self.roll)
            controls["fcs/elevator-cmd-norm"] *= signs.get("elevator_sign", 1.0)
            controls["fcs/aileron-cmd-norm"] *= signs.get("aileron_sign", 1.0)
            controls["fcs/rudder-cmd-norm"] *= signs.get("rudder_sign", 1.0)
            controls["fcs/throttle-cmd-norm"] *= signs.get("throttle_sign", 1.0)
            for prop, value in controls.items():
                self._set_property(prop, value)
            if not self.jsbsim_exec.run():
                raise RuntimeError(f"JSBSim run failed for {self.agent_id}")
            self._update_from_jsbsim()
            if self.position[2] <= 0.0:
                self.kill("crash")
                break
        self.missile_cooldown = max(0, self.missile_cooldown - 1)

    def _update_from_jsbsim(self) -> None:
        lon = self._get_property("position/long-gc-deg")
        lat = self._get_property("position/lat-geod-deg")
        alt_m = self._get_property("position/h-sl-ft") * self.FT2M
        self.position = lla_to_local(
            lon, lat, alt_m, self.reference_lat, self.reference_lon, self.reference_alt)
        vn = self._get_property("velocities/v-north-fps") * self.FT2M
        ve = self._get_property("velocities/v-east-fps") * self.FT2M
        vd = self._get_property("velocities/v-down-fps") * self.FT2M
        self.velocity = np.array([vn, ve, -vd], dtype=np.float32)
        self.roll = self._get_property("attitude/roll-rad")
        self.pitch = self._get_property("attitude/pitch-rad")
        self.heading = self._get_property("attitude/heading-true-rad")

    def _get_property(self, name: str) -> float:
        return float(self.jsbsim_exec.get_property_value(name))

    def _set_property(self, name: str, value: float) -> None:
        try:
            self.jsbsim_exec.set_property_value(name, float(value))
        except Exception:
            pass

    def close(self) -> None:
        self.jsbsim_exec = None


AircraftPlatform = SimpleKinematicAircraftPlatform
