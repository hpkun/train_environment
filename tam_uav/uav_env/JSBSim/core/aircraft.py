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

    def __post_init__(self) -> None:
        self.partners = []
        self.enemies = []
        self.launch_missiles = []
        self.under_missiles = []
        self._geodetic = np.zeros(3, dtype=np.float32)
        self._direct_fcs_command = np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float64)
        self.death_reason = None
        self.max_speed_observed_mps = self.speed
        self.max_abs_load_factor_g = 1.0
        self.speed_limit_exceedance_count = 0
        self.overload_limit_exceedance_count = 0

    def reset_runtime(self) -> None:
        self.alive = True
        self.crashed = False
        self.out_of_boundary = False
        self.missile_left = self.aircraft_type.missile_num
        self.missile_cooldown = 0
        self.death_reason = None
        self.max_speed_observed_mps = self.speed
        self.max_abs_load_factor_g = 1.0
        self.speed_limit_exceedance_count = 0
        self.overload_limit_exceedance_count = 0
        self.launch_missiles.clear()
        self.under_missiles.clear()

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

    def step_direct_fcs(self, command: np.ndarray, physics_steps: int,
                        simulation_frequency: int, max_speed_mps: float) -> None:
        """Apply [throttle, aileron, elevator, rudder] without a PID layer.

        The simple backend is a deterministic test approximation. The JSBSim
        subclass below writes the same command directly to the four FCS inputs.
        """
        del max_speed_mps
        self.apply_direct_fcs_command(command)
        for _ in range(physics_steps):
            self.step_physics_once(1.0 / float(simulation_frequency))

    def apply_direct_fcs_command(self, command: np.ndarray) -> None:
        values = np.asarray(command, dtype=np.float64).reshape(4).copy()
        values[0] = np.clip(values[0], 0.4, 0.9)
        values[1:] = np.clip(values[1:], -1.0, 1.0)
        self._direct_fcs_command = values

    def step_physics_once(self, dt: float) -> bool:
        if not self.alive:
            return False
        throttle, aileron, elevator, rudder = self._direct_fcs_command
        speed = max(1.0, self.speed + (90.0 * (throttle - 0.4) - 12.0) * dt)
        self.roll = float(np.clip(self.roll + aileron * np.deg2rad(90.0) * dt,
                                  -np.deg2rad(80.0), np.deg2rad(80.0)))
        self.pitch = float(np.clip(self.pitch - elevator * np.deg2rad(30.0) * dt,
                                   -np.deg2rad(45.0), np.deg2rad(45.0)))
        self.heading = wrap_pi(self.heading +
                               (np.sin(self.roll) * np.deg2rad(25.0) +
                                rudder * np.deg2rad(12.0)) * dt)
        self.velocity = heading_to_unit(self.heading, self.pitch) * speed
        self.position = self.position + self.velocity * dt
        self.max_speed_observed_mps = max(self.max_speed_observed_mps, self.speed)
        self.max_abs_load_factor_g = max(self.max_abs_load_factor_g, self.load_factor_g)
        return True

    @property
    def load_factor_g(self) -> float:
        return float(1.0 + 8.0 * min(1.0, abs(self._direct_fcs_command[2])))

    @property
    def direct_fcs_command(self) -> np.ndarray:
        return self._direct_fcs_command.copy()

    def kill(self, reason: str = "killed") -> None:
        self.alive = False
        self.death_reason = reason
        if reason in {"crash", "nonfinite"}:
            self.crashed = True
        if reason == "boundary":
            self.out_of_boundary = True

    @property
    def is_alive(self) -> bool:
        return self.alive

    @property
    def speed_violation_count(self) -> int:
        """Deprecated alias for speed_limit_exceedance_count."""
        return self.speed_limit_exceedance_count

    @property
    def overload_violation_count(self) -> int:
        """Deprecated alias for overload_limit_exceedance_count."""
        return self.overload_limit_exceedance_count

    @property
    def is_crash(self) -> bool:
        return self.crashed

    @property
    def is_shotdown(self) -> bool:
        return (not self.alive) and (not self.crashed) and (not self.out_of_boundary)

    def get_position(self) -> np.ndarray:
        return self.position

    def get_velocity(self) -> np.ndarray:
        return self.velocity

    def get_rpy(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.heading], dtype=np.float32)

    def get_geodetic(self) -> np.ndarray:
        return self._geodetic

    def shotdown(self) -> None:
        self.kill("killed")

    def crash(self) -> None:
        self.kill("crash")

    def check_missile_warning(self):
        for missile in self.under_missiles:
            if getattr(missile, "is_alive", False):
                return missile
        return None

    def clone_for_prediction(self):
        """Deterministic test-backend clone; formal configs use JSBSim below."""
        clone = SimpleKinematicAircraftPlatform(
            self.agent_id, self.side, self.aircraft_type, self.position.copy(),
            self.velocity.copy(), float(self.heading), pitch=float(self.pitch),
            roll=float(self.roll), alive=bool(self.alive),
            missile_left=int(self.missile_left))
        clone.apply_direct_fcs_command(self.direct_fcs_command)
        return clone

    def close(self) -> None:
        return None


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
        self._set_property("gear/gear-pos-norm", 0.0)
        for index in range(3):
            self._set_property(f"gear/unit[{index}]/pos-norm", 0.0)
        self._set_property("fcs/flap-cmd-norm", 0.0)
        self._set_property("fcs/flap-pos-norm", 0.0)
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

    def step_direct_fcs(self, command: np.ndarray, physics_steps: int,
                        simulation_frequency: int, max_speed_mps: float) -> None:
        del simulation_frequency, max_speed_mps
        self.apply_direct_fcs_command(command)
        for _ in range(physics_steps):
            if not self.step_physics_once(self.physics_dt):
                break

    def apply_direct_fcs_command(self, command: np.ndarray) -> None:
        super().apply_direct_fcs_command(command)
        throttle, aileron, elevator, rudder = self._direct_fcs_command
        values = {
            "fcs/throttle-cmd-norm": throttle,
            "fcs/aileron-cmd-norm": aileron,
            "fcs/elevator-cmd-norm": elevator,
            "fcs/rudder-cmd-norm": rudder,
        }
        for name, value in values.items():
            self._set_property(name, float(value))

    def control_initialization_status(self) -> dict:
        propulsion = self.jsbsim_exec.get_propulsion()
        engine_count = propulsion.get_num_engines()
        return {
            "model": self.model_name,
            "gear_command_norm": self._get_property("gear/gear-cmd-norm"),
            "gear_position_norm": self._get_property("gear/gear-pos-norm"),
            "flap_command_norm": self._get_property("fcs/flap-cmd-norm"),
            "flap_position_norm": self._get_property("fcs/flap-pos-norm"),
            "engine_running": [bool(self._get_property(
                "propulsion/engine/set-running" if index == 0
                else f"propulsion/engine[{index}]/set-running"))
                for index in range(engine_count)],
            "throttle_command_norm": self._get_property("fcs/throttle-cmd-norm"),
            "throttle_position_norm": self._get_property("fcs/throttle-pos-norm"),
            "throttle_adapter": "command_only_no_adapter",
        }

    def clone_for_prediction(self):
        """Create an independent JSBSim carrier at the exact public flight state.

        Candidate rollout state never touches the official aircraft FDM, time,
        counters, events, weapons, or runtime diagnostics.
        """
        clone = JSBSimAircraftPlatform(
            self.agent_id, self.side, self.aircraft_type,
            self.position.copy(), self.velocity.copy(), float(self.heading),
            pitch=float(self.pitch), roll=float(self.roll),
            model_root=str(self.model_root), model_name=self.model_name,
            reference_lat=self.reference_lat, reference_lon=self.reference_lon,
            reference_alt=self.reference_alt,
            simulation_frequency=self.simulation_frequency,
            suppress_output=self.suppress_output)
        clone.position = self.position.copy()
        clone.velocity = self.velocity.copy()
        clone.heading = float(self.heading)
        clone.pitch = float(self.pitch)
        clone.roll = float(self.roll)
        clone._reset_fdm()
        clone.apply_direct_fcs_command(self.direct_fcs_command)
        return clone

    def step_physics_once(self, dt: float | None = None) -> bool:
        del dt
        if not self.alive:
            return False
        self.apply_direct_fcs_command(self._direct_fcs_command)
        if not self.jsbsim_exec.run():
            self.kill("nonfinite")
            return False
        self._update_from_jsbsim()
        if not np.isfinite(np.concatenate([self.position, self.velocity])).all():
            self.kill("nonfinite")
            return False
        if self.position[2] <= 0.0:
            self.kill("crash")
            return False
        self.max_speed_observed_mps = max(self.max_speed_observed_mps, self.speed)
        self.max_abs_load_factor_g = max(self.max_abs_load_factor_g, self.load_factor_g)
        return True

    @property
    def load_factor_g(self) -> float:
        return abs(self._get_property("accelerations/Nz"))

    def _update_from_jsbsim(self) -> None:
        lon = self._get_property("position/long-gc-deg")
        lat = self._get_property("position/lat-geod-deg")
        alt_m = self._get_property("position/h-sl-ft") * self.FT2M
        self.position = lla_to_local(
            lon, lat, alt_m, self.reference_lat, self.reference_lon, self.reference_alt)
        self._geodetic = np.array([lon, lat, alt_m], dtype=np.float32)
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
