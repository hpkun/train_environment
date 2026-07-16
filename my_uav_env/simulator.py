"""
Aircraft and Missile simulators wrapping JSBSim flight dynamics.
Based on the CloseAirCombat project, with all neural-network dependencies removed.
"""
import contextlib
import os
import sys
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Union

from my_uav_env.alignment.state_extractor import body_vector_to_inertial_neu


def compute_los_angles_and_rates(relative_position, relative_velocity,
                                 epsilon: float = 1e-8):
    """Paper Eq.10/Eq.11 LOS angles and rates for missile-to-target vectors."""
    r = np.asarray(relative_position, dtype=np.float64)
    r_dot = np.asarray(relative_velocity, dtype=np.float64)
    if r.shape != (3,) or r_dot.shape != (3,) or not np.all(
            np.isfinite(np.concatenate([r, r_dot]))):
        raise FloatingPointError("non-finite LOS input")
    rx, ry, rz = r
    rdx, rdy, rdz = r_dot
    horizontal_sq = float(rx * rx + ry * ry)
    horizontal = float(np.sqrt(horizontal_sq))
    range_sq = float(np.dot(r, r))
    beta = float(np.arctan2(ry, rx))
    epsilon_angle = float(np.arctan2(rz, horizontal))
    beta_dot = float((rdy * rx - ry * rdx) / max(horizontal_sq, epsilon))
    epsilon_dot = float(
        (horizontal_sq * rdz - rz * (rdx * rx + rdy * ry))
        / max(range_sq * horizontal, epsilon)
    )
    values = np.asarray(
        [beta, epsilon_angle, beta_dot, epsilon_dot], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite LOS output")
    return tuple(float(value) for value in values)


def compute_paper_eq9_overloads(relative_position, relative_velocity,
                                missile_velocity, navigation_constant=3.0,
                                gravity=9.81, epsilon: float = 1e-8):
    """Return literal paper Eq.9 lateral and pitch overload commands."""
    beta, los_epsilon, beta_dot, epsilon_dot = compute_los_angles_and_rates(
        relative_position, relative_velocity, epsilon=epsilon)
    velocity = np.asarray(missile_velocity, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("non-finite missile velocity")
    speed = float(np.linalg.norm(velocity))
    if speed <= epsilon:
        return 0.0, 0.0
    gamma_m = float(np.arcsin(np.clip(velocity[2] / speed, -1.0, 1.0)))
    cos_sum = float(np.cos(los_epsilon + beta))
    if abs(cos_sum) < epsilon:
        cos_sum = epsilon if cos_sum >= 0.0 else -epsilon
    n_mc = (
        navigation_constant * speed * np.cos(gamma_m) / gravity
        * (beta_dot
           + np.tan(los_epsilon) * np.tan(los_epsilon + beta) * epsilon_dot)
    )
    n_mh = (
        speed / gravity * navigation_constant / cos_sum * epsilon_dot
    )
    values = np.asarray([n_mc, n_mh], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite Eq.9 command")
    return float(values[0]), float(values[1])

# jsbsim is imported at module level.  JSBSim's C++ startup banner is printed
# to stdout during ``FGFDMExec()`` construction.  The ``SuppressOutput`` context
# manager (below) can temporarily silence it; use ``suppress_jsbsim_output=True``
# on ``AircraftSimulator`` or ``UavCombatEnv`` to enable this.

# The ``import jsbsim`` step may print a one-line banner to C++ stdout.
# We suppress it with a Python-level redirect only — no CRT/Win32 handle
# manipulation (which is fragile and can permanently break stdout on some
# Windows/MSVC configurations, causing "no output" failures).
# Per-aircraft FGFDMExec construction banners are handled by SuppressOutput
# further below.  KMP_DUPLICATE_LIB_OK must be set before importing this module.
_saved_py_out = sys.stdout
_saved_py_err = sys.stderr
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
try:
    import jsbsim
finally:
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _saved_py_out
    sys.stderr = _saved_py_err

from .catalog import Catalog, Property
from .utils import get_package_data_dir, LLA2NEU, NEU2LLA

TeamColors = str  # "Red", "Blue", etc.


# ==============================================================================
#  OS-level stdout/stderr suppressor for JSBSim C++ banner
# ==============================================================================
class SuppressOutput:
    """Context manager that silences C/C++ stdout/stderr (e.g. JSBSim banner).

    Uses two independent mechanisms that together cover all output paths:

    1. ``SetStdHandle(STD_OUTPUT_HANDLE, NUL)`` — redirects ``WriteConsoleW()``
       which ``std::cout`` uses on real consoles and which survives ``_dup2``.
    2. ``_dup2(nul_fd, 1)`` — redirects CRT-based output (``printf``, ``fprintf``,
       ``os.write(1, …)``).

    **Neither mechanism calls ``_close(1)``**, which is fragile on Windows because
    CRT ``_dup`` does not always call ``DuplicateHandle`` — closing fd 1 can
    destroy the underlying console handle and permanently break stdout.

    Used in ``AircraftSimulator.reload()`` to wrap ``jsbsim.FGFDMExec()``
    which emits a C++ startup banner on every instantiation.
    """

    def __init__(self, suppress: bool = True):
        self._suppress = suppress

    def __enter__(self):
        if not self._suppress:
            return self

        # ---- 0. Flush everything ----
        sys.stdout.flush()
        sys.stderr.flush()
        self._fflush_all()

        # ---- 1. Save Python stdout/stderr objects ----
        self._saved_py_stdout = sys.stdout
        self._saved_py_stderr = sys.stderr

        # ---- 2. Redirect Python to devnull ----
        self._devnull_out = open(os.devnull, "w")
        self._devnull_err = open(os.devnull, "w")
        sys.stdout = self._devnull_out
        sys.stderr = self._devnull_err

        # ---- 3. Redirect C-level output ----
        self._crt_ok = False
        self._win32_ok = False

        if sys.platform == "win32":
            try:
                import ctypes
                self._crt = ctypes.CDLL("msvcrt")
                self._crt._dup.restype = ctypes.c_int
                self._crt._dup2.restype = ctypes.c_int
                self._crt._close.restype = ctypes.c_int

                # Save CRT fds via _dup (creates new fd referencing same handle)
                self._saved_fd_out = self._crt._dup(1)
                self._saved_fd_err = self._crt._dup(2)
                if self._saved_fd_out != -1 and self._saved_fd_err != -1:
                    # Open NUL for CRT redirection
                    self._nul_fd = os.open(os.devnull, os.O_WRONLY)
                    self._crt._dup2(self._nul_fd, 1)
                    self._crt._dup2(self._nul_fd, 2)
                    self._crt_ok = True

                # Win32 std handle redirect — catches WriteConsoleW() which
                # survives _dup2 on real consoles.
                self._krn = ctypes.WinDLL("kernel32", use_last_error=True)
                self._saved_h_out = self._krn.GetStdHandle(-11)
                self._saved_h_err = self._krn.GetStdHandle(-12)
                self._h_nul = self._krn.CreateFileW(
                    "NUL", 0x40000000, 3, None, 3, 0x80, None,
                )
                if self._h_nul not in (-1, None):
                    self._krn.SetStdHandle(-11, self._h_nul)
                    self._krn.SetStdHandle(-12, self._h_nul)
                    self._win32_ok = True
            except Exception:
                pass
        else:
            try:
                self._saved_fd_out_posix = os.dup(1)
                self._saved_fd_err_posix = os.dup(2)
                self._dev_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self._dev_fd, 1)
                os.dup2(self._dev_fd, 2)
                self._crt_ok = True
            except OSError:
                pass

        return self

    def __exit__(self, *args):
        if not self._suppress:
            return

        self._fflush_all()

        # ---- Close devnull Python wrappers ----
        try:
            self._devnull_out.close()
        except Exception:
            pass
        try:
            self._devnull_err.close()
        except Exception:
            pass

        # ---- Restore C-level output ----
        try:
            if hasattr(self, "_saved_fd_out_posix"):
                # POSIX path
                os.dup2(self._saved_fd_out_posix, 1)
                os.dup2(self._saved_fd_err_posix, 2)
                try:
                    os.close(self._saved_fd_out_posix)
                except OSError:
                    pass
                try:
                    os.close(self._saved_fd_err_posix)
                except OSError:
                    pass
                os.close(self._dev_fd)
            elif hasattr(self, "_crt") and self._crt_ok:
                # Windows CRT restore: dup2 saved fds back
                try:
                    self._crt._dup2(self._saved_fd_out, 1)
                except OSError:
                    pass
                try:
                    self._crt._dup2(self._saved_fd_err, 2)
                except OSError:
                    pass
                # Close saved fd copies
                try:
                    self._crt._close(self._saved_fd_out)
                except OSError:
                    pass
                try:
                    self._crt._close(self._saved_fd_err)
                except OSError:
                    pass
        except Exception:
            pass

        # ---- Restore Win32 std handles ----
        try:
            if getattr(self, "_win32_ok", False):
                self._krn.SetStdHandle(-11, self._saved_h_out)
                self._krn.SetStdHandle(-12, self._saved_h_err)
        except Exception:
            pass

        # ---- Cleanup NUL handle ----
        if hasattr(self, "_h_nul") and self._h_nul not in (None, -1):
            try:
                self._krn.CloseHandle(self._h_nul)
            except Exception:
                pass
        if hasattr(self, "_nul_fd"):
            try:
                os.close(self._nul_fd)
            except OSError:
                pass

        # ---- ALWAYS restore Python stdout/stderr ----
        sys.stdout = self._saved_py_stdout
        sys.stderr = self._saved_py_stderr

    @staticmethod
    def _fflush_all():
        try:
            import ctypes
            libc = ctypes.CDLL("msvcrt" if sys.platform == "win32" else None)
            libc.fflush(None)
        except Exception:
            pass


class BaseSimulator(ABC):
    """Abstract base for all simulation entities (aircraft, missiles)."""

    def __init__(self, uid: str, color: TeamColors, dt: float):
        self._uid = uid
        self._color = color
        self._dt = dt
        self.model = ""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)
        logging.debug(f"{self.__class__.__name__}:{self._uid} created")

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def color(self) -> str:
        return self._color

    @property
    def dt(self) -> float:
        return self._dt

    def get_geodetic(self):
        """(longitude, latitude, altitude) — degrees, meters"""
        return self._geodetic

    def get_position(self):
        """(north, east, up) — meters"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw) — radians"""
        return self._posture

    def get_velocity(self):
        """(v_north, v_east, v_up) — m/s"""
        return self._velocity

    def reload(self):
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, **kwargs):
        pass

    def log(self):
        lon, lat, alt = self.get_geodetic()
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi
        return f"{self._uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw}," \
               f"Name={self.model.upper()}," \
               f"Color={self._color}"

    @abstractmethod
    def close(self):
        if hasattr(self, "jsbsim_exec") and self.jsbsim_exec is not None:
            try:
                self.jsbsim_exec = None
            except Exception:
                pass

    def __del__(self):
        self.close()
        logging.debug(f"{self.__class__.__name__}:{self._uid} deleted")


class AircraftSimulator(BaseSimulator):
    """Wraps a JSBSim FGFDMExec instance for one aircraft."""

    ALIVE = 0
    CRASH = 1
    SHOTDOWN = 2

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = "f16",
                 init_state: dict = None,
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60,
                 num_missiles: int = 0,
                 suppress_jsbsim_output: bool = True,
                 **kwargs):
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model
        self.init_state = init_state if init_state is not None else {}
        self.lon0, self.lat0, self.alt0 = origin
        self.bloods = 100
        self._status = AircraftSimulator.ALIVE
        self.num_missiles = num_missiles
        self.num_left_missiles = num_missiles

        # Linked simulators
        self.partners: List[AircraftSimulator] = []
        self.enemies: List[AircraftSimulator] = []
        self.launch_missiles: List[MissileSimulator] = []
        self.under_missiles: List[MissileSimulator] = []

        # Initialize JSBSim
        self.jsbsim_exec = None
        self._suppress_jsbsim = suppress_jsbsim_output
        self._reload_total = 0        # total reload() calls
        self._jsbsim_created = 0      # times FGFDMExec was constructed
        self.reload()

    @property
    def is_alive(self):
        return self._status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        return self._status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        return self._status == AircraftSimulator.SHOTDOWN

    def crash(self):
        self._status = AircraftSimulator.CRASH

    def shotdown(self):
        self._status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """Reset the aircraft to initial conditions without recreating JSBSim."""
        super().reload()
        self.bloods = 100
        self._status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()
        self.under_missiles.clear()
        self.num_left_missiles = self.num_missiles

        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin

        self._reload_total += 1

        # First-time: create JSBSim instance. Subsequent: reuse (avoids C++ memory leak).
        if self.jsbsim_exec is None:
            self._jsbsim_created += 1
            if self._jsbsim_created > 1:
                with open(f"_jsbsim_recreate_{os.getpid()}.log", "a") as _f:
                    _f.write(f"{self.uid}: JSBSim recreated (create #{self._jsbsim_created}, "
                             f"total reloads={self._reload_total})\n")
            data_dir = get_package_data_dir()
            _ctx = SuppressOutput() if self._suppress_jsbsim else contextlib.nullcontext()
            with _ctx:
                self.jsbsim_exec = jsbsim.FGFDMExec(data_dir)
                self.jsbsim_exec.set_debug_level(0)
                self.jsbsim_exec.load_model(self.model)
                Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
            self.jsbsim_exec.set_dt(self.dt)
        else:
            self.jsbsim_exec.reset_to_initial_conditions(0)

        self._clear_default_condition()

        # Apply (possibly new) initial state
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)

        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # Restart propulsion
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()

        self._update_properties()

    def _clear_default_condition(self):
        """Reset JSBSim initial condition properties to defaults."""
        defaults = {
            "ic/long-gc-deg": 120.0,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000,
            "ic/psi-true-deg": 0.0,
            "ic/u-fps": 800.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/p-rad_sec": 0.0,
            "ic/q-rad_sec": 0.0,
            "ic/r-rad_sec": 0.0,
            "ic/roc-fpm": 0.0,
            "ic/terrain-elevation-ft": 0,
        }
        for prop_path, value in defaults.items():
            self.jsbsim_exec.set_property_value(prop_path, value)

    def run(self):
        """Advance JSBSim by one physics frame. Returns False if sim terminated."""
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()
            return result
        return True

    def close(self):
        if self.jsbsim_exec is not None:
            self.jsbsim_exec = None
        self.partners = []
        self.enemies = []

    def _update_properties(self):
        FT2M = 0.3048
        # position — read imperial (only units this JSBSim build provides)
        lon = self.get_property_value("position/long-gc-deg")
        lat = self.get_property_value("position/lat-geod-deg")
        alt_m = self.get_property_value("position/h-sl-ft") * FT2M
        self._geodetic[:] = [lon, lat, alt_m]
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # posture
        self._posture[:] = self.get_property_values([
            "attitude/roll-rad", "attitude/pitch-rad", "attitude/heading-true-rad",
        ])
        # velocity — read imperial, convert to m/s
        vn = self.get_property_value("velocities/v-north-fps") * FT2M
        ve = self.get_property_value("velocities/v-east-fps") * FT2M
        vd = self.get_property_value("velocities/v-down-fps") * FT2M
        self._velocity[:] = [vn, ve, -vd]  # down -> up

    def get_sim_time(self):
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        if len(props) != len(values):
            raise ValueError("Mismatched lengths")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        if isinstance(prop, Property):
            if prop.access == "R" and prop.update:
                prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        elif isinstance(prop, str):
            # Direct JSBSim property path
            return self.jsbsim_exec.get_property_value(prop)
        raise ValueError(f"Unhandled property type: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max
            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
            if "W" in prop.access and prop.update:
                prop.update(self)
        elif isinstance(prop, str):
            self.jsbsim_exec.set_property_value(prop, value)
        else:
            raise ValueError(f"Unhandled property type: {type(prop)} ({prop})")

    def project_velocity_magnitude(self, maximum_speed_mps: float) -> float:
        """Scale the JSBSim velocity vector without changing its direction."""
        velocity = np.asarray(self.get_velocity(), dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        maximum = float(maximum_speed_mps)
        if not np.isfinite(speed) or speed <= maximum or speed <= 1e-12:
            return speed
        scale = maximum / speed
        feet_per_meter = 1.0 / 0.3048
        projected = velocity * scale
        lon, lat, altitude = self.get_geodetic()
        roll, pitch, heading = self.get_rpy()
        angular_rates = self.get_property_values((
            "velocities/p-rad_sec", "velocities/q-rad_sec",
            "velocities/r-rad_sec"))
        fcs_paths = (
            "fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
            "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm")
        fcs_commands = self.get_property_values(fcs_paths)
        initial_state = {
            "ic/long-gc-deg": lon,
            "ic/lat-geod-deg": lat,
            "ic/h-sl-ft": altitude * feet_per_meter,
            "ic/phi-rad": roll,
            "ic/theta-rad": pitch,
            "ic/psi-true-rad": heading,
            "ic/vn-fps": projected[0] * feet_per_meter,
            "ic/ve-fps": projected[1] * feet_per_meter,
            "ic/vd-fps": -projected[2] * feet_per_meter,
            "ic/p-rad_sec": angular_rates[0],
            "ic/q-rad_sec": angular_rates[1],
            "ic/r-rad_sec": angular_rates[2],
        }
        for path, value in initial_state.items():
            self.jsbsim_exec.set_property_value(path, float(value))
        self.jsbsim_exec.reset_to_initial_conditions(1)
        self.set_property_values(fcs_paths, fcs_commands)
        self._update_properties()
        return float(np.linalg.norm(self._velocity))

    def check_missile_warning(self):
        from my_uav_env.sensors import select_most_dangerous_missile
        missile, _diag = select_most_dangerous_missile(self, self.under_missiles)
        return missile

    def get_missile_warning_diagnostic(self):
        from my_uav_env.sensors import select_most_dangerous_missile
        return select_most_dangerous_missile(self, self.under_missiles)


class MissileSimulator(BaseSimulator):
    """Proportional-navigation missile with physical dynamics."""

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator,
               uid: str, missile_model: str = "AIM-9L",
               guidance_mode: str = "paper_eq9", config=None, rng=None,
               launch_speed_mps: float | None = None,
               overshoot_window_s: float | None = None,
               overshoot_distance_hysteresis_m: float = 0.0,
               positive_closing_threshold_mps: float = 0.0):
        assert parent.dt == target.dt
        missile = MissileSimulator(
            uid, parent.color, missile_model, parent.dt,
            guidance_mode=guidance_mode, config=config, rng=rng,
            launch_speed_mps=launch_speed_mps,
            overshoot_window_s=overshoot_window_s,
            overshoot_distance_hysteresis_m=overshoot_distance_hysteresis_m,
            positive_closing_threshold_mps=positive_closing_threshold_mps)
        missile.launch(parent)
        missile.target(target)
        if (guidance_mode == "paper_minimal_point_mass_v1"
                and launch_speed_mps is not None):
            los = np.asarray(target.get_position(), dtype=np.float64) - np.asarray(
                missile.get_position(), dtype=np.float64)
            distance = float(np.linalg.norm(los))
            if distance > 1e-9:
                missile._velocity[:] = los / distance * float(launch_speed_mps)
                speed = float(np.linalg.norm(missile._velocity))
                pitch = float(np.arcsin(np.clip(
                    missile._velocity[2] / max(speed, 1e-9), -1.0, 1.0)))
                heading = float(np.arctan2(
                    missile._velocity[1], missile._velocity[0]))
                missile._posture[:] = [0.0, pitch, heading]
        elif (guidance_mode == "paper_learnable_point_mass_v1"
              and launch_speed_mps is not None):
            roll, pitch, heading = np.asarray(parent.get_rpy(), dtype=np.float64)
            body_x = body_vector_to_inertial_neu(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                roll, pitch, heading)
            norm = float(np.linalg.norm(body_x))
            if np.isfinite(norm) and norm > 1e-9:
                missile._velocity[:] = (
                    body_x / norm * float(launch_speed_mps))
                missile._posture[:] = [0.0, pitch, heading]
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-9L", dt=1 / 12,
                 guidance_mode: str = "paper_eq9", config=None, rng=None,
                 launch_speed_mps: float | None = None,
                 overshoot_window_s: float | None = None,
                 overshoot_distance_hysteresis_m: float = 0.0,
                 positive_closing_threshold_mps: float = 0.0):
        super().__init__(uid, color, dt)
        if guidance_mode not in (
                "paper_eq9", "legacy_simplified", "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1"):
            raise ValueError(
                "guidance_mode must be 'paper_eq9', 'legacy_simplified', or "
                "'paper_minimal_point_mass_v1'")
        self.guidance_mode = guidance_mode
        self._status = MissileSimulator.INACTIVE
        from configs.brma_mappo_paper_spec import MissileConfig
        self.config = config or MissileConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.model = self.config.model.value if config is not None else model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False
        self._kill_rewarded = False
        self._parent_id: str = ""
        self._target_id: str = ""
        self._termination_reason: str = ""  # "hit", "p_hit_fail", "timeout", "low_speed", "overshoot", "target_dead"

        # Missile physical parameters (modern short-range AAM)
        # Isp=240 → Δv ≈ 568 m/s (rocket equation), peak speed ≈ 850–900 m/s (M2.5+)
        # cD=0.22 → energy retention over 10 km tail-chase, still lethal at 60 s
        self._g = 9.81
        self._t_max = float(self.config.maximum_flight_time_s.value)
        self._t_thrust = float(self.config.thrust_time_s.value)
        self._Isp = float(self.config.specific_impulse_s.value)
        self._Length = float(self.config.length_m.value)
        self._Diameter = float(self.config.diameter_m.value)
        self._cD = float(self.config.drag_coefficient.value)
        self._m0 = float(self.config.initial_mass_kg.value)
        self._dm = float(self.config.mass_flow_kg_s.value)
        self._K = float(self.config.navigation_constant.value)
        self._nyz_max = float(self.config.maximum_overload_g.value)
        self._Rc = float(self.config.hit_radius_m.value)
        self._v_min = float(self.config.minimum_speed_mps.value)
        self._t_arm = float(self.config.arming_time_s.value)
        self._launch_speed_mps = launch_speed_mps
        self._overshoot_window_s = float(
            5.0 if overshoot_window_s is None else overshoot_window_s)
        self._overshoot_distance_hysteresis_m = float(
            overshoot_distance_hysteresis_m)
        self._positive_closing_threshold_mps = float(
            positive_closing_threshold_mps)
        self._trajectory_sink = None
        self._historical_min_range_m = float("inf")
        self._has_ever_positive_closing = False
        self._overshoot_counter = 0
        self._pn_guidance_frames = 0
        self._pn_nonzero_command_frames = 0
        self._maximum_command_g = 0.0

    @property
    def is_alive(self):
        return self._status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        return self._status == MissileSimulator.HIT

    @property
    def is_done(self):
        return self._status in (MissileSimulator.HIT, MissileSimulator.MISS)

    @property
    def Isp(self):
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        return self._K  # constant 3.0 — paper §2.1.3 proportional guidance law

    @property
    def S(self):
        S0 = np.pi * (self._Diameter / 2) ** 2
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        self.parent_aircraft = parent
        self._parent_id = parent.uid
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = parent.get_velocity()
        if self._launch_speed_mps is not None:
            speed = float(np.linalg.norm(self._velocity))
            if speed > 1e-12:
                self._velocity[:] *= float(self._launch_speed_mps) / speed
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        self._t = 0
        self._m = self._m0
        self._dtheta, self._dphi = 0, 0
        self._status = MissileSimulator.LAUNCHED
        self._distance_pre = np.inf
        self._distance_increment = deque(maxlen=max(
            1, int(round(self._overshoot_window_s / self.dt))))
        self._historical_min_range_m = float("inf")
        self._has_ever_positive_closing = False
        self._overshoot_counter = 0
        self._pn_guidance_frames = 0
        self._pn_nonzero_command_frames = 0
        self._maximum_command_g = 0.0
        self._left_t = int(1 / self.dt)
        self.render_explosion = False

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self._target_id = target.uid
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        self._t += self.dt
        zero_action = np.zeros(2, dtype=np.float64)
        if self.target_aircraft is None or not self.target_aircraft.is_alive:
            self._status = MissileSimulator.MISS
            self._termination_reason = "target_dead"
            return

        distance = float(np.linalg.norm(
            self.target_aircraft.get_position() - self.get_position()))
        if not np.isfinite(distance):
            self._status = MissileSimulator.MISS
            self._termination_reason = "numerical_invalid"
            return
        if self._t > self._t_max:
            self._status = MissileSimulator.MISS
            self._termination_reason = "timeout"
            self._emit_trajectory_sample(self._intercept_diagnostic(
                zero_action, distance, self._distance_pre))
            return

        armed = (
            self.guidance_mode == "paper_minimal_point_mass_v1"
            or (self.guidance_mode == "paper_learnable_point_mass_v1"
                and self._t >= self._t_arm)
            or (self.guidance_mode not in (
                    "paper_minimal_point_mass_v1",
                    "paper_learnable_point_mass_v1")
                and self._t > self._t_arm))
        if armed and distance <= self._Rc:
            if self._roll_hit_probability():
                self._status = MissileSimulator.HIT
                self.target_aircraft.shotdown()
                self._termination_reason = "hit"
            else:
                self._status = MissileSimulator.MISS
                self._termination_reason = "p_hit_fail"
            self._emit_trajectory_sample(self._intercept_diagnostic(
                zero_action, distance, self._distance_pre))
            return

        try:
            action, _ = self._guidance()
        except (FloatingPointError, ValueError, OverflowError):
            self._status = MissileSimulator.MISS
            self._termination_reason = "numerical_invalid"
            return
        command_norm = float(np.linalg.norm(action))
        if not np.isfinite(command_norm):
            self._status = MissileSimulator.MISS
            self._termination_reason = "numerical_invalid"
            return
        self._pn_guidance_frames += 1
        self._pn_nonzero_command_frames += int(command_norm > 1e-9)
        self._maximum_command_g = max(self._maximum_command_g, command_norm)
        previous_distance = self._distance_pre
        self._state_trans(action)
        state = np.concatenate([
            np.asarray(self.get_position(), dtype=np.float64),
            np.asarray(self.get_velocity(), dtype=np.float64)])
        if not np.all(np.isfinite(state)):
            self._status = MissileSimulator.MISS
            self._termination_reason = "numerical_invalid"
            return

        distance = float(np.linalg.norm(
            self.target_aircraft.get_position() - self.get_position()))
        increasing = bool(distance > previous_distance)
        self._distance_increment.append(increasing)
        self._distance_pre = distance
        diagnostic = self._intercept_diagnostic(
            action, distance, previous_distance)

        if (self.guidance_mode not in (
                "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1")
                and np.linalg.norm(self.get_velocity()) < self._v_min):
            self._status = MissileSimulator.MISS
            self._termination_reason = "low_speed"
        elif (self.guidance_mode in (
                "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1")
              and self._overshoot_counter >= self._distance_increment.maxlen):
            self._status = MissileSimulator.MISS
            self._termination_reason = "overshoot"
        elif (self.guidance_mode not in (
                "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1")
              and len(self._distance_increment) == self._distance_increment.maxlen
              and np.sum(self._distance_increment) >= self._distance_increment.maxlen):
            self._status = MissileSimulator.MISS
            self._termination_reason = "overshoot"
        self._emit_trajectory_sample(diagnostic)

    def _intercept_diagnostic(self, action, distance: float,
                              previous_distance: float) -> dict:
        missile_position = np.asarray(self.get_position(), dtype=np.float64)
        target_position = np.asarray(
            self.target_aircraft.get_position(), dtype=np.float64)
        missile_velocity = np.asarray(self.get_velocity(), dtype=np.float64)
        target_velocity = np.asarray(
            self.target_aircraft.get_velocity(), dtype=np.float64)
        los = target_position - missile_position
        relative_velocity = target_velocity - missile_velocity
        los_norm = max(float(np.linalg.norm(los)), 1e-9)
        missile_speed = float(np.linalg.norm(missile_velocity))
        target_speed = float(np.linalg.norm(target_velocity))
        closing = float(-np.dot(los, relative_velocity) / los_norm)
        if closing > self._positive_closing_threshold_mps:
            self._has_ever_positive_closing = True
        self._historical_min_range_m = min(
            self._historical_min_range_m, float(distance))
        causal_overshoot = (
            self._has_ever_positive_closing
            and np.isfinite(self._historical_min_range_m)
            and float(distance) > (
                self._historical_min_range_m
                + self._overshoot_distance_hysteresis_m)
            and closing <= 0.0
        )
        self._overshoot_counter = (
            self._overshoot_counter + 1 if causal_overshoot else 0)
        cosine = (float(np.dot(missile_velocity, los))
                  / max(missile_speed * los_norm, 1e-9))
        velocity_to_los_angle_deg = float(np.rad2deg(np.arccos(
            np.clip(cosine, -1.0, 1.0))))
        los_angular_rate = float(
            np.linalg.norm(np.cross(los, relative_velocity))
            / max(los_norm * los_norm, 1e-9))
        range_delta = (None if not np.isfinite(previous_distance)
                       else float(distance - previous_distance))
        return {
            "missile_id": self.uid,
            "shooter_id": self._parent_id,
            "target_id": self._target_id,
            "team": "red" if self._parent_id.startswith("red") else "blue",
            "flight_time_sec": float(self._t),
            "range_m": float(distance),
            "historical_min_range_m": float(self._historical_min_range_m),
            "range_delta_m": range_delta,
            "closing_speed_mps": closing,
            "has_ever_positive_closing": bool(
                self._has_ever_positive_closing),
            "missile_speed_mps": missile_speed,
            "target_speed_mps": target_speed,
            "missile_velocity_to_LOS_angle_deg": velocity_to_los_angle_deg,
            "LOS_angular_rate_rad_s": los_angular_rate,
            "PN_lateral_command_g": float(action[0]),
            "PN_vertical_command_g": float(action[1]),
            "missile_position": missile_position.tolist(),
            "target_position": target_position.tolist(),
            "missile_velocity": missile_velocity.tolist(),
            "target_velocity": target_velocity.tolist(),
            "overshoot_counter": int(self._overshoot_counter),
            "termination_reason": self._termination_reason,
        }

    def _emit_trajectory_sample(self, diagnostic: dict) -> None:
        if self._trajectory_sink is not None:
            diagnostic["termination_reason"] = self._termination_reason
            self._trajectory_sink(dict(diagnostic))

    def _roll_hit_probability(self) -> bool:
        """Paper 2.1.3 hit probability using missile velocity and LOS.

        P_hit = 0.05 + 0.95 * max(0, Vm dot Los / (|Vm| |Los|)).
        This replaces the previous heading-difference approximation.
        """
        return float(self.rng.random()) < self._hit_probability()

    def _hit_probability(self) -> float:
        """Return the paper hit probability for the current geometry."""
        vm = self.get_velocity()
        los = self.target_aircraft.get_position() - self.get_position()
        vm_norm = float(np.linalg.norm(vm))
        los_norm = float(np.linalg.norm(los))

        if los_norm < 1e-8:
            directional_match = 1.0
        elif vm_norm < 1e-8:
            directional_match = 0.0
        else:
            directional_match = float(np.dot(vm, los) / (vm_norm * los_norm))
            directional_match = float(np.clip(directional_match, 0.0, 1.0))

        return float(0.05 + 0.95 * directional_match)

    def detach_references(self):
        """Remove this missile from aircraft-owned lists after termination."""
        if self.parent_aircraft is not None:
            self.parent_aircraft.launch_missiles = [
                m for m in self.parent_aircraft.launch_missiles if m is not self]
        if self.target_aircraft is not None:
            self.target_aircraft.under_missiles = [
                m for m in self.target_aircraft.under_missiles if m is not self]

    def log(self):
        if self.is_alive:
            return super().log()
        elif self.is_done and not self.render_explosion:
            self.render_explosion = True
            if not self.is_success:
                # MISS (target dead / timeout / lost lock) — no explosion
                return f"-{self._uid}"
            log_msg = f"-{self._uid}\n"
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg += f"{self._uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self._color},Radius={self._Rc}"
            return log_msg
        return None

    def close(self):
        self.target_aircraft = None

    def _guidance(self):
        """Proportional navigation guidance law."""
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])
        theta_m = np.arcsin(dz_m / v_m) if v_m > 0 else 0
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])
        Rxyz = max(np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m]), 1e-8)

        if self.guidance_mode in (
                "paper_eq9", "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1"):
            relative_position = np.array([x_t - x_m, y_t - y_m, z_t - z_m])
            relative_velocity = np.array([dx_t - dx_m, dy_t - dy_m, dz_t - dz_m])
            ny, nz = compute_paper_eq9_overloads(
                relative_position, relative_velocity, self.get_velocity(),
                navigation_constant=self.K, gravity=self._g)
        else:
            dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / (Rxy ** 2 + 1e-8)
            deps = ((dz_t - dz_m) * Rxy ** 2 - (z_t - z_m) * (
                (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz ** 2 * Rxy + 1e-8)
            ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
            nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        command = np.asarray([ny, nz], dtype=np.float64)
        if self.guidance_mode in (
                "paper_eq9", "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1"):
            norm = float(np.linalg.norm(command))
            if not np.isfinite(norm):
                raise FloatingPointError("non-finite PN command")
            if norm > self._nyz_max:
                command *= self._nyz_max / norm
        else:
            command = np.clip(command, -self._nyz_max, self._nyz_max)
        return command, Rxyz

    def _state_trans(self, action):
        """Update missile position, velocity, and attitude."""
        if self.guidance_mode in (
                "paper_minimal_point_mass_v1",
                "paper_learnable_point_mass_v1"):
            self._state_trans_minimal_point_mass(action)
            return
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        v = np.linalg.norm(self.get_velocity())
        theta, phi = self.get_rpy()[1:]
        T = self._g * self.Isp * self._dm
        D = 0.5 * self._cD * self.S * self.rho * v ** 2
        nx = (T - D) / (self._m * self._g)
        ny, nz = action

        dv = self._g * (nx - np.sin(theta))
        self._dphi = self._g / max(v, 1e-8) * (ny / max(np.cos(theta), 1e-8))
        self._dtheta = self._g / max(v, 1e-8) * (nz - np.cos(theta))

        v += self.dt * dv
        phi += self.dt * self._dphi
        theta += self.dt * self._dtheta
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta),
        ])
        self._posture[:] = np.array([0, theta, phi])
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm

    def _state_trans_minimal_point_mass(self, action):
        """Constant-speed point mass with the published PN steering command."""
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(
            *self.get_position(), self.lon0, self.lat0, self.alt0)
        speed = float(np.linalg.norm(self.get_velocity()))
        if speed <= 1e-9:
            return
        theta, heading = self.get_rpy()[1:]
        ny, nz = action
        heading_rate = self._g * ny / max(speed * np.cos(theta), 1e-8)
        pitch_rate = self._g * (nz - np.cos(theta)) / speed
        heading += self.dt * heading_rate
        theta += self.dt * pitch_rate
        self._velocity[:] = np.array([
            speed * np.cos(theta) * np.cos(heading),
            speed * np.cos(theta) * np.sin(heading),
            speed * np.sin(theta),
        ])
        self._posture[:] = np.array([0.0, theta, heading])
