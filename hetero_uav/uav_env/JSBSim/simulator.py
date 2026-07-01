"""Aircraft and Missile simulators wrapping JSBSim flight dynamics."""
from __future__ import annotations

import contextlib
import os
import sys
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union

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

    def check_missile_warning(self):
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):
    """Scripted close-range AAM with proportional guidance."""

    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator,
               uid: str, missile_model: str = "AIM-9L",
               guidance_config: dict | None = None):
        assert parent.dt == target.dt
        missile = MissileSimulator(uid, parent.color, missile_model, parent.dt)
        missile.configure_guidance(guidance_config or {})
        missile.launch(parent)
        missile.target(target)
        return missile

    def __init__(self, uid="A0101", color="Red", model="AIM-9L", dt=1 / 12):
        super().__init__(uid, color, dt)
        self._status = MissileSimulator.INACTIVE
        self.model = model
        self.parent_aircraft = None
        self.target_aircraft = None
        self.render_explosion = False
        self._kill_rewarded = False
        self._parent_id: str = ""
        self._target_id: str = ""
        self._termination_reason: str = ""  # "hit", "p_hit_fail", "timeout", "target_dead", "unknown"

        # Scripted close-range AAM parameters. Fixed speed is a simulation
        # constant, not a real weapon performance table.
        self._g = 9.81
        self._t_max = 60
        self._K = 3
        self._nyz_max = 30
        self._Rc = 300
        self._missile_speed_mps = 600.0
        self._guidance_mode = "legacy"
        self._min_range_m = np.nan
        self._last_closing_speed_mps = np.nan
        self._directional_match_at_hit_check = np.nan
        self._p_hit_at_hit_check = np.nan
        self._speed_at_termination_mps = np.nan
        self._t_arm = 0.15  # warhead safety-arming delay (s) — prevents same-frame detonation at launch

    def configure_guidance(self, cfg: dict) -> None:
        cfg = cfg or {}
        mode = str(cfg.get("mode", self._guidance_mode)).lower()
        self._guidance_mode = mode if mode in {"legacy", "pn"} else "legacy"
        self._K = float(cfg.get("navigation_gain", self._K))
        self._nyz_max = float(cfg.get("max_overload_g", self._nyz_max))
        self._missile_speed_mps = float(cfg.get("speed_mps", self._missile_speed_mps))


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
    def K(self):
        return self._K  # navigation gain for PN guidance; default 3.0 is a project/TAM-supported parameter, not a BRMA numeric table value

    @property
    def target_distance(self) -> float:
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        self.parent_aircraft = parent
        self._parent_id = parent.uid
        self.parent_aircraft.launch_missiles.append(self)
        self._geodetic[:] = parent.get_geodetic()
        self._position[:] = parent.get_position()
        self._velocity[:] = self._initial_velocity(parent)
        self._posture[:] = parent.get_rpy()
        self._posture[0] = 0
        self._update_posture_from_velocity()
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        self._t = 0
        self._status = MissileSimulator.LAUNCHED
        self._termination_reason = ""
        self._min_range_m = np.nan
        self._last_closing_speed_mps = np.nan
        self._directional_match_at_hit_check = np.nan
        self._p_hit_at_hit_check = np.nan
        self._speed_at_termination_mps = np.nan
        self.render_explosion = False

    def target(self, target: AircraftSimulator):
        self.target_aircraft = target
        self._target_id = target.uid
        self.target_aircraft.under_missiles.append(self)

    def run(self):
        self._t += self.dt
        action, distance = self._guidance()
        self._min_range_m = (
            distance if not np.isfinite(self._min_range_m)
            else min(float(self._min_range_m), float(distance))
        )

        if (distance < self._Rc and self.target_aircraft.is_alive
                and self._t > self._t_arm):  # warhead must be armed before detonation
            # Paper: P_hit = 0.05 + 0.95 · dir_match — probabilistic kill filter
            # even when the physical missile reaches the target.
            if self._roll_hit_probability():
                self._status = MissileSimulator.HIT
                self.target_aircraft.shotdown()
                self._termination_reason = "hit"
            else:
                self._status = MissileSimulator.MISS  # warhead fails, target survives
                self._termination_reason = "p_hit_fail"
        elif self._t > self._t_max:
            self._status = MissileSimulator.MISS
            self._termination_reason = "timeout"
        elif not self.target_aircraft.is_alive:
            self._status = MissileSimulator.MISS
            self._termination_reason = "target_dead"
        else:
            self._state_trans(action)
            return
        self._speed_at_termination_mps = float(np.linalg.norm(self.get_velocity()))

    def _roll_hit_probability(self) -> bool:
        """Paper 2.1.3 hit probability using missile velocity and LOS.

        P_hit = 0.05 + 0.95 * max(0, Vm dot Los / (|Vm| |Los|)).
        This replaces the previous heading-difference approximation.
        """
        vm = self.get_velocity()
        los = self.target_aircraft.get_position() - self.get_position()
        vm_norm = float(np.linalg.norm(vm))
        los_norm = float(np.linalg.norm(los))

        if vm_norm < 1e-8 or los_norm < 1e-8:
            directional_match = 1.0
        else:
            directional_match = float(np.sum(vm * los) / (vm_norm * los_norm + 1e-8))
            directional_match = max(0.0, directional_match)

        P_hit = 0.05 + 0.95 * directional_match
        self._directional_match_at_hit_check = float(directional_match)
        self._p_hit_at_hit_check = float(P_hit)
        return np.random.random() < P_hit

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

    @staticmethod
    def _unit(vec, fallback=None):
        arr = np.asarray(vec, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if np.isfinite(norm) and norm >= 1e-8:
            return arr / norm
        if fallback is None:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return MissileSimulator._unit(fallback)

    @staticmethod
    def _direction_from_posture(posture):
        _roll, pitch, yaw = np.asarray(posture, dtype=np.float64)
        return np.asarray([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ], dtype=np.float64)

    def _initial_velocity(self, parent: AircraftSimulator):
        parent_velocity = np.asarray(parent.get_velocity(), dtype=np.float64)
        parent_speed = float(np.linalg.norm(parent_velocity))
        if np.isfinite(parent_speed) and parent_speed > 1e-6:
            direction = self._unit(parent_velocity)
        else:
            direction = self._unit(self._direction_from_posture(parent.get_rpy()))
        return direction * self._missile_speed_mps

    def _update_posture_from_velocity(self):
        vel = np.asarray(self.get_velocity(), dtype=np.float64)
        speed = float(np.linalg.norm(vel))
        if not np.isfinite(speed) or speed < 1e-8:
            return
        yaw = float(np.arctan2(vel[1], vel[0]))
        pitch = float(np.arcsin(np.clip(vel[2] / speed, -1.0, 1.0)))
        self._posture[:] = np.asarray([0.0, pitch, yaw], dtype=np.float64)

    def _guidance(self):
        los = self.target_aircraft.get_position() - self.get_position()
        distance = float(np.linalg.norm(los))
        if self._guidance_mode == "pn":
            a_cmd, diag = self.compute_pn_lateral_acceleration(
                self.get_position(),
                self.get_velocity(),
                self.target_aircraft.get_position(),
                self.target_aircraft.get_velocity(),
                navigation_gain=self._K,
                max_overload_g=self._nyz_max,
            )
            self._last_closing_speed_mps = float(diag.get("closing_speed_mps", np.nan))
            return a_cmd, max(distance, 1e-8)
        desired_dir = self._unit(los, fallback=self.get_velocity())
        return desired_dir, max(distance, 1e-8)

    @staticmethod
    def compute_pn_lateral_acceleration(
        missile_pos,
        missile_vel,
        target_pos,
        target_vel,
        *,
        navigation_gain: float = 3.0,
        max_overload_g: float = 30.0,
    ):
        r = np.asarray(target_pos, dtype=np.float64) - np.asarray(missile_pos, dtype=np.float64)
        vm = np.asarray(missile_vel, dtype=np.float64)
        vt = np.asarray(target_vel, dtype=np.float64)
        v_rel = vt - vm
        R = float(np.linalg.norm(r))
        if not np.isfinite(R) or R < 1e-8:
            return np.zeros(3, dtype=np.float64), {"range_m": np.nan, "closing_speed_mps": 0.0}
        r_hat = r / R
        closing_speed = -float(np.dot(v_rel, r_hat))
        vm_norm = float(np.linalg.norm(vm))
        if not np.isfinite(vm_norm) or vm_norm < 1e-8:
            return np.zeros(3, dtype=np.float64), {"range_m": R, "closing_speed_mps": closing_speed}
        if closing_speed <= 0.0:
            return np.zeros(3, dtype=np.float64), {
                "range_m": R,
                "closing_speed_mps": closing_speed,
                "omega_los_norm": 0.0,
                "acc_cmd_norm": 0.0,
            }
        missile_dir = vm / vm_norm
        omega_los = np.cross(r, v_rel) / max(R * R, 1e-8)
        a_cmd = float(navigation_gain) * closing_speed * np.cross(omega_los, missile_dir)
        a_cmd = a_cmd - float(np.dot(a_cmd, missile_dir)) * missile_dir
        acc_norm = float(np.linalg.norm(a_cmd))
        max_acc = float(max_overload_g) * 9.81
        if np.isfinite(acc_norm) and acc_norm > max_acc > 0.0:
            a_cmd = a_cmd / acc_norm * max_acc
        if not np.all(np.isfinite(a_cmd)):
            a_cmd = np.zeros(3, dtype=np.float64)
        return a_cmd.astype(np.float64), {
            "range_m": R,
            "closing_speed_mps": closing_speed,
            "omega_los_norm": float(np.linalg.norm(omega_los)),
            "acc_cmd_norm": float(np.linalg.norm(a_cmd)),
        }

    def _state_trans(self, action):
        if self._guidance_mode == "pn":
            current_vel = np.asarray(self.get_velocity(), dtype=np.float64)
            new_velocity = current_vel + np.asarray(action, dtype=np.float64) * self.dt
            new_dir = self._unit(new_velocity, fallback=current_vel)
            self._velocity[:] = new_dir * self._missile_speed_mps
            self._position[:] += self.dt * self.get_velocity()
            self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
            self._update_posture_from_velocity()
            return

        desired_dir = action
        current_dir = self._unit(self.get_velocity(), fallback=desired_dir)
        desired_dir = self._unit(desired_dir, fallback=current_dir)
        dot = float(np.clip(np.sum(current_dir * desired_dir), -1.0, 1.0))
        angle = float(np.arccos(dot))

        if angle > 1e-8:
            max_turn = self.K * self._nyz_max * self._g / max(self._missile_speed_mps, 1.0) * self.dt
            frac = min(1.0, max_turn / angle)
            sin_angle = max(float(np.sin(angle)), 1e-8)
            new_dir = (
                np.sin((1.0 - frac) * angle) / sin_angle * current_dir
                + np.sin(frac * angle) / sin_angle * desired_dir
            )
            new_dir = self._unit(new_dir, fallback=desired_dir)
        else:
            new_dir = desired_dir

        self._velocity[:] = new_dir * self._missile_speed_mps
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        self._update_posture_from_velocity()
