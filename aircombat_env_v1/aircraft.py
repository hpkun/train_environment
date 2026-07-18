"""Minimal reusable JSBSim F-16 wrapper with SI-valued state access."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import jsbsim
except ImportError:  # Allows geometry/PID tests without JSBSim installed.
    jsbsim = None


FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
RAD_TO_DEG = 180.0 / np.pi
DATA_DIR = Path(__file__).resolve().parent / "data"


class AircraftSimulator:
    """Own one FGFDMExec and reuse it for every reset."""

    def __init__(self, sim_frequency_hz=60, model="f16", data_dir=DATA_DIR):
        if sim_frequency_hz <= 0:
            raise ValueError("sim_frequency_hz must be positive")
        if jsbsim is None:
            raise ImportError("jsbsim is required to construct AircraftSimulator")
        self.dt = 1.0 / float(sim_frequency_hz)
        self.model = model
        self.data_dir = Path(data_dir)
        self.fdm = jsbsim.FGFDMExec(str(self.data_dir))
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model(self.model):
            raise RuntimeError(f"failed to load JSBSim model {self.model!r}")
        self.fdm.set_dt(self.dt)
        self._has_initialized = False

    def get_property(self, name):
        return float(self.fdm.get_property_value(name))

    def set_property(self, name, value):
        self.fdm.set_property_value(name, float(value))

    def reset(self, altitude_m=6000.0, speed_mps=250.0, heading_deg=0.0,
              roll_deg=0.0, pitch_deg=0.0, latitude_deg=60.0,
              longitude_deg=120.0, elevator_trim=0.0, throttle_base=0.3,
              alpha_deg=None, beta_deg=0.0, flight_path_angle_deg=None):
        """Reset initial conditions without constructing another FGFDMExec."""
        initial = {
            "ic/long-gc-deg": longitude_deg,
            "ic/lat-geod-deg": latitude_deg,
            "ic/h-sl-ft": altitude_m * M_TO_FT,
            "ic/psi-true-deg": heading_deg,
            "ic/phi-deg": roll_deg,
            "ic/theta-deg": pitch_deg,
            "ic/p-rad_sec": 0.0,
            "ic/q-rad_sec": 0.0,
            "ic/r-rad_sec": 0.0,
            "ic/roc-fpm": 0.0,
            "ic/terrain-elevation-ft": 0.0,
        }
        if alpha_deg is None and flight_path_angle_deg is None:
            initial.update({"ic/u-fps": speed_mps * M_TO_FT,
                            "ic/v-fps": 0.0,"ic/w-fps": 0.0})
        else:
            alpha = float(alpha_deg if alpha_deg is not None else
                          pitch_deg - flight_path_angle_deg)
            gamma = float(flight_path_angle_deg if flight_path_angle_deg is not None else
                          pitch_deg - alpha)
            initial.update({"ic/vt-fps": speed_mps * M_TO_FT,"ic/alpha-deg":alpha,
                            "ic/beta-deg":beta_deg,"ic/gamma-deg":gamma})
        # The first engine/FCS initialization of a newly loaded FDM differs
        # slightly from later resets. Prime it once through this same path so
        # the first recorded episode and every reused episode are reproducible.
        passes = 1 if self._has_initialized else 2
        for _ in range(passes):
            self.fdm.reset_to_initial_conditions(0)
            for name, value in initial.items():
                self.set_property(name, value)
            self.set_controls(0.0, elevator_trim, 0.0, throttle_base)
            if not self.fdm.run_ic():
                raise RuntimeError("JSBSim failed to initialize")
            propulsion = self.fdm.get_propulsion()
            for index in range(propulsion.get_num_engines()):
                propulsion.get_engine(index).init_running()
            propulsion.get_steady_state()
        self._has_initialized = True
        self.set_controls(0.0, elevator_trim, 0.0, throttle_base)
        return self.state()

    def set_controls(self, aileron, elevator, rudder, throttle):
        controls = {
            "fcs/aileron-cmd-norm": np.clip(aileron, -1.0, 1.0),
            "fcs/elevator-cmd-norm": np.clip(elevator, -1.0, 1.0),
            "fcs/rudder-cmd-norm": np.clip(rudder, -1.0, 1.0),
            "fcs/throttle-cmd-norm": np.clip(throttle, 0.0, 1.0),
        }
        for name, value in controls.items():
            self.set_property(name, value)

    def run(self):
        """Advance exactly one physics step."""
        if not self.fdm.run():
            raise RuntimeError("JSBSim physics step failed")
        return self.state()

    def state(self):
        """Return the experiment's observable state using SI units and radians."""
        return {
            "time": self.get_property("simulation/sim-time-sec"),
            "longitude": self.get_property("position/long-gc-deg"),
            "latitude": self.get_property("position/lat-geod-deg"),
            "altitude": self.get_property("position/h-sl-ft") * FT_TO_M,
            "roll": self.get_property("attitude/roll-rad"),
            "pitch": self.get_property("attitude/pitch-rad"),
            "heading": self.get_property("attitude/heading-true-rad"),
            "true_airspeed": self.get_property("velocities/vtrue-fps") * FT_TO_M,
            "v_north": self.get_property("velocities/v-north-fps") * FT_TO_M,
            "v_east": self.get_property("velocities/v-east-fps") * FT_TO_M,
            "v_down": self.get_property("velocities/v-down-fps") * FT_TO_M,
            "p": self.get_property("velocities/p-rad_sec"),
            "q": self.get_property("velocities/q-rad_sec"),
            "r": self.get_property("velocities/r-rad_sec"),
            "alpha": self.get_property("aero/alpha-rad"),
            "beta": self.get_property("aero/beta-rad"),
            "load_factor": self.get_property("accelerations/n-pilot-z-norm"),
        }
