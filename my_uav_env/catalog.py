"""
Minimal property catalog for JSBSim simulation.
Provides named access to JSBSim properties via a MixedCatalog dict.
"""
import re
from collections import namedtuple
from numpy.linalg import norm

Property = namedtuple("Property", "name_jsbsim description min max access spaces clipped update")
Property.__new__.__defaults__ = (None, None, float("-inf"), float("+inf"), "RW", None, True, None)


class MixedCatalog(dict):
    """
    A dict-like catalog that lazily resolves property names.
    Supports both attribute access (Catalog.foo_bar) and dict access (Catalog["foo_bar"]).
    JSBSim native properties are auto-registered via add_jsbsim_props().
    Extra/computed properties are defined in _EXTRA_PROPS.
    """

    # Custom properties not natively in JSBSim (metric conversions, extreme state detection)
    _EXTRA_PROPS = {}

    def __init__(self):
        super().__init__()
        self._init_extra_props()

    def _init_extra_props(self):
        for name, prop in self._EXTRA_PROPS.items():
            self[name] = prop

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            # Fall back: treat name as a raw JSBSim property path
            self[name] = Property(name_jsbsim=name, access="RW")
        return super().__getitem__(name)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Catalog has no property '{name}'")

    def add_jsbsim_props(self, jsbsim_props):
        """
        Register native JSBSim properties from the FDM property catalog.
        Called by AircraftSimulator.reload().
        Format per line: "property/path (access)"
        """
        for line in jsbsim_props:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            jsbsim_path, access = parts[0], parts[1]
            access = re.sub(r"[\(\)]", "", access)
            py_name = re.sub(r"_$", "", re.sub(r"[\-/\]\[]+", "_", jsbsim_path))
            if py_name not in self:
                self[py_name] = Property(name_jsbsim=jsbsim_path, access=access)


# ---- Define extra/computed properties ----

def _update_detect_extreme_state(sim):
    extreme_velocity = sim.get_property_value("velocities/eci-velocity-mag-fps") >= 1e10
    extreme_rotation = norm(sim.get_property_values([
        "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec"
    ])) >= 1000
    extreme_altitude = sim.get_property_value("position/h-sl-ft") >= 1e10
    extreme_acc = max(
        abs(sim.get_property_value(p))
        for p in ["accelerations/n-pilot-x-norm",
                   "accelerations/n-pilot-y-norm",
                   "accelerations/n-pilot-z-norm"]
    ) > 1e1
    sim.set_property_value(
        "detect/extreme-state",
        extreme_altitude or extreme_rotation or extreme_velocity or extreme_acc,
    )


def _make_ft2m_update(jsbsim_ft, extra_m):
    def update(sim):
        sim.set_property_value(extra_m, sim.get_property_value(jsbsim_ft) * 0.3048)
    return update


MixedCatalog._EXTRA_PROPS = {
    "position_h_sl_m": Property("position/h-sl-m", "altitude MSL [m]", -500, 26000, access="R",
                                 update=_make_ft2m_update("position/h-sl-ft", "position/h-sl-m")),
    "velocities_v_north_mps": Property("velocities/v-north-mps", "velocity north [m/s]", -700, 700, access="R",
                                        update=_make_ft2m_update("velocities/v-north-fps", "velocities/v-north-mps")),
    "velocities_v_east_mps": Property("velocities/v-east-mps", "velocity east [m/s]", -700, 700, access="R",
                                       update=_make_ft2m_update("velocities/v-east-fps", "velocities/v-east-mps")),
    "velocities_v_down_mps": Property("velocities/v-down-mps", "velocity down [m/s]", -700, 700, access="R",
                                       update=_make_ft2m_update("velocities/v-down-fps", "velocities/v-down-mps")),
    "velocities_vc_mps": Property("velocities/vc-mps", "airspeed [m/s]", 0, 1400, access="R",
                                   update=_make_ft2m_update("velocities/vc-fps", "velocities/vc-mps")),
    "velocities_u_mps": Property("velocities/u-mps", "body x velocity [m/s]", -700, 700, access="R",
                                  update=_make_ft2m_update("velocities/u-fps", "velocities/u-mps")),
    "velocities_v_mps": Property("velocities/v-mps", "body y velocity [m/s]", -700, 700, access="R",
                                  update=_make_ft2m_update("velocities/v-fps", "velocities/v-mps")),
    "velocities_w_mps": Property("velocities/w-mps", "body z velocity [m/s]", -700, 700, access="R",
                                  update=_make_ft2m_update("velocities/w-fps", "velocities/w-mps")),
    "detect_extreme_state": Property("detect/extreme-state", "extreme state flag", 0, 1, access="R",
                                      update=_update_detect_extreme_state),
}

# Singleton catalog instance
Catalog = MixedCatalog()
