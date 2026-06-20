from __future__ import annotations

import numpy as np
import pytest
import inspect

from scripts import audit_tam_direct_control_response as audit
from uav_env import make_env


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def _surface_after(agent_id: str, action: list[float], frames: int = 60) -> dict[str, float]:
    env = make_env(CONFIG)
    env.reset(seed=0)
    sim = env._get_sim(agent_id)
    normalized = np.asarray(action, dtype=np.float64)
    indices = np.rint((normalized + 1.0) * 0.5 * (env.tam_action_levels - 1)).astype(np.int64)
    targets = env._parse_actions({agent_id: indices})
    for _ in range(frames):
        env._apply_pid_controls(targets)
        sim.run()
    values = {
        "left_aileron": float(sim.get_property_value("fcs/left-aileron-pos-rad")),
        "right_aileron": float(sim.get_property_value("fcs/right-aileron-pos-rad")),
        "elevator": float(sim.get_property_value("fcs/elevator-pos-rad")),
        "rudder": float(sim.get_property_value("fcs/rudder-pos-rad")),
        "throttle": float(sim.get_property_value("fcs/throttle-cmd-norm")),
    }
    env.close()
    return values


@pytest.mark.parametrize(
    ("negative", "positive", "surface_names"),
    [
        ([0.75, -0.4, 0.0, 0.0], [0.75, 0.4, 0.0, 0.0], ("left_aileron", "right_aileron")),
        ([0.75, 0.0, -0.3, 0.0], [0.75, 0.0, 0.3, 0.0], ("elevator",)),
        ([0.75, 0.0, 0.0, -0.4], [0.75, 0.0, 0.0, 0.4], ("rudder",)),
    ],
)
def test_f22_direct_axis_changes_real_surface(negative, positive, surface_names):
    negative_surface = _surface_after("red_0", negative)
    positive_surface = _surface_after("red_0", positive)

    assert any(
        abs(negative_surface[name] - positive_surface[name]) > 1e-3
        for name in surface_names
    )


def test_f16_direct_surfaces_still_respond():
    left = _surface_after("red_1", [0.75, -0.4, 0.3, -0.4])
    right = _surface_after("red_1", [0.75, 0.4, -0.3, 0.4])

    assert abs(left["left_aileron"] - right["left_aileron"]) > 1e-3
    assert abs(left["elevator"] - right["elevator"]) > 1e-3
    assert abs(left["rudder"] - right["rudder"]) > 1e-3


def test_throttle_mapping_remains_in_tam_range():
    low = _surface_after("red_0", [-1.0, 0.0, 0.0, 0.0])
    high = _surface_after("red_0", [1.0, 0.0, 0.0, 0.0])

    assert low["throttle"] == pytest.approx(0.4)
    assert high["throttle"] == pytest.approx(0.9)


def test_audit_uses_model_specific_surface_property_adapter():
    assert audit.FCS_SURFACE_PROPERTIES["f22"] == {
        "left_aileron": "fcs/left-aileron-pos-rad",
        "right_aileron": "fcs/right-aileron-pos-rad",
        "elevator": "fcs/elevator-pos-rad",
        "rudder": "fcs/rudder-pos-rad",
        "throttle": "fcs/throttle-pos-norm",
    }
    assert audit.FCS_SURFACE_PROPERTIES["f16"]["elevator"] == "fcs/elevator-pos-rad"


def test_audit_explicitly_rejects_nonfinite_crash_reason():
    assert "no_crash_nonfinite_state" in inspect.getsource(audit._checks)
