from __future__ import annotations

import numpy as np

from uav_env import make_env


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def test_airborne_stabilization_runs_for_formal_f22_reset():
    env = make_env(CONFIG)
    env.reset(seed=0)
    report = env.red_planes["red_0"]._initial_stabilization_report
    assert report["enabled"] is True
    assert report["model"] == "f22"
    env.close()


def test_red_direct_action_is_not_overridden_after_reset_stabilization():
    env = make_env(CONFIG)
    env.reset(seed=0)
    indices = np.array([0, 0, 39, 39], dtype=np.int64)
    targets = env._parse_actions({"red_0": indices})
    env._apply_pid_controls(targets)
    sim = env.red_planes["red_0"]
    assert np.isclose(sim.get_property_value("fcs/throttle-cmd-norm"), 0.4)
    assert np.isclose(sim.get_property_value("fcs/aileron-cmd-norm"), -1.0)
    assert np.isclose(sim.get_property_value("fcs/elevator-cmd-norm"), 1.0)
    assert np.isclose(sim.get_property_value("fcs/rudder-cmd-norm"), 1.0)
    env.close()
