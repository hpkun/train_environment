from __future__ import annotations

import numpy as np

from uav_env import make_env
from scripts.validate_tam_airborne_initialization import (
    classify_flight_outcome,
    summarize_reset_reports,
)


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


def test_reset_report_summary_requires_each_aircraft_contract_to_pass():
    reports = {
        "red_0": {"passed_reset_contract": True},
        "blue_0": {"passed_reset_contract": False},
    }
    summary = summarize_reset_reports(reports)
    assert summary["aircraft_count"] == 2
    assert summary["passed_reset_contract"] is False
    assert summary["failed_aircraft"] == ["blue_0"]


def test_flight_outcome_classification_is_explicit():
    assert classify_flight_outcome(True, "", True) == "alive"
    assert classify_flight_outcome(False, "Crash_LowAlt", True) == "long_horizon_trim_failure"
    assert classify_flight_outcome(False, "Missile_Kill", True) == "missile_kill"
    assert classify_flight_outcome(False, "", False) == "nonfinite"
    assert classify_flight_outcome(False, "LowSpeed", True) == "policy_or_flight_failure:LowSpeed"
