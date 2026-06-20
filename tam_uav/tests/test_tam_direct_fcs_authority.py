from __future__ import annotations

import numpy as np

from uav_env import make_env
from scripts.experiment_logging_schema import TAM_ACTION_TIMESERIES_COLUMNS


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def test_direct_fcs_writer_discovers_paths_and_separates_high_low_throttle():
    env = make_env(CONFIG)
    env.reset(seed=0)
    for agent_id in ("red_0", "red_1", "blue_0"):
        sim = env.red_planes.get(agent_id) or env.blue_planes[agent_id]
        low = sim.set_tam_direct_fcs_command({
            "throttle_cmd_norm": 0.4, "aileron_cmd_norm": 0.0,
            "elevator_cmd_norm": 0.0, "rudder_cmd_norm": 0.0,
        })
        high = sim.set_tam_direct_fcs_command({
            "throttle_cmd_norm": 0.9, "aileron_cmd_norm": 0.0,
            "elevator_cmd_norm": 0.0, "rudder_cmd_norm": 0.0,
        })
        assert high["written_fcs_paths"]
        assert any("throttle-pos" in path for path in high["written_fcs_paths"])
        shared_paths = set(low["readback_values"]) & set(high["readback_values"])
        throttle_paths = [path for path in shared_paths if "throttle" in path]
        assert throttle_paths
        assert any(
            high["readback_values"][path] > low["readback_values"][path]
            for path in throttle_paths
        )
    env.close()


def test_reset_and_step_paths_both_expose_unified_writer_reports():
    env = make_env(CONFIG)
    env.reset(seed=0)
    reset_report = env.red_planes["red_0"]._initial_stabilization_report
    assert reset_report["writer_report_before_run_ic"]["written_fcs_paths"]

    targets = env._parse_actions({"red_0": np.array([39, 20, 6, 20], np.int64)})
    env._apply_pid_controls(targets)
    command = env._last_tam_action_commands["red_0"]
    assert command["written_fcs_paths"]
    assert command["readback_values"]
    env.close()


def test_static_calibration_is_state_independent_and_f16_defaults_to_identity():
    env = make_env(CONFIG)
    raw = {
        "throttle_cmd_norm": 0.9, "aileron_cmd_norm": 0.1,
        "elevator_cmd_norm": -0.2, "rudder_cmd_norm": 0.3,
    }
    first = env._calibrate_tam_direct_command(raw, "f16")
    second = env._calibrate_tam_direct_command(raw, "f16")
    assert first == second
    assert first["calibrated_throttle_cmd_norm"] == raw["throttle_cmd_norm"]
    assert first["calibrated_aileron_cmd_norm"] == raw["aileron_cmd_norm"]
    assert first["calibration_profile"]["model"] == "f16"
    env.close()


def test_rich_action_schema_keeps_calibration_and_writer_readback():
    required = {
        "raw_throttle_cmd_norm", "raw_aileron_cmd_norm",
        "raw_elevator_cmd_norm", "raw_rudder_cmd_norm",
        "calibrated_throttle_cmd_norm", "calibrated_aileron_cmd_norm",
        "calibrated_elevator_cmd_norm", "calibrated_rudder_cmd_norm",
        "calibration_profile", "written_fcs_paths", "readback_values",
    }
    assert required <= set(TAM_ACTION_TIMESERIES_COLUMNS)
