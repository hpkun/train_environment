import numpy as np
import pytest

from aircombat_env_v1.metrics import step_response_metrics


def row(time, heading, pitch=0.0, speed=250.0, target_heading=0.2,
        target_pitch=0.0, target_speed=250.0):
    return {"time": time, "heading": heading, "pitch": pitch,
            "true_airspeed": speed, "target_heading": target_heading,
            "target_pitch": target_pitch, "target_true_airspeed": target_speed}


def test_overshoot_is_measured_only_after_target_is_reached():
    rows = [row(i, value) for i, value in enumerate((0.05, 0.15, 0.21, 0.25, 0.20))]
    metrics = step_response_metrics(rows, {"heading": 0.0, "pitch": 0.0,
                                           "true_airspeed": 250.0})
    assert metrics["heading_reached_target"] is True
    assert metrics["heading_overshoot_deg"] == pytest.approx(np.rad2deg(0.05))
    assert metrics["heading_settling_time_s"] == 1


def test_heading_step_wraps_across_pi():
    initial = np.deg2rad(179.0)
    target = np.deg2rad(-179.0)
    rows = [row(i, value, target_heading=target) for i, value in enumerate(
        np.deg2rad((179.5, -179.5, -178.5, -179.0)))]
    metrics = step_response_metrics(rows, {"heading": initial, "pitch": 0.0,
                                           "true_airspeed": 250.0})
    assert metrics["heading_reached_target"] is True
    assert metrics["heading_overshoot_deg"] == pytest.approx(0.5)
