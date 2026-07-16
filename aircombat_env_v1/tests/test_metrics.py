import numpy as np
import pytest

from aircombat_env_v1.metrics import aggregate_case_metrics, step_response_metrics


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


def test_case_aggregation_averages_costs_and_keeps_worst_safety_value():
    base = {
        "roll_error_integral": 1.0, "pitch_error_integral": 2.0,
        "speed_error_integral": 3.0, "control_increment_energy": 4.0,
        "control_rate_energy": 5.0, "aileron_saturation_ratio": 0.1,
        "elevator_saturation_ratio": 0.2, "throttle_saturation_ratio": 0.3,
        "altitude_loss_m": 10.0, "maximum_alpha_deg": 5.0,
        "maximum_beta_deg": 2.0, "maximum_load_factor": 3.0,
        "heading_final_error_deg": 4.0, "pitch_final_error_deg": 2.0,
        "speed_final_error_mps": 6.0, "heading_error_rms_deg": 3.0,
        "pitch_error_rms_deg": 2.0, "speed_error_rms_mps": 5.0,
        "heading_error_p95_deg": 4.0, "pitch_error_p95_deg": 3.0,
        "speed_error_p95_mps": 7.0, "crashed": False, "has_nan_or_inf": False,
    }
    worse = dict(base, roll_error_integral=3.0, maximum_alpha_deg=20.0)
    aggregate = aggregate_case_metrics([
        {"name": "a", "complete": True, "metrics": base},
        {"name": "b", "complete": True, "metrics": worse},
    ])
    assert aggregate["roll_error_integral"] == 2.0
    assert aggregate["maximum_alpha_deg"] == 20.0
