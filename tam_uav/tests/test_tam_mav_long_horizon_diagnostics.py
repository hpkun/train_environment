from __future__ import annotations

from scripts.validate_tam_categorical_initial_policy_flight import (
    choose_stable_elevator,
    summarize_flight_trace,
)
from scripts.diagnose_tam_mav_policy_drift import summarize_policy_traces


def test_trace_summary_reports_long_horizon_stability_fields():
    trace = [
        {"step": 1, "altitude_m": 6000.0, "vertical_speed_mps": -1.0, "speed_mps": 250.0},
        {"step": 2, "altitude_m": 5900.0, "vertical_speed_mps": -2.0, "speed_mps": 245.0},
    ]
    summary = summarize_flight_trace(trace, death_reason="alive", death_step=-1)
    assert summary["final_altitude_m"] == 5900.0
    assert summary["min_altitude_m"] == 5900.0
    assert summary["final_speed_mps"] == 245.0
    assert summary["min_speed_mps"] == 245.0
    assert summary["mean_vertical_speed_mps"] == -1.5
    assert summary["death_reason"] == "alive"
    assert summary["death_step"] == -1


def test_stable_elevator_selection_prefers_survival_then_altitude_margin():
    sweep = {
        "10": {"survived_1000": True, "min_altitude_m": 3000.0},
        "11": {"survived_1000": True, "min_altitude_m": 3400.0},
        "12": {"survived_1000": False, "min_altitude_m": 2490.0},
    }
    assert choose_stable_elevator(sweep) == 11


def test_stable_elevator_selection_falls_back_to_latest_crash():
    sweep = {
        "6": {"survived_1000": False, "death_step": 592, "min_altitude_m": 2498.0},
        "8": {"survived_1000": False, "death_step": 437, "min_altitude_m": 2497.0},
    }
    assert choose_stable_elevator(sweep) == 6


def test_policy_trace_summary_reports_dominant_elevator_and_predeath_descent():
    traces = [[
        {"action_indices": [39, 20, 9, 20], "altitude_m": 3000.0, "vertical_speed_mps": -20.0},
        {"action_indices": [39, 20, 9, 20], "altitude_m": 2490.0, "vertical_speed_mps": -40.0},
    ]]
    summary = summarize_policy_traces(traces)
    assert summary["dominant_elevator_bin"] == 9
    assert summary["predeath_100_mean_vertical_speed_mps"] == -30.0
