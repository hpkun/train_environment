from scripts.analyze_tam_training_trend import (
    classify_stage,
    summarize_missile_event_rows,
    staged_values,
    summarize_training_rows,
)


def _rows():
    return [
        {
            "total_steps": str(step),
            "avg_return": str(value),
            "red_win": str(index / 40),
            "blue_win": str(1 - index / 40),
            "mav_survival": str(index / 80),
            "red_missiles_fired": str(index),
            "missile_hits": str(index // 2),
            "blue_alive_final": str(2 - index / 10),
            "entropy_mav": "8.0", "entropy_uav": "7.0",
            "action_bin_usage_mav": "0.5",
            "action_bin_usage_uav": "0.6",
            "correction_factor_mean": "1.0",
            "approx_kl_mav": "0.01", "approx_kl_uav": "0.02",
            "nan_detected": "0",
        }
        for index, (step, value) in enumerate(
            [(100, -10), (200, -8), (300, -6), (400, -4), (500, -2)]
        )
    ]


def test_staged_values_reports_start_quartiles_and_end():
    assert staged_values(_rows(), "avg_return") == {
        "start": -10.0, "25%": -8.0, "50%": -6.0,
        "75%": -4.0, "end": -2.0,
    }


def test_training_summary_reports_positive_return_slope_and_finite_metrics():
    summary = summarize_training_rows(_rows())
    assert summary["final_return"] == -2.0
    assert summary["rolling_return_slope_per_10k_steps"] > 0
    assert summary["red_fired"]["end"] == 4.0
    assert summary["red_hit_rate"]["end"] == 0.5
    assert summary["collapse_detected"] is False
    assert summary["stage_decision"] == "A"


def test_stage_classifier_prioritizes_training_semantic_anomalies():
    summary = summarize_training_rows(_rows())
    summary["correction_factor"]["end"] = 50.0
    assert classify_stage(summary) == "C"


def test_stage_classifier_marks_persistent_mav_crash_as_stability_work():
    summary = summarize_training_rows(_rows())
    summary["rolling_return_slope_per_10k_steps"] = 0.0
    summary["mav_survival"]["end"] = 0.0
    summary["mav_death_step"] = {"start": 500.0, "end": 400.0, "slope": -1.0}
    assert classify_stage(summary) == "B"


def test_stage_classifier_uses_airborne_audit_to_separate_trim_from_reset():
    summary = summarize_training_rows(_rows())
    summary["environment_audit"] = {
        "passed": False,
        "reset_contract_passed": True,
        "f22_speed_at_60s_passed": False,
    }
    assert classify_stage(summary) == "B"
    summary["environment_audit"]["reset_contract_passed"] = False
    assert classify_stage(summary) == "D"


def test_rich_missile_events_provide_red_only_launch_and_hit_trends():
    rows = [
        {"step": "10", "owner_team": "red", "event_type": "launch", "hit_success": ""},
        {"step": "20", "owner_team": "blue", "event_type": "hit", "hit_success": "1"},
        {"step": "30", "owner_team": "red", "event_type": "hit", "hit_success": "1"},
    ]
    summary = summarize_missile_event_rows(rows)
    assert summary["red_fired"]["end"] == 1.0
    assert summary["red_hits"]["end"] == 1.0
    assert summary["red_hit_rate"]["end"] == 1.0
