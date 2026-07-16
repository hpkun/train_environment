import pytest

from aircombat_env_v1.config import load_config
from aircombat_env_v1.execution import bounded_combined_command
from aircombat_env_v1.scripts import tune_pid, validate_pid
from aircombat_env_v1 import validation


def test_attitude_integrals_are_always_zero_in_tuning():
    config = load_config()
    config["pid"]["roll"]["ki"] = 0.07
    config["pid"]["pitch"]["ki"] = 0.08
    prepared = tune_pid.prepare_config(config)
    assert prepared["pid"]["roll"]["ki"] == 0.0
    assert prepared["pid"]["pitch"]["ki"] == 0.0


def test_failed_stage_rolls_back_to_baseline():
    baseline = {"pid": "baseline"}
    proposed = {"pid": "proposed"}
    metrics = {"failed_case_count": 1}
    selected, accepted, reason = tune_pid.select_stage_config(
        baseline, proposed, 10.0, 20.0, metrics)
    assert selected is baseline
    assert accepted is False
    assert reason == "candidate_has_failed_cases"


def test_invalid_joint_cost_rejects_accept_candidate():
    stages = [{"stage_accepted": True}] * 3
    with pytest.raises(RuntimeError, match="refusing --accept-candidate"):
        tune_pid.require_acceptance_eligible(
            stages, 1e12, {"failed_case_count": 0})


def test_joint_health_rejects_unstable_final_segment():
    thresholds = load_config()["validation_thresholds"]
    metrics = {
        "maximum_alpha_deg": 1.0, "maximum_beta_deg": 1.0,
        "maximum_load_factor": 1.0, "aileron_saturation_ratio": 0.0,
        "elevator_saturation_ratio": 0.0, "throttle_saturation_ratio": 0.0,
        "heading_final_error_deg": 0.0, "pitch_final_error_deg": 0.0,
        "speed_final_error_mps": 0.0, "stable_heading_error_deg": 0.0,
        "stable_pitch_error_deg": 6.0, "stable_speed_error_mps": 0.0,
    }
    assert tune_pid.joint_health_reasons(metrics, thresholds, dynamic=True) == [
        "pitch_final_error"]


def test_bounded_combined_has_final_stable_segment():
    assert bounded_combined_command(250, 250, 60, 5, stable_duration=10) == (
        0.0, 0.0, 250.0)


def test_descent_then_level_does_not_descend_for_200_seconds():
    assert validate_pid.command_at_decision_step(
        "descent_then_level", 100, 250, 200, 5) == (0.0, 0.0, 250.0)
    assert validate_pid.command_at_decision_step(
        "descent_then_level", 999, 250, 200, 5) == (0.0, 0.0, 250.0)


def test_dynamic_case_uses_stable_segment_not_dynamic_final_field():
    thresholds = load_config()["validation_thresholds"]
    metrics = {
        "has_nan_or_inf": False, "crashed": False, "maximum_alpha_deg": 0.0,
        "maximum_beta_deg": 0.0, "maximum_load_factor": 1.0,
        "aileron_saturation_ratio": 0.0, "elevator_saturation_ratio": 0.0,
        "throttle_saturation_ratio": 0.0, "heading_final_error_deg": 999.0,
        "pitch_final_error_deg": 999.0, "speed_final_error_mps": 999.0,
        "stable_heading_error_deg": 1.0, "stable_pitch_error_deg": 1.0,
        "stable_speed_error_mps": 1.0,
    }
    assert validate_pid.failure_reasons(metrics, thresholds, dynamic=True) == []


def test_quick_mode_contains_exactly_eight_cases():
    assert len(validate_pid.quick_cases()) == 8


def test_mark_validated_rejects_quick_mode():
    with pytest.raises(ValueError, match="only with --mode full"):
        validate_pid.ensure_mark_validated_allowed("quick", True)


def test_pitch_integral_only_preserves_all_other_tuned_parameters():
    config = load_config()
    original = {
        "roll": dict(config["pid"]["roll"]),
        "pitch_kp": config["pid"]["pitch"]["kp"],
        "pitch_kd": config["pid"]["pitch"]["kd"],
        "speed": dict(config["pid"]["speed"]),
    }
    candidate = tune_pid.prepare_pitch_integral_config(config, 0.003)
    assert candidate["pid"]["roll"]["kp"] == original["roll"]["kp"]
    assert candidate["pid"]["roll"]["kd"] == original["roll"]["kd"]
    assert candidate["pid"]["roll"]["ki"] == 0.0
    assert candidate["pid"]["pitch"]["kp"] == original["pitch_kp"]
    assert candidate["pid"]["pitch"]["kd"] == original["pitch_kd"]
    assert candidate["pid"]["pitch"]["ki"] == 0.003
    assert candidate["pid"]["speed"]["kp"] == original["speed"]["kp"]
    assert candidate["pid"]["speed"]["ki"] == original["speed"]["ki"]
    assert candidate["pid"]["speed"]["kd"] == 0.0


def test_pitch_integral_stage_a_explicitly_contains_zero():
    assert 0.0 in tune_pid.PITCH_INTEGRAL_VALUES


def test_invalid_pitch_integral_candidate_cannot_be_accepted():
    with pytest.raises(RuntimeError, match="no valid pitch integral candidate"):
        tune_pid.require_pitch_integral_candidate(False)


def test_stage_a_uses_lexicographic_health_sort_not_cost():
    worse_health_low_cost = {
        "pitch_ki": 0.001, "crashed": False, "has_nan_or_inf": False,
        "failure_reasons": ["pitch_final_error"],
        "stable_pitch_error_deg": 1.0, "elevator_saturation_ratio": 0.0,
        "pitch_error_rms_deg": 1.0, "total_cost": 0.0,
    }
    healthy_high_cost = {
        "pitch_ki": 0.0025, "crashed": False, "has_nan_or_inf": False,
        "failure_reasons": [], "stable_pitch_error_deg": 4.0,
        "elevator_saturation_ratio": 0.1, "pitch_error_rms_deg": 4.0,
        "total_cost": 999999.0,
    }
    assert tune_pid.select_stage_a_nonzero(
        [worse_health_low_cost, healthy_high_cost])[0]["pitch_ki"] == 0.0025


def test_stage_b_uses_same_quick_cases_as_validate_script():
    assert tune_pid.quick_cases() == validate_pid.quick_cases()
    assert tune_pid.quick_cases() == validation.quick_cases()


def test_only_quick_eight_of_eight_is_valid():
    record = {"pitch_ki": 0.001, "quick_passed": 7,
              "crashed": False, "has_nan_or_inf": False}
    assert tune_pid.quick_gate_candidate_valid(record) is False
    record["quick_passed"] = 8
    assert tune_pid.quick_gate_candidate_valid(record) is True


def test_no_valid_candidate_uses_best_attempt_filename():
    assert tune_pid.candidate_output_name(False) == "best_attempt_config.yaml"
    assert tune_pid.candidate_output_name(True) == "candidate_config.yaml"
