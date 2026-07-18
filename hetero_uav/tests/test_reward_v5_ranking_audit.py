from __future__ import annotations

import pytest

from scripts.audit_hetero_3v2_reward_v5_ranking import (
    RULE_POLICY_LABEL,
    _observational_ranking,
    _paired_scenarios,
    _ranking_checks,
    _red_combat_counts,
    _undiscounted_task_return,
    _validate_scenario_uniqueness,
)


def _episode(**updates):
    row = {
        "seed": 1,
        "outcome": "draw",
        "red_kills": 0,
        "red_attack_alive_final": 2,
        "blue_attack_alive_final": 2,
        "mav_survived": 1,
        "undiscounted_task_return": 0.0,
        "discounted_team_return": 0.0,
        "potential_telescoping_error": 0.0,
    }
    row.update(updates)
    return row


def _result(label, rows, discounted_mean=0.0):
    return {
        "label": label,
        "discounted_team_return_mean": discounted_mean,
        "episode_rows": rows,
    }


def test_empty_sign_checks_are_na_not_pass():
    checks = _ranking_checks([_result("policy", [_episode()])])
    assert checks["red_win_positive"] == {"sample_count": 0, "pass": None}
    assert checks["blue_win_without_red_kill_negative"] == {
        "sample_count": 0, "pass": None}


@pytest.mark.parametrize(
    ("outcome", "task_return", "check_name", "expected"),
    [
        ("red_win", 1.0, "red_win_positive", True),
        ("red_win", -0.1, "red_win_positive", False),
        ("blue_win", -1.0, "blue_win_without_red_kill_negative", True),
        ("blue_win", 0.1, "blue_win_without_red_kill_negative", False),
    ],
)
def test_sign_checks_report_true_and_false(
        outcome, task_return, check_name, expected):
    checks = _ranking_checks([_result(
        "policy", [_episode(outcome=outcome,
                            undiscounted_task_return=task_return)])])
    assert checks[check_name] == {"sample_count": 1, "pass": expected}


def test_monotonic_checks_use_only_same_seed_matched_pairs():
    left = _result("left", [
        _episode(seed=10, blue_attack_alive_final=2,
                 undiscounted_task_return=0.0),
        _episode(seed=11, red_attack_alive_final=2,
                 undiscounted_task_return=1.0),
        _episode(seed=12, mav_survived=1,
                 undiscounted_task_return=1.0),
    ])
    right = _result("right", [
        _episode(seed=10, blue_attack_alive_final=1,
                 undiscounted_task_return=1.0),
        _episode(seed=11, red_attack_alive_final=1,
                 undiscounted_task_return=0.0),
        _episode(seed=12, mav_survived=0,
                 undiscounted_task_return=0.0),
    ])
    checks = _ranking_checks([left, right])
    for name in (
        "more_blue_losses_raise_return_when_matched",
        "more_red_attack_losses_lower_return_when_matched",
        "mav_death_lowers_return_when_matched",
    ):
        assert checks[name] == {"comparison_count": 1, "pass": True}


def test_different_seeds_do_not_form_monotonic_pairs():
    checks = _ranking_checks([
        _result("left", [_episode(seed=20, blue_attack_alive_final=2)]),
        _result("right", [_episode(seed=21, blue_attack_alive_final=1,
                                  undiscounted_task_return=5.0)]),
    ])
    assert checks["more_blue_losses_raise_return_when_matched"] == {
        "comparison_count": 0, "pass": None}


def test_task_return_is_event_plus_terminal():
    assert _undiscounted_task_return(1.25, -0.4) == pytest.approx(0.85)


def test_checkpoint_comparison_and_observational_ranking_remain_discounted():
    results = [
        _result("checkpoint_100k_attack", [], discounted_mean=0.4),
        _result("checkpoint_250k_collapsed", [], discounted_mean=0.1),
        _result("other", [], discounted_mean=0.8),
    ]
    checks = _ranking_checks(results)
    assert checks["checkpoint_100k_not_below_250k"] is True
    assert [row["label"] for row in _observational_ranking(results)] == [
        "other", "checkpoint_100k_attack", "checkpoint_250k_collapsed"]


def test_perturbations_are_strictly_unique_and_duplicates_are_rejected():
    scenarios = _paired_scenarios(2_000, 20)
    assert _validate_scenario_uniqueness(scenarios) == 20
    with pytest.raises(ValueError, match="perturbations must be unique"):
        _validate_scenario_uniqueness([scenarios[0], scenarios[0]])


def test_hits_kills_and_rule_label_remain_distinct():
    events = [
        {"event": "launch", "shooter_id": "red_1"},
        {"event": "hit", "shooter_id": "red_1"},
        {"event": "hit", "shooter_id": "red_2"},
    ]
    assert _red_combat_counts(events, red_kills=1) == {
        "red_launches": 1, "red_hits": 2, "red_kills": 1}
    assert RULE_POLICY_LABEL == "rule_red_policy"


def test_telescoping_check_is_structured():
    checks = _ranking_checks([_result("policy", [
        _episode(potential_telescoping_error=1e-12),
        _episode(seed=2, potential_telescoping_error=2e-6),
    ])])
    assert checks["potential_telescoping_within_tolerance"] == {
        "sample_count": 2, "pass": False}
