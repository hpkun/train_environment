from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hetero_3v2_environment_stability_common import (
    append_jsonl, build_report, combat_event_counts, compare_continuous,
    compare_discrete, compare_traces, finite_failures, load_completed_cases,
    read_jsonl_strict, stable_signature, validate_active_mask,
    validate_missile_step, validate_reset_state, validate_termination,
    validate_unique_perturbations,
)
from scripts.hetero_3v2_v2_audit_common import perturbation
from scripts.audit_hetero_3v2_environment_stability import _warmup_seed


def _context(**updates):
    row = {"policy_type": "test", "case_id": "case", "seed": 1,
           "episode": 0, "step": 1}
    row.update(updates)
    return row


def test_warmup_seed_is_never_negative():
    assert _warmup_seed(0) == 0
    assert _warmup_seed(1) == 0
    assert _warmup_seed(7) == 6


def _trace(value=1.0):
    return {
        "step": 1, "alive": {"red_0": True}, "active_mask": [1],
        "selected_targets": {"red_0": None}, "fire_gate_flags": {},
        "events": [], "newly_dead": [], "death_reasons": {},
        "missile_count": 0, "outcome": "ongoing", "end_reason": "",
        "terminated": False, "truncated": False, "team_done": False,
        "ammo": {"red_0": 0}, "position": {"red_0": [value, 0, 0]},
        "velocity": {"red_0": [0, 0, 0]}, "rpy": {"red_0": [0, 0, 0]},
        "geodetic": {"red_0": [0, 0, 0]}, "critic_state": [0.0],
        "actor_observation": {"red_0": [0.0]}, "reward": {"red_0": 0.0},
        "control_target": {"red_0": None}, "sim_time_sec": 0.2,
        "fire_gate_geometry": {}, "actions": {"red_0": [0, 0, 0]},
    }


def test_stable_perturbation_signature_and_duplicate_rejection():
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}
    assert stable_signature(left) == stable_signature(right)
    rows = [perturbation(100 + index) for index in range(20)]
    assert validate_unique_perturbations(rows) == 20
    with pytest.raises(ValueError, match="must be unique"):
        validate_unique_perturbations([rows[0], rows[0]])


def test_continuous_allclose_and_discrete_strict_comparison():
    mismatch, maxima = compare_continuous([1.0], [1.0 + 1e-9])
    assert mismatch is None
    assert maxima[""] == pytest.approx(1e-9)
    mismatch, _ = compare_continuous([1.0], [1.1])
    assert mismatch["field"] == ""
    assert compare_discrete({"value": 1}, {"value": 1}) is None
    assert compare_discrete({"value": 1}, {"value": 2})["field"] == "value"


def test_deterministic_trace_mismatch_locates_step_and_field():
    expected = [_trace(), _trace()]
    actual = [_trace(), _trace(2.0)]
    result = compare_traces(expected, actual)
    assert result["match"] is False
    assert result["first_mismatch_step"] == 1
    assert result["first_mismatch_field"] == "position.red_0"


def test_active_mask_resurrection_fails():
    rows = validate_active_mask(
        [0], [1], {"red_0": True}, ["red_0"], _context())
    assert any(row["check_name"] == "active_mask_resurrection" for row in rows)


def test_termination_contract_rejects_both_flags_and_team_done_mismatch():
    rows = validate_termination(
        True, True, False, "draw", "timeout", 10, 10, 1, 1, _context())
    names = {row["check_name"] for row in rows}
    assert "termination_both_true" in names
    assert "termination_team_done_mismatch" in names


class _Plane:
    def __init__(self, alive=True, ammo=0):
        self.is_alive = alive
        self.num_left_missiles = ammo


def _reset_env():
    ids = ["red_0", "red_1", "red_2", "blue_0", "blue_1"]
    ammo = {"red_0": 0, "red_1": 2, "red_2": 2, "blue_0": 2, "blue_1": 2}
    return SimpleNamespace(
        step_count=0, sim_time_sec=0.0, missiles=[], event_log=[], newly_dead=set(),
        death_reasons={}, selected_targets={aid: None for aid in ids},
        last_fire_gates={}, v5_terminal_applied=False, v5_last_event_step=-1,
        controllers={aid: object() for aid in ids},
        previous_missile_risk={aid: 0.0 for aid in ids[:3]},
        previous_missile_speed={}, aircraft={aid: _Plane(True, ammo[aid]) for aid in ids},
        red_ids=ids[:3], blue_ids=ids[3:],
    )


def test_reset_leak_is_detected():
    env = _reset_env()
    env.event_log.append({"event": "launch"})
    failures, unavailable = validate_reset_state(
        env, {}, {"active_mask": [1, 1, 1]})
    assert unavailable == []
    assert any(row["check_name"] == "reset_event_log_leak" for row in failures)


def test_non_finite_value_is_detected():
    rows = finite_failures(
        {"obs": [0.0, np.nan]}, **_context())
    assert rows[0]["check_name"] == "non_finite"


def test_mav_and_dead_shooter_launches_fail():
    rows = validate_missile_step(
        previous_alive={"red_0": False}, alive={"red_0": False},
        previous_ammo={"red_0": 0}, ammo={"red_0": 0},
        events=[{"event": "launch", "shooter_id": "red_0", "target_id": "blue_0"}],
        death_reasons={}, roles={"red_0": "mav"}, agent_ids={"red_0", "blue_0"},
        context=_context())
    names = {row["check_name"] for row in rows}
    assert "missile_mav_launch" in names
    assert "missile_dead_shooter_launch" in names


def test_missile_hit_death_without_event_fails():
    rows = validate_missile_step(
        previous_alive={"blue_0": True}, alive={"blue_0": False},
        previous_ammo={"blue_0": 2}, ammo={"blue_0": 2}, events=[],
        death_reasons={"blue_0": "missile_hit"},
        roles={"blue_0": "attack_uav"}, agent_ids={"blue_0"}, context=_context())
    assert any(row["check_name"] == "missile_death_without_hit" for row in rows)


def test_hits_and_kills_are_independent_counts():
    events = [
        {"event": "hit", "shooter_id": "red_1", "target_id": "blue_0"},
        {"event": "hit", "shooter_id": "red_2", "target_id": "blue_1"},
    ]
    counts = combat_event_counts(events, {"blue_0": "missile_hit"})
    assert counts["red_hits"] == 2
    assert counts["red_kills"] == 1


def test_jsonl_resume_skip_duplicate_and_corrupt_tail(tmp_path):
    raw = tmp_path / "raw.jsonl"
    append_jsonl(raw, {"case_id": "rule_000", "case_completed": True})
    assert set(load_completed_cases(raw)) == {"rule_000"}
    append_jsonl(raw, {"case_id": "rule_000", "case_completed": True})
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_completed_cases(raw)
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"case_id":"ok"}\n{"case_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt JSONL"):
        read_jsonl_strict(corrupt)


def test_report_na_fail_semantics_and_low_win_rate_is_not_failure(tmp_path):
    (tmp_path / "audit_meta.json").write_text(
        json.dumps({"formal_contract": "test"}), encoding="utf-8")
    append_jsonl(tmp_path / "environment_stability_raw.jsonl", {
        "case_id": "rule_000", "case_completed": True, "runtime_ok": True,
        "case_type": "rule", "policy_type": "rule_policy", "outcome": "blue_win",
        "red_launches": 0, "red_hits": 0, "red_kills": 0,
    })
    (tmp_path / "environment_stability_failures.jsonl").touch()
    payload = build_report(tmp_path)
    assert payload["checks"]["RULE_POLICY_RUNTIME"]["status"] == "PASS"
    assert payload["checks"]["ZERO_ACTION_RUNTIME"]["status"] == "N/A"
    assert payload["overall_status"] == "N/A"
    append_jsonl(tmp_path / "environment_stability_failures.jsonl", {
        "check_name": "non_finite", "case_id": "rule_000", "step": 1,
        "message": "nan"})
    payload = build_report(tmp_path)
    assert payload["checks"]["NUMERICAL_FINITE"]["status"] == "FAIL"
    assert payload["overall_status"] == "FAIL"
