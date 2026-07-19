from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hetero_3v2_environment_stability_common import (
    append_jsonl, build_report, combat_event_counts, compare_continuous,
    compare_discrete, compare_traces, finite_failures, load_completed_cases,
    read_jsonl_strict, stable_signature, validate_active_mask,
    validate_action_replay_completion, validate_boundary_state,
    validate_case_completion, validate_missile_step, validate_reset_state, validate_termination,
    validate_unique_perturbations,
)
from scripts.hetero_3v2_v2_audit_common import perturbation
from scripts.audit_hetero_3v2_environment_stability import (_actions,_prepare_output,
  _warmup_seed,parse_args)


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


def _report_dir(tmp_path, rows, failures=()):
    (tmp_path / "audit_meta.json").write_text(json.dumps({"formal_contract":"test"}),encoding="utf-8")
    for row in rows:append_jsonl(tmp_path / "environment_stability_raw.jsonl",row)
    (tmp_path / "environment_stability_failures.jsonl").touch()
    for row in failures:append_jsonl(tmp_path / "environment_stability_failures.jsonl",row)
    return tmp_path


def _runtime_row(case_id,case_type,policy,runtime_ok=True,**extra):
    return {"case_id":case_id,"case_completed":True,"runtime_ok":runtime_ok,
      "case_type":case_type,"policy_type":policy,"steps":0,"outcome":"ongoing",
      "end_reason":"","terminated":False,"truncated":False,"team_done":False,
      "reset_failure_count":0,**extra}


def test_deterministic_runtime_failures_are_fail_closed(tmp_path):
    rows=[_runtime_row("deterministic_000_run_a","deterministic","rule_policy",False,deterministic_run="a",trace=[]),
      _runtime_row("deterministic_000_run_b","deterministic","rule_policy",False,deterministic_run="b",trace=[],
        deterministic_match=True,action_replay_contract_ok=True)]
    payload=build_report(_report_dir(tmp_path,rows))
    assert payload["checks"]["DETERMINISTIC_REPLAY"]["status"]=="FAIL"
    assert payload["checks"]["CASE_RUNTIME_INTEGRITY"]["status"]=="FAIL"
    assert payload["overall_status"]=="FAIL"


def test_deterministic_action_contract_is_required_for_match(tmp_path):
    rows=[_runtime_row("deterministic_000_run_a","deterministic","rule_policy",True,deterministic_run="a",trace=[]),
      _runtime_row("deterministic_000_run_b","deterministic","rule_policy",True,deterministic_run="b",trace=[],
        deterministic_match=True,action_replay_contract_ok=False)]
    assert build_report(_report_dir(tmp_path,rows))["checks"]["DETERMINISTIC_REPLAY"]["status"]=="FAIL"


@pytest.mark.parametrize("case_type,policy,check",[("reset","reset","RESET_ISOLATION"),("rule","rule_policy","RULE_POLICY_RUNTIME"),
  ("zero","zero_action","ZERO_ACTION_RUNTIME"),("random","bounded_random","RANDOM_ACTION_RUNTIME")])
def test_case_type_runtime_failure_reaches_fixed_check(tmp_path,case_type,policy,check):
    payload=build_report(_report_dir(tmp_path,[_runtime_row(f"{case_type}_000",case_type,policy,False)]))
    assert payload["checks"][check]["status"]=="FAIL" and payload["overall_status"]=="FAIL"


def test_unmatched_runtime_exception_is_not_ignored_and_failure_count_is_records(tmp_path):
    row=_runtime_row("rule_000","rule","rule_policy",False)
    failures=[{"check_name":"runtime_exception","case_id":"rule_000","step":1,"message":"boom"},
      {"check_name":"unexpected_audit_failure","case_id":"rule_000","step":1,"message":"also boom"}]
    path=_report_dir(tmp_path,[row],failures);before=(path/"environment_stability_failures.jsonl").read_text()
    payload=build_report(path)
    assert payload["checks"]["CASE_RUNTIME_INTEGRITY"]["status"]=="FAIL"
    assert payload["failure_record_count"]==2
    assert (path/"environment_stability_failures.jsonl").read_text()==before


def test_max_steps_requires_timeout_but_smoke_limit_does_not():
    rows=validate_termination(False,False,False,"ongoing","",10,10,1,1,_context(step=10))
    assert any(row["check_name"]=="termination_missing_timeout" for row in rows)
    failures,artificial,complete=validate_case_completion(
      steps=2,terminated=False,truncated=False,env_max_steps=1000,max_case_steps=2,
      outcome="ongoing",end_reason="",context=_context())
    assert failures==[] and artificial is True and complete is False
    failures,artificial,complete=validate_case_completion(
      steps=1000,terminated=False,truncated=False,env_max_steps=1000,max_case_steps=0,
      outcome="ongoing",end_reason="",context=_context())
    assert failures[0]["check_name"]=="termination_missing_timeout" and artificial is False and complete is False


def test_run_b_actions_are_strictly_replayed_without_calling_rule():
    class Rule:
        def actions(self,*args):raise AssertionError("rule must not be called")
    env=SimpleNamespace(red_ids=["red_0","red_1"])
    source=[{"red_0":[0.1,0.2,0.3],"red_1":[-0.1,-0.2,-0.3]}]
    actions=_actions(env,"rule_policy",None,Rule(),source,0)
    assert all(value.dtype==np.float32 and value.shape==(3,) for value in actions.values())
    assert actions["red_0"].tolist()==pytest.approx(source[0]["red_0"])
    with pytest.raises(IndexError,match="exhausted"):_actions(env,"rule_policy",None,Rule(),source,1)


def test_run_b_early_end_with_remaining_actions_fails():
    rows=validate_action_replay_completion(
      expected_length=3,consumed_steps=2,environment_episode_complete=True,context=_context())
    assert rows[0]["check_name"]=="deterministic_action_replay_early_end"


def _boundary_state(position=(0,0,1000),velocity=(1,0,0),rpy=(0,0,0),geodetic=(0,0,1000)):
    return {"position":position,"velocity":velocity,"rpy":rpy,"geodetic":geodetic}


def test_boundary_uses_combined_state_and_frozen_limits():
    nan_velocity=_boundary_state(velocity=(np.nan,0,0))
    assert validate_boundary_state(alive={"a":False},actual_newly_dead=["a"],death_reasons={"a":"numeric_anomaly"},states={"a":nan_velocity},context=_context())==[]
    rows=validate_boundary_state(alive={"a":True},actual_newly_dead=[],death_reasons={},states={"a":nan_velocity},context=_context())
    assert rows[0]["check_name"]=="boundary_nonfinite_aircraft_still_alive"
    for state,name in ((_boundary_state(position=(0,0,99)),"boundary_crash_aircraft_still_alive"),
      (_boundary_state(position=(50001,0,1000)),"boundary_out_of_zone_aircraft_still_alive"),
      (_boundary_state(position=(0,0,10001)),"boundary_out_of_zone_aircraft_still_alive")):
        rows=validate_boundary_state(alive={"a":True},actual_newly_dead=[],death_reasons={},states={"a":state},context=_context())
        assert rows[0]["check_name"]==name
    assert validate_boundary_state(alive={"a":False},actual_newly_dead=["a"],death_reasons={"a":"missile_hit"},states={"a":_boundary_state()},context=_context())==[]


def test_report_every_cases_validation_and_resume_contract(tmp_path,monkeypatch):
    monkeypatch.setattr("sys.argv",["audit","--output-dir",str(tmp_path),"--report-every-cases","0"])
    with pytest.raises(SystemExit):parse_args()
    meta={"report_every_cases":5};(tmp_path/"audit_meta.json").write_text(json.dumps(meta),encoding="utf-8")
    (tmp_path/"environment_stability_raw.jsonl").touch()
    args=SimpleNamespace(output_dir=str(tmp_path),resume=True)
    with pytest.raises(ValueError,match="report_every_cases"):_prepare_output(args,{"report_every_cases":10})
