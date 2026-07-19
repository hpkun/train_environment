"""Pure checks and reporting helpers for the formal 3v2 stability audit."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_VERSION = "hetero_3v2_environment_stability_v1"
DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-8

CHECK_NAMES = (
    "DETERMINISTIC_REPLAY",
    "RESET_ISOLATION",
    "NUMERICAL_FINITE",
    "DYNAMICS_BOUNDARY",
    "ACTIVE_MASK_MONOTONICITY",
    "TERMINATION_CONSISTENCY",
    "DEATH_ACCOUNTING",
    "MISSILE_EVENT_ACCOUNTING",
    "RULE_POLICY_RUNTIME",
    "ZERO_ACTION_RUNTIME",
    "RANDOM_ACTION_RUNTIME",
)

CHECK_PREFIX = {
    "DETERMINISTIC_REPLAY": ("deterministic_",),
    "RESET_ISOLATION": ("reset_",),
    "NUMERICAL_FINITE": ("non_finite",),
    "DYNAMICS_BOUNDARY": ("boundary_",),
    "ACTIVE_MASK_MONOTONICITY": ("active_mask_",),
    "TERMINATION_CONSISTENCY": ("termination_",),
    "DEATH_ACCOUNTING": ("death_",),
    "MISSILE_EVENT_ACCOUNTING": ("missile_",),
}

DISCRETE_TRACE_FIELDS = (
    "step", "alive", "active_mask", "selected_targets", "fire_gate_flags",
    "events", "newly_dead", "death_reasons", "missile_count", "outcome",
    "end_reason", "terminated", "truncated", "team_done", "ammo",
)
CONTINUOUS_TRACE_FIELDS = (
    "position", "velocity", "rpy", "geodetic", "critic_state",
    "actor_observation", "reward", "control_target", "sim_time_sec",
    "fire_gate_geometry", "actions",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def stable_signature(value: Any) -> str:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )


def validate_unique_perturbations(rows: list[dict]) -> int:
    signatures = [stable_signature(row) for row in rows]
    unique_count = len(set(signatures))
    if unique_count != len(rows):
        raise ValueError(
            "audit perturbations must be unique: "
            f"requested={len(rows)}, unique={unique_count}")
    return unique_count


def failure(
    check_name: str,
    policy_type: str,
    case_id: str,
    seed: int,
    episode: int,
    step: int,
    message: str,
    agent_id: str = "",
    expected: Any = None,
    actual: Any = None,
) -> dict:
    return {
        "check_name": check_name,
        "policy_type": policy_type,
        "case_id": case_id,
        "seed": int(seed),
        "episode": int(episode),
        "step": int(step),
        "agent_id": agent_id,
        "expected": jsonable(expected),
        "actual": jsonable(actual),
        "message": message,
    }


def _path_values(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _path_values(value[key], path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _path_values(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def finite_failures(
    values: dict,
    *,
    policy_type: str,
    case_id: str,
    seed: int,
    episode: int,
    step: int,
) -> list[dict]:
    failures = []
    for path, value in _path_values(values):
        if value is None or isinstance(value, (str, bool)):
            continue
        try:
            finite = bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
        except (TypeError, ValueError):
            continue
        if not finite:
            failures.append(failure(
                "non_finite", policy_type, case_id, seed, episode, step,
                f"non-finite numeric value at {path}", actual=value,
            ))
    return failures


def compare_discrete(expected: Any, actual: Any, path: str = "") -> dict | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return {"field": path, "expected": expected, "actual": actual}
        for key in sorted(expected):
            mismatch = compare_discrete(
                expected[key], actual[key], f"{path}.{key}" if path else key)
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return {"field": path, "expected": expected, "actual": actual}
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = compare_discrete(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if expected != actual:
        return {"field": path, "expected": expected, "actual": actual}
    return None


def compare_continuous(
    expected: Any,
    actual: Any,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    path: str = "",
) -> tuple[dict | None, dict[str, float]]:
    maxima: dict[str, float] = {}
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return {"field": path, "expected": expected, "actual": actual}, maxima
        for key in sorted(expected):
            mismatch, child_max = compare_continuous(
                expected[key], actual[key], rtol=rtol, atol=atol,
                path=f"{path}.{key}" if path else key,
            )
            maxima.update(child_max)
            if mismatch:
                return mismatch, maxima
        return None, maxima
    if expected is None or actual is None:
        if expected is actual:
            return None, maxima
        return {"field": path, "expected": expected, "actual": actual}, maxima
    try:
        left = np.asarray(expected, dtype=np.float64)
        right = np.asarray(actual, dtype=np.float64)
    except (TypeError, ValueError):
        return compare_discrete(expected, actual, path), maxima
    if left.shape != right.shape:
        return {"field": path, "expected": expected, "actual": actual}, maxima
    difference = float(np.max(np.abs(left - right))) if left.size else 0.0
    maxima[path] = difference
    if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
        return {"field": path, "expected": expected, "actual": actual,
                "max_absolute_difference": difference}, maxima
    return None, maxima


def compare_traces(
    expected: list[dict],
    actual: list[dict],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict:
    maxima: dict[str, float] = {}
    if len(expected) != len(actual):
        return {
            "match": False,
            "first_mismatch_step": min(len(expected), len(actual)),
            "first_mismatch_field": "trace_length",
            "max_absolute_difference_by_field": maxima,
        }
    for index, (left, right) in enumerate(zip(expected, actual)):
        for field in DISCRETE_TRACE_FIELDS:
            mismatch = compare_discrete(left.get(field), right.get(field), field)
            if mismatch:
                return {
                    "match": False, "first_mismatch_step": index,
                    "first_mismatch_field": mismatch["field"],
                    "max_absolute_difference_by_field": maxima,
                }
        for field in CONTINUOUS_TRACE_FIELDS:
            mismatch, field_max = compare_continuous(
                left.get(field), right.get(field), rtol=rtol, atol=atol,
                path=field,
            )
            for key, value in field_max.items():
                maxima[key] = max(maxima.get(key, 0.0), value)
            if mismatch:
                return {
                    "match": False, "first_mismatch_step": index,
                    "first_mismatch_field": mismatch["field"],
                    "max_absolute_difference_by_field": maxima,
                }
    return {
        "match": True,
        "first_mismatch_step": None,
        "first_mismatch_field": None,
        "max_absolute_difference_by_field": maxima,
    }


def validate_active_mask(
    previous: list[int], current: list[int], alive: dict[str, bool], red_ids: list[str],
    context: dict,
) -> list[dict]:
    rows = []
    for index, agent_id in enumerate(red_ids):
        if current[index] > previous[index]:
            rows.append(failure(
                "active_mask_resurrection", **context, agent_id=agent_id,
                expected="monotonic 1->0", actual=[previous[index], current[index]],
                message="active mask returned from inactive to active"))
        expected = int(bool(alive[agent_id]))
        if current[index] != expected:
            rows.append(failure(
                "active_mask_alive_mismatch", **context, agent_id=agent_id,
                expected=expected, actual=current[index],
                message="active mask disagrees with aircraft is_alive"))
    return rows


def validate_termination(
    terminated: bool, truncated: bool, team_done: bool,
    outcome: str, end_reason: str, step_count: int, max_steps: int,
    red_attack_alive: int, blue_attack_alive: int, context: dict,
) -> list[dict]:
    rows = []
    if terminated and truncated:
        rows.append(failure(
            "termination_both_true", **context, expected=False, actual=True,
            message="terminated and truncated cannot both be true"))
    if team_done != (terminated or truncated):
        rows.append(failure(
            "termination_team_done_mismatch", **context,
            expected=terminated or truncated, actual=team_done,
            message="team_done must equal terminated or truncated"))
    if step_count > max_steps:
        rows.append(failure(
            "termination_step_limit", **context, expected=f"<= {max_steps}",
            actual=step_count, message="episode exceeded max_steps"))
    capability_lost = red_attack_alive == 0 or blue_attack_alive == 0
    if capability_lost and not terminated:
        rows.append(failure(
            "termination_capability_not_terminated", **context,
            expected=True, actual=terminated,
            message="combat capability loss must terminate"))
    if truncated and (step_count != max_steps or outcome != "draw" or end_reason != "timeout"):
        rows.append(failure(
            "termination_timeout_contract", **context,
            expected={"step": max_steps, "outcome": "draw", "end_reason": "timeout"},
            actual={"step": step_count, "outcome": outcome, "end_reason": end_reason},
            message="time-limit truncation contract mismatch"))
    expected_outcome = None
    if red_attack_alive == 0 and blue_attack_alive == 0:
        expected_outcome = "mutual_elimination"
    elif red_attack_alive == 0:
        expected_outcome = "blue_win"
    elif blue_attack_alive == 0:
        expected_outcome = "red_win"
    elif truncated:
        expected_outcome = "draw"
    if team_done and expected_outcome and outcome != expected_outcome:
        rows.append(failure(
            "termination_outcome_mismatch", **context,
            expected=expected_outcome, actual=outcome,
            message="outcome disagrees with final combat capability"))
    return rows


def validate_reset_state(env, obs: dict, info: dict, previous_controller_ids=None) -> tuple[list[dict], list[str]]:
    failures, unavailable = [], []
    context = {
        "policy_type": "reset", "case_id": "reset", "seed": 0,
        "episode": 0, "step": 0,
    }
    required = (
        "step_count", "sim_time_sec", "missiles", "event_log", "newly_dead",
        "death_reasons", "selected_targets", "last_fire_gates",
        "v5_terminal_applied", "v5_last_event_step", "controllers",
        "previous_missile_risk", "previous_missile_speed",
    )
    for name in required:
        if not hasattr(env, name):
            unavailable.append(name)
            failures.append(failure(
                "reset_missing_core_field", **context, expected="field available",
                actual="missing", message=f"required reset field is unavailable: {name}"))
    if failures:
        return failures, unavailable
    checks = {
        "step_count": env.step_count == 0,
        "sim_time_sec": env.sim_time_sec == 0.0,
        "missiles": len(env.missiles) == 0,
        "event_log": len(env.event_log) == 0,
        "newly_dead": len(env.newly_dead) == 0,
        "death_reasons": len(env.death_reasons) == 0,
        "selected_targets": all(value is None for value in env.selected_targets.values()),
        "last_fire_gates": len(env.last_fire_gates) == 0,
        "v5_terminal_applied": env.v5_terminal_applied is False,
        "v5_last_event_step": env.v5_last_event_step == -1,
        "aircraft_alive": all(item.is_alive for item in env.aircraft.values()),
        "previous_missile_risk": all(value == 0.0 for value in env.previous_missile_risk.values()),
        "previous_missile_speed": len(env.previous_missile_speed) == 0,
        "active_mask": np.array_equal(
            np.asarray(info["active_mask"], dtype=np.int8),
            np.asarray([env.aircraft[aid].is_alive for aid in env.red_ids], dtype=np.int8)),
    }
    try:
        from uav_env.JSBSim.formal_v1.scenario import MISSILE_COUNTS
        checks["ammunition"] = all(
            env.aircraft[aid].num_left_missiles == MISSILE_COUNTS[aid]
            for aid in (*env.red_ids, *env.blue_ids))
    except (ImportError, AttributeError, KeyError):
        unavailable.append("initial_ammunition_contract")
    if previous_controller_ids is not None:
        checks["controllers_rebuilt"] = all(
            id(env.controllers[aid]) != previous_controller_ids.get(aid)
            for aid in env.controllers)
    for name, passed in checks.items():
        if not passed:
            failures.append(failure(
                f"reset_{name}_leak", **context, expected=True, actual=False,
                message=f"reset isolation check failed: {name}"))
    return failures, unavailable


def validate_missile_step(
    *, previous_alive: dict[str, bool], alive: dict[str, bool],
    previous_ammo: dict[str, int], ammo: dict[str, int], events: list[dict],
    death_reasons: dict[str, str], roles: dict[str, str], agent_ids: set[str],
    context: dict,
) -> list[dict]:
    rows = []
    launches = [event for event in events if event.get("event") == "launch"]
    hits = [event for event in events if event.get("event") == "hit"]
    for event in launches:
        shooter = str(event.get("shooter_id", ""))
        if shooter == "red_0":
            rows.append(failure(
                "missile_mav_launch", **context, agent_id=shooter,
                expected="MAV never launches", actual=event,
                message="weaponless MAV produced a launch event"))
        if shooter not in agent_ids or not previous_alive.get(shooter, False):
            rows.append(failure(
                "missile_dead_shooter_launch", **context, agent_id=shooter,
                expected="alive known shooter", actual=event,
                message="launch shooter was dead or unknown at step start"))
        if roles.get(shooter) != "attack_uav":
            rows.append(failure(
                "missile_non_attack_uav_launch", **context, agent_id=shooter,
                expected="attack_uav", actual=roles.get(shooter),
                message="non-attack role produced launch"))
    for shooter in agent_ids:
        launch_count = sum(event.get("shooter_id") == shooter for event in launches)
        ammo_delta = previous_ammo.get(shooter, 0) - ammo.get(shooter, 0)
        if launch_count != ammo_delta:
            rows.append(failure(
                "missile_ammo_launch_mismatch", **context, agent_id=shooter,
                expected=launch_count, actual=ammo_delta,
                message="launch count does not equal ammunition decrease"))
    hit_targets = []
    for event in hits:
        shooter = str(event.get("shooter_id", ""))
        target = str(event.get("target_id", ""))
        if shooter not in agent_ids or target not in agent_ids:
            rows.append(failure(
                "missile_hit_unknown_agent", **context,
                expected=sorted(agent_ids), actual=event,
                message="hit references an unknown shooter or target"))
            continue
        hit_targets.append(target)
        if not previous_alive.get(target, False) or alive.get(target, True):
            rows.append(failure(
                "missile_hit_without_death_transition", **context, agent_id=target,
                expected="alive->dead", actual=[previous_alive.get(target), alive.get(target)],
                message="hit target did not transition from alive to dead"))
    if len(hit_targets) != len(set(hit_targets)):
        rows.append(failure(
            "missile_duplicate_hit_target", **context,
            expected="one hit per target", actual=hit_targets,
            message="same target was counted by multiple hit events"))
    for agent_id, reason in death_reasons.items():
        if reason == "missile_hit" and previous_alive.get(agent_id, False) and not alive.get(agent_id, True):
            if agent_id not in hit_targets:
                rows.append(failure(
                    "missile_death_without_hit", **context, agent_id=agent_id,
                    expected="matching hit event", actual=events,
                    message="missile_hit death has no hit event in transition"))
    return rows


def combat_event_counts(events: list[dict], death_reasons: dict[str, str]) -> dict:
    return {
        "red_launches": sum(
            event.get("event") == "launch"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events),
        "blue_launches": sum(
            event.get("event") == "launch"
            and str(event.get("shooter_id", "")).startswith("blue")
            for event in events),
        "red_hits": sum(
            event.get("event") == "hit"
            and str(event.get("shooter_id", "")).startswith("red")
            for event in events),
        "blue_hits": sum(
            event.get("event") == "hit"
            and str(event.get("shooter_id", "")).startswith("blue")
            for event in events),
        "red_kills": sum(
            reason == "missile_hit" and agent_id.startswith("blue")
            for agent_id, reason in death_reasons.items()),
        "blue_kills": sum(
            reason == "missile_hit" and agent_id.startswith("red")
            for agent_id, reason in death_reasons.items()),
    }


def read_jsonl_strict(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"corrupt JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_completed_cases(path: Path) -> dict[str, dict]:
    completed = {}
    for row in read_jsonl_strict(path):
        case_id = row.get("case_id")
        if not case_id:
            raise ValueError("raw JSONL record is missing case_id")
        if case_id in completed:
            raise ValueError(f"duplicate case_id in raw JSONL: {case_id}")
        if row.get("case_completed") is True:
            completed[case_id] = row
    return completed


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        handle.flush()


def _status(sample_count: int, failure_count: int, summary: str) -> dict:
    if sample_count == 0:
        state = "N/A"
    else:
        state = "FAIL" if failure_count else "PASS"
    return {"status": state, "sample_count": sample_count,
            "failure_count": failure_count, "summary": summary}


def build_report(input_dir: Path) -> dict:
    input_dir = Path(input_dir)
    raw = read_jsonl_strict(input_dir / "environment_stability_raw.jsonl")
    failures = read_jsonl_strict(input_dir / "environment_stability_failures.jsonl")
    meta_path = input_dir / "audit_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    checks = {}
    for name in CHECK_NAMES:
        if name.endswith("_RUNTIME"):
            case_type = {
                "RULE_POLICY_RUNTIME": "rule",
                "ZERO_ACTION_RUNTIME": "zero",
                "RANDOM_ACTION_RUNTIME": "random",
            }[name]
            samples = [row for row in raw if row.get("case_type") == case_type]
            count = sum(not row.get("runtime_ok", False) for row in samples)
        elif name == "DETERMINISTIC_REPLAY":
            samples = [row for row in raw
                       if row.get("case_type") == "deterministic"
                       and row.get("deterministic_run") == "b"]
            count = sum(not row.get("deterministic_match", False) for row in samples)
        elif name == "RESET_ISOLATION":
            samples = [row for row in raw if row.get("case_type") == "reset"]
            count = sum(row.get("reset_failure_count", 0) for row in samples)
        else:
            samples = raw
            prefixes = CHECK_PREFIX[name]
            count = sum(any(
                str(item.get("check_name", "")).startswith(prefix)
                for prefix in prefixes) for item in failures)
        checks[name] = _status(
            len(samples), int(count),
            f"{len(samples)} completed case(s), {int(count)} failure(s)")
    states = {row["status"] for row in checks.values()}
    overall = "FAIL" if "FAIL" in states else ("N/A" if "N/A" in states else "PASS")
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "configuration": meta,
        "checks": checks,
        "completed_case_count": len(raw),
        "failure_record_count": len(failures),
        "policy_runtime": {},
        "state_extrema": {},
        "failure_index": failures,
    }
    deterministic = [row for row in raw
                     if row.get("case_type") == "deterministic"
                     and row.get("deterministic_run") == "b"]
    mismatch = [row for row in deterministic
                if not row.get("deterministic_match", False)]
    maxima = {}
    for row in deterministic:
        for key, value in row.get("max_absolute_difference_by_field", {}).items():
            maxima[key] = max(maxima.get(key, 0.0), float(value))
    payload["deterministic_replay"] = {
        "deterministic_case_count": len(deterministic),
        "deterministic_match_count": len(deterministic) - len(mismatch),
        "deterministic_mismatch_count": len(mismatch),
        "first_mismatch_step": mismatch[0].get("first_mismatch_step") if mismatch else None,
        "first_mismatch_field": mismatch[0].get("first_mismatch_field") if mismatch else None,
        "max_absolute_difference_by_field": maxima,
    }
    for policy in ("rule_policy", "zero_action", "bounded_random"):
        case_type = {"rule_policy": "rule", "zero_action": "zero",
                     "bounded_random": "random"}[policy]
        rows = [row for row in raw if row.get("case_type") == case_type]
        payload["policy_runtime"][policy] = {
            "episodes": len(rows),
            "red_win": sum(row.get("outcome") == "red_win" for row in rows),
            "blue_win": sum(row.get("outcome") == "blue_win" for row in rows),
            "draw": sum(row.get("outcome") == "draw" for row in rows),
            "red_launches": sum(row.get("red_launches", 0) for row in rows),
            "red_hits": sum(row.get("red_hits", 0) for row in rows),
            "red_kills": sum(row.get("red_kills", 0) for row in rows),
        }
    extrema_keys = (
        "max_speed_mps", "max_altitude_m", "max_horizontal_distance_m",
        "max_abs_position", "max_abs_velocity",
    )
    for key in extrema_keys:
        values = [row[key] for row in raw if key in row and row[key] is not None]
        payload["state_extrema"][key] = max(values) if values else None
    minimums = [row["min_altitude_m"] for row in raw
                if row.get("min_altitude_m") is not None]
    payload["state_extrema"]["min_altitude_m"] = min(minimums) if minimums else None
    (input_dir / "environment_stability_audit.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_episode_csv(input_dir / "environment_stability_episodes.csv", raw)
    _write_markdown(input_dir / "environment_stability_report.md", payload)
    return payload


def _write_episode_csv(path: Path, rows: list[dict]) -> None:
    excluded = {"trace", "perturbation", "failures", "max_absolute_difference_by_field"}
    flat_rows = [{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                  else value for key, value in row.items() if key not in excluded}
                 for row in rows]
    if not flat_rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in flat_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_markdown(path: Path, payload: dict) -> None:
    config = payload["configuration"]
    lines = [
        "# Heterogeneous 3V2 Environment Stability Audit", "",
        "## Audit configuration", "",
        f"- Config: `{config.get('config_path', 'N/A')}`",
        f"- Contract: `{config.get('formal_contract', 'N/A')}` / "
        f"`{config.get('reward_contract', 'N/A')}`",
        f"- Seed: {config.get('seed', 'N/A')}", "",
        "## Overall result", "", f"**{payload['overall_status']}**", "",
        "## Stability checks", "",
        "| Check | Status | Samples | Failures | Summary |",
        "|---|---|---:|---:|---|",
    ]
    for name in CHECK_NAMES:
        row = payload["checks"][name]
        lines.append(
            f"| {name} | {row['status']} | {row['sample_count']} | "
            f"{row['failure_count']} | {row['summary']} |")
    lines.extend(["", "## Policy runtime statistics", ""])
    for policy, row in payload["policy_runtime"].items():
        lines.append(f"- `{policy}`: {json.dumps(row, sort_keys=True)}")
    lines.extend(["", "## Failure index", ""])
    if payload["failure_index"]:
        for row in payload["failure_index"][:100]:
            lines.append(
                f"- `{row.get('case_id')}` step {row.get('step')}: "
                f"{row.get('check_name')} - {row.get('message')}")
    else:
        lines.append("No failures recorded.")
    lines.extend(["", "## State extrema", "", "```json",
                  json.dumps(payload["state_extrema"], indent=2), "```", "",
                  "## Interpretation boundaries", "",
                  "This audit checks implementation stability, consistency and reproducibility. "
                  "Win rate and combat effectiveness are descriptive only and are not PASS criteria. "
                  "The audit does not assess whether Reward V5 is learnable or optimal.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
