"""Run a resumable implementation-stability audit for formal heterogeneous 3v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hetero_3v2_environment_stability_common import (
    AUDIT_VERSION, DEFAULT_ATOL, DEFAULT_RTOL, append_jsonl, build_report,
    combat_event_counts, compare_traces, failure, finite_failures, jsonable,
    load_completed_cases,
    stable_signature, validate_active_mask, validate_missile_step,
    validate_reset_state, validate_termination, validate_unique_perturbations,
)
from scripts.hetero_3v2_v2_audit_common import perturbation
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from uav_env.JSBSim.formal_v1.scenario import MISSILE_COUNTS
from uav_env.make_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2_reward_v5.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--deterministic-cases", type=int, default=10)
    parser.add_argument("--reset-episodes", type=int, default=30)
    parser.add_argument("--rule-episodes", type=int, default=30)
    parser.add_argument("--zero-episodes", type=int, default=10)
    parser.add_argument("--random-episodes", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-case-steps", type=int, default=0,
        help="Audit-only step cap; 0 uses the environment max_steps.")
    args = parser.parse_args()
    for name in (
        "deterministic_cases", "reset_episodes", "rule_episodes",
        "zero_episodes", "random_episodes", "max_case_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def _warmup_seed(seed: int) -> int:
    return max(0, int(seed) - 1)


def _case_plan(args) -> tuple[list[dict], dict[str, int]]:
    groups = {
        "deterministic": [perturbation(args.seed + index)
                          for index in range(args.deterministic_cases)],
        "reset": [perturbation(args.seed + 10_000 + index)
                  for index in range(args.reset_episodes)],
        "rule": [perturbation(args.seed + 20_000 + index)
                 for index in range(args.rule_episodes)],
        "zero": [perturbation(args.seed + 30_000 + index)
                 for index in range(args.zero_episodes)],
        "random": [perturbation(args.seed + 40_000 + index)
                   for index in range(args.random_episodes)],
    }
    unique_counts = {
        name: validate_unique_perturbations(rows) for name, rows in groups.items()
    }
    plan = []
    for index, row in enumerate(groups["deterministic"]):
        for run in ("a", "b"):
            plan.append({
                "case_id": f"deterministic_{index:03d}_run_{run}",
                "case_type": "deterministic", "policy_type": "rule_policy",
                "episode": index, "seed": args.seed + index,
                "perturbation": row, "deterministic_run": run,
            })
    for group, policy, offset in (
        ("reset", "reset", 10_000),
        ("rule", "rule_policy", 20_000),
        ("zero", "zero_action", 30_000),
        ("random", "bounded_random", 40_000),
    ):
        for index, row in enumerate(groups[group]):
            plan.append({
                "case_id": f"{group}_{index:03d}", "case_type": group,
                "policy_type": policy, "episode": index,
                "seed": args.seed + offset + index, "perturbation": row,
            })
    return plan, unique_counts


def _stable_config_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _expected_meta(args, env, config: Path, unique_counts: dict) -> dict:
    return {
        "audit_script_version": AUDIT_VERSION,
        "formal_contract": env.formal_contract,
        "reward_contract": env.reward_contract,
        "observation_contract": env.observation_contract,
        "action_dim": env.action_dim,
        "actor_obs_dim": env.actor_obs_dim,
        "critic_state_dim": env.critic_state_dim,
        "max_steps": env.max_steps,
        "sim_freq": env.sim_freq,
        "agent_interaction_steps": env.agent_interaction_steps,
        "seed": args.seed,
        "deterministic_cases": args.deterministic_cases,
        "reset_episodes": args.reset_episodes,
        "rule_episodes": args.rule_episodes,
        "zero_episodes": args.zero_episodes,
        "random_episodes": args.random_episodes,
        "max_case_steps": args.max_case_steps,
        "rtol": DEFAULT_RTOL,
        "atol": DEFAULT_ATOL,
        "config_path": _stable_config_path(config),
        "unique_perturbation_count_by_group": unique_counts,
    }


def _prepare_output(args, expected_meta: dict) -> tuple[Path, dict[str, dict]]:
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    entries = list(output.iterdir())
    meta_path = output / "audit_meta.json"
    raw_path = output / "environment_stability_raw.jsonl"
    if entries and not args.resume:
        raise ValueError(
            f"output directory is non-empty: {output}; use --resume to continue")
    if args.resume:
        if not meta_path.is_file():
            raise ValueError("--resume requires audit_meta.json")
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in expected_meta.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"resume metadata mismatch for {key}: "
                    f"{existing.get(key)!r} != {value!r}")
    else:
        meta = {**expected_meta, "created_at": datetime.now(timezone.utc).isoformat()}
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        raw_path.touch()
        (output / "environment_stability_failures.jsonl").touch()
    return output, load_completed_cases(raw_path)


def _canonical_events(events: list[dict]) -> list[dict]:
    keys = ("event", "shooter_id", "target_id", "missile_id", "reason")
    return [{key: event.get(key) for key in keys} for event in events]


def _snapshot(env, obs, rewards, terms, truncs, info, actions) -> dict:
    bool_keys = (
        "observable", "range_ok", "ata_ok", "ta_ok", "geometry_ok",
        "cooldown_ready", "duplicate_target_blocked", "allowed",
    )
    geometry_keys = ("range_m", "ata_rad", "ta_rad")
    return jsonable({
        "step": env.step_count,
        "alive": {aid: bool(env.aircraft[aid].is_alive) for aid in env.aircraft},
        "active_mask": np.asarray(info["active_mask"], dtype=np.int8),
        "selected_targets": info["selected_targets"],
        "fire_gate_flags": {
            aid: {key: bool(row[key]) for key in bool_keys if key in row}
            for aid, row in info["fire_gates"].items()},
        "events": _canonical_events(info["step_events"]),
        "newly_dead": sorted(env.newly_dead),
        "death_reasons": info["death_reasons"],
        "missile_count": len(env.missiles),
        "ammo": {aid: int(env.aircraft[aid].num_left_missiles) for aid in env.aircraft},
        "outcome": info["outcome"], "end_reason": info["end_reason"],
        "terminated": bool(any(terms.values())),
        "truncated": bool(any(truncs.values())),
        "team_done": bool(info["team_done"]),
        "position": {aid: env.aircraft[aid].get_position() for aid in env.aircraft},
        "velocity": {aid: env.aircraft[aid].get_velocity() for aid in env.aircraft},
        "rpy": {aid: env.aircraft[aid].get_rpy() for aid in env.aircraft},
        "geodetic": {aid: env.aircraft[aid].get_geodetic() for aid in env.aircraft},
        "critic_state": info["critic_state"],
        "actor_observation": {aid: obs[aid]["flat"] for aid in env.red_ids},
        "reward": rewards,
        "control_target": info["control_targets"],
        "sim_time_sec": env.sim_time_sec,
        "fire_gate_geometry": {
            aid: {key: row[key] for key in geometry_keys if key in row}
            for aid, row in info["fire_gates"].items()},
        "actions": actions,
    })


def _actions(env, policy_type: str, rng, rule) -> dict[str, np.ndarray]:
    if policy_type in ("rule_policy", "reset"):
        return rule.actions(env, "red")
    if policy_type == "zero_action":
        return {aid: np.zeros(3, np.float32) for aid in env.red_ids}
    if policy_type == "bounded_random":
        return {aid: rng.uniform(-1.0, 1.0, 3).astype(np.float32)
                for aid in env.red_ids}
    raise ValueError(f"unsupported policy type: {policy_type}")


def _state_extrema(env, extrema: dict) -> None:
    for aircraft in env.aircraft.values():
        position = np.asarray(aircraft.get_position(), dtype=np.float64)
        velocity = np.asarray(aircraft.get_velocity(), dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        horizontal = float(np.linalg.norm(position[:2]))
        extrema["max_speed_mps"] = max(extrema["max_speed_mps"], speed)
        extrema["max_altitude_m"] = max(extrema["max_altitude_m"], float(position[2]))
        extrema["min_altitude_m"] = min(extrema["min_altitude_m"], float(position[2]))
        extrema["max_horizontal_distance_m"] = max(
            extrema["max_horizontal_distance_m"], horizontal)
        extrema["max_abs_position"] = max(
            extrema["max_abs_position"], float(np.max(np.abs(position))))
        extrema["max_abs_velocity"] = max(
            extrema["max_abs_velocity"], float(np.max(np.abs(velocity))))


def _run_case(env, case: dict, max_case_steps: int, previous_controller_ids) -> tuple[dict, list[dict]]:
    policy_type = case["policy_type"]
    context_base = {
        "policy_type": policy_type, "case_id": case["case_id"],
        "seed": case["seed"], "episode": case["episode"],
    }
    obs, info = env.reset(
        seed=case["seed"],
        options={"audit_initial_perturbation": case["perturbation"]})
    reset_rows, unavailable = validate_reset_state(
        env, obs, info, previous_controller_ids=previous_controller_ids)
    for row in reset_rows:
        row.update(context_base)
    failures = reset_rows if case["case_type"] == "reset" else []
    rule = PaperGreedyOpponent()
    rng = np.random.default_rng(case["seed"] + 1_000_000)
    trace = []
    previous_alive = {aid: bool(env.aircraft[aid].is_alive) for aid in env.aircraft}
    previous_active = np.asarray(info["active_mask"], dtype=np.int8).tolist()
    previous_ammo = {aid: int(env.aircraft[aid].num_left_missiles) for aid in env.aircraft}
    dead_states = {}
    extrema = {
        "max_speed_mps": 0.0, "max_altitude_m": -np.inf,
        "min_altitude_m": np.inf, "max_horizontal_distance_m": 0.0,
        "max_abs_position": 0.0, "max_abs_velocity": 0.0,
    }
    event_steps = []
    step_limit = max_case_steps or env.max_steps
    runtime_ok = True
    for audit_step in range(step_limit):
        actions = _actions(env, policy_type, rng, rule)
        try:
            obs, rewards, terms, truncs, info = env.step(actions)
        except Exception as exc:
            failures.append(failure(
                "runtime_exception", **context_base, step=audit_step,
                expected="successful env.step", actual=type(exc).__name__,
                message=str(exc)))
            runtime_ok = False
            break
        snapshot = _snapshot(env, obs, rewards, terms, truncs, info, actions)
        trace.append(snapshot)
        alive = snapshot["alive"]
        current_active = snapshot["active_mask"]
        current_ammo = snapshot["ammo"]
        context = {**context_base, "step": env.step_count}
        failures.extend(validate_active_mask(
            previous_active, current_active, alive, env.red_ids, context))
        actual_newly_dead = sorted(
            aid for aid in env.aircraft if previous_alive[aid] and not alive[aid])
        if actual_newly_dead != snapshot["newly_dead"]:
            failures.append(failure(
                "death_newly_dead_mismatch", **context,
                expected=actual_newly_dead, actual=snapshot["newly_dead"],
                message="newly_dead is not the alive-to-dead transition set"))
        for aid in actual_newly_dead:
            if aid not in snapshot["death_reasons"]:
                failures.append(failure(
                    "death_reason_missing", **context, agent_id=aid,
                    expected="one final death reason", actual=None,
                    message="newly dead aircraft has no death reason"))
        failures.extend(validate_missile_step(
            previous_alive=previous_alive, alive=alive,
            previous_ammo=previous_ammo, ammo=current_ammo,
            events=info["step_events"], death_reasons=info["death_reasons"],
            roles=env.roles, agent_ids=set(env.aircraft), context=context))
        terminated = bool(any(terms.values()))
        truncated = bool(any(truncs.values()))
        failures.extend(validate_termination(
            terminated, truncated, bool(info["team_done"]),
            info["outcome"], info["end_reason"], env.step_count, env.max_steps,
            int(info["red_attack_alive"]), int(info["blue_attack_alive"]), context))
        numeric_values = {
            "actor_observation": snapshot["actor_observation"],
            "critic_state": snapshot["critic_state"], "reward": rewards,
            "position": snapshot["position"], "velocity": snapshot["velocity"],
            "rpy": snapshot["rpy"], "geodetic": snapshot["geodetic"],
            "control_target": snapshot["control_target"],
            "sim_time_sec": snapshot["sim_time_sec"],
            "fire_gate_geometry": snapshot["fire_gate_geometry"],
            "reward_components": info["reward_components"],
        }
        nonfinite = finite_failures(numeric_values, **context)
        failures.extend(nonfinite)
        _state_extrema(env, extrema)
        for aid in env.aircraft:
            state = np.r_[env.aircraft[aid].get_position(),
                          env.aircraft[aid].get_velocity(), env.aircraft[aid].get_rpy()]
            if not alive[aid]:
                if aid in dead_states and not np.allclose(
                        dead_states[aid], state, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
                    failures.append(failure(
                        "death_inactive_state_changed", **context, agent_id=aid,
                        expected=dead_states[aid], actual=state,
                        message="dead aircraft state changed after death"))
                dead_states.setdefault(aid, state.copy())
        for aid in actual_newly_dead:
            reason = info["death_reasons"].get(aid)
            position = np.asarray(env.aircraft[aid].get_position(), dtype=np.float64)
            finite = bool(np.isfinite(position).all())
            valid = True
            if reason == "crash":
                valid = finite and position[2] < 100.0
            elif reason == "out_of_zone":
                valid = finite and (
                    np.linalg.norm(position[:2]) > 50_000.0 or position[2] > 10_000.0)
            elif reason == "numeric_anomaly":
                valid = not finite
            if not valid:
                failures.append(failure(
                    "boundary_death_reason_mismatch", **context, agent_id=aid,
                    expected=reason, actual=position,
                    message="death reason does not match frozen hard boundary"))
        event_steps.extend(int(event.get("step", env.step_count))
                           for event in info["step_events"])
        if event_steps != sorted(event_steps):
            failures.append(failure(
                "missile_event_step_not_monotonic", **context,
                expected="non-decreasing", actual=event_steps,
                message="event steps are not monotonic"))
        previous_alive = alive
        previous_active = current_active
        previous_ammo = current_ammo
        if nonfinite:
            runtime_ok = False
            break
        if terminated or truncated:
            break
    events = list(env.event_log)
    combat_counts = combat_event_counts(events, env.death_reasons)
    for aid in env.aircraft:
        launches = sum(event.get("event") == "launch"
                       and event.get("shooter_id") == aid for event in events)
        if launches > MISSILE_COUNTS[aid]:
            failures.append(failure(
                "missile_launch_count_exceeds_ammo", **context_base,
                step=env.step_count, agent_id=aid,
                expected=f"<= {MISSILE_COUNTS[aid]}", actual=launches,
                message="aircraft launched more missiles than initial ammunition"))
    record = {
        **case, "perturbation_signature": stable_signature(case["perturbation"]),
        "case_completed": True, "runtime_ok": runtime_ok,
        "steps": len(trace), "outcome": info.get("outcome", "incomplete"),
        "end_reason": info.get("end_reason", "audit_step_limit"),
        "terminated": bool(trace and trace[-1]["terminated"]),
        "truncated": bool(trace and trace[-1]["truncated"]),
        "team_done": bool(trace and trace[-1]["team_done"]),
        "reset_failure_count": len(reset_rows),
        "reset_unavailable_fields": unavailable,
        "non_finite_count": sum(row["check_name"] == "non_finite" for row in failures),
        "numeric_anomaly_count": sum(
            reason == "numeric_anomaly" for reason in env.death_reasons.values()),
        "crash_count": sum(reason == "crash" for reason in env.death_reasons.values()),
        "out_of_zone_count": sum(
            reason == "out_of_zone" for reason in env.death_reasons.values()),
        "missile_death_count": sum(
            reason == "missile_hit" for reason in env.death_reasons.values()),
        **combat_counts,
        **{key: (None if not np.isfinite(value) else float(value))
           for key, value in extrema.items()},
        "trace": trace if case["case_type"] == "deterministic" else [],
    }
    return record, failures


def main() -> None:
    args = parse_args()
    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    plan, unique_counts = _case_plan(args)
    probe = make_env(str(config))
    try:
        probe.reset(seed=args.seed)
        expected_meta = _expected_meta(args, probe, config, unique_counts)
    finally:
        probe.close()
    output, completed = _prepare_output(args, expected_meta)
    raw_path = output / "environment_stability_raw.jsonl"
    failure_path = output / "environment_stability_failures.jsonl"
    env = make_env(str(config))
    env.reset(seed=_warmup_seed(args.seed),
              options={"audit_initial_perturbation": {}})
    start = time.monotonic()
    previous_controller_ids = {aid: id(controller)
                               for aid, controller in env.controllers.items()}
    try:
        for plan_index, case in enumerate(plan):
            if case["case_id"] in completed:
                continue
            case_env = env
            case_previous_controller_ids = previous_controller_ids
            if case["case_type"] == "deterministic":
                case_env = make_env(str(config))
                case_previous_controller_ids = None
            try:
                record, rows = _run_case(
                    case_env, case, args.max_case_steps,
                    case_previous_controller_ids)
            except KeyboardInterrupt:
                build_report(output)
                raise
            except Exception as exc:
                rows = [failure(
                    "runtime_exception", case["policy_type"], case["case_id"],
                    case["seed"], case["episode"], -1,
                    f"case execution raised {type(exc).__name__}: {exc}",
                    expected="completed case", actual=type(exc).__name__)]
                record = {
                    **case, "case_completed": True, "runtime_ok": False,
                    "steps": 0, "outcome": "audit_exception",
                    "end_reason": type(exc).__name__, "trace": [],
                }
            finally:
                if case_env is not env:
                    case_env.close()
            for row in rows:
                append_jsonl(failure_path, row)
            if case["case_type"] == "deterministic" and case.get("deterministic_run") == "b":
                baseline_id = case["case_id"].replace("_run_b", "_run_a")
                baseline = completed.get(baseline_id)
                if baseline is None:
                    baseline_rows = load_completed_cases(raw_path)
                    baseline = baseline_rows.get(baseline_id)
                comparison = compare_traces(
                    baseline.get("trace", []) if baseline else [], record.get("trace", []))
                record.update({
                    "deterministic_match": comparison["match"],
                    "first_mismatch_step": comparison["first_mismatch_step"],
                    "first_mismatch_field": comparison["first_mismatch_field"],
                    "max_absolute_difference_by_field": comparison[
                        "max_absolute_difference_by_field"],
                })
                if not comparison["match"]:
                    row = failure(
                        "deterministic_replay_mismatch", case["policy_type"],
                        case["case_id"], case["seed"], case["episode"],
                        comparison["first_mismatch_step"] or 0,
                        "deterministic replay differs",
                        expected="matching run_a trace",
                        actual=comparison["first_mismatch_field"])
                    append_jsonl(failure_path, row)
                    rows.append(row)
            append_jsonl(raw_path, record)
            completed[case["case_id"]] = record
            if case_env is env:
                previous_controller_ids = {
                    aid: id(controller) for aid, controller in env.controllers.items()}
            build_report(output)
            done = len(completed)
            elapsed = time.monotonic() - start
            remaining = max(len(plan) - done, 0)
            eta = elapsed / max(done, 1) * remaining
            total_failures = len(read_jsonl_count(failure_path))
            print(
                f"[stability] {case['case_id']} policy={case['policy_type']} "
                f"steps={record.get('steps', 0)} outcome={record.get('outcome')} "
                f"failures={total_failures} progress={done}/{len(plan)} "
                f"eta_sec={eta:.1f}", flush=True)
    finally:
        env.close()
    payload = build_report(output)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def read_jsonl_count(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


if __name__ == "__main__":
    main()
