"""Diagnose missile launch accounting for the hetero JSBSim environment.

This script is an environment audit utility. It does not train policies, load
MAPPO checkpoints, or change combat mechanics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy


def _red_actions(env, policy_name: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if policy_name == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.red_ids}
    if policy_name == "random":
        return {
            aid: rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            for aid in env.red_ids
        }
    raise ValueError(f"unsupported red policy: {policy_name}")


def _all_done(terminated: dict, truncated: dict) -> bool:
    return all(bool(v) for v in terminated.values()) or all(
        bool(v) for v in truncated.values()
    )


def _agent_sims(env) -> dict[str, Any]:
    sims = {}
    sims.update(getattr(env, "red_planes", {}))
    sims.update(getattr(env, "blue_planes", {}))
    return sims


def _initial_missiles(env) -> dict[str, int]:
    out = {}
    for aid, sim in _agent_sims(env).items():
        out[aid] = int(getattr(sim, "num_missiles", 0))
    return out


def _final_missiles_left(env) -> dict[str, int]:
    out = {}
    for aid, sim in _agent_sims(env).items():
        out[aid] = int(getattr(sim, "num_left_missiles", 0))
    return out


def _record_key(record: dict) -> tuple:
    return (
        record.get("missile_id"),
        record.get("shooter_id"),
        record.get("target_id"),
        record.get("launch_step"),
        record.get("physics_frame"),
    )


def _merge_launch_records(records: list[dict], new_records: list[dict]) -> None:
    seen = {_record_key(record) for record in records}
    for record in new_records:
        key = _record_key(record)
        if key not in seen:
            records.append(dict(record))
            seen.add(key)


def _count_by(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field, ""))
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sum_launch_diag(total: dict, step_diag: dict) -> None:
    for team, counters in step_diag.items():
        team_total = total.setdefault(team, {})
        for key, value in counters.items():
            team_total[key] = int(team_total.get(key, 0)) + int(value)


def _launch_intervals(records: list[dict]) -> tuple[dict[str, list[int]], dict[str, int]]:
    by_shooter: dict[str, list[int]] = {}
    for record in records:
        shooter = str(record.get("shooter_id", ""))
        frame = record.get("physics_frame", "")
        if not shooter or frame == "":
            continue
        by_shooter.setdefault(shooter, []).append(int(frame))

    intervals: dict[str, list[int]] = {}
    min_intervals: dict[str, int] = {}
    for shooter, frames in by_shooter.items():
        frames = sorted(frames)
        diffs = [b - a for a, b in zip(frames, frames[1:])]
        intervals[shooter] = diffs
        if diffs:
            min_intervals[shooter] = min(diffs)
    return intervals, min_intervals


def _build_summary(
    *,
    args,
    env,
    steps_executed: int,
    launch_records: list[dict],
    launch_diag_total: dict,
    initial_missiles: dict[str, int],
) -> dict:
    roles = getattr(env, "agent_roles", {})
    for record in launch_records:
        shooter = str(record.get("shooter_id", ""))
        target = str(record.get("target_id", ""))
        record.setdefault("shooter_role", str(roles.get(shooter, "")))
        record.setdefault("target_role", str(roles.get(target, "")))

    launches_by_shooter = _count_by(launch_records, "shooter_id")
    launches_by_target = _count_by(launch_records, "target_id")
    launches_by_team = _count_by(launch_records, "shooter_team")
    if not launches_by_team:
        launches_by_team = _count_by(launch_records, "team")
    launches_by_target_role = _count_by(launch_records, "target_role")
    launches_by_shooter_role = _count_by(launch_records, "shooter_role")

    ammo_violations = []
    for shooter, count in launches_by_shooter.items():
        configured = int(initial_missiles.get(shooter, 0))
        if count > configured:
            ammo_violations.append({
                "shooter_id": shooter,
                "launches": int(count),
                "configured_num_missiles": configured,
            })

    mav_launch_violations = []
    for shooter, count in launches_by_shooter.items():
        if str(roles.get(shooter, "")) == "mav" and count > 0:
            mav_launch_violations.append({
                "shooter_id": shooter,
                "launches": int(count),
            })

    launches_against_mav = sum(
        1 for record in launch_records
        if str(record.get("target_role") or roles.get(str(record.get("target_id", "")), "")) == "mav"
    )
    launch_intervals, min_intervals = _launch_intervals(launch_records)
    warnings = []
    if not launch_records:
        warnings.append("no launches observed")
    cooldown = int(getattr(env, "missile_cooldown_frames", 0))
    for shooter, min_interval in min_intervals.items():
        if cooldown and min_interval < cooldown:
            warnings.append(
                f"{shooter} launch interval {min_interval} physics frames is below cooldown {cooldown}"
            )

    return {
        "config": args.config,
        "steps_executed": int(steps_executed),
        "red_policy": args.red_policy,
        "blue_policy": args.blue_policy,
        "total_launches": int(len(launch_records)),
        "launches_by_team": launches_by_team,
        "launches_by_shooter": launches_by_shooter,
        "launches_by_target": launches_by_target,
        "launches_by_target_role": launches_by_target_role,
        "launches_against_mav": int(launches_against_mav),
        "launches_by_shooter_role": launches_by_shooter_role,
        "max_launches_by_single_shooter": int(max(launches_by_shooter.values(), default=0)),
        "ammo_violations": ammo_violations,
        "mav_launch_violations": mav_launch_violations,
        "target_role_records_available": any(
            bool(record.get("target_role")) for record in launch_records
        ),
        "min_launch_interval_by_shooter": min_intervals,
        "launch_intervals_by_shooter": launch_intervals,
        "num_left_missiles_final_by_agent": _final_missiles_left(env),
        "initial_num_missiles_by_agent": initial_missiles,
        "launch_diag_totals": launch_diag_total,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--red-policy", choices=["zero", "random"], default="zero")
    parser.add_argument(
        "--blue-policy",
        choices=["zero", "rule_nearest", "greedy_fsm", "random"],
        default="rule_nearest",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/missile_launch_logic.json",
    )
    parser.add_argument(
        "--output-acmi",
        default="outputs/tacview/missile_launch_logic.acmi",
    )
    args = parser.parse_args()

    from uav_env import make_env

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.config, env_type="jsbsim_hetero")
    rng = np.random.default_rng(args.seed)
    blue_policy = OpponentPolicy(mode=args.blue_policy, seed=args.seed + 17)
    launch_records: list[dict] = []
    launch_diag_total: dict[str, dict[str, int]] = {}
    steps_executed = 0

    try:
        obs, _info = env.reset(seed=args.seed)
        initial_missiles = _initial_missiles(env)
        for step in range(1, args.steps + 1):
            actions = _red_actions(env, args.red_policy, rng)
            actions.update(blue_policy.act(obs, env.blue_ids))
            obs, _rewards, terminated, truncated, info = env.step(actions)
            steps_executed = step
            _merge_launch_records(launch_records, info.get("__launch_quality_step__", []))
            _merge_launch_records(launch_records, info.get("__launch_quality_done__", []))
            _sum_launch_diag(launch_diag_total, info.get("__launch_diag__", {}))
            if _all_done(terminated, truncated):
                break

        summary = _build_summary(
            args=args,
            env=env,
            steps_executed=steps_executed,
            launch_records=launch_records,
            launch_diag_total=launch_diag_total,
            initial_missiles=initial_missiles,
        )
        output_json.write_text(
            json.dumps({"summary": summary, "launch_records": launch_records}, indent=2),
            encoding="utf-8",
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    print(f"output_json: {output_json}")
    print(f"steps_executed: {steps_executed}")
    print(f"total_launches: {summary['total_launches']}")
    print(f"launches_by_shooter: {summary['launches_by_shooter']}")
    print(f"ammo_violations: {summary['ammo_violations']}")
    print(f"mav_launch_violations: {summary['mav_launch_violations']}")
    print(f"launches_against_mav: {summary['launches_against_mav']}")

    if summary["ammo_violations"] or summary["mav_launch_violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
