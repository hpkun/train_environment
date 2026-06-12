"""Probe whether a red UAV can fire from an explicit launch envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from red_attack_audit_utils import (
    DEFAULT_CONFIG,
    blue_actions,
    collect_step_counts,
    direct_chase_action,
    geometry,
    make_env,
    safe_mav_action,
    team_done,
    write_json,
    write_md,
)


def _forced_initial_states(config: str) -> dict:
    raw = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    states = dict(raw.get("initial_states", {}))
    states.update({
        # red_1 starts behind blue_0, within 10 km and pointing north.
        "red_0": {"lat": 59.98, "lon": 120.02, "altitude_m": 6000, "speed_mps": 250, "yaw_deg": 0.0},
        "red_1": {"lat": 60.000, "lon": 120.000, "altitude_m": 6000, "speed_mps": 280, "yaw_deg": 0.0},
        "red_2": {"lat": 60.000, "lon": 120.040, "altitude_m": 6000, "speed_mps": 260, "yaw_deg": 0.0},
        "blue_0": {"lat": 60.030, "lon": 120.000, "altitude_m": 6000, "speed_mps": 240, "yaw_deg": 0.0},
        "blue_1": {"lat": 60.030, "lon": 120.040, "altitude_m": 6000, "speed_mps": 240, "yaw_deg": 0.0},
    })
    return states


def infer_failure_reason(g: dict) -> str | None:
    if not g.get("alive", False):
        return "shooter_not_alive"
    if not g.get("target_id"):
        return "no_alive_target"
    if g.get("missile_count", 0) <= 0:
        return "no_missile"
    if not g.get("launch_condition_distance", False):
        return "distance_outside_launch_envelope"
    if not g.get("launch_condition_angle", False):
        return "angle_outside_launch_envelope"
    if g.get("cooldown", 0) > 0:
        return "cooldown"
    return None


def build_probe(config: str, steps: int, blue_policy: str) -> dict:
    env = make_env(config, initial_states=_forced_initial_states(config))
    records = []
    red_fired = red_hits = 0
    try:
        obs, _info = env.reset(seed=0)
        for step in range(1, steps + 1):
            before = geometry(env, "red_1", "blue_0")
            actions = {
                "red_0": safe_mav_action(),
                "red_1": direct_chase_action(env, "red_1", speed=0.9),
                "red_2": direct_chase_action(env, "red_2", speed=0.8),
            }
            actions.update(blue_actions(env, obs, blue_policy))
            obs, _rewards, terminated, truncated, info = env.step(actions)
            counts = collect_step_counts(info)
            red_fired += counts["red_fired"]
            red_hits = max(red_hits, counts["red_hits_total"])
            after = geometry(env, "red_1", "blue_0")
            records.append({
                "step": step,
                **{f"before_{k}": v for k, v in before.items()},
                "missile_fired_this_step": counts["red_fired"] > 0,
                "red_fired_this_step": counts["red_fired"],
                "red_hits_total": counts["red_hits_total"],
                "fire_failure_reason": None if counts["red_fired"] > 0 else infer_failure_reason(before),
                **{f"after_{k}": v for k, v in after.items()},
            })
            if team_done(terminated, truncated):
                break

        any_in_envelope = any(r.get("before_launch_condition_all") for r in records)
        any_fire_in_envelope = any(r.get("before_launch_condition_all") and r["missile_fired_this_step"] for r in records)
        if red_fired > 0:
            conclusion = "red_uav_can_fire"
        elif any_in_envelope:
            conclusion = "red_uav_entered_envelope_but_did_not_fire"
        else:
            conclusion = "red_uav_never_entered_envelope"
        return {
            "config": config,
            "steps_requested": steps,
            "steps_recorded": len(records),
            "blue_policy": blue_policy,
            "red_fired_total": red_fired,
            "red_hits_total": red_hits,
            "any_launch_envelope_satisfied": bool(any_in_envelope),
            "any_fire_while_envelope_satisfied": bool(any_fire_in_envelope),
            "red_uav_can_fire_in_theoretical_envelope": bool(red_fired > 0),
            "attack_window_reward_matches_launch_envelope": "not_evaluated_in_this_probe",
            "unit_or_normalization_suspect": bool(any_in_envelope and red_fired == 0),
            "conclusion": conclusion,
            "records": records,
        }
    finally:
        env.close()


def write_report_md(data: dict, output_md: str) -> None:
    first = data["records"][0] if data["records"] else {}
    lines = [
        "# Red Missile Launch Envelope Probe",
        "",
        f"- conclusion: {data['conclusion']}",
        f"- red_fired_total: {data['red_fired_total']}",
        f"- red_hits_total: {data['red_hits_total']}",
        f"- any_launch_envelope_satisfied: {data['any_launch_envelope_satisfied']}",
        f"- any_fire_while_envelope_satisfied: {data['any_fire_while_envelope_satisfied']}",
        "",
        "## First Step Geometry",
        "```json",
        json.dumps(first, indent=2),
        "```",
    ]
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe red UAV missile launch envelope")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--blue-policy", choices=["zero", "brma_rule"], default="zero")
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/red_launch_envelope/red_missile_launch_envelope_probe.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/red_launch_envelope/red_missile_launch_envelope_probe.md",
    )
    args = parser.parse_args()
    data = build_probe(args.config, args.steps, args.blue_policy)
    out_json = write_json(args.output_json, data)
    write_report_md(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"conclusion: {data['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

