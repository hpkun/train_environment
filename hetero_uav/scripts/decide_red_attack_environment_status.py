"""Decide red attack environment status from audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from red_attack_audit_utils import write_json, write_md


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"missing": True, "path": path}
    return json.loads(p.read_text(encoding="utf-8"))


def decide(pipeline: dict, oracle: dict, envelope: dict, happo: dict) -> dict:
    red_fire_logic = bool(pipeline.get("red_auto_fire_logic_enabled"))
    logging_ok = bool(pipeline.get("logging_fields_present_for_red"))
    oracle_cases = oracle.get("cases", [])
    oracle_can_fire = any(c.get("red_missiles_fired_mean", 0.0) > 0.0 for c in oracle_cases)
    oracle_can_hit = any(c.get("red_missile_hits_mean", 0.0) > 0.0 for c in oracle_cases)
    envelope_can_fire = bool(envelope.get("red_uav_can_fire_in_theoretical_envelope"))
    happo_records = happo.get("records", [])
    happo_fires = any(r.get("red_missiles_fired_total", 0) > 0 for r in happo_records)
    happo_enters_envelope = any(r.get("launch_envelope_rate", 0.0) > 0.0 for r in happo_records)
    happo_avoids = any(r.get("policy_avoids_engagement", False) for r in happo_records)

    primary: list[str] = []
    if not red_fire_logic:
        status = "red_fire_chain_broken"
        primary.append("red_auto_fire_not_enabled")
        next_action = "A"
    elif not logging_ok:
        status = "logging_bug_only"
        primary.append("logging_miscounts_red_fire")
        next_action = "B"
    elif not oracle_can_fire and not envelope_can_fire:
        status = "red_fire_chain_broken"
        primary.extend(["launch_envelope_too_strict", "red_target_assignment_missing"])
        next_action = "A"
    elif envelope_can_fire and not oracle_can_fire:
        status = "red_fire_chain_working_policy_not_engaging"
        primary.extend(["initial_geometry_too_hard", "policy_avoids_engagement"])
        next_action = "C"
    elif oracle_can_fire and not happo_fires:
        status = "red_fire_chain_working_policy_not_engaging"
        primary.append("policy_avoids_engagement")
        if not happo_enters_envelope:
            primary.append("initial_geometry_too_hard")
        next_action = "D"
    elif oracle_can_fire and oracle_can_hit:
        status = "red_fire_chain_working_reward_problem"
        primary.append("policy_avoids_engagement" if happo_avoids else "attack_window_reward_mismatch")
        next_action = "D"
    else:
        status = "inconclusive"
        primary.append("unknown")
        next_action = "E"

    labels = {
        "A": "fix red fire-control or target assignment",
        "B": "fix logging",
        "C": "use easier initial geometry",
        "D": "add combat reward/curriculum",
        "E": "build scripted red attack oracle baseline",
        "F": "stop current environment training",
    }
    return {
        "environment_attack_pipeline_status": status,
        "primary_issue": primary[:2],
        "next_action": next_action,
        "next_action_label": labels[next_action],
        "evidence": {
            "red_auto_fire_logic_enabled": red_fire_logic,
            "logging_fields_present_for_red": logging_ok,
            "oracle_can_fire": oracle_can_fire,
            "oracle_can_hit": oracle_can_hit,
            "envelope_can_fire": envelope_can_fire,
            "happo_fires": happo_fires,
            "happo_enters_envelope": happo_enters_envelope,
            "happo_avoids_engagement": happo_avoids,
        },
    }


def write_report_md(data: dict, output_md: str) -> None:
    lines = [
        "# Red Attack Environment Decision",
        "",
        f"- environment_attack_pipeline_status: {data['environment_attack_pipeline_status']}",
        f"- primary_issue: {data['primary_issue']}",
        f"- next_action: {data['next_action']} ({data['next_action_label']})",
        "",
        "## Evidence",
        "```json",
        json.dumps(data["evidence"], indent=2),
        "```",
    ]
    write_md(output_md, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide red attack environment status")
    parser.add_argument("--pipeline-json", default="outputs/environment_audit/red_attack_pipeline/red_attack_pipeline.json")
    parser.add_argument("--oracle-json", default="outputs/environment_audit/red_attack_oracle/red_attack_oracle_sanity.json")
    parser.add_argument("--envelope-json", default="outputs/environment_audit/red_launch_envelope/red_missile_launch_envelope_probe.json")
    parser.add_argument("--happo-json", default="outputs/environment_audit/happo_engagement_geometry/happo_policy_engagement_geometry.json")
    parser.add_argument("--output-json", default="outputs/environment_audit/red_attack_environment_decision.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/red_attack_environment_decision.md")
    args = parser.parse_args()
    data = decide(
        _load(args.pipeline_json),
        _load(args.oracle_json),
        _load(args.envelope_json),
        _load(args.happo_json),
    )
    out_json = write_json(args.output_json, data)
    write_report_md(data, args.output_md)
    print(f"output_json: {out_json}")
    print(f"environment_attack_pipeline_status: {data['environment_attack_pipeline_status']}")
    print(f"next_action: {data['next_action']} ({data['next_action_label']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

