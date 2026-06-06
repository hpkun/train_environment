"""Generate a heterogeneous environment gap report from readiness audit JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMPLEMENTED = [
    "JSBSim backend",
    "high-level [pitch, heading, speed] action",
    "60Hz simulation and 5Hz decision",
    "A-4 MAV",
    "f16 UAV",
    "MAV missiles=0",
    "attack UAV missiles=2",
    "paper-aligned 3v2/5v4 composition",
    "V2 mav_shared_geo observation",
    "combat metrics",
]

PARTIAL = [
    "MAV shared observation has no communication delay/noise",
    "V2 geometry observation is abstracted, not full radar/RCS/EO sensor model",
    "blue greedy finite-state greedy_fsm opponent is implemented as an engineering approximation; still not a full paper reproduction",
    "reward/termination inherited from BRMA and not yet audited for MAV/UAV roles",
    "balanced protocol is hard ablation, not main paper-aligned setting",
]

MISSING = [
    "audit reward and termination for heterogeneous MAV/UAV objective",
    "add training log outcome metrics if still needed",
    "finalize paper-aligned environment protocol before new methods",
    "V1 brma_sensor audit is optional reference only",
]


def _paper_record(records: list[dict], token: str) -> dict | None:
    for record in records:
        if record.get("protocol_type") != "paper_aligned":
            continue
        if token in Path(record.get("config", "")).name:
            return record
    return None


def _uav_missiles(record: dict) -> dict[str, int]:
    missiles = record.get("missile_counts", {})
    out = {}
    for aid, type_name in zip(
        record.get("red_ids", []) + record.get("blue_ids", []),
        record.get("red_agent_types", []) + record.get("blue_agent_types", []),
    ):
        if type_name == "attack_uav":
            out[aid] = int(missiles.get(aid, -1))
    return out


def _mav_missiles(record: dict) -> dict[str, int]:
    missiles = record.get("missile_counts", {})
    out = {}
    for aid, type_name in zip(record.get("red_ids", []), record.get("red_agent_types", [])):
        if type_name == "mav":
            out[aid] = int(missiles.get(aid, -1))
    return out


def _protocol_status(record: dict | None, label: str) -> dict:
    if record is None:
        return {"label": label, "exists": False}
    return {
        "label": label,
        "exists": True,
        "config": record["config"],
        "passed": not bool(record.get("error")),
        "red_count": record.get("red_count"),
        "blue_count": record.get("blue_count"),
        "red_attack_uav_count": record.get("red_attack_uav_count"),
        "blue_attack_uav_count": record.get("blue_attack_uav_count"),
        "mav_count": record.get("mav_count"),
        "mav_missiles": _mav_missiles(record),
        "uav_missiles": _uav_missiles(record),
        "max_steps": record.get("max_steps"),
        "sim_freq": record.get("sim_freq"),
        "agent_interaction_steps": record.get("agent_interaction_steps"),
        "decision_dt": record.get("decision_dt"),
        "actor_dim": record.get("actor_dim"),
        "critic_dim": record.get("critic_dim"),
        "observation_mode": record.get("observation_mode"),
        "warnings": record.get("warnings", []),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Heterogeneous Environment Gap Report",
        "",
        "The current environment is not ready for method module work. The next "
        "environment task is greedy_fsm diagnostics followed by reward/termination audit.",
        "",
        "## Protocol Status",
    ]
    for item in report["protocol_status"]:
        lines.extend([
            "",
            f"### {item['label']}",
            "",
            f"- exists: {item.get('exists', False)}",
        ])
        if not item.get("exists"):
            continue
        for key in [
            "passed",
            "red_count",
            "blue_count",
            "red_attack_uav_count",
            "blue_attack_uav_count",
            "mav_count",
            "mav_missiles",
            "uav_missiles",
            "max_steps",
            "sim_freq",
            "agent_interaction_steps",
            "decision_dt",
            "actor_dim",
            "critic_dim",
            "observation_mode",
            "warnings",
        ]:
            lines.append(f"- {key}: {item.get(key)}")

    for title, key in [
        ("Implemented", "implemented"),
        ("Abstracted / Partially Aligned", "abstracted_partially_aligned"),
        ("Missing / Next Actions", "missing_next_actions"),
    ]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {value}" for value in report["alignment_summary"][key])

    lines.extend([
        "",
        "## Next Environment Task",
        "",
        f"- next_environment_task: {report['next_environment_task']}",
        "- Do not start attention, HAPPO, GRU, long training, or other method-module work.",
    ])
    return "\n".join(lines) + "\n"


def build_report(audit: dict) -> dict:
    records = audit.get("records", [])
    status = [
        _protocol_status(_paper_record(records, "3v2"), "paper-aligned 3v2"),
        _protocol_status(_paper_record(records, "5v4"), "paper-aligned 5v4"),
    ]
    return {
        "source_summary": audit.get("summary", {}),
        "protocol_status": status,
        "alignment_summary": {
            "implemented": IMPLEMENTED,
            "abstracted_partially_aligned": PARTIAL,
            "missing_next_actions": MISSING,
        },
        "next_environment_task": "blue_greedy_fsm_diagnostics_then_reward_termination_audit",
        "ready_for_method_module": False,
        "method_module_blocked_reason": (
            "not ready for method module until paper-aligned environment protocol, "
            "blue opponent, reward/termination, and outcome metrics are finalized"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default="outputs/environment_audit/paper_aligned_v2_readiness.json",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/hetero_environment_gap_report.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/hetero_environment_gap_report.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    audit = json.loads(input_path.read_text(encoding="utf-8"))
    report = build_report(audit)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(report), encoding="utf-8")

    print(f"input_json: {input_path}", flush=True)
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"next_environment_task: {report['next_environment_task']}", flush=True)


if __name__ == "__main__":
    main()
