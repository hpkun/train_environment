"""Read-only audit of blue opponent logic alignment.

The script records current scripted opponent capabilities and gaps against the
environment expectations documented for BRMA-MAPPO and TAM-HAPPO inspired
setups. It does not train and does not modify the environment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy

GREEDY_FSM_STATES = [
    "evade",
    "recover_altitude",
    "attack_mav_priority",
    "attack_nearest",
    "search_acquire",
    "patrol",
]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _ensure_visibility_json(path: Path) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        cmd = [
            "python",
            "scripts/diagnose_hetero_visibility_geometry.py",
            "--steps",
            "100",
            "--blue-policy",
            "greedy_fsm",
            "--output-json",
            str(path),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        warnings.append("visibility json was missing and has been generated")

    if not path.exists():
        warnings.append("visibility json missing after generation attempt")
        return {}, warnings
    return json.loads(path.read_text(encoding="utf-8")), warnings


def _paper_aligned_visibility_summary(data: dict) -> dict:
    records = [
        record for record in data.get("records", [])
        if "hetero_mav_shared_geo_3v2" in record.get("config", "")
        or "hetero_mav_shared_geo_5v4" in record.get("config", "")
    ]
    warnings = []
    for record in records:
        warnings.extend(record.get("warnings", []))
    return {
        "paper_aligned_blue_observed_any": any(
            bool(record.get("blue_observed_any", False)) for record in records),
        "paper_aligned_first_step_blue_observed": [
            int(record.get("first_step_blue_observed", -1)) for record in records
        ],
        "paper_aligned_red_mav_shared_fraction": [
            float(record.get("red_mav_shared_fraction", 0.0)) for record in records
        ],
        "paper_aligned_blue_direct_fraction": [
            float(record.get("blue_direct_fraction", 0.0)) for record in records
        ],
        "visibility_warnings": sorted(set(warnings)),
    }


def build_audit(visibility_json: Path) -> dict:
    opponent_source = _read_text(ROOT / "algorithms/mappo/opponent_policy.py")
    env_source = _read_text(ROOT / "uav_env/JSBSim/env.py")
    hetero_source = _read_text(
        ROOT / "uav_env/JSBSim/envs/hetero_uav_combat_env.py")
    visibility, visibility_generation_warnings = _ensure_visibility_json(
        visibility_json)
    visibility_summary = _paper_aligned_visibility_summary(visibility)

    modes = sorted(OpponentPolicy.MODES)
    support_flags = {
        "has_last_states": "last_states" in opponent_source,
        "has_missile_warning_branch": "missile_warning" in opponent_source,
        "has_altitude_recover_branch": "recover_altitude" in opponent_source,
        "has_mav_priority_branch": "attack_mav_priority" in opponent_source,
        "has_nearest_attack_branch": "attack_nearest" in opponent_source,
        "has_search_acquisition_behavior": "search_acquire" in opponent_source,
        "has_patrol_branch": "patrol" in opponent_source,
        "has_target_assignment": False,
        "has_candidate_maneuver_scoring": False,
        "directly_controls_missile": False,
        "relies_on_env_fire_control": "_check_missile_launch" in env_source,
        "relies_on_env_evasion": "Missile Evasion Script" in env_source
        or "check_missile_warning" in env_source,
        "blue_has_gcas_in_env": "enable_gcas_for_blue" in env_source,
        "blue_direct_only_v2_visibility": "mav_shared" in hetero_source
        and "ego_is_red" in hetero_source,
    }

    blue_never_observed = not visibility_summary[
        "paper_aligned_blue_observed_any"]
    gap_flags = {
        "gap_blue_patrol_only_if_no_visibility": blue_never_observed,
        "gap_no_target_assignment": not support_flags["has_target_assignment"],
        "gap_no_candidate_maneuver_scoring": not support_flags[
            "has_candidate_maneuver_scoring"],
        "gap_no_search_acquisition_behavior": not support_flags[
            "has_search_acquisition_behavior"],
        "gap_visibility_asymmetry": any(
            "asymmetric information" in warning
            for warning in visibility_summary["visibility_warnings"]),
        "gap_geometry_or_sensor_protocol_unresolved": blue_never_observed,
    }

    recommended = [
        "add controlled branch tests before changing default opponent",
        "horizon-sweep visibility after adding search_acquire",
        "decide whether paper-aligned geometry should be closer or blue direct range should differ",
        "audit reward/termination after opponent logic",
        "do not train with greedy_fsm until explicit opponent validation passes",
    ]

    return {
        "current_opponent_modes": modes,
        "greedy_fsm_states": GREEDY_FSM_STATES,
        "support_flags": support_flags,
        "visibility_audit": {
            **visibility_summary,
            "visibility_json": str(visibility_json),
            "visibility_generation_warnings": visibility_generation_warnings,
        },
        "gap_flags": gap_flags,
        "recommended_next_actions": recommended,
        "notes": [
            "No original heterogeneous paper text was found in this script; literature evidence is taken from repository notes and current code comments.",
            "The audit is read-only and does not modify initial_states, observation range, missile, reward, termination, PID, or aircraft XML.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/blue_opponent_logic_alignment.json",
    )
    parser.add_argument(
        "--visibility-json",
        default="outputs/environment_audit/hetero_visibility_geometry.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json)
    visibility_path = Path(args.visibility_json)
    audit = build_audit(visibility_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"opponent_modes: {audit['current_opponent_modes']}")
    print(f"greedy_fsm_states: {audit['greedy_fsm_states']}")
    print("key_gap_flags:")
    for key, value in audit["gap_flags"].items():
        print(f"  {key}: {value}")
    print("recommended_next_actions:")
    for item in audit["recommended_next_actions"]:
        print(f"  - {item}")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
