from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_audit_help_mentions_outputs():
    result = subprocess.run(
        ["python", "scripts/audit_blue_opponent_logic_alignment.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--output-json" in result.stdout
    assert "--visibility-json" in result.stdout


def test_alignment_audit_outputs_expected_flags():
    output_json = ROOT / "outputs/test_environment_audit/blue_opponent_logic_alignment.json"
    result = subprocess.run(
        [
            "python",
            "scripts/audit_blue_opponent_logic_alignment.py",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert output_json.exists(), result.stdout + result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))

    for key in [
        "current_opponent_modes",
        "greedy_fsm_states",
        "support_flags",
        "gap_flags",
        "recommended_next_actions",
    ]:
        assert key in data

    assert {"zero", "random", "rule_nearest", "greedy_fsm"}.issubset(
        set(data["current_opponent_modes"])
    )
    flags = data["support_flags"]
    assert flags["has_missile_warning_branch"] is True
    assert flags["has_altitude_recover_branch"] is True
    assert flags["has_mav_priority_branch"] is True
    assert flags["has_nearest_attack_branch"] is True
    assert flags["has_patrol_branch"] is True
    assert flags["has_search_acquisition_behavior"] is True
    assert flags["has_target_assignment"] is False
    assert flags["has_candidate_maneuver_scoring"] is False
    assert flags["directly_controls_missile"] is False
    assert flags["relies_on_env_fire_control"] is True
    assert data["gap_flags"]["gap_no_search_acquisition_behavior"] is False
    assert "search_acquire" in data["greedy_fsm_states"]


def test_greedy_fsm_controlled_branches_pass():
    output_json = ROOT / "outputs/test_environment_audit/greedy_fsm_controlled_branches.json"
    result = subprocess.run(
        [
            "python",
            "scripts/diagnose_greedy_fsm_controlled_branches.py",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert output_json.exists(), result.stdout + result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))
    records = {record["case"]: record for record in data["records"]}
    for case in [
        "search_acquire_case",
        "nearest_attack_case",
        "mav_priority_case",
        "evade_case",
        "recover_altitude_case",
    ]:
        assert records[case]["passed"] is True
        assert records[case]["nan_detected"] is False
    assert data["summary"]["all_passed"] is True


def test_blue_opponent_alignment_docs_exist():
    audit_doc = ROOT / "docs/blue_opponent_paper_alignment_audit.md"
    assert audit_doc.exists()
    audit_text = audit_doc.read_text(encoding="utf-8").lower()
    for phrase in [
        "brma-mappo",
        "tam-happo",
        "target assignment",
        "candidate maneuver",
        "visibility asymmetry",
        "not final opponent",
    ]:
        assert phrase in audit_text

    design_doc = ROOT / "docs/blue_greedy_fsm_opponent_design.md"
    design_text = design_doc.read_text(encoding="utf-8").lower()
    for phrase in [
        "controlled branch diagnostics",
        "patrol-only",
        "not final opponent",
    ]:
        assert phrase in design_text
