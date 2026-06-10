import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_happo_3v2_reference_alignment.py"


def test_audit_happo_3v2_reference_alignment_help_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "HAPPO 3v2" in result.stdout


def test_audit_happo_3v2_reference_alignment_outputs_expected_schema():
    out_json = ROOT / "outputs" / "test_protocol_audit" / "happo_3v2_reference_alignment.json"
    out_md = ROOT / "outputs" / "test_protocol_audit" / "happo_3v2_reference_alignment.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-json",
            str(out_json.relative_to(ROOT)),
            "--output-md",
            str(out_md.relative_to(ROOT)),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "happo_ref_v0_reward_implemented: False" in result.stdout
    assert "happo_smoke_implemented: False" in result.stdout
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in [
        "paper_reference_setup",
        "current_env_setup",
        "gaps",
        "success_criteria",
        "next_steps",
    ]:
        assert key in data

    assert data["happo_ref_v0_reward_design"]["implemented_this_round"] is False
    assert data["minimal_happo_v0_plan"]["implemented_this_round"] is False
    assert any(gap["name"] == "happo_update_gap" for gap in data["gaps"])

    md = out_md.read_text(encoding="utf-8")
    assert "HAPPO" in md
    assert "3v2" in md
    assert "MAV" in md
    assert "UAV" in md
    assert "Timeout draw behavior" in md


def test_happo_3v2_reference_validation_plan_documents_scope():
    doc = ROOT / "docs" / "happo_3v2_reference_validation_plan.md"
    text = doc.read_text(encoding="utf-8")
    assert "separate MAV actor" in text
    assert "separate UAV actor" in text
    assert "centralized critic" in text
    assert "no attention in the first stage" in text
    assert "Do not switch now to low-level 4D control" in text
