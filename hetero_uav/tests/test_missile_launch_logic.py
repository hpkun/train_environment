from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tacview_export_has_type_and_metadata_fields():
    output_acmi = ROOT / "outputs/test_environment_audit/typecheck.acmi"
    output_json = ROOT / "outputs/test_environment_audit/typecheck_meta.json"
    subprocess.run(
        [
            "python",
            "scripts/export_hetero_tacview_acmi.py",
            "--config",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "--steps",
            "3",
            "--red-policy",
            "zero",
            "--blue-policy",
            "rule_nearest",
            "--output-acmi",
            str(output_acmi),
            "--output-json",
            str(output_json),
            "--record-missiles",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    text = output_acmi.read_text(encoding="utf-8-sig")
    assert "Type=Air+FixedWing" in text
    meta = json.loads(output_json.read_text(encoding="utf-8"))
    assert meta["acmi_entity_type_fix"] is True
    assert "missile_launch_counts" in meta
    assert "mav_targeted_by_missile_count" in meta


def test_missile_launch_logic_help():
    result = subprocess.run(
        ["python", "scripts/diagnose_missile_launch_logic.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for flag in ["--config", "--steps", "--blue-policy", "--output-json"]:
        assert flag in result.stdout


def test_missile_launch_logic_short_diagnostic():
    output_json = ROOT / "outputs/test_environment_audit/missile_launch_logic.json"
    result = subprocess.run(
        [
            "python",
            "scripts/diagnose_missile_launch_logic.py",
            "--steps",
            "50",
            "--blue-policy",
            "rule_nearest",
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
    summary = data["summary"]
    for key in [
        "total_launches",
        "launches_by_shooter",
        "ammo_violations",
        "mav_launch_violations",
        "launches_against_mav",
    ]:
        assert key in summary
    assert "launch_records" in data
    assert summary["ammo_violations"] == []
    assert summary["mav_launch_violations"] == []
    for shooter, count in summary["launches_by_shooter"].items():
        configured = summary["initial_num_missiles_by_agent"][shooter]
        assert count <= configured


def test_missile_launch_doc_exists():
    doc = ROOT / "docs/missile_launch_logic_audit.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for phrase in [
        "remaining ammo",
        "MAV carries 0",
        "UAV carries 2",
        "lock delay",
        "cooldown",
        "Tacview",
    ]:
        assert phrase in text
