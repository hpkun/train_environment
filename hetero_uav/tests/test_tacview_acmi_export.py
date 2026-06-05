from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_tacview_help():
    result = subprocess.run(
        ["python", "scripts/export_hetero_tacview_acmi.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for flag in [
        "--config",
        "--steps",
        "--red-policy",
        "--blue-policy",
        "--output-acmi",
        "--record-missiles",
    ]:
        assert flag in result.stdout


def test_export_tacview_short_rollout():
    output_acmi = ROOT / "outputs/test_tacview/test_rollout.acmi"
    output_json = ROOT / "outputs/test_tacview/test_rollout_meta.json"
    result = subprocess.run(
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
            "greedy_fsm",
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
    assert output_acmi.exists(), result.stdout + result.stderr
    assert output_json.exists(), result.stdout + result.stderr

    text = output_acmi.read_text(encoding="utf-8-sig")
    assert "FileType=text/acmi/tacview" in text
    assert "FileVersion=2.1" in text
    assert "ReferenceTime" in text
    assert "#0" in text
    assert "Name=red_0" in text or "Name=red_0_mav" in text
    assert "Name=blue_0" in text
    assert "Color=Red" in text
    assert "Color=Blue" in text

    meta = json.loads(output_json.read_text(encoding="utf-8"))
    assert meta["steps_executed"] >= 1
    assert meta["frames_recorded"] >= 2
    assert "final_red_alive" in meta
    assert "final_blue_alive" in meta
    assert meta["record_missiles"] is True


def test_tacview_doc_exists():
    doc = ROOT / "docs/tacview_acmi_export.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for phrase in ["Tacview", "ACMI", "initial geometry", "MAV", "not training"]:
        assert phrase in text
