from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mav_flight_stability_help():
    result = subprocess.run(
        ["python", "scripts/diagnose_mav_flight_stability.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for flag in [
        "--configs",
        "--steps",
        "--blue-policy",
        "--output-json",
        "--export-acmi",
        "--disable-config-trim",
    ]:
        assert flag in result.stdout


def test_mav_flight_stability_short_diagnostic():
    output_json = ROOT / "outputs/test_environment_audit/mav_flight_stability.json"
    subprocess.run(
        [
            "python",
            "scripts/diagnose_mav_flight_stability.py",
            "--steps",
            "20",
            "--blue-policy",
            "zero",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "records" in data
    summary = data["summary"]
    assert "zero_action_mav_stable" in summary
    assert "recommended_next_actions" in summary
    assert data["records"]
    for record in data["records"]:
        for key in [
            "case",
            "trim_enabled",
            "mav_action_trim_applied",
            "effective_mav_action",
            "mav_initial_altitude_m",
            "mav_final_altitude_m",
            "mav_min_altitude_m",
            "mav_alive_final",
            "nan_detected",
        ]:
            assert key in record
        assert record["nan_detected"] is False


def test_aircraft_level_hold_comparison_help_and_smoke():
    help_result = subprocess.run(
        ["python", "scripts/diagnose_aircraft_level_hold_comparison.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--output-json" in help_result.stdout

    output_json = ROOT / "outputs/test_environment_audit/aircraft_level_hold_comparison.json"
    subprocess.run(
        [
            "python",
            "scripts/diagnose_aircraft_level_hold_comparison.py",
            "--duration",
            "2",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert "records" in data
    assert "conclusion" in data


def test_mav_flight_stability_doc_exists():
    doc = ROOT / "docs/mav_flight_stability_audit.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for phrase in [
        "red-policy zero",
        "not caused by untrained RL",
        "A-4",
        "pitch bias",
        "not training",
    ]:
        assert phrase in text
