from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    candidates = [sys.executable]
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import gymnasium"],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


PYTHON = _find_python()


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_f22_mav_control_response_audit_help_runs():
    result = subprocess.run(
        [PYTHON, "scripts/audit_f22_mav_control_response.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--steps" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout


def test_f22_mav_control_response_short_audit_outputs_expected_fields():
    output_json = "outputs/test_environment_audit/f22_control_response.json"
    output_md = "outputs/test_environment_audit/f22_control_response.md"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/audit_f22_mav_control_response.py",
            "--steps",
            "5",
            "--output-json",
            output_json,
            "--output-md",
            output_md,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in [
        "resource_status",
        "control_response",
        "warnings",
        "blocking_issues",
        "next_action",
    ]:
        assert key in data

    for scenario in ["level", "climb", "speed_up"]:
        assert scenario in data["control_response"]
        record = data["control_response"][scenario]
        assert "altitude_delta_m" in record
        assert "speed_delta_mps" in record
        assert "heading_delta_rad" in record
        assert "nan_detected" in record

    md = md_path.read_text(encoding="utf-8")
    for token in [
        "F-22",
        "control response",
        "action_trim_by_role",
        "missile audit should wait",
    ]:
        assert token in md
