"""Test F-22 action path check script. No training."""
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
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


PYTHON = _find_python()


def test_help_runs():
    result = subprocess.run(
        [PYTHON, "scripts/check_f22_action_path.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_short_run_produces_expected_output():
    output_json = "outputs/test_environment_audit/f22_action_path_check.json"
    output_md = "outputs/test_environment_audit/f22_action_path_check.md"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/check_f22_action_path.py",
            "--steps", "2",
            "--output-json", output_json,
            "--output-md", output_md,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, (
        f"failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "scenarios" in data
    assert "summary" in data
    assert "recommendation" in data["summary"]
    assert "trim_pitch_value" in data["summary"]

    md = md_path.read_text(encoding="utf-8")
    for token in ["F-22", "Action Path", "trim", "missile audit"]:
        assert token in md, f"missing '{token}' in markdown"


def test_doc_exists():
    doc = ROOT / "docs/f22_action_path_check.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for token in ["F-22", "MAV", "control properties", "missile audit"]:
        assert token in text, f"missing '{token}' in doc"
