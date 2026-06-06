"""Test blue opponent protocol validation script. No training."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    """Find a Python executable that can import gymnasium."""
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


def test_validation_script_help_lists_expected_args():
    result = subprocess.run(
        [PYTHON, "scripts/validate_blue_opponent_protocol.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in [
        "--blue-opponents",
        "--red-policies",
        "--steps",
        "--episodes",
        "--output-json",
        "--output-md",
    ]:
        assert token in result.stdout, f"missing {token} in --help"


def test_validation_script_short_run():
    output_json = "outputs/test_environment_audit/blue_opponent_validation.json"
    output_md = "outputs/test_environment_audit/blue_opponent_validation.md"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/validate_blue_opponent_protocol.py",
            "--configs",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "--blue-opponents",
            "rule_nearest",
            "--red-policies",
            "zero",
            "--steps",
            "5",
            "--episodes",
            "1",
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
        timeout=120,
    )
    assert result.returncode == 0, (
        f"validation failed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists(), f"JSON missing: {json_path}"
    assert md_path.exists(), f"MD missing: {md_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "records" in data
    assert "summary" in data
    assert len(data["records"]) >= 1

    rec = data["records"][0]
    for key in [
        "blue_opponent_policy",
        "red_policy",
        "nan_detected",
        "action_min",
        "action_max",
        "mav_survival_rate",
        "opponent_difficulty_label",
    ]:
        assert key in rec, f"missing key {key} in record"

    assert rec["blue_opponent_policy"] == "rule_nearest"
    assert rec["red_policy"] == "zero"
    assert rec["nan_detected"] is False
    assert -1.0 <= rec["action_min"] <= 1.0
    assert -1.0 <= rec["action_max"] <= 1.0

    md = md_path.read_text(encoding="utf-8")
    for token in [
        "rule_nearest",
        "greedy_fsm",
        "not a method module",
        "not a training run",
    ]:
        assert token in md, f"missing '{token}' in markdown"


def test_validation_doc_exists_and_has_required_content():
    doc_path = ROOT / "docs" / "blue_opponent_protocol_validation.md"
    assert doc_path.exists(), f"missing doc: {doc_path}"
    text = doc_path.read_text(encoding="utf-8")
    for token in [
        "rule_nearest",
        "greedy_fsm",
        "not a method module",
        "not a training run",
    ]:
        assert token in text, f"missing '{token}' in doc"
