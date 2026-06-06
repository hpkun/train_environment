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
    candidates = []
    candidates.append(sys.executable)
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


def test_reward_termination_audit_help_lists_expected_args():
    result = subprocess.run(
        [_find_python(), "scripts/audit_reward_termination_hetero.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout
    for token in [
        "--configs",
        "--steps",
        "--output-json",
        "--output-md",
        "--opponent-policy",
    ]:
        assert token in help_text


def test_reward_termination_audit_short_rollout_outputs_json_and_markdown():
    output_json = "outputs/test_environment_audit/reward_termination_audit.json"
    output_md = "outputs/test_environment_audit/reward_termination_audit.md"
    result = subprocess.run(
        [
            _find_python(),
            "scripts/audit_reward_termination_hetero.py",
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
    assert "records" in data
    assert "summary" in data
    assert "warnings" in data
    assert "hetero_role_assessment" in data
    assert data["records"]
    for record in data["records"]:
        for key in [
            "config",
            "static_config",
            "reward_components_seen",
            "termination_behavior",
            "hetero_role_assessment",
            "warnings",
        ]:
            assert key in record

    md = md_path.read_text(encoding="utf-8")
    for token in ["reward", "termination", "MAV", "UAV", "not modifying reward"]:
        assert token in md
