"""Test visibility/geometry audit. No training, no env changes."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_hetero_visibility_geometry.py"


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=10)
    for flag in ["--steps", "--red-policy", "--blue-policy", "--output-json"]:
        assert flag in result.stdout


def test_short_run():
    out = "outputs/test_environment_audit/hetero_visibility_geometry.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--steps", "3", "--red-policy", "zero", "--blue-policy", "greedy_fsm",
         "--output-json", out],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=120,
        env=_subprocess_env())
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert Path(out).exists()
    data = json.loads(Path(out).read_text())
    assert "records" in data
    assert "summary" in data
    for r in data["records"]:
        for key in ["config", "red_observed_any", "blue_observed_any",
                    "first_step_red_observed", "first_step_blue_observed",
                    "red_mav_shared_fraction", "blue_direct_fraction",
                    "warnings"]:
            assert key in r, f"Missing {key}"
        assert not any(np.isnan(v) if isinstance(v, float) else False
                       for v in r.values() if isinstance(v, (int, float)))


def test_doc_exists():
    doc = ROOT / "docs" / "hetero_visibility_geometry_audit.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "MAV shared" in text
    assert "blue" in text.lower()
    assert "initial geometry" in text.lower()
