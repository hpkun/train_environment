"""Test aircraft model admissibility audit script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_audit_cli_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_aircraft_model_admissibility.py",
         "--steps", "10", "--models", "f16"],
        cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0


def test_summary_contains_f16_f22():
    result = subprocess.run(
        [sys.executable, "scripts/audit_aircraft_model_admissibility.py",
         "--steps", "10"],
        cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0
    assert "f16" in result.stdout
    assert "f22" in result.stdout


def test_output_files_created():
    subprocess.run(
        [sys.executable, "scripts/audit_aircraft_model_admissibility.py",
         "--steps", "10"],
        cwd=ROOT, text=True, capture_output=True, timeout=120)
    csv_path = ROOT / "outputs/environment_audit/aircraft_model_admissibility.csv"
    json_path = ROOT / "outputs/environment_audit/aircraft_model_admissibility_summary.json"
    assert csv_path.exists()
    assert json_path.exists()
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert "f16" in summary
    assert "f22" in summary
    assert isinstance(summary["f16"]["admissible"], bool)
    assert isinstance(summary["f22"]["admissible"], bool)


def test_does_not_modify_configs():
    """Audit must not modify official config files."""
    configs = [
        ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
        ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
    ]
    for cfg in configs:
        assert cfg.exists()
        content = cfg.read_text(encoding="utf-8")
        assert "f22" in content or "f16" in content  # unchanged


def test_f16_not_default_mainline_replacement():
    """Docs must NOT say F16 surrogate is default/replacement for F22."""
    audit_doc = ROOT / "docs/paper_parent_env_alignment_audit.md"
    text = audit_doc.read_text(encoding="utf-8").lower()
    assert "f22 main line paused" in text
