from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_role_v1_reward_effects.py"
ROLE_DIR = ROOT / "outputs" / "main_mappo_experiment_f22_50k_role_v1"
LEGACY_DIR = ROOT / "outputs" / "main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix"


def test_audit_role_v1_reward_effects_help_runs():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "role_v1" in result.stdout
    assert "--role-dir" in result.stdout
    assert "--legacy-dir" in result.stdout


def test_audit_role_v1_reward_effects_outputs_json_and_markdown():
    if not ROLE_DIR.exists() or not LEGACY_DIR.exists():
        return

    out_json = ROOT / "outputs" / "test_reward_audit" / "role_v1_reward_effects_audit.json"
    out_md = ROOT / "outputs" / "test_reward_audit" / "role_v1_reward_effects_audit.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--role-dir",
            str(ROLE_DIR.relative_to(ROOT)),
            "--legacy-dir",
            str(LEGACY_DIR.relative_to(ROOT)),
            "--output-json",
            str(out_json.relative_to(ROOT)),
            "--output-md",
            str(out_md.relative_to(ROOT)),
            "--component-steps",
            "5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "role_v1_weaker_than_legacy" in result.stdout
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "comparison_summary" in data
    assert "role_component_audit" in data
    assert "paper_alignment" in data
    assert "recommended_changes" in data
    assert isinstance(data["recommended_changes"], list)
    assert len(data["recommended_changes"]) <= 5

    md = out_md.read_text(encoding="utf-8")
    for token in ("brma_legacy", "role_v1", "MAV", "UAV", "support", "reward"):
        assert token in md
