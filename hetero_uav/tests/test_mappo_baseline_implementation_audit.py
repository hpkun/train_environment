"""Tests for the MAPPO baseline implementation audit script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mappo_baseline_audit_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_mappo_baseline_implementation.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mappo_baseline_audit_outputs_expected_modules(tmp_path):
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_mappo_baseline_implementation.py",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in [
        "summary",
        "network_architecture",
        "action_distribution_and_logprob",
        "rollout_mask_logic",
        "done_and_gae_logic",
        "reward_and_value_target_logic",
        "centralized_critic_input",
        "advantage_and_ppo_update",
        "multiagent_parameter_sharing",
        "evaluation_consistency",
        "paper_alignment",
    ]:
        assert key in data

    md = out_md.read_text(encoding="utf-8")
    for token in [
        "MAPPO baseline",
        "BRMA-MAPPO",
        "TAM-HAPPO",
        "alive mask",
        "team done",
        "clipped Gaussian",
        "not the final method",
    ]:
        assert token in md
