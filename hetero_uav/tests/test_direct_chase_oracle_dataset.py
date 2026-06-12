from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_collect_direct_chase_oracle_dataset_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/collect_direct_chase_oracle_dataset.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--episodes" in result.stdout
    assert "--output" in result.stdout


def test_collect_direct_chase_oracle_dataset_one_episode_outputs_npz_and_summary():
    pytest.importorskip("gymnasium")
    pytest.importorskip("jsbsim")
    out = ROOT / "outputs/test_oracle_dataset/direct_chase_oracle_3v2.npz"
    summary = ROOT / "outputs/test_oracle_dataset/direct_chase_oracle_3v2_summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_direct_chase_oracle_dataset.py",
            "--episodes",
            "1",
            "--max-steps",
            "20",
            "--output",
            str(out.relative_to(ROOT)),
            "--summary-json",
            str(summary.relative_to(ROOT)),
            "--opponent-policy",
            "zero",
            "--max-samples",
            "200",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists()
    assert summary.exists()

    data = np.load(out, allow_pickle=True)
    assert "actor_obs" in data
    assert "oracle_action" in data
    assert data["actor_obs"].shape[-1] == 96
    assert data["oracle_action"].shape[-1] == 3

    meta = json.loads(summary.read_text(encoding="utf-8"))
    for key in [
        "num_samples",
        "episodes",
        "red_missiles_fired_mean",
        "red_missile_hits_mean",
        "blue_dead_mean",
        "launch_range_rate",
        "launch_angle_rate",
        "launch_envelope_rate",
    ]:
        assert key in meta
