from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_happo_oracle_pretrain_runner_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/run_happo_oracle_pretrain_finetune_200k.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dry-run" in result.stdout


def test_happo_oracle_pretrain_runner_dry_run_prints_three_steps():
    result = subprocess.run(
        [sys.executable, "scripts/run_happo_oracle_pretrain_finetune_200k.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stdout = result.stdout.lower()
    assert "collect" in stdout
    assert "pretrain" in stdout
    assert "finetune" in stdout
    assert "train_happo_reference.py" in stdout
