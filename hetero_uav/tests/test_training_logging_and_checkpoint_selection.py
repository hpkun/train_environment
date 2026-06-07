"""Tests for main experiment logging and checkpoint selection tools."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_main_experiment_help_has_core_arguments():
    text = _run_help("scripts/run_main_mappo_experiment.py")
    for token in [
        "--total-env-steps",
        "--rollout-length",
        "--eval-episodes",
        "--output-dir",
    ]:
        assert token in text


def test_train_baseline_help_has_console_log_interval():
    text = _run_help("scripts/train_mappo_baseline.py")
    assert "--console-log-interval" in text


def test_checkpoint_eval_help_has_fast_selection_arguments():
    text = _run_help("scripts/evaluate_main_mappo_checkpoints.py")
    for token in [
        "--selection-mode",
        "--max-checkpoints",
        "--quick-eval-episodes",
        "--resume",
    ]:
        assert token in text
