from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_train_happo_help_exposes_stall_watchdog_flags():
    result = _run(["scripts/train_happo_reference.py", "--help"])
    assert result.returncode == 0
    assert "--heartbeat-stall-timeout-sec" in result.stdout
    assert "--exit-on-heartbeat-stall" in result.stdout


def test_run_4env_stability_debug_dry_run():
    result = _run(["scripts/run_4env_stability_debug.py", "--dry-run"])
    assert result.returncode == 0
    assert "--num-envs 4" in result.stdout
    assert "--max-steps 1000" in result.stdout
    assert "--debug-rollout-heartbeat" in result.stdout
    assert "outputs/debug_4env_max1000_500k" in result.stdout


def test_run_4env_reset_frequency_compare_dry_run():
    result = _run(["scripts/run_4env_reset_frequency_compare.py", "--dry-run"])
    assert result.returncode == 0
    assert "outputs/debug_4env_max64_100k" in result.stdout
    assert "outputs/debug_4env_max1000_500k" in result.stdout
    assert "outputs/debug_2env_max1000_500k" in result.stdout


def test_analyze_heartbeat_stall_help_runs():
    result = _run(["scripts/analyze_heartbeat_stall.py", "--help"])
    assert result.returncode == 0
    assert "--output-dir" in result.stdout
