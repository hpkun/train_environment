from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 60):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_runtime_speedup_tool_help_and_dry_runs():
    commands = [
        ["scripts/run_oracle_pretrain_fast_check.py", "--dry-run"],
        ["scripts/profile_runtime_hotspots.py", "--help"],
        ["scripts/run_happo_oracle_pretrain_finetune_200k.py", "--dry-run"],
        ["scripts/collect_direct_chase_oracle_dataset.py", "--help"],
        ["scripts/pretrain_uav_actor_from_oracle.py", "--help"],
    ]
    for cmd in commands:
        result = _run(cmd)
        assert result.returncode == 0, result.stdout + result.stderr


def test_oracle_runner_dry_run_supports_skip_existing_and_force():
    result = _run([
        "scripts/run_happo_oracle_pretrain_finetune_200k.py",
        "--dry-run",
        "--skip-existing",
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout.lower()
    assert "collect" in text or "skip collect" in text
    assert "pretrain" in text or "skip pretrain" in text
    assert "finetune" in text or "skip finetune" in text

    help_result = _run(["scripts/run_happo_oracle_pretrain_finetune_200k.py", "--help"])
    assert "--skip-existing" in help_result.stdout
    assert "--force" in help_result.stdout


def test_eval_fast_help_and_pretrain_speed_args():
    eval_help = _run(["scripts/evaluate_happo_3v2_reference_checkpoints.py", "--help"])
    assert eval_help.returncode == 0, eval_help.stdout + eval_help.stderr
    assert "--fast" in eval_help.stdout

    pretrain_help = _run(["scripts/pretrain_uav_actor_from_oracle.py", "--help"])
    assert pretrain_help.returncode == 0, pretrain_help.stdout + pretrain_help.stderr
    for token in [
        "--val-ratio",
        "--max-train-samples",
        "--early-stop-patience",
        "--target-val-loss",
    ]:
        assert token in pretrain_help.stdout

