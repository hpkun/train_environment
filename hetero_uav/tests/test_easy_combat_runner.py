from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


def test_easy_combat_runner_help():
    result = _run(["scripts/run_happo_easy_combat_100k.py", "--help"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dry-run" in result.stdout


def test_easy_combat_runner_dry_run_prints_train_and_eval():
    result = _run(["scripts/run_happo_easy_combat_100k.py", "--dry-run"])
    assert result.returncode == 0, result.stdout + result.stderr
    text = result.stdout
    assert "[train]" in text
    assert "[fast_eval]" in text
    assert "hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml" in text
    assert "--total-env-steps 100000" in text
    assert "--device cuda" in text
