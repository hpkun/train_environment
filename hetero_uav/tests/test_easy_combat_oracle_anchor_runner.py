from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_easy_combat_oracle_anchor_runner_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_happo_easy_combat_oracle_anchor_50k.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--uav-imitation-dataset" in result.stdout
    assert "--uav-imitation-coef" in result.stdout
    assert "outputs/happo_easy_combat_oracle_anchor_50k" in result.stdout

