from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_oracle_pretrained_closed_loop_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_oracle_pretrained_closed_loop.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--checkpoint" in result.stdout
    assert "--mav-safe-fixed" in result.stdout
    assert "--stochastic" in result.stdout

