import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_policy_mode_eval_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_happo_policy_modes.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "policy modes" in result.stdout
    assert "--modes" in result.stdout


def test_policy_mode_eval_missing_checkpoint_exits_cleanly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_happo_policy_modes.py",
            "--output-dir",
            str(tmp_path / "missing"),
            "--episodes",
            "1",
            "--modes",
            "deterministic",
            "stochastic",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in (result.stderr + result.stdout).lower()
