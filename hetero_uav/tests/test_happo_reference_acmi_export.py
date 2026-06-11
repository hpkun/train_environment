import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_happo_reference_acmi_export_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/export_happo_reference_acmi.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "Export HAPPO reference checkpoint" in result.stdout


def test_happo_reference_acmi_export_missing_checkpoint_exits_cleanly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_happo_reference_acmi.py",
            "--experiment-dir",
            str(tmp_path / "missing"),
            "--checkpoint",
            "best",
            "--output",
            str(tmp_path / "out.acmi"),
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
