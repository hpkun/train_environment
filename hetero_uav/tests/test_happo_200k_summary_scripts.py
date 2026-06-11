import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def test_summary_script_help_runs():
    result = _run(["scripts/summarize_happo_3v2_reference_200k.py", "--help"])
    assert result.returncode == 0
    assert "Summarize HAPPO 3v2 reference" in result.stdout


def test_checkpoint_eval_script_help_runs():
    result = _run(["scripts/evaluate_happo_3v2_reference_checkpoints.py", "--help"])
    assert result.returncode == 0
    assert "Evaluate HAPPO 3v2 reference checkpoints" in result.stdout


def test_summary_missing_output_dir_exits_cleanly(tmp_path):
    result = _run(
        [
            "scripts/summarize_happo_3v2_reference_200k.py",
            "--output-dir",
            str(tmp_path / "missing"),
        ]
    )
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)


def test_checkpoint_missing_output_dir_exits_cleanly(tmp_path):
    result = _run(
        [
            "scripts/evaluate_happo_3v2_reference_checkpoints.py",
            "--experiment-dir",
            str(tmp_path / "missing"),
            "--episodes",
            "1",
        ]
    )
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)
