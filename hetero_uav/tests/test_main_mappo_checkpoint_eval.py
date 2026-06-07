"""Test checkpoint evaluation script. No full eval in tests."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    candidates = [sys.executable]
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import gymnasium"],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


PYTHON = _find_python()


def test_help_runs():
    result = subprocess.run(
        [PYTHON, "scripts/evaluate_main_mappo_checkpoints.py", "--help"],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_dir_gives_clear_error():
    """Script should exit cleanly (not crash) when experiment dir is missing."""
    result = subprocess.run(
        [
            PYTHON, "scripts/evaluate_main_mappo_checkpoints.py",
            "--experiment-dir", "outputs/_nonexistent_dir_for_test",
            "--eval-episodes", "1", "--device", "cpu",
        ],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    # Should exit with error but not a Python traceback
    assert result.returncode != 0
    assert "no checkpoints" in (result.stdout + result.stderr).lower()


def test_doc_exists():
    doc = ROOT / "docs/main_mappo_checkpoint_selection.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for token in ["checkpoint", "latest", "not a method module"]:
        assert token in text, f"missing '{token}' in doc"
