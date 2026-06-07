"""Test training-time logging, periodic eval, and best checkpoint."""
from __future__ import annotations

import csv
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


def test_train_help_includes_eval_args():
    result = subprocess.run(
        [PYTHON, "scripts/train_mappo_baseline.py", "--help"],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ["--eval-during-training", "--eval-interval-steps", "--train-eval-episodes"]:
        assert token in result.stdout, f"missing {token}"


def test_runner_help_includes_eval_args():
    result = subprocess.run(
        [PYTHON, "scripts/run_main_mappo_experiment.py", "--help"],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for token in ["--eval-during-training", "--eval-interval-steps", "--train-eval-episodes"]:
        assert token in result.stdout, f"missing {token}"


def test_smoke_online_eval():
    output_dir = "outputs/test_main_mappo_experiment_online_eval"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/run_main_mappo_experiment.py",
            "--total-env-steps", "64",
            "--rollout-length", "16",
            "--max-steps", "64",
            "--eval-episodes", "1",
            "--device", "cpu",
            "--opponent-policy", "rule_nearest",
            "--eval-during-training",
            "--eval-interval-steps", "32",
            "--train-eval-episodes", "1",
            "--output-dir", output_dir,
        ],
        cwd=ROOT, env=_env(),
        text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=600,
    )
    assert result.returncode == 0, (
        f"runner failed:\nstdout={result.stdout[-800:]}\nstderr={result.stderr[-800:]}"
    )

    out = ROOT / output_dir
    train_csv_path = out / "train_log.csv"
    assert train_csv_path.exists(), f"missing {train_csv_path}"

    with open(train_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for field in ["action_saturation_rate", "train_red_win_rate_recent", "train_mav_survival_rate_recent"]:
            assert field in header, f"missing field '{field}' in train_log.csv"
        for row in reader:
            assert int(row.get("nan_detected", "1")) == 0

    eval_log_path = out / "eval_log.csv"
    assert eval_log_path.exists(), f"missing {eval_log_path}"
