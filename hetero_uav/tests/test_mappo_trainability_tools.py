"""Test trainability diagnostic tools. No HAPPO, no attention, no GRU."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
CONFIG = "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml"


def test_train_cli_accepts_log_csv():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=10)
    assert "--log-csv" in result.stdout
    assert "--save-interval" in result.stdout
    assert "--eval-interval" in result.stdout


def test_train_smoke_with_log_csv():
    out_dir = "outputs/test_train_log"
    log_csv = f"{out_dir}/train_log.csv"
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT),
         "--config", CONFIG, "--iterations", "1", "--rollout-length", "8",
         "--opponent-policy", "rule_nearest",
         "--log-csv", log_csv, "--output-dir", out_dir,
         "--device", "cpu", "--debug"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert Path(log_csv).exists()


def test_log_csv_has_required_columns():
    log_csv = "outputs/test_train_log/train_log.csv"
    assert Path(log_csv).exists()
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    for col in ["iteration", "average_team_return", "average_red_alive",
                "average_blue_alive", "actor_loss", "critic_loss",
                "entropy", "action_mean_abs", "nan_detected"]:
        assert col in rows[0], f"Missing column: {col}"


def test_model_saved():
    model_path = "outputs/test_train_log/latest/model.pt"
    assert Path(model_path).exists()


def test_diagnose_trainability_runs():
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "diagnose_mappo_trainability.py"),
         "--iterations", "2", "--rollout-length", "8", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "trainability smoke: PASSED" in result.stdout


def test_diagnose_eval_runs():
    model_path = "outputs/mappo_trainability/latest/model.pt"
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "diagnose_mappo_trainability_eval.py"),
         "--model", model_path, "--episodes", "1", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    # Should have output for at least 2 configs
    assert result.stdout.count("ret=") >= 2


def test_no_nan_in_log():
    log_csv = "outputs/mappo_trainability/train_log.csv"
    if Path(log_csv).exists():
        with open(log_csv) as f:
            for row in csv.DictReader(f):
                assert int(row["nan_detected"]) == 0
