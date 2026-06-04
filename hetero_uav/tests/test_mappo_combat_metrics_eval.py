"""Tests for MAPPO combat-metrics evaluation outputs."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"
POSTHOC_SCRIPT = ROOT / "scripts" / "evaluate_saved_mappo_with_combat_metrics.py"
LONGRUN_SCRIPT = ROOT / "scripts" / "run_mappo_balanced_baseline_longrun.py"
MODEL = ROOT / "outputs" / "test_mappo_longrun_baseline" / "seed_0" / "latest" / "model.pt"
OUT_DIR = ROOT / "outputs" / "test_mappo_combat_metrics"
CONFIG_3V3 = "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml"

REQUIRED_FIELDS = {
    "red_win_rate",
    "blue_win_rate",
    "draw_rate",
    "timeout_rate",
    "mav_survival_rate",
    "episode_end_reason_counts",
    "winner_counts",
}


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _ensure_model():
    if MODEL.exists():
        return
    result = subprocess.run(
        [
            "python", str(LONGRUN_SCRIPT),
            "--seeds", "0",
            "--total-env-steps", "32",
            "--rollout-length", "8",
            "--max-steps", "16",
            "--eval-episodes", "1",
            "--device", "cpu",
            "--opponent-policy", "rule_nearest",
            "--output-dir", "outputs/test_mappo_longrun_baseline",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=300,
        env=_env(),
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )


def test_eval_help_has_summary_json():
    result = subprocess.run(
        ["python", str(EVAL_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=20,
        env=_env(),
    )
    assert result.returncode == 0
    assert "--summary-json" in result.stdout


def test_eval_zero_shot_summary_contains_combat_metrics():
    _ensure_model()
    summary_json = OUT_DIR / "eval_summary.json"
    result = subprocess.run(
        [
            "python", str(EVAL_SCRIPT),
            "--model", str(MODEL.relative_to(ROOT)),
            "--obs-adapter-version", "v2",
            "--episodes", "1",
            "--device", "cpu",
            "--opponent-policy", "rule_nearest",
            "--configs", CONFIG_3V3,
            "--summary-json", str(summary_json.relative_to(ROOT)),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=300,
        env=_env(),
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert data
    assert REQUIRED_FIELDS.issubset(data[0])


def test_posthoc_help():
    result = subprocess.run(
        ["python", str(POSTHOC_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=20,
        env=_env(),
    )
    assert result.returncode == 0
    for flag in ("--model", "--episodes", "--output-json", "--output-csv"):
        assert flag in result.stdout


def test_posthoc_eval_outputs_json_and_csv():
    _ensure_model()
    output_json = OUT_DIR / "posthoc.json"
    output_csv = OUT_DIR / "posthoc.csv"
    result = subprocess.run(
        [
            "python", str(POSTHOC_SCRIPT),
            "--model", str(MODEL.relative_to(ROOT)),
            "--episodes", "1",
            "--device", "cpu",
            "--opponent-policy", "rule_nearest",
            "--output-json", str(output_json.relative_to(ROOT)),
            "--output-csv", str(output_csv.relative_to(ROOT)),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=600,
        env=_env(),
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert output_json.exists()
    assert output_csv.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data
    assert REQUIRED_FIELDS.issubset(data[0])
    with output_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert REQUIRED_FIELDS.issubset(rows[0])
