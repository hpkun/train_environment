"""Tests for MAPPO balanced baseline long-run tooling."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_mappo_baseline.py"
LONGRUN_SCRIPT = ROOT / "scripts" / "run_mappo_balanced_baseline_longrun.py"
OUT_DIR = ROOT / "outputs" / "test_mappo_longrun_baseline"


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_smoke():
    report = OUT_DIR / "longrun_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("status") == "passed" and data.get("total_env_steps") == 32:
            return

    result = subprocess.run(
        [
            "python",
            str(LONGRUN_SCRIPT),
            "--seeds",
            "0",
            "--total-env-steps",
            "32",
            "--rollout-length",
            "8",
            "--max-steps",
            "16",
            "--eval-episodes",
            "1",
            "--device",
            "cpu",
            "--opponent-policy",
            "rule_nearest",
            "--output-dir",
            str(OUT_DIR),
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


def test_train_help_contains_total_env_steps():
    result = subprocess.run(
        ["python", str(TRAIN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=20,
        env=_env(),
    )
    assert result.returncode == 0
    assert "--total-env-steps" in result.stdout


def test_longrun_help():
    result = subprocess.run(
        ["python", str(LONGRUN_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=20,
        env=_env(),
    )
    assert result.returncode == 0
    for flag in ("--total-env-steps", "--rollout-length",
                 "--eval-episodes", "--output-dir"):
        assert flag in result.stdout


def test_longrun_smoke_outputs():
    _run_smoke()
    assert (OUT_DIR / "longrun_train_summary.csv").exists()
    assert (OUT_DIR / "longrun_eval_summary.csv").exists()
    assert (OUT_DIR / "longrun_report.json").exists()
    assert (OUT_DIR / "seed_0" / "latest" / "model.pt").exists()
    assert (OUT_DIR / "seed_0" / "latest" / "meta.json").exists()


def test_longrun_report():
    _run_smoke()
    data = json.loads((OUT_DIR / "longrun_report.json").read_text(
        encoding="utf-8"
    ))
    assert data["status"] == "passed"
    assert data["total_env_steps"] == 32
    assert data["obs_adapter_version"] == "v2"
    assert data["actor_dim"] == 96
    assert data["critic_dim"] == 480
    assert data["actor_arch"] == "mlp"
    assert data["any_train_nan"] is False
    assert data["any_eval_nan"] is False
    assert data["all_actor_dim_ok"] is True
    assert data["all_critic_dim_ok"] is True
    assert data["min_total_env_steps_actual"] >= 32


def test_longrun_meta():
    _run_smoke()
    meta = json.loads((OUT_DIR / "seed_0" / "latest" / "meta.json").read_text(
        encoding="utf-8"
    ))
    assert meta["total_env_steps_target"] == 32
    assert meta["total_env_steps_actual"] >= 32
    assert meta["obs_adapter_version"] == "v2"
    assert meta["actor_obs_dim"] == 96
    assert meta["critic_state_dim"] == 480


def test_longrun_csvs_are_clean():
    _run_smoke()
    with (OUT_DIR / "longrun_train_summary.csv").open(encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    assert train_rows
    assert all(int(float(row["nan_detected"])) == 0 for row in train_rows)

    with (OUT_DIR / "longrun_eval_summary.csv").open(encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    assert eval_rows
    assert all(row["nan_detected"] == "False" for row in eval_rows)
    assert all(row["actor_dim_ok"] == "True" for row in eval_rows)
    assert all(row["critic_dim_ok"] == "True" for row in eval_rows)


def test_longrun_doc_exists():
    doc = ROOT / "docs" / "mappo_balanced_baseline_longrun.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "500k" in text
    assert "env steps" in text
    assert "not a formal zero-shot claim" in text
    assert "balanced 3v3" in text
    assert "balanced 4v4" in text
