"""Tests for MAPPO baseline environment stability validation."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_mappo_baseline_environment_stability.py"
OUT_DIR = ROOT / "outputs" / "test_mappo_env_stability"
BALANCED_TRAIN = "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml"
BALANCED_EVAL = {
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
}


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_minimal_validation():
    report = OUT_DIR / "stability_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        if (data.get("status") == "passed"
                and data.get("train_config") == BALANCED_TRAIN
                and set(data.get("eval_configs", [])) == BALANCED_EVAL):
            return

    result = subprocess.run(
        [
            "python",
            str(SCRIPT),
            "--seeds",
            "0",
            "--iterations",
            "1",
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


def test_validate_script_help():
    result = subprocess.run(
        ["python", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=20,
        env=_env(),
    )
    assert result.returncode == 0
    for flag in (
        "--seeds",
        "--iterations",
        "--rollout-length",
        "--max-steps",
        "--eval-episodes",
        "--output-dir",
    ):
        assert flag in result.stdout


def test_minimal_stability_validation_outputs():
    _run_minimal_validation()
    assert (OUT_DIR / "stability_train_summary.csv").exists()
    assert (OUT_DIR / "stability_eval_summary.csv").exists()
    assert (OUT_DIR / "stability_report.json").exists()


def test_stability_report_passed():
    _run_minimal_validation()
    data = json.loads((OUT_DIR / "stability_report.json").read_text(
        encoding="utf-8"
    ))
    assert data["status"] == "passed"
    assert data["obs_adapter_version"] == "v2"
    assert data["actor_dim"] == 96
    assert data["critic_dim"] == 480
    assert data["train_config"] == BALANCED_TRAIN
    assert set(data["eval_configs"]) == BALANCED_EVAL
    assert data["any_train_nan"] is False
    assert data["any_eval_nan"] is False
    assert data["all_actor_dim_ok"] is True
    assert data["all_critic_dim_ok"] is True


def test_stability_train_summary():
    _run_minimal_validation()
    with (OUT_DIR / "stability_train_summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        assert int(float(row["nan_detected"])) == 0
        assert Path(row["model_path"]).exists()
        assert Path(row["log_csv"]).exists()


def test_stability_eval_summary():
    _run_minimal_validation()
    with (OUT_DIR / "stability_eval_summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        assert row["nan_detected"] == "False"
        assert row["actor_dim_ok"] == "True"
        assert row["critic_dim_ok"] == "True"


def test_stability_doc_exists():
    doc = ROOT / "docs" / "mappo_baseline_environment_stability.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "MAPPO baseline" in text
    assert "environment stability" in text
    assert "not a formal zero-shot experiment" in text
    assert "actor_obs_dim=96" in text
    assert "critic_state_dim=480" in text
