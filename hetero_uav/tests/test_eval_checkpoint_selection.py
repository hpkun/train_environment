from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python"


def test_eval_checkpoint_scores_select_different_configs():
    from eval_checkpoint_selection import compute_eval_scores, best_metric_name

    records = [
        {
            "config": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "red_win_rate": 0.1,
            "red_missile_hits_mean": 0.0,
            "blue_dead_mean": 0.0,
        },
        {
            "config": "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
            "red_win_rate": 0.0,
            "red_missile_hits_mean": 1.5,
            "blue_dead_mean": 1.0,
        },
    ]
    scores = compute_eval_scores(records)
    assert scores["score_3v2"] == 0.1
    assert scores["score_5v4"] == 0.5
    assert scores["score_combined"] == 0.3
    assert best_metric_name("best_5v4") == "score_5v4"


def test_eval_checkpoint_meta_contains_step_and_metrics(tmp_path):
    from eval_checkpoint_selection import build_eval_checkpoint_meta

    records = [
        {"config": "3v2.yaml", "red_win_rate": 0.2, "red_missile_hits_mean": 1.0, "blue_dead_mean": 1.0},
        {"config": "5v4.yaml", "red_win_rate": 0.4, "red_missile_hits_mean": 0.5, "blue_dead_mean": 0.5},
    ]
    meta = build_eval_checkpoint_meta(
        step=50000,
        iteration=10,
        policy_arch="brma_recurrent_masked",
        records=records,
        extra={"random_scale_mask": True},
    )
    assert meta["step"] == 50000
    assert meta["policy_arch"] == "brma_recurrent_masked"
    assert meta["scores"]["score_3v2"] == pytest.approx(0.6)
    assert meta["scores"]["score_5v4"] == pytest.approx(0.6)
    assert meta["scores"]["score_combined"] == pytest.approx(0.6)
    assert meta["metrics"]["3v2"]["red_win_rate"] == 0.2
    assert meta["random_scale_mask"] is True

    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["5v4"]["blue_dead_mean"] == 0.5


def test_summarize_eval_checkpoints_help_runs():
    result = subprocess.run(
        [PYTHON, "scripts/summarize_eval_checkpoints.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--output-dir" in result.stdout
