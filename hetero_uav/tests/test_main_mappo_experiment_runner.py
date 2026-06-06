"""Smoke-test the main experiment runner.  No training claim."""
from __future__ import annotations

import json
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
                capture_output=True,
                timeout=15,
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


def test_main_experiment_short_smoke():
    output_dir = "outputs/test_main_mappo_experiment"
    result = subprocess.run(
        [
            PYTHON,
            "scripts/run_main_mappo_experiment.py",
            "--total-env-steps", "64",
            "--rollout-length", "16",
            "--max-steps", "64",
            "--eval-episodes", "1",
            "--device", "cpu",
            "--output-dir", output_dir,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert result.returncode == 0, (
        f"runner failed:\nstdout={result.stdout[-800:]}\nstderr={result.stderr[-800:]}"
    )

    out = ROOT / output_dir
    for fname in ["train_log.csv", "eval_summary.json",
                  "main_experiment_summary.json", "main_experiment_summary.csv"]:
        assert (out / fname).exists(), f"missing {out / fname}"

    summary = json.loads((out / "main_experiment_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary, list) and len(summary) > 0
    for rec in summary:
        assert rec["opponent_policy"] == "greedy_fsm"
        assert rec["obs_adapter_version"] == "v2"
        assert rec["actor_dim"] == 96
        assert rec["critic_dim"] == 480
        assert rec["nan_detected"] is False
        assert rec["actor_dim_ok"] is True
        assert rec["critic_dim_ok"] is True
        for k in ["avg_return", "red_win_rate", "blue_win_rate", "draw_rate",
                  "timeout_rate", "mav_survival_rate"]:
            assert k in rec, f"missing {k}"
