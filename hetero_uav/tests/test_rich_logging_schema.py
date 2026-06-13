from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_logging_schema_importable() -> None:
    schema = importlib.import_module("scripts.experiment_logging_schema")
    assert "train_metrics.csv" in schema.FILE_SCHEMAS
    assert "eval_episode_metrics.csv" in schema.FILE_SCHEMAS
    assert "aircraft_timeseries.csv" in schema.FILE_SCHEMAS
    assert "attention_metrics.csv" in schema.FILE_SCHEMAS


def test_run_rich_logging_smoke_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_rich_logging_smoke.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--enable-rich-logging" in result.stdout
    assert "outputs/rich_logging_smoke" in result.stdout
    assert "total-env-steps 1024" in result.stdout
