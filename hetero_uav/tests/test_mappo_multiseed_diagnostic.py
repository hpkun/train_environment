"""Test multiseed diagnostic. No HAPPO, no attention, no GRU."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"
DIAG_SCRIPT = ROOT / "scripts" / "run_mappo_v1_v2_multiseed_diagnostic.py"


def test_eval_supports_summary_json():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=10)
    assert "--summary-json" in result.stdout


def test_multiseed_diagnostic_runs():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(DIAG_SCRIPT),
         "--seeds", "0",
         "--iterations", "1", "--rollout-length", "8", "--max-steps", "16",
         "--eval-episodes", "1",
         "--device", "cpu", "--opponent-policy", "rule_nearest",
         "--output-dir", "outputs/test_mappo_multiseed"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_multiseed_outputs_exist():
    out = Path("outputs/test_mappo_multiseed")
    for fname in ["train_summary.csv", "eval_summary.csv",
                  "aggregate_summary.json"]:
        assert (out / fname).exists(), f"Missing {fname}"

    with open(out / "train_summary.csv") as f:
        rows = list(csv.DictReader(f))
    versions = {r["version"] for r in rows}
    assert "v1" in versions and "v2" in versions

    with open(out / "eval_summary.csv") as f:
        e_rows = list(csv.DictReader(f))
    assert len(e_rows) >= 2

    agg = json.loads((out / "aggregate_summary.json").read_text())
    assert set(agg) == {"v1", "v2"}
    for v in ("v1", "v2"):
        assert "train_best_return_mean" in agg[v]
        assert "eval_by_config" in agg[v]


def test_doc_exists():
    doc = ROOT / "docs" / "mappo_v1_v2_multiseed_diagnostic.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "brma_sensor" in text
    assert "mav_shared_geo" in text
    assert "not" in text and "formal" in text
