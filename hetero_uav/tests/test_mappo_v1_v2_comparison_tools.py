"""Test V1/V2 comparison tools. No HAPPO, no attention, no GRU."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compare_trainability_runs():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "compare_mappo_v1_v2_trainability.py"),
         "--iterations", "1", "--rollout-length", "8", "--max-steps", "16",
         "--device", "cpu", "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert "v1" in result.stdout and "v2" in result.stdout


def test_compare_outputs_exist():
    for v in ("v1", "v2"):
        assert Path(f"outputs/compare_mappo_v1_v2/{v}/latest/model.pt").exists()
        assert Path(f"outputs/compare_mappo_v1_v2/{v}/train_log.csv").exists()


def test_compare_zero_shot_runs():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "compare_mappo_v1_v2_zero_shot_smoke.py"),
         "--episodes", "1", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_comparison_doc_exists():
    doc = ROOT / "docs" / "mappo_v1_v2_trainability_comparison.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "brma_sensor" in text
    assert "mav_shared_geo" in text
    assert "formal experiment" in text.lower()


def test_no_nan_in_comparison_logs():
    for v in ("v1", "v2"):
        log = Path(f"outputs/compare_mappo_v1_v2/{v}/train_log.csv")
        if not log.exists():
            continue
        import csv
        with open(log) as f:
            for row in csv.DictReader(f):
                assert int(float(row["nan_detected"])) == 0
