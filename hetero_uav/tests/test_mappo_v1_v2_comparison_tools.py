"""Test V1/V2 comparison tools — order-independent, fail-fast checks."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = "outputs/compare_mappo_v1_v2"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_mappo_v1_v2_trainability.py"
ZERO_SHOT_SCRIPT = ROOT / "scripts" / "compare_mappo_v1_v2_zero_shot_smoke.py"


def _ensure_comparison_outputs():
    """Ensure v1/v2 models exist; train if missing."""
    if (Path(f"{OUT_DIR}/v1/latest/model.pt").exists()
            and Path(f"{OUT_DIR}/v2/latest/model.pt").exists()):
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT),
         "--iterations", "1", "--rollout-length", "8", "--max-steps", "16",
         "--device", "cpu", "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_compare_trainability_runs():
    _ensure_comparison_outputs()
    assert "v1" in "".join(os.listdir(OUT_DIR))  # at minimum


def test_summary_json_exists():
    _ensure_comparison_outputs()
    p = Path(f"{OUT_DIR}/trainability_summary.json")
    assert p.exists(), str(p)
    data = json.loads(p.read_text())
    assert len(data) == 2
    versions = {d["version"] for d in data}
    assert versions == {"v1", "v2"}
    for d in data:
        if d["version"] == "v1":
            assert d["actor_dim"] == 140
            assert d["critic_dim"] == 700
        else:
            assert d["actor_dim"] == 96
            assert d["critic_dim"] == 480
        assert d["nan_detected"] == 0


def test_summary_csv_exists():
    _ensure_comparison_outputs()
    p = Path(f"{OUT_DIR}/trainability_summary.csv")
    assert p.exists()
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_compare_zero_shot_runs():
    _ensure_comparison_outputs()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(ZERO_SHOT_SCRIPT),
         "--episodes", "1", "--device", "cpu",
         "--opponent-policy", "rule_nearest"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_zero_shot_summary_json():
    # Ensure zero_shot has run (may need prior train output too)
    _ensure_comparison_outputs()
    if not Path(f"{OUT_DIR}/zero_shot_smoke_summary.json").exists():
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run(
            [sys.executable, str(ZERO_SHOT_SCRIPT),
             "--episodes", "1", "--device", "cpu",
             "--opponent-policy", "rule_nearest"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(ROOT), timeout=300, env=env)
    p = Path(f"{OUT_DIR}/zero_shot_smoke_summary.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert len(data) >= 2
    for d in data:
        assert not d["nan_detected"]


def test_comparison_doc_exists():
    doc = ROOT / "docs" / "mappo_v1_v2_trainability_comparison.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "brma_sensor" in text
    assert "mav_shared_geo" in text
