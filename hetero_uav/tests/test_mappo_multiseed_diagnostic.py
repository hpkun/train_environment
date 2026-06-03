"""Test multiseed diagnostic — order-independent. No HAPPO/attention/GRU."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("outputs/test_mappo_multiseed")
DIAG_SCRIPT = ROOT / "scripts" / "run_mappo_v1_v2_multiseed_diagnostic.py"


def _ensure_multiseed_outputs():
    required = [
        OUT / "train_summary.csv",
        OUT / "eval_summary.csv",
        OUT / "aggregate_summary.json",
    ]
    if all(p.exists() for p in required):
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(DIAG_SCRIPT),
         "--seeds", "0",
         "--iterations", "1", "--rollout-length", "8", "--max-steps", "16",
         "--eval-episodes", "1",
         "--device", "cpu", "--opponent-policy", "rule_nearest",
         "--output-dir", str(OUT)],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=300, env=env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"


def test_eval_supports_summary_json():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
         "--help"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=10)
    assert "--summary-json" in result.stdout


def test_multiseed_outputs_exist():
    _ensure_multiseed_outputs()
    for fname in ["train_summary.csv", "eval_summary.csv",
                  "aggregate_summary.json"]:
        assert (OUT / fname).exists(), f"Missing {fname}"


def test_train_summary_fields():
    _ensure_multiseed_outputs()
    with open(OUT / "train_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2
    by_version = {r["version"]: r for r in rows}
    assert set(by_version) == {"v1", "v2"}
    assert int(float(by_version["v1"]["actor_dim"])) == 140
    assert int(float(by_version["v2"]["actor_dim"])) == 96
    assert int(float(by_version["v1"]["critic_dim"])) == 700
    assert int(float(by_version["v2"]["critic_dim"])) == 480
    for r in rows:
        assert int(float(r["nan_detected"])) == 0, f"NaN in {r['version']}"


def test_eval_summary_fields():
    _ensure_multiseed_outputs()
    with open(OUT / "eval_summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2
    for r in rows:
        assert r["actor_dim_ok"] == "True"
        assert r["critic_dim_ok"] == "True"
        assert r["nan_detected"] == "False"


def test_aggregate_json_fields():
    _ensure_multiseed_outputs()
    agg = json.loads((OUT / "aggregate_summary.json").read_text())
    assert set(agg) == {"v1", "v2"}
    for v in ("v1", "v2"):
        d = agg[v]
        for key in ["train_best_return_mean", "train_best_return_std",
                    "train_last_return_mean", "train_last_return_std",
                    "episodes_completed_mean", "final_red_alive_mean",
                    "final_blue_alive_mean", "eval_by_config"]:
            assert key in d, f"Missing {key} in {v}"
        assert len(d["eval_by_config"]) >= 1


def test_doc_exists():
    doc = ROOT / "docs" / "mappo_v1_v2_multiseed_diagnostic.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "brma_sensor" in text
    assert "mav_shared_geo" in text
    assert "not" in text and "formal" in text
    assert "entity attention" in text.lower()
