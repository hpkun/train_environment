"""Test V1/V2 comparison tools without relying on pytest execution order."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = "outputs/compare_mappo_v1_v2"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_mappo_v1_v2_trainability.py"
ZERO_SHOT_SCRIPT = ROOT / "scripts" / "compare_mappo_v1_v2_zero_shot_smoke.py"


def _ensure_comparison_outputs():
    required = [
        Path(f"{OUT_DIR}/v1/latest/model.pt"),
        Path(f"{OUT_DIR}/v2/latest/model.pt"),
        Path(f"{OUT_DIR}/trainability_summary.json"),
        Path(f"{OUT_DIR}/trainability_summary.csv"),
    ]
    if all(path.exists() for path in required):
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [
            "python",
            str(COMPARE_SCRIPT),
            "--iterations",
            "1",
            "--rollout-length",
            "8",
            "--max-steps",
            "16",
            "--device",
            "cpu",
            "--opponent-policy",
            "rule_nearest",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )


def _run_zero_shot_once():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [
            "python",
            str(ZERO_SHOT_SCRIPT),
            "--episodes",
            "1",
            "--device",
            "cpu",
            "--opponent-policy",
            "rule_nearest",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
    )


def test_compare_trainability_outputs_exist():
    _ensure_comparison_outputs()
    assert Path(f"{OUT_DIR}/v1/latest/model.pt").exists()
    assert Path(f"{OUT_DIR}/v2/latest/model.pt").exists()


def test_trainability_summary_json_and_csv():
    _ensure_comparison_outputs()
    json_path = Path(f"{OUT_DIR}/trainability_summary.json")
    csv_path = Path(f"{OUT_DIR}/trainability_summary.csv")
    assert json_path.exists(), str(json_path)
    assert csv_path.exists(), str(csv_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_version = {d["version"]: d for d in data}
    assert set(by_version) == {"v1", "v2"}
    assert by_version["v1"]["actor_dim"] == 140
    assert by_version["v2"]["actor_dim"] == 96
    assert by_version["v1"]["critic_dim"] == 700
    assert by_version["v2"]["critic_dim"] == 480
    assert by_version["v1"]["nan_detected"] == 0
    assert by_version["v2"]["nan_detected"] == 0

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_compare_zero_shot_runs():
    _ensure_comparison_outputs()
    _run_zero_shot_once()

    summary = Path(f"{OUT_DIR}/zero_shot_smoke_summary.json")
    v1_stdout = Path(f"{OUT_DIR}/v1_zero_shot_stdout.txt")
    v2_stdout = Path(f"{OUT_DIR}/v2_zero_shot_stdout.txt")
    assert summary.exists()
    assert v1_stdout.exists()
    assert v2_stdout.exists()
    for path in (v1_stdout, v2_stdout):
        text = path.read_text(encoding="utf-8")
        assert "nan_detected: True" not in text
        assert "actor_dim_ok: False" not in text
        assert "critic_dim_ok: False" not in text


def test_zero_shot_summary_json():
    _ensure_comparison_outputs()
    if not Path(f"{OUT_DIR}/zero_shot_smoke_summary.json").exists():
        _run_zero_shot_once()

    data = json.loads(
        Path(f"{OUT_DIR}/zero_shot_smoke_summary.json").read_text(
            encoding="utf-8"
        )
    )
    by_version = {d["version"]: d for d in data}
    assert set(by_version) == {"v1", "v2"}
    for d in by_version.values():
        assert d["returncode"] == 0
        assert not d["nan_detected_found_in_stdout"]
        assert d["actor_dim_ok_found_in_stdout"]
        assert d["critic_dim_ok_found_in_stdout"]


def test_comparison_doc_exists():
    doc = ROOT / "docs" / "mappo_v1_v2_trainability_comparison.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "brma_sensor" in text
    assert "mav_shared_geo" in text
    assert "diagnostic" in text
