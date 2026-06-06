"""Tests for heterogeneous environment readiness audit."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_hetero_environment_readiness.py"
OUT_JSON = ROOT / "outputs" / "test_environment_audit" / "hetero_environment_readiness.json"


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_audit():
    result = subprocess.run(
        [
            "python",
            str(SCRIPT),
            "--steps",
            "1",
            "--output-json",
            str(OUT_JSON.relative_to(ROOT)),
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


def _records_by_name():
    if not OUT_JSON.exists():
        _run_audit()
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {Path(r["config"]).name: r for r in data["records"]}


def test_audit_help():
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
        "--output-json",
        "--include-v1",
        "--steps",
        "--configs",
        "--protocol-type",
        "--skip-step-check",
    ):
        assert flag in result.stdout


def _run_specific_audit(args: list[str], output_json: str):
    result = subprocess.run(
        ["python", str(SCRIPT), *args, "--output-json", output_json],
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
    assert "[AUDIT] start" in result.stdout
    assert "[AUDIT] done" in result.stdout
    data = json.loads((ROOT / output_json).read_text(encoding="utf-8"))
    assert data["records"]
    assert data["summary"]["failed_configs"] == 0
    return data


def test_single_paper_aligned_config_audit():
    data = _run_specific_audit(
        [
            "--configs",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "--protocol-type",
            "paper_aligned",
            "--steps",
            "1",
        ],
        "outputs/test_environment_audit/single_3v2.json",
    )
    assert len(data["records"]) == 1
    assert data["records"][0]["protocol_type"] == "paper_aligned"


def test_single_balanced_config_audit():
    data = _run_specific_audit(
        [
            "--configs",
            "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
            "--protocol-type",
            "balanced",
            "--steps",
            "1",
        ],
        "outputs/test_environment_audit/single_balanced_3v3.json",
    )
    assert len(data["records"]) == 1
    assert data["records"][0]["protocol_type"] == "balanced"


def test_skip_step_check_audit():
    data = _run_specific_audit(
        [
            "--configs",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "--protocol-type",
            "paper_aligned",
            "--skip-step-check",
        ],
        "outputs/test_environment_audit/skip_step_3v2.json",
    )
    record = data["records"][0]
    assert record["reset_ok"] is True
    assert record["zero_step_ok"] is None
    assert record["bounded_random_step_ok"] is None


def test_audit_outputs_json():
    _run_audit()
    assert OUT_JSON.exists()


def test_paper_aligned_3v2_and_5v4():
    records = _records_by_name()
    r3 = records["hetero_mav_shared_geo_3v2.yaml"]
    r5 = records["hetero_mav_shared_geo_5v4.yaml"]

    assert r3["protocol_type"] == "paper_aligned"
    assert r3["red_count"] == 3
    assert r3["blue_count"] == 2
    assert r3["red_attack_uav_count"] == 2
    assert r3["blue_attack_uav_count"] == 2
    assert r3["mav_count"] == 1
    assert r3["actor_dim"] == 96
    assert r3["critic_dim"] == 480
    assert r3["nan_detected"] is False

    assert r5["protocol_type"] == "paper_aligned"
    assert r5["red_count"] == 5
    assert r5["blue_count"] == 4
    assert r5["red_attack_uav_count"] == 4
    assert r5["blue_attack_uav_count"] == 4
    assert r5["mav_count"] == 1
    assert r5["actor_dim"] == 96
    assert r5["critic_dim"] == 480
    assert r5["nan_detected"] is False


def test_balanced_warnings():
    records = _records_by_name()
    for name in (
        "hetero_balanced_mav_shared_geo_3v3.yaml",
        "hetero_balanced_mav_shared_geo_4v4.yaml",
    ):
        rec = records[name]
        assert rec["protocol_type"] == "balanced"
        assert rec["red_count"] == rec["blue_count"]
        assert any("fewer red attack UAV" in w for w in rec["warnings"])


def test_finalization_doc_exists():
    doc = ROOT / "docs" / "hetero_environment_finalization_plan.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "paper-aligned" in text.lower()
    assert "balanced total-count" in text.lower()
    assert "blue opponent" in text.lower()
    assert "reward/termination audit" in text.lower()
    assert "not ready for method module" in text.lower()
