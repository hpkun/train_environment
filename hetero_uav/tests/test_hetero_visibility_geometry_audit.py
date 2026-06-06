"""Test visibility/geometry audit. No training, no env changes."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_hetero_visibility_geometry.py"


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=10)
    for flag in [
        "--steps",
        "--steps-list",
        "--red-policy",
        "--blue-policy",
        "--output-json",
        "--disable-config-trim",
    ]:
        assert flag in result.stdout


def test_short_run():
    out = "outputs/test_environment_audit/hetero_visibility_geometry.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--steps", "3", "--red-policy", "zero", "--blue-policy", "greedy_fsm",
         "--output-json", out],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=120,
        env=_subprocess_env())
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    assert Path(out).exists()
    data = json.loads(Path(out).read_text())
    assert "records" in data
    assert "summary" in data
    for r in data["records"]:
        for key in ["config", "red_observed_any", "blue_observed_any",
                    "first_step_red_observed", "first_step_blue_observed",
                    "red_mav_shared_fraction", "blue_direct_fraction",
                    "warnings"]:
            assert key in r, f"Missing {key}"
        assert not any(np.isnan(v) if isinstance(v, float) else False
                       for v in r.values() if isinstance(v, (int, float)))
        assert r["step_geometry"]
        for key in [
            "red_mav_shared_tracks_total",
            "blue_direct_tracks_total",
        ]:
            assert key in r["step_geometry"][0], f"Missing step field {key}"


def test_steps_list_run():
    out = "outputs/test_environment_audit/hetero_visibility_geometry_after_trim.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--steps-list", "3", "5", "--blue-policy", "greedy_fsm",
         "--output-json", out],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=120,
        env=_subprocess_env())
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    data = json.loads(Path(out).read_text())
    assert "records" in data
    assert "summary" in data
    assert "horizon_summary_by_config" in data["summary"]
    assert data["summary"]["horizons"] == [3, 5]
    for r in data["records"]:
        assert "horizon_steps" in r
        assert r["horizon_steps"] in [3, 5]
        for key in [
            "initial_min_red_blue_distance_m",
            "final_min_red_blue_distance_m",
            "min_red_blue_distance_delta_m",
            "closest_red_blue_distance_m",
            "blue_closing_fraction",
            "blue_ever_within_direct_range",
            "direct_range_margin_closest_m",
            "action_trim_enabled",
            "mav_altitude_final_m",
            "mav_alive_final",
            "heading_wrap_used",
            "turn_back_heading_delta_mean_abs",
            "post_pass_separation_m",
        ]:
            assert key in r, f"Missing {key}"
        assert not any(np.isnan(v) if isinstance(v, float) else False
                       for v in r.values() if isinstance(v, (int, float)))
    for item in data["summary"]["horizon_summary_by_config"].values():
        assert "blue_observed_by_3" in item
        assert "blue_observed_by_5" in item
        assert "first_horizon_blue_observed" in item


def test_disable_config_trim_run():
    out = "outputs/test_environment_audit/hetero_visibility_geometry_no_trim.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--steps", "3", "--blue-policy", "greedy_fsm", "--disable-config-trim",
         "--output-json", out],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(ROOT), timeout=120,
        env=_subprocess_env())
    assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
    data = json.loads(Path(out).read_text())
    assert data["records"]
    for r in data["records"]:
        assert r["action_trim_enabled"] is False


def test_doc_exists():
    doc = ROOT / "docs" / "hetero_visibility_geometry_audit.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "MAV shared" in text
    assert "blue" in text.lower()
    assert "initial geometry" in text.lower()
