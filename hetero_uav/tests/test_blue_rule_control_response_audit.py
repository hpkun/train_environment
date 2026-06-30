from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


def test_delta10_target_heading_moves_only_ten_degrees():
    from scripts.audit_blue_rule_control_response import (
        _delta10_target_heading,
        _wrap_pi,
    )

    current_heading = math.radians(170.0)
    target = _delta10_target_heading(current_heading, 1.0)
    assert abs(_wrap_pi(target - math.radians(180.0))) < 1e-6


def test_direct_heading_probe_points_to_bearing_without_delta_limit():
    from scripts.audit_blue_rule_control_response import (
        _direct_heading_probe,
        _wrap_pi,
    )

    assert abs(_wrap_pi(_direct_heading_probe(0.0, math.pi) - math.pi)) < 1e-6


def test_control_response_report_schema_from_fake_rows(tmp_path):
    from scripts.audit_blue_rule_control_response import _write_outputs

    rows = [
        {
            "episode": 0,
            "step": 0,
            "blue_id": "blue_0",
            "branch_state": "combat",
            "heading_error_to_target_rad": math.pi,
            "heading_cmd_saturated": 1,
            "range_delta_next_m": 100.0,
            "actual_heading_delta_next_rad": 0.01,
            "command_tracking_error_rad": math.pi - 0.01,
            "blue_roll_rad": 0.05,
            "blue_speed_mps": 300.0,
            "boundary_pressure": 0.0,
            "death_or_outcome_if_terminal": "",
        }
    ]
    _write_outputs(tmp_path, rows, config="fake.yaml", episodes=1, max_steps=1, red_mode="zero")

    summary = list(csv.DictReader((tmp_path / "blue_rule_control_response_summary.csv").open()))
    metrics = {row["metric"] for row in summary}
    assert "high_error_cmd_saturation_rate" in metrics
    assert "high_error_range_increase_rate_next" in metrics
    assert (tmp_path / "blue_rule_control_response_report.md").exists()


def test_control_response_audit_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_blue_rule_control_response.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--red-mode" in result.stdout


def test_control_response_short_smoke_outputs_required_fields(tmp_path):
    out_dir = tmp_path / "blue_control"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_blue_rule_control_response.py",
            "--episodes",
            "1",
            "--max-steps",
            "3",
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-1000:]
    rows = list(csv.DictReader((out_dir / "blue_rule_control_response_steps.csv").open()))
    assert rows
    required = {
        "target_heading_cmd_rad",
        "blue_yaw_rad",
        "blue_track_heading_rad",
        "actual_heading_next_rad",
        "command_tracking_error_rad",
        "range_delta_next_m",
        "delta10_heading_error_after_cmd_rad",
        "direct_probe_heading_error_after_cmd_rad",
    }
    assert required.issubset(rows[0].keys())
