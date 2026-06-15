from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_analyze_action_heading_alignment_help_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/analyze_action_heading_alignment.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--fake-input" in proc.stdout
    assert "--skip-rollout" in proc.stdout


def test_analyze_action_heading_alignment_fake_summary(tmp_path):
    fake = tmp_path / "fake_rows.csv"
    fields = [
        "source",
        "episode_id",
        "step",
        "red_id",
        "target_id",
        "current_heading_rad",
        "target_bearing_rad",
        "heading_error_rad",
        "action_heading",
        "decoded_target_heading_rad",
        "command_error_to_bearing_rad",
        "AO_rad",
        "AO_next_rad",
        "delta_AO_rad",
        "range_m",
        "range_next_m",
        "delta_range_m",
        "launch_allowed",
        "block_reason",
        "action_pitch",
        "action_speed",
    ]
    with fake.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "source": "fake_policy",
            "episode_id": 0,
            "step": 1,
            "red_id": "red_1",
            "target_id": "blue_0",
            "current_heading_rad": 0.0,
            "target_bearing_rad": 0.2,
            "heading_error_rad": 0.2,
            "action_heading": 0.064,
            "decoded_target_heading_rad": 0.201,
            "command_error_to_bearing_rad": 0.001,
            "AO_rad": 0.5,
            "AO_next_rad": 0.4,
            "delta_AO_rad": -0.1,
            "range_m": 8000,
            "range_next_m": 7800,
            "delta_range_m": -200,
            "launch_allowed": False,
            "block_reason": "lock_delay",
            "action_pitch": 0.0,
            "action_speed": 0.8,
        })

    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    detail_csv = tmp_path / "detail.csv"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_action_heading_alignment.py",
            "--fake-input",
            str(fake),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
            "--detail-csv",
            str(detail_csv),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_csv.exists()
    assert out_md.exists()
    assert detail_csv.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "Action Decode Chain" in text
    assert "absolute target heading" in text
