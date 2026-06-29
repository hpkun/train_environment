from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_mav_shared_audit_scripts_help():
    for script in (
        "scripts/audit_mav_shared_observation_quality.py",
        "scripts/audit_mav_shared_launch_gate.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--config" in result.stdout


def test_mav_shared_launch_gate_audit_smoke(tmp_path):
    out_dir = tmp_path / "launch_gate"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_mav_shared_launch_gate.py",
            "--episodes",
            "1",
            "--max-steps",
            "5",
            "--output-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "launch_gate_by_track_source.csv",
        "launch_block_reason_by_track_source.csv",
        "launch_quality_by_track_source.csv",
        "lock_continuity_by_track_source.csv",
        "mav_shared_launch_candidates.csv",
        "launch_events_by_track_source.csv",
        "launch_interval_paper_alignment.md",
    ):
        assert (out_dir / name).exists(), name
