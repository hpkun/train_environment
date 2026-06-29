"""Tests for mav_shared launch gate audit script correctness."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class FakeEnv:
    """Minimal fake env to test lock_ready and launch gate audit logic."""
    def __init__(self, lock_target=None, lock_timer=0, missile_lock_delay_frames=15,
                 use_boresight_launch_gate=False):
        self._lock_target = dict(lock_target or {})
        self._lock_timer = dict(lock_timer or {})
        self.missile_lock_delay_frames = missile_lock_delay_frames
        self.use_boresight_launch_gate = use_boresight_launch_gate


class TestLockReady:
    def test_lock_ready_uses_frames_not_seconds(self):
        """_lock_ready must compare _lock_timer (frames) against missile_lock_delay_frames."""
        from scripts.audit_mav_shared_launch_gate import _lock_ready

        env = FakeEnv(
            lock_target={"red_1": "blue_0"},
            lock_timer={"red_1": 15},
            missile_lock_delay_frames=15,
        )
        assert _lock_ready(env, "red_1", "blue_0") is True

    def test_lock_not_ready_when_timer_insufficient(self):
        from scripts.audit_mav_shared_launch_gate import _lock_ready

        env = FakeEnv(
            lock_target={"red_1": "blue_0"},
            lock_timer={"red_1": 10},
            missile_lock_delay_frames=15,
        )
        assert _lock_ready(env, "red_1", "blue_0") is False

    def test_lock_not_ready_when_target_mismatch(self):
        from scripts.audit_mav_shared_launch_gate import _lock_ready

        env = FakeEnv(
            lock_target={"red_1": "blue_1"},
            lock_timer={"red_1": 20},
            missile_lock_delay_frames=15,
        )
        assert _lock_ready(env, "red_1", "blue_0") is False

    def test_lock_timer_default_zero(self):
        from scripts.audit_mav_shared_launch_gate import _lock_ready

        env = FakeEnv()
        assert _lock_ready(env, "red_1", "blue_0") is False


class TestLaunchEventTrackSource:
    def test_reads_launch_track_source_field(self):
        """The real environment uses 'launch_track_source' not 'track_source' in launch quality records."""
        rec = {"launch_track_source": "mav_shared", "shooter_id": "red_1", "target_id": "blue_0"}
        track_source = rec.get("launch_track_source", rec.get("track_source", "unknown"))
        assert track_source == "mav_shared"

    def test_fallback_to_track_source_when_launch_track_source_absent(self):
        rec = {"track_source": "direct", "shooter_id": "red_1", "target_id": "blue_0"}
        track_source = rec.get("launch_track_source", rec.get("track_source", "unknown"))
        assert track_source == "direct"

    def test_unknown_when_both_absent(self):
        rec = {"shooter_id": "red_1"}
        track_source = rec.get("launch_track_source", rec.get("track_source", "unknown"))
        assert track_source == "unknown"


class TestGeometryOk:
    def test_geometry_ok_without_boresight(self):
        use_boresight = False
        range_ok, ao_ok, ta_ok, boresight_ok = True, True, True, False
        if use_boresight:
            geometry_ok = range_ok and ao_ok and ta_ok and boresight_ok
        else:
            geometry_ok = range_ok and ao_ok and ta_ok
        assert geometry_ok is True

    def test_geometry_ok_with_boresight_fails(self):
        use_boresight = True
        range_ok, ao_ok, ta_ok, boresight_ok = True, True, True, False
        if use_boresight:
            geometry_ok = range_ok and ao_ok and ta_ok and boresight_ok
        else:
            geometry_ok = range_ok and ao_ok and ta_ok
        assert geometry_ok is False

    def test_geometry_ok_with_boresight_passes(self):
        use_boresight = True
        range_ok, ao_ok, ta_ok, boresight_ok = True, True, True, True
        if use_boresight:
            geometry_ok = range_ok and ao_ok and ta_ok and boresight_ok
        else:
            geometry_ok = range_ok and ao_ok and ta_ok
        assert geometry_ok is True


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
