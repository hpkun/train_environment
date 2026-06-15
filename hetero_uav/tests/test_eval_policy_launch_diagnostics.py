from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eval_policy_launch_diagnostics_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_policy_launch_diagnostics.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Evaluate learned policy launch-envelope diagnostics" in result.stdout
    assert "--scenario" in result.stdout
    assert "--diagnostic-output-dir" in result.stdout


def test_eval_policy_launch_diagnostics_summary():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    rows = [
        {
            "episode_id": 0,
            "range_ok": True,
            "ao_ok": False,
            "ta_ok": True,
            "lock_ready": False,
            "launch_allowed": False,
            "launch_block_reason": "ao_blocked",
            "action_pitch": 0.1,
            "action_heading": 0.2,
            "action_speed": 0.3,
            "missiles_fired": 0,
            "missile_hits": 0,
            "blue_dead": 0,
        },
        {
            "episode_id": 0,
            "range_ok": True,
            "ao_ok": True,
            "ta_ok": True,
            "lock_ready": True,
            "launch_allowed": True,
            "launch_block_reason": "allowed",
            "action_pitch": 1.0,
            "action_heading": 0.0,
            "action_speed": 0.0,
            "missiles_fired": 1,
            "missile_hits": 1,
            "blue_dead": 1,
        },
    ]
    summary = script._summarize(rows, 1, "model", "3v2", "flat")
    assert summary["range_ok_rate"] == 1.0
    assert summary["ao_ok_rate"] == 0.5
    assert summary["lock_ready_rate"] == 0.5
    assert summary["red_missiles_fired"] == 1
    assert summary["missile_hits"] == 1
    assert summary["dominant_block_reason"] in {"ao_blocked", "allowed"}
