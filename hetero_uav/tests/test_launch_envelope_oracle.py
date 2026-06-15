from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_check_launch_envelope_oracle_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_launch_envelope_oracle.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Check scripted red launch envelope entry" in result.stdout
    assert "--episodes" in result.stdout
    assert "--output-dir" in result.stdout


def test_case_summary_counts_block_reasons():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_launch_envelope_oracle as script

    episodes = [
        {
            "red_missiles_fired": 1,
            "blue_missiles_fired": 0,
            "red_missile_hits": 1,
            "blue_dead": 1,
            "range_ok_rate": 0.5,
            "ao_ok_rate": 0.25,
            "ta_ok_rate": 0.75,
            "predicted_allowed_rate": 0.1,
            "min_range_m": 4000.0,
            "block_reason_counts": {"ao_blocked": 3, "lock_delay": 2},
        },
        {
            "red_missiles_fired": 0,
            "blue_missiles_fired": 1,
            "red_missile_hits": 0,
            "blue_dead": 0,
            "range_ok_rate": 1.0,
            "ao_ok_rate": 0.5,
            "ta_ok_rate": 0.5,
            "predicted_allowed_rate": 0.0,
            "min_range_m": 3000.0,
            "block_reason_counts": {"ao_blocked": 1},
        },
    ]
    summary = script._case_summary("direct_chase", "zero", episodes)
    assert summary["red_missiles_fired_mean"] == 0.5
    assert summary["red_missile_hits_mean"] == 0.5
    assert summary["min_range_m"] == 3000.0
    assert summary["launch_block_reason_counts"] == {"ao_blocked": 4, "lock_delay": 2}
