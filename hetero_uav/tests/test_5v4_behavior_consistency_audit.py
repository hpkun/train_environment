from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_audit_5v4_behavior_consistency_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_5v4_behavior_consistency.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "5v4" in result.stdout


def test_audit_5v4_behavior_consistency_one_episode(tmp_path: Path) -> None:
    output_dir = tmp_path / "behavior_audit_5v4"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_5v4_behavior_consistency.py",
            "--episodes",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    data = json.loads((output_dir / "behavior_consistency_5v4.json").read_text(encoding="utf-8"))
    for key in [
        "blue_targeting_behavior",
        "mav_role_behavior",
        "blue_pursuit_behavior",
        "mav_frontmost_rate",
        "time_blue_nearest_target_is_mav_rate",
    ]:
        assert key in data
