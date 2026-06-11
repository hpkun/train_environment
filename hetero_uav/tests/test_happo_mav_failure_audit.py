import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mav_failure_audit_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/audit_happo_mav_failure_modes.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "MAV failure modes" in result.stdout


def test_mav_failure_audit_missing_checkpoint_exits_cleanly(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_happo_mav_failure_modes.py",
            "--output-dir",
            str(tmp_path / "missing"),
            "--episodes",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in (result.stderr + result.stdout).lower()


def test_mav_failure_audit_schema_on_mock_output(tmp_path):
    data = {
        "summary": {
            "mav_first_death_rate": 0.0,
            "mav_missile_death_rate": 0.0,
            "mav_crash_death_rate": 0.0,
            "conclusion": "mock",
        }
    }
    p = tmp_path / "mock.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    for key in [
        "mav_first_death_rate",
        "mav_missile_death_rate",
        "mav_crash_death_rate",
        "conclusion",
    ]:
        assert key in loaded["summary"]
