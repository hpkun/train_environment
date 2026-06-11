import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_happo_1m_readiness_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/decide_happo_1m_readiness.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "1M readiness" in result.stdout


def test_happo_1m_readiness_accepts_mock_inputs(tmp_path):
    consistency = tmp_path / "consistency.json"
    policy = tmp_path / "policy.json"
    mav = tmp_path / "mav.json"
    checkpoint = tmp_path / "checkpoint.json"
    summary = tmp_path / "summary.json"
    consistency.write_text(json.dumps({"consistency_status": "inconsistent", "blocking_issues": [], "likely_causes": ["stochastic training"]}), encoding="utf-8")
    policy.write_text(json.dumps({"records": [], "mode_comparison": {"stochastic_latest_close_to_train": False}}), encoding="utf-8")
    mav.write_text(json.dumps({"summary": {"mav_first_death_rate": 1.0, "mav_missile_death_rate": 0.0, "mav_crash_death_rate": 0.0, "conclusion": "mock"}}), encoding="utf-8")
    checkpoint.write_text(json.dumps([{"checkpoint": "best", "blue_dead_mean": 0.1, "red_missile_hits_mean": 0.0}]), encoding="utf-8")
    summary.write_text(json.dumps({"judgment": {"still_timeout_survival": True}}), encoding="utf-8")
    out_json = tmp_path / "decision.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/decide_happo_1m_readiness.py",
            "--consistency-json",
            str(consistency),
            "--policy-mode-json",
            str(policy),
            "--mav-failure-json",
            str(mav),
            "--checkpoint-json",
            str(checkpoint),
            "--summary-json",
            str(summary),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "decision.md"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ["run_1m_recommended", "blocking_issues", "warnings", "required_fixes_before_1m"]:
        assert key in data
