import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mav_failure_fix_decision_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/decide_mav_failure_fix_path.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "failure fix path" in result.stdout.lower()


def test_mav_failure_fix_decision_accepts_mock_inputs(tmp_path):
    control = tmp_path / "control.json"
    ablation = tmp_path / "ablation.json"
    blue = tmp_path / "blue.json"
    failure = tmp_path / "failure.json"
    readiness = tmp_path / "readiness.json"
    control.write_text(json.dumps({"summary": {"f22_stable": False, "f16_surrogate_more_stable": True}}), encoding="utf-8")
    ablation.write_text(json.dumps({"summary": {"best_realistic_mav_survival_rate": 0.0, "safe_mav_can_survive": False, "f16_surrogate_improves": True}}), encoding="utf-8")
    blue.write_text(json.dumps({"mav_target_fraction": 0.1, "unavailable_fields": []}), encoding="utf-8")
    failure.write_text(json.dumps({"records": [{"summary": {"mav_death_rate": 1.0}}]}), encoding="utf-8")
    readiness.write_text(json.dumps({"run_1m_recommended": False}), encoding="utf-8")
    out_json = tmp_path / "decision.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/decide_mav_failure_fix_path.py",
            "--control-json", str(control),
            "--ablation-json", str(ablation),
            "--blue-target-json", str(blue),
            "--mav-failure-json", str(failure),
            "--readiness-json", str(readiness),
            "--output-json", str(out_json),
            "--output-md", str(tmp_path / "decision.md"),
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
    for key in ["primary_failure_hypothesis", "recommended_next_action", "run_1m_allowed", "blocking_issues"]:
        assert key in data
