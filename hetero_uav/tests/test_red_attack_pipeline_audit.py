import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, timeout=180):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=True,
    )


def test_red_attack_pipeline_help_runs():
    out = run_cmd(["scripts/audit_red_attack_pipeline.py", "--help"], timeout=30)
    assert "red attack" in out.stdout.lower()


def test_red_attack_pipeline_short_run_schema():
    output = ROOT / "outputs/test_environment_audit/red_attack_pipeline.json"
    md = ROOT / "outputs/test_environment_audit/red_attack_pipeline.md"
    run_cmd([
        "scripts/audit_red_attack_pipeline.py",
        "--steps", "1",
        "--output-json", str(output.relative_to(ROOT)),
        "--output-md", str(md.relative_to(ROOT)),
    ])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["red_0_mav_num_missiles_zero"] is True
    assert data["red_uav_num_missiles_two"] is True
    assert data["blue_num_missiles_two"] is True
    assert "static_fire_control" in data
    assert "red_observation_visibility" in data
    assert "blocking_issues" in data

