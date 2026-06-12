import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, timeout=240):
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


def test_happo_engagement_geometry_help_runs():
    out = run_cmd(["scripts/audit_happo_policy_engagement_geometry.py", "--help"], timeout=30)
    assert "engagement geometry" in out.stdout.lower()


def test_happo_engagement_geometry_one_episode_schema():
    output = ROOT / "outputs/test_environment_audit/happo_engagement_geometry.json"
    md = ROOT / "outputs/test_environment_audit/happo_engagement_geometry.md"
    run_cmd([
        "scripts/audit_happo_policy_engagement_geometry.py",
        "--episodes", "1",
        "--max-steps", "30",
        "--output-json", str(output.relative_to(ROOT)),
        "--output-md", str(md.relative_to(ROOT)),
    ])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "records" in data
    assert "overall_conclusion" in data
    for rec in data["records"]:
        assert "launch_range_rate" in rec
        assert "launch_envelope_rate" in rec
        assert "action_saturation_rate" in rec
        assert "conclusion" in rec

