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


def test_red_launch_envelope_probe_help_runs():
    out = run_cmd(["scripts/probe_red_missile_launch_envelope.py", "--help"], timeout=30)
    assert "envelope" in out.stdout.lower()


def test_red_launch_envelope_probe_short_schema():
    output = ROOT / "outputs/test_environment_audit/red_launch_envelope.json"
    md = ROOT / "outputs/test_environment_audit/red_launch_envelope.md"
    run_cmd([
        "scripts/probe_red_missile_launch_envelope.py",
        "--steps", "5",
        "--output-json", str(output.relative_to(ROOT)),
        "--output-md", str(md.relative_to(ROOT)),
    ])
    data = json.loads(output.read_text(encoding="utf-8"))
    for key in [
        "records",
        "red_uav_can_fire_in_theoretical_envelope",
        "any_launch_envelope_satisfied",
        "conclusion",
    ]:
        assert key in data
    assert data["records"]
    assert "before_distance_m" in data["records"][0]

