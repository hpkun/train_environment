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


def test_red_attack_oracle_help_runs():
    out = run_cmd(["scripts/run_red_attack_oracle_sanity.py", "--help"], timeout=30)
    assert "oracle" in out.stdout.lower()


def test_red_attack_oracle_one_episode_schema():
    output = ROOT / "outputs/test_environment_audit/red_attack_oracle.json"
    md = ROOT / "outputs/test_environment_audit/red_attack_oracle.md"
    run_cmd([
        "scripts/run_red_attack_oracle_sanity.py",
        "--episodes", "1",
        "--max-steps", "40",
        "--output-json", str(output.relative_to(ROOT)),
        "--output-md", str(md.relative_to(ROOT)),
    ])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "cases" in data
    assert len(data["cases"]) == 4
    for record in data["cases"]:
        for key in [
            "red_missiles_fired_mean",
            "blue_missiles_fired_mean",
            "red_missile_hits_mean",
            "blue_dead_mean",
            "first_red_fire_step_mean",
            "red_fire_possible_rate",
            "conclusion",
        ]:
            assert key in record

