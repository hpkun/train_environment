import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, timeout=60):
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


def test_red_attack_environment_decision_help_runs():
    out = run_cmd(["scripts/decide_red_attack_environment_status.py", "--help"], timeout=30)
    assert "decide" in out.stdout.lower()


def test_red_attack_environment_decision_schema(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    oracle = tmp_path / "oracle.json"
    envelope = tmp_path / "envelope.json"
    happo = tmp_path / "happo.json"
    pipeline.write_text(json.dumps({
        "red_auto_fire_logic_enabled": True,
        "logging_fields_present_for_red": True,
    }), encoding="utf-8")
    oracle.write_text(json.dumps({
        "cases": [{"red_missiles_fired_mean": 1.0, "red_missile_hits_mean": 1.0}]
    }), encoding="utf-8")
    envelope.write_text(json.dumps({
        "red_uav_can_fire_in_theoretical_envelope": True
    }), encoding="utf-8")
    happo.write_text(json.dumps({
        "records": [{"red_missiles_fired_total": 0, "launch_envelope_rate": 0.0, "policy_avoids_engagement": True}]
    }), encoding="utf-8")
    output = tmp_path / "decision.json"
    md = tmp_path / "decision.md"
    run_cmd([
        "scripts/decide_red_attack_environment_status.py",
        "--pipeline-json", str(pipeline),
        "--oracle-json", str(oracle),
        "--envelope-json", str(envelope),
        "--happo-json", str(happo),
        "--output-json", str(output),
        "--output-md", str(md),
    ])
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["environment_attack_pipeline_status"] in {
        "red_fire_chain_broken",
        "red_fire_chain_working_policy_not_engaging",
        "red_fire_chain_working_reward_problem",
        "logging_bug_only",
        "inconclusive",
    }
    assert len(data["primary_issue"]) <= 2
    assert data["next_action"] in {"A", "B", "C", "D", "E", "F"}

