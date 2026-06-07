from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_missile_launch_contract_audit_help_runs():
    result = subprocess.run(
        ["python", "scripts/audit_missile_launch_contract.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--steps" in result.stdout
    assert "--episodes" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout


def test_missile_launch_contract_audit_short_run_outputs_contract():
    output_json = "outputs/test_environment_audit/missile_launch_contract_audit.json"
    output_md = "outputs/test_environment_audit/missile_launch_contract_audit.md"
    result = subprocess.run(
        [
            "python",
            "scripts/audit_missile_launch_contract.py",
            "--steps",
            "5",
            "--episodes",
            "1",
            "--output-json",
            output_json,
            "--output-md",
            output_md,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for key in [
        "static_contract",
        "rollout_diagnostics",
        "open_questions",
        "blocking_violations",
        "status",
    ]:
        assert key in data

    static = data["static_contract"]
    assert static["launch_range_max_m"] == 10000
    assert abs(static["launch_ao_thresh_deg"] - 45) < 1e-6
    assert abs(static["launch_ta_thresh_deg"] - 90) < 1e-6
    assert abs(static["missile_lock_delay_sec"] - 0.25) < 0.02
    assert abs(static["missile_cooldown_sec"] - 0.5) < 0.02
    assert static["mav_num_missiles"] == 0
    assert static["attack_uav_num_missiles"] == 2
    assert data["blocking_violations"] == []

    md = md_path.read_text(encoding="utf-8")
    for token in ["missile", "10 km", "0.25", "0.5", "3-9 line", "Open Questions"]:
        assert token in md
