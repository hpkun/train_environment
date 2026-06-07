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


def test_f22_mav_model_audit_help_runs():
    result = subprocess.run(
        ["python", "scripts/audit_f22_mav_model.py", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout


def test_f22_mav_model_audit_short_run_outputs_expected_contract():
    output_json = "outputs/test_environment_audit/f22_mav_model_audit.json"
    output_md = "outputs/test_environment_audit/f22_mav_model_audit.md"
    result = subprocess.run(
        [
            "python",
            "scripts/audit_f22_mav_model.py",
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
        timeout=420,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    json_path = ROOT / output_json
    md_path = ROOT / output_md
    assert json_path.exists()
    assert md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    summary = data["summary"]

    assert summary["f22_model_exists"] is True
    assert summary["f22_load_model_ok"] is True
    assert summary["f22_run_ic_ok"] is True
    assert summary["main_3v2_red0_model"] == "f22"
    assert summary["main_5v4_red0_model"] == "f22"
    assert summary["mav_missiles"] == 0
    assert summary["attack_uav_model"] == "f16"
    assert summary["attack_uav_missiles"] == 2
    assert summary["actor_dim"] == 96
    assert summary["critic_dim"] == 480
    assert summary["nan_detected"] is False

    md = md_path.read_text(encoding="utf-8")
    assert "F-22" in md
    assert "red_0 model: f22" in md
