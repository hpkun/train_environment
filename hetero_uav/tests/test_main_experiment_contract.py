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


def test_main_experiment_contract_help_runs():
    result = subprocess.run(
        ["python", "scripts/audit_main_experiment_contract.py", "--help"],
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


def test_main_experiment_contract_passes_and_documents_protocol():
    output_json = "outputs/test_main_experiment_contract/audit.json"
    output_md = "outputs/test_main_experiment_contract/audit.md"
    result = subprocess.run(
        [
            "python",
            "scripts/audit_main_experiment_contract.py",
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
    assert data["summary"]["contract_passed"] is True
    assert data["summary"]["blocking_violations"] == []

    assert "hetero_mav_shared_geo_3v2.yaml" in data["config_contract"]["train_config"]
    assert any(
        "hetero_mav_shared_geo_3v2.yaml" in cfg
        for cfg in data["config_contract"]["eval_configs"]
    )
    assert any(
        "hetero_mav_shared_geo_5v4.yaml" in cfg
        for cfg in data["config_contract"]["eval_configs"]
    )
    assert data["config_contract"]["observation_mode"] == "mav_shared_geo"
    assert data["config_contract"]["hetero_reward_mode"] == "brma_legacy"
    assert data["adapter_contract"]["obs_adapter_version"] == "v2"
    assert data["adapter_contract"]["actor_dim"] == 96
    assert data["adapter_contract"]["critic_dim"] == 480
    assert data["adapter_contract"]["action_dim"] == 3
    assert data["runner_contract"]["default_opponent_policy"] == "greedy_fsm"

    md = md_path.read_text(encoding="utf-8")
    for token in [
        "3v2",
        "5v4",
        "mav_shared_geo",
        "brma_legacy",
        "greedy_fsm",
        "shared MAPPO",
        "not a method module",
    ]:
        assert token in md
