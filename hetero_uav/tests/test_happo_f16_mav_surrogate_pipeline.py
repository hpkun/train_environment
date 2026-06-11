import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "uav_env" / "JSBSim" / "configs" / "hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def test_f16_mav_surrogate_config_contract():
    assert CONFIG.exists()
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mav = data["aircraft_type_params"]["mav"]
    attack = data["aircraft_type_params"]["attack_uav"]
    assert mav["aircraft_model"] == "f16"
    assert mav["role"] == "mav"
    assert mav["num_missiles"] == 0
    assert attack["aircraft_model"] == "f16"
    assert attack["num_missiles"] == 2
    assert data["hetero_reward_mode"] == "happo_ref_v0"
    assert data["observation_mode"] == "mav_shared_geo"


def test_pipeline_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/run_happo_f16_mav_surrogate_validation_pipeline.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    assert "F-16 MAV surrogate" in result.stdout


def test_pipeline_dry_run_outputs_commands_and_gate(tmp_path):
    out_json = tmp_path / "dry_run.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_happo_f16_mav_surrogate_validation_pipeline.py",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--decision-json",
            str(out_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0
    text = result.stdout
    assert "run_happo_3v2_reference_f16_mav_surrogate_200k.py" in text
    assert "run_happo_3v2_reference_f16_mav_surrogate_1m_fast.py" in text
    assert "mav_survival_rate >= 0.3" in text
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["dry_run"] is True
    assert "gate_conditions" in data
    assert "train_200k_command" in data
    assert "train_1m_command" in data
