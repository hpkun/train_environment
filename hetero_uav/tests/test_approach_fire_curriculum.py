from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_approach_fire_easy_f16_mav_surrogate.yaml"


def test_approach_fire_easy_config_loads():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["hetero_reward_mode"] == "happo_ref_v0"
    assert data["observation_mode"] == "mav_shared_geo"
    assert data["max_steps"] == 1000
    assert data["aircraft_type_params"]["mav"]["num_missiles"] == 0
    assert data["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2
    assert data["initial_states"]["blue_0"]["lat"] == 60.07
    assert data["initial_states"]["red_1"]["yaw_deg"] == 0.0
    assert data["initial_states"]["blue_0"]["yaw_deg"] == 180.0


def test_approach_fire_runner_dry_run_outputs_commands(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_approach_fire_curriculum.py"),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "approach_fire"),
            "--total-env-steps",
            "16",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch flat" in result.stdout
    assert "--policy-arch entity_attention" in result.stdout
    assert "eval_policy_launch_diagnostics.py" in result.stdout
    assert "summarize_approach_fire_curriculum.py" in result.stdout


def test_approach_fire_summary_reads_fake_diagnostics(tmp_path):
    base = tmp_path / "base"
    for name, fired, hits in [
        ("flat_easy_imitation", 2, 1),
        ("entity_easy_imitation", 0, 0),
    ]:
        d = base / name / "launch_diagnostics_3v2"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "red_missiles_fired": fired,
            "missile_hits": hits,
            "blue_dead_mean": hits,
            "range_ok_rate": 0.5 if fired else 0.1,
            "ao_ok_rate": 0.4 if fired else 0.1,
            "ta_ok_rate": 0.3 if fired else 0.1,
            "lock_ready_rate": 0.01,
            "launch_allowed_rate": 0.0,
            "dominant_block_reason": "no_missile" if fired else "out_of_range",
            "action_saturation_rate": 0.2,
        }), encoding="utf-8")

    out_csv = tmp_path / "summary.csv"
    out_md = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_approach_fire_curriculum.py"),
            "--base-dir",
            str(base),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
            "--steps",
            "16",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    text = out_md.read_text(encoding="utf-8")
    assert "flat_easy_imitation" in text
    assert "entity_easy_imitation" in text
    assert "out_of_range" in text
