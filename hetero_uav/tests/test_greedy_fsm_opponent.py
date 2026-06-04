from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_opponent_policy_supports_greedy_fsm_and_existing_modes():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    assert {"zero", "random", "rule_nearest", "greedy_fsm"}.issubset(
        OpponentPolicy.MODES
    )


def test_greedy_fsm_empty_obs_returns_valid_action():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy = OpponentPolicy("greedy_fsm", seed=0)
    actions = policy.act({"blue_0": {}}, ["blue_0"])
    action = actions["blue_0"]

    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert policy.last_states["blue_0"] == "patrol"


def test_greedy_fsm_enemy_obs_returns_valid_action_and_state():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    obs = {
        "blue_0": {
            "enemy_states": np.array(
                [
                    [0.8, -0.2, 0.1, 0, 0, 0],
                    [0.2, 0.3, -0.1, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            "altitude": np.array([0.5], dtype=np.float32),
            "missile_warning": np.array([0.0], dtype=np.float32),
        }
    }
    policy = OpponentPolicy("greedy_fsm")
    actions = policy.act(obs, ["blue_0"])
    action = actions["blue_0"]

    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert policy.last_states["blue_0"] in {
        "attack_nearest",
        "attack_mav_priority",
    }


def test_greedy_fsm_diagnosis_script_outputs_json(tmp_path):
    output_json = tmp_path / "greedy_fsm_opponent_diagnostic.json"
    result = subprocess.run(
        [
            "python",
            "scripts/diagnose_greedy_fsm_opponent.py",
            "--steps",
            "3",
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert output_json.exists(), result.stdout + result.stderr
    data = json.loads(output_json.read_text(encoding="utf-8"))
    policies = {record["opponent_policy"] for record in data["records"]}
    assert {"rule_nearest", "greedy_fsm"}.issubset(policies)
    assert all(not record["nan_detected"] for record in data["records"])
    assert "greedy_fsm_state_coverage" in data["summary"]
    assert "greedy_fsm_has_non_patrol_state" in data["summary"]
    assert "greedy_fsm_action_saturation_mean" in data["summary"]

    for record in data["records"]:
        assert "blue_action_mean" in record
        assert "blue_action_std" in record
        assert "blue_action_saturation_rate" in record
        assert "dominant_state" in record
        assert "dominant_state_ratio" in record
        assert 0.0 <= record["blue_action_saturation_rate"] <= 1.0

    greedy_records = [
        record for record in data["records"]
        if record["opponent_policy"] == "greedy_fsm"
    ]
    assert greedy_records
    assert all(record["blue_state_counts"] for record in greedy_records)


def test_greedy_fsm_design_doc_exists():
    doc = ROOT / "docs" / "blue_greedy_fsm_opponent_design.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8").lower()
    assert "finite-state" in text
    assert "rule_nearest" in text
    assert "not a new algorithm" in text
    assert "target assignment" in text
    assert "candidate maneuver" in text
    assert "not final opponent" in text
    assert "rule_nearest remains default" in text
