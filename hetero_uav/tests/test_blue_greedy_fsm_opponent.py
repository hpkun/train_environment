from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_greedy_fsm_mode_and_synthetic_action_validity():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    assert "greedy_fsm" in OpponentPolicy.MODES
    policy = OpponentPolicy("greedy_fsm", seed=0)
    action = policy.act(
        {
            "blue_0": {
                "enemy_states": np.array([[0.5, 0.2, 0.1]], dtype=np.float32),
                "enemy_observed_mask": np.array([1.0], dtype=np.float32),
                "missile_warning": np.array([0.0], dtype=np.float32),
                "altitude": np.array([1.0], dtype=np.float32),
            }
        },
        ["blue_0"],
    )["blue_0"]
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_missile_warning_changes_action():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    base_obs = {
        "enemy_states": np.array([[0.5, 0.2, 0.1]], dtype=np.float32),
        "enemy_observed_mask": np.array([1.0], dtype=np.float32),
        "altitude": np.array([1.0], dtype=np.float32),
    }
    policy = OpponentPolicy("greedy_fsm", seed=0)
    no_warning = policy.act(
        {"blue_0": {**base_obs, "missile_warning": np.array([0.0], dtype=np.float32)}},
        ["blue_0"],
    )["blue_0"]
    warning = policy.act(
        {"blue_0": {**base_obs, "missile_warning": np.array([1.0], dtype=np.float32)}},
        ["blue_0"],
    )["blue_0"]
    assert not np.allclose(no_warning, warning)
    assert warning[2] >= 0.9


def test_target_assignment_deconflicts_multiple_blue_agents():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    obs = {
        "enemy_states": np.array(
            [[0.2, 0.1, 0.0], [0.3, -0.1, 0.0], [0.4, 0.2, 0.0]],
            dtype=np.float32,
        ),
        "enemy_observed_mask": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "missile_warning": np.array([0.0], dtype=np.float32),
        "altitude": np.array([1.0], dtype=np.float32),
    }
    policy = OpponentPolicy("greedy_fsm", seed=0)
    actions = policy.act({"blue_0": obs, "blue_1": obs}, ["blue_0", "blue_1"])
    assert set(actions) == {"blue_0", "blue_1"}
    assert policy.last_assigned_targets["blue_0"] != policy.last_assigned_targets["blue_1"]


def test_rule_nearest_still_available():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy = OpponentPolicy("rule_nearest", seed=0)
    action = policy.act(
        {
            "blue_0": {
                "enemy_states": np.array([[0.5, -0.2, 0.1]], dtype=np.float32),
            }
        },
        ["blue_0"],
    )["blue_0"]
    assert action.shape == (3,)
    assert np.isfinite(action).all()


def test_diagnose_blue_greedy_fsm_opponent_runs():
    output_json = "outputs/test_environment_audit/blue_greedy_fsm_opponent.json"
    result = subprocess.run(
        [
            "python",
            "scripts/diagnose_blue_greedy_fsm_opponent.py",
            "--steps",
            "5",
            "--output-json",
            output_json,
        ],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((ROOT / output_json).read_text(encoding="utf-8"))
    assert data["summary"]["nan_records"] == 0
    configs = {Path(record["config"]).name for record in data["records"]}
    assert "hetero_mav_shared_geo_3v2.yaml" in configs
    assert "hetero_mav_shared_geo_5v4.yaml" in configs
    for record in data["records"]:
        assert -1.0 <= record["blue_action_min"] <= 1.0
        assert -1.0 <= record["blue_action_max"] <= 1.0
        assert record["nan_detected"] is False
