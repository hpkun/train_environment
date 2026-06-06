from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _find_python():
    """Find a Python executable that can import gymnasium."""
    candidates = []
    # 1. sys.executable (correct when running under brmamappo)
    candidates.append(sys.executable)
    # 2. 'python' from PATH (may be shadowed by Anaconda in subprocess)
    found = shutil.which("python")
    if found and found not in candidates:
        candidates.append(found)
    # Test each candidate
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import gymnasium"],
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


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


def test_greedy_fsm_reads_optional_env_ownship_context():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    class MockEnv:
        def refresh_engaged_targets(self):
            return {"red_1"}

        def get_blue_own_kinematics(self):
            return {
                "blue_0": {
                    "heading": np.pi / 2.0,
                    "position": np.array([20000.0, 0.0, 6000.0], dtype=np.float32),
                }
            }

        def get_blue_own_positions(self):
            return {
                "blue_0": np.array([20000.0, 0.0, 6000.0], dtype=np.float32),
                "blue_1": np.array([0.0, 0.0, 6000.0], dtype=np.float32),
            }

    policy = OpponentPolicy("greedy_fsm", seed=0)
    action = policy.act({"blue_0": {}}, ["blue_0"], env=MockEnv())["blue_0"]
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert policy.used_env_refresh_engaged_targets is True
    assert policy.used_env_own_kinematics is True
    assert policy.used_env_own_positions is True
    assert policy.last_states["blue_0"] == "missing_obs"


def test_search_acquire_uses_env_heading_when_no_target_visible():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    class MockEnv:
        def get_blue_own_kinematics(self):
            return {"blue_0": {"heading": np.pi / 2.0}}

        def get_blue_own_positions(self):
            return {"blue_0": np.array([0.0, 0.0, 6000.0], dtype=np.float32)}

    policy = OpponentPolicy("greedy_fsm", seed=0)
    obs = {
        "blue_0": {
            "enemy_states": np.zeros((1, 3), dtype=np.float32),
            "enemy_observed_mask": np.zeros(1, dtype=np.float32),
            "altitude": np.array([1.0], dtype=np.float32),
            "ego_geo_state": np.zeros(6, dtype=np.float32),
        }
    }
    action = policy.act(obs, ["blue_0"], env=MockEnv())["blue_0"]
    assert policy.last_states["blue_0"] == "search_acquire"
    assert np.isclose(action[1], 0.52, atol=1e-5)


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


def test_close_range_diagnostic_config_exists():
    path = (
        ROOT
        / "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml"
    )
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "enable_gcas_for_blue: true" in text
    assert "max_num_red: 3" in text
    assert "max_num_blue: 2" in text


def test_diagnose_blue_greedy_fsm_opponent_runs():
    output_json = "outputs/test_environment_audit/blue_greedy_fsm_opponent_close_range.json"
    result = subprocess.run(
        [
            _find_python(),
            "scripts/diagnose_blue_greedy_fsm_opponent.py",
            "--steps",
            "20",
            "--include-close-range",
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
    assert "hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml" in configs
    for record in data["records"]:
        assert -1.0 <= record["blue_action_min"] <= 1.0
        assert -1.0 <= record["blue_action_max"] <= 1.0
        assert record["nan_detected"] is False
        assert "blue_action_mean" in record
        assert "state_counts" in record
        assert "assigned_target_counts" in record
        assert "used_env_refresh_engaged_targets" in record
        assert "used_env_own_kinematics" in record
        assert "used_env_own_positions" in record
        assert record["used_env_refresh_engaged_targets"] is True
        assert record["used_env_own_kinematics"] is True
        assert record["used_env_own_positions"] is True
    close_records = [
        record
        for record in data["records"]
        if Path(record["config"]).name
        == "hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml"
    ]
    assert close_records
    assert any(record["state_counts"] for record in close_records)
