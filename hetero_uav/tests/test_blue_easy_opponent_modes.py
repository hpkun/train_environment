from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _obs_with_target(
    *,
    range_norm: float = 0.20,
    lateral_norm: float = 0.30,
    vertical_norm: float = 0.02,
    missile_warning: float = 0.0,
) -> dict:
    enemy_states = np.zeros((3, 11), dtype=np.float32)
    enemy_states[0, :3] = np.asarray([range_norm, lateral_norm, vertical_norm], dtype=np.float32)
    enemy_alive = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return {
        "ego_geo_state": np.asarray([0.0, 0.0, 7000.0, 0.0, 0.0, 0.0, 320.0], dtype=np.float32),
        "ego_state": np.zeros(11, dtype=np.float32),
        "enemy_states": enemy_states,
        "enemy_alive_mask": enemy_alive,
        "enemy_observed_mask": enemy_alive,
        "missile_warning": np.asarray([missile_warning], dtype=np.float32),
    }


def test_easy_modes_are_registered_and_constructible():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    assert "tam_greedy_easy" in OpponentPolicy.MODES
    assert "brma_rule_safe_pursuit_easy" in OpponentPolicy.MODES
    assert OpponentPolicy("tam_greedy_easy").mode == "tam_greedy_easy"
    assert OpponentPolicy("brma_rule_safe_pursuit_easy").mode == "brma_rule_safe_pursuit_easy"


def test_tam_greedy_easy_outputs_finite_clipped_actions_for_missing_and_normal_obs():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy = OpponentPolicy("tam_greedy_easy", seed=7)
    actions = policy.act(
        {
            "blue_0": {},
            "blue_1": _obs_with_target(range_norm=0.18, lateral_norm=0.8, vertical_norm=0.5),
        },
        ["blue_0", "blue_1"],
    )
    for action in actions.values():
        arr = np.asarray(action, dtype=np.float32)
        assert arr.shape == (3,)
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= -1.0
        assert float(arr.max()) <= 1.0
        assert -0.25 <= float(arr[0]) <= 0.25
        assert float(arr[2]) <= 0.65


def test_tam_greedy_easy_enters_extend_after_close_pass_and_reset_clears_state():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    policy = OpponentPolicy("tam_greedy_easy", seed=8)
    obs = {"blue_0": _obs_with_target(range_norm=0.03, lateral_norm=0.0)}
    policy.act(obs, ["blue_0"])
    assert policy.easy_mode_states
    assert policy.easy_extend_steps
    policy.reset_memory()
    assert not policy.easy_mode_states
    assert not policy.easy_extend_steps


def test_safe_pursuit_easy_postprocess_limits_speed_pitch_and_heading_delta():
    from algorithms.mappo.opponent_policy import OpponentPolicy, _wrap_heading_norm

    policy = OpponentPolicy("brma_rule_safe_pursuit_easy", seed=9)
    action = policy._postprocess_easy_safe_pursuit_action(
        np.asarray([0.9, 0.8, 1.0], dtype=np.float32),
        current_heading_norm=0.0,
    )
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert -0.35 <= float(action[0]) <= 0.35
    assert float(action[2]) <= 0.70
    assert abs(_wrap_heading_norm(float(action[1]) - 0.0)) <= 0.150001


def test_easy_modes_are_available_in_main_cli_choices():
    result = subprocess.run(
        [sys.executable, "scripts/train_happo_reference.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "tam_greedy_easy" in result.stdout
    assert "brma_rule_safe_pursuit_easy" in result.stdout


def test_diagnose_blue_pressure_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_blue_pressure.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--opponent-policies" in result.stdout
    assert "tam_greedy_easy" in result.stdout
