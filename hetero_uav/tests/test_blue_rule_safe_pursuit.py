from __future__ import annotations

import csv
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


import rule_based_agent  # noqa: E402
from rule_based_agent import blue_coordinated_actions, _blue_pursuit_action_impl  # noqa: E402
from scripts.audit_blue_rule_control_response import _lead_bearing_from_obs  # noqa: E402


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _base_obs() -> dict:
    enemy_states = np.zeros((3, 11), dtype=np.float32)
    enemy_states[0] = np.asarray(
        [0.125, 0.0, 0.0, 0.95, 0.20, 0.125, 0.50, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return {
        "ego_state": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        "enemy_states": enemy_states,
        "death_mask": np.ones(5, dtype=np.float32),
        "altitude": np.asarray([7000.0], dtype=np.float32),
        "velocity": np.asarray([330.0, 0.0, 0.0], dtype=np.float32),
    }


def _reset_rule_memory() -> None:
    for name in (
        "_last_target_bearing",
        "_lost_target_steps",
        "_prev_heading_cmd",
        "_prev_lead_bearing",
        "_simple_last_seen_bearing",
        "_simple_lost_steps",
        "_simple_debug_state",
    ):
        getattr(rule_based_agent, name, {}).clear()


def _heading_delta(action: np.ndarray, own_heading: float = 0.0) -> float:
    return abs(_wrap_pi(float(action[1]) * math.pi - own_heading))


def test_opponent_policy_exposes_safe_pursuit_mode():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    assert "brma_rule_safe_pursuit" in OpponentPolicy.MODES


def test_default_brma_rule_path_matches_explicit_delta10():
    _reset_rule_memory()
    obs = _base_obs()
    default = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
    )
    explicit = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="delta10",
    )

    np.testing.assert_allclose(default, explicit, atol=1e-7)


def test_safe_pursuit_uses_current_ao_not_lead_for_radar_track():
    _reset_rule_memory()
    obs = _base_obs()
    delta10 = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="delta10",
    )
    safe = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    desired = float(obs["enemy_states"][0, 3]) * math.pi

    assert _heading_delta(delta10) <= math.radians(10.1)
    assert -1.0 <= float(safe[1]) <= 1.0
    assert abs(_wrap_pi(float(safe[1]) * math.pi - desired)) < math.radians(1.0)


def test_safe_pursuit_heading_ignores_target_velocity_and_ta():
    _reset_rule_memory()
    obs_a = _base_obs()
    obs_b = _base_obs()
    obs_b["enemy_states"][0, 4] = -0.95
    obs_b["enemy_states"][0, 6] = 0.05

    safe_a = _blue_pursuit_action_impl(
        obs_a, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    safe_b = _blue_pursuit_action_impl(
        obs_b, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    np.testing.assert_allclose(safe_a[1], safe_b[1], atol=1e-7)

    obs_b["enemy_states"][0, 3] = -0.5
    safe_c = _blue_pursuit_action_impl(
        obs_b, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    assert abs(float(safe_c[1]) - float(safe_a[1])) > 0.2


def test_safe_pursuit_selects_nearest_valid_target_not_weighted_score():
    _reset_rule_memory()
    obs = _base_obs()
    # Target 0 has better AO/TA but is farther; target 1 is nearest.
    obs["enemy_states"][0] = np.asarray(
        [0.10, 0.0, 0.0, 0.05, 0.80, 0.40, 0.5, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    obs["enemy_states"][1] = np.asarray(
        [0.05, 0.0, 0.0, -0.45, 0.01, 0.08, 0.5, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    assert abs(_wrap_pi(float(action[1]) * math.pi - (-0.45 * math.pi))) < math.radians(1.0)
    assert rule_based_agent._simple_debug_state[0]["selected_target_idx"] == 1


def test_safe_pursuit_coordinated_assignment_uses_simple_step_deconfliction():
    _reset_rule_memory()
    obs0 = _base_obs()
    obs1 = _base_obs()
    for obs in (obs0, obs1):
        obs["enemy_states"][0] = np.asarray(
            [0.05, 0.0, 0.0, 0.10, 0.01, 0.08, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        )
        obs["enemy_states"][1] = np.asarray(
            [0.07, 0.0, 0.0, -0.35, 0.01, 0.12, 0.5, 0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        )
    actions = blue_coordinated_actions(
        {"blue_0": obs0, "blue_1": obs1},
        num_blue=2,
        num_red=3,
        engaged_targets={"red_0"},
        own_positions={"blue_0": np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
                       "blue_1": np.asarray([0.0, 0.0, 7000.0], dtype=np.float32)},
        own_headings={"blue_0": 0.0, "blue_1": 0.0},
        pursuit_mode="safe_pursuit",
    )
    assert abs(_wrap_pi(float(actions["blue_0"][1]) * math.pi - 0.10 * math.pi)) < math.radians(1.0)
    assert abs(_wrap_pi(float(actions["blue_1"][1]) * math.pi - -0.35 * math.pi)) < math.radians(1.0)


def test_safe_pursuit_keeps_precombat_safety_branches_before_direct_heading():
    _reset_rule_memory()
    low_alt = _base_obs()
    low_alt["altitude"] = np.asarray([300.0], dtype=np.float32)

    delta10 = _blue_pursuit_action_impl(
        low_alt, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 300.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="delta10",
    )
    safe = _blue_pursuit_action_impl(
        low_alt, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 300.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    np.testing.assert_allclose(safe, delta10, atol=1e-6)


def test_safe_pursuit_does_not_keep_complex_multistage_heading_limiter():
    assert not hasattr(rule_based_agent, "_safe_pursuit_heading_limit_deg")
    assert not hasattr(rule_based_agent, "_safe_pursuit_heading_command")
    source = inspect.getsource(rule_based_agent._blue_pursuit_action_impl)
    assert "_safe_pursuit_heading_limit_deg" not in source
    assert "_safe_pursuit_heading_command" not in source
    assert "_prev_heading_cmd" not in inspect.getsource(rule_based_agent._blue_simple_pursuit_action_impl)
    assert "_prev_lead_bearing" not in inspect.getsource(rule_based_agent._blue_simple_pursuit_action_impl)


def test_safe_pursuit_does_not_override_boundary_safety():
    _reset_rule_memory()
    obs = _base_obs()
    own_pos = np.asarray([39500.0, 0.0, 7000.0], dtype=np.float32)
    safe = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=own_pos,
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )

    assert rule_based_agent._simple_debug_state[0]["desired_heading_source"] == "safety"
    assert abs(_wrap_pi(float(safe[1]) * math.pi - math.pi)) < math.radians(1.0)


def test_awacs_safe_pursuit_uses_ao_bearing_not_velocity_lead():
    _reset_rule_memory()
    obs = _base_obs()
    obs["enemy_states"][0, 4] = 0.0
    obs["enemy_states"][0, 6] = 1.0
    safe = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([0.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )

    assert abs(_wrap_pi(float(safe[1]) * math.pi - float(obs["enemy_states"][0, 3]) * math.pi)) < math.radians(1.0)


def test_safe_pursuit_updates_and_uses_last_seen_for_15_steps_then_center_cruise():
    _reset_rule_memory()
    obs = _base_obs()
    visible = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    last_heading = float(visible[1]) * math.pi
    no_target = _base_obs()
    no_target["enemy_states"][:] = 0.0

    for step in range(15):
        reacquire = _blue_pursuit_action_impl(
            no_target, 2, 3, 0, forced_target_idx=None,
            own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
            own_heading=0.0,
            pursuit_mode="safe_pursuit",
        )
        assert abs(_wrap_pi(float(reacquire[1]) * math.pi - last_heading)) < math.radians(1.0)
        assert rule_based_agent._simple_debug_state[0]["desired_heading_source"] == "reacquire_last_seen"
        assert rule_based_agent._simple_debug_state[0]["simple_lost_steps"] == step + 1

    cruise = _blue_pursuit_action_impl(
        no_target, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    assert rule_based_agent._simple_debug_state[0]["desired_heading_source"] == "center_cruise"
    assert abs(_wrap_pi(float(cruise[1]) * math.pi - math.pi)) < math.radians(1.0)


def test_primary_entrypoint_help_includes_safe_pursuit():
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for script in (
        "scripts/train_happo_reference.py",
        "scripts/train_happo_reference_parallel.py",
        "scripts/eval_happo_reference.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr[-1000:]
        assert "brma_rule_safe_pursuit" in result.stdout


def test_safe_pursuit_audit_smoke_writes_report(tmp_path):
    out_dir = tmp_path / "safe_audit"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_blue_rule_control_response.py",
            "--opponent-policy",
            "brma_rule_safe_pursuit",
            "--episodes",
            "1",
            "--max-steps",
            "3",
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )

    assert result.returncode == 0, result.stderr[-1000:]
    assert (out_dir / "blue_rule_safe_pursuit_report.md").exists()
    rows = list(csv.DictReader((out_dir / "blue_rule_safe_pursuit_steps.csv").open()))
    assert rows
    assert "safe_pursuit_mode_active" in rows[0]
