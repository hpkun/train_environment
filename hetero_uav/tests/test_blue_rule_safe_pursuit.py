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
from rule_based_agent import _blue_pursuit_action_impl  # noqa: E402
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


def _heading_delta(action: np.ndarray, own_heading: float = 0.0) -> float:
    return abs(_wrap_pi(float(action[1]) * math.pi - own_heading))


def test_opponent_policy_exposes_safe_pursuit_mode():
    from algorithms.mappo.opponent_policy import OpponentPolicy

    assert "brma_rule_safe_pursuit" in OpponentPolicy.MODES


def test_default_brma_rule_path_matches_explicit_delta10():
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


def test_safe_pursuit_uses_absolute_red_action_heading_bounds_for_radar_track():
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
    desired = _lead_bearing_from_obs(obs, obs["enemy_states"][0], 0.0)

    assert _heading_delta(delta10) <= math.radians(10.1)
    assert -1.0 <= float(safe[1]) <= 1.0
    assert abs(_wrap_pi(float(safe[1]) * math.pi - desired)) < math.radians(1.0)


def test_safe_pursuit_keeps_precombat_safety_branches_before_direct_heading():
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


def test_safe_pursuit_does_not_override_boundary_safety():
    obs = _base_obs()
    own_pos = np.asarray([39500.0, 0.0, 7000.0], dtype=np.float32)
    delta10 = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=own_pos,
        own_heading=0.0,
        pursuit_mode="delta10",
    )
    safe = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=own_pos,
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )

    np.testing.assert_allclose(safe, delta10, atol=1e-6)


def test_awacs_safe_pursuit_uses_ao_bearing_not_velocity_lead():
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
