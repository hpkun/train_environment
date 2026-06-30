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


def test_safe_pursuit_audit_smoke_writes_report():
    import tempfile
    import shutil
    tmp_dir = Path(tempfile.mkdtemp(prefix="safe_audit_"))
    out_dir = tmp_dir / "safe_audit"
    try:
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
        # Roll audit fields present
        assert "high_roll_active" in rows[0]
        assert "roll_recovery_active" in rows[0]
        assert "blue_roll_abs_deg" in rows[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Roll safety guard tests
# ---------------------------------------------------------------------------

def _obs_with_roll(roll_rad: float, enemy_offset_rad: float = 0.10) -> dict:
    """Build an observation with a specific ego roll."""
    ego_state = np.zeros(11, dtype=np.float32)
    ego_state[6] = 0.55
    ego_state[7] = float(np.sin(roll_rad))
    ego_state[8] = float(np.cos(roll_rad))
    ego_state[9] = 0.0
    ego_state[10] = 1.0
    enemy_states = np.zeros((3, 11), dtype=np.float32)
    enemy_states[0] = np.asarray(
        [0.125, 0.0, 0.0, enemy_offset_rad / np.pi, 0.20, 0.125, 0.50, 0.0, 1.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return {
        "ego_state": ego_state,
        "enemy_states": enemy_states,
        "death_mask": np.ones(5, dtype=np.float32),
        "altitude": np.asarray([7000.0], dtype=np.float32),
        "velocity": np.asarray([330.0, 0.0, 0.0], dtype=np.float32),
    }


def test_roll_recovery_triggers_at_high_roll():
    """When abs(roll) > 75 deg, safe_pursuit outputs roll_recovery not current_target."""
    _reset_rule_memory()
    obs = _obs_with_roll(np.deg2rad(80.0), 0.10)
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    dbg = rule_based_agent._simple_debug_state[0]
    assert dbg["desired_heading_source"] == "roll_recovery", f"got {dbg['desired_heading_source']}"
    assert dbg["roll_recovery_active"] == 1
    assert abs(float(action[1])) <= 1.0
    assert float(action[2]) >= 0.62  # throttle at max (vel_int=1.0 → 190/306)


def test_extreme_roll_recovery_triggers_above_105_deg():
    """When abs(roll) > 105 deg, safe_pursuit outputs extreme_roll_recovery."""
    _reset_rule_memory()
    obs = _obs_with_roll(np.deg2rad(110.0), 0.10)
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    dbg = rule_based_agent._simple_debug_state[0]
    assert dbg["desired_heading_source"] == "extreme_roll_recovery"
    assert dbg["extreme_roll_recovery_active"] == 1
    assert abs(float(action[1])) <= 1.0


def test_roll_recovery_does_not_update_last_seen():
    """Roll recovery must not preserve target bearing in last_seen."""
    _reset_rule_memory()
    # First step: normal pursuit to set last_seen
    obs_normal = _obs_with_roll(0.0, 0.10)
    _blue_pursuit_action_impl(
        obs_normal, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    assert 0 in rule_based_agent._simple_last_seen_bearing
    # Second step: high roll should clear last_seen
    obs_roll = _obs_with_roll(np.deg2rad(80.0), 0.10)
    _blue_pursuit_action_impl(
        obs_roll, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    assert 0 not in rule_based_agent._simple_last_seen_bearing


def test_roll_recovery_action_in_bounds():
    """action[1] must remain in [-1, 1] during roll recovery."""
    _reset_rule_memory()
    for roll_deg in [80.0, 95.0, 110.0, 130.0, 160.0]:
        obs = _obs_with_roll(np.deg2rad(roll_deg), 0.10)
        action = _blue_pursuit_action_impl(
            obs, 2, 3, 0, forced_target_idx=None,
            own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
            own_heading=0.0,
            pursuit_mode="safe_pursuit",
        )
        assert -1.0 <= float(action[1]) <= 1.0, f"roll={roll_deg} action[1]={action[1]}"
        assert float(action[2]) >= 0.62, f"roll={roll_deg} throttle={action[2]}"


def test_low_speed_safety_still_triggers():
    """Low speed safety (< 220 m/s) still triggers even with roll recovery code present."""
    _reset_rule_memory()
    obs = _base_obs()
    # Set speed to 200 m/s via ego_state[6] (normalized speed)
    obs["ego_state"][6] = 200.0 / 600.0  # ~0.333
    obs["velocity"] = np.asarray([200.0, 0.0, 0.0], dtype=np.float32)
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=0,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    dbg = rule_based_agent._simple_debug_state[0]
    assert dbg["desired_heading_source"] == "low_speed_recovery"


def test_normal_pursuit_not_affected_by_roll_guard():
    """Normal safe_pursuit still uses current_target nearest when roll is low."""
    _reset_rule_memory()
    obs = _obs_with_roll(0.0, 0.10)
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="safe_pursuit",
    )
    dbg = rule_based_agent._simple_debug_state[0]
    assert dbg["desired_heading_source"] == "current_target"
    assert dbg["roll_recovery_active"] == 0
    assert dbg["extreme_roll_recovery_active"] == 0


def test_roll_guard_not_active_in_legacy_brma_rule():
    """Legacy brma_rule (delta10) should not trigger roll_recovery."""
    _reset_rule_memory()
    obs = _obs_with_roll(np.deg2rad(80.0), 0.10)
    action = _blue_pursuit_action_impl(
        obs, 2, 3, 0, forced_target_idx=None,
        own_position=np.asarray([1000.0, 0.0, 7000.0], dtype=np.float32),
        own_heading=0.0,
        pursuit_mode="delta10",
    )
    # Legacy path should not have _simple_debug_state entry for roll_recovery
    if 0 in rule_based_agent._simple_debug_state:
        dbg = rule_based_agent._simple_debug_state[0]
        assert dbg.get("roll_recovery_active", 0) == 0
