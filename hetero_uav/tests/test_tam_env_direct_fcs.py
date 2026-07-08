from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest
from gymnasium import spaces


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _make_tam_env(max_steps: int = 20, **kwargs):
    from uav_env import make_env

    return make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml",
        env_type="tam",
        max_steps=max_steps,
        suppress_jsbsim_output=True,
        num_missiles_per_plane=0,
        **kwargs,
    )


def test_tam_env_entrypoints_construct():
    from uav_env import TamCombatEnv, make_tam_env

    env = make_tam_env(max_num_red=1, max_num_blue=1, max_steps=5, suppress_jsbsim_output=True)
    try:
        assert isinstance(env, TamCombatEnv)
    finally:
        env.close()


def test_tam_env_factory_env_type_tam():
    from uav_env import TamCombatEnv, make_env

    env = make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml",
        env_type="tam",
        max_steps=5,
        suppress_jsbsim_output=True,
    )
    try:
        assert isinstance(env, TamCombatEnv)
    finally:
        env.close()


def test_tam_env_action_space_is_raw_direct_fcs():
    env = _make_tam_env(max_steps=5)
    try:
        assert set(env.action_space.spaces) == set(env.agent_ids)
        for aid in env.agent_ids:
            space = env.action_space.spaces[aid]
            assert isinstance(space, spaces.Box)
            assert space.shape == (4,)
            assert space.dtype == np.float32
            np.testing.assert_allclose(space.low, np.full(4, -1.0, dtype=np.float32))
            np.testing.assert_allclose(space.high, np.full(4, 1.0, dtype=np.float32))
    finally:
        env.close()


def test_tam_env_reset_and_neutral_steps():
    env = _make_tam_env(max_steps=8)
    try:
        obs, info = env.reset(seed=0)
        assert set(obs) == set(env.agent_ids)
        assert isinstance(info, dict)
        neutral = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(3):
            obs, rewards, terminated, truncated, info = env.step(neutral)
            assert set(obs) == set(env.agent_ids)
            assert set(rewards) == set(env.agent_ids)
            assert set(terminated) == set(env.agent_ids)
            assert set(truncated) == set(env.agent_ids)
            assert "reward_components" in info
    finally:
        env.close()


def test_tam_reset_clears_action_logs():
    env = _make_tam_env(max_steps=8)
    try:
        env.reset(seed=10)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_raw_actions"]

        _obs, info = env.reset(seed=11)
        assert info["tam_raw_actions"] == {}
        assert info["tam_sanitized_actions"] == {}
        assert info["tam_fcs_commands"] == {}
        assert info["tam_applied_fcs"] == {}
        assert info["tam_action_warnings"] == {}
        assert env._last_effective_actions == {}
        assert env._last_action_trim_applied == {}
        assert info["tam_control_mode"] == "raw_direct_fcs"
        assert "tam_control_diagnostics" in info
    finally:
        env.close()


def test_tam_env_strict_rejects_3d_action():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=2)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions[env.agent_ids[0]] = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError, match="shape=\\(4,\\)"):
            env.step(actions)
    finally:
        env.close()


@pytest.mark.parametrize(
    "bad_action",
    [
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((4, 1), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.float32(0.0),
        [[0.0, 0.0, 0.0, 0.0]],
    ],
)
def test_tam_strict_rejects_2d_shape_even_size4(bad_action):
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=12)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        aid = env.agent_ids[0]
        actions[aid] = bad_action
        with pytest.raises(ValueError, match=rf"{aid}.*shape=\(4,\).*got shape"):
            env.step(actions)
    finally:
        env.close()


def test_tam_env_strict_rejects_missing_alive_action():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=3)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions.pop(env.agent_ids[0])
        with pytest.raises(ValueError, match="missing action"):
            env.step(actions)
    finally:
        env.close()


def test_tam_non_strict_padding_allows_flatten_only_when_enabled():
    env = _make_tam_env(
        max_steps=5,
        strict_action_shape=False,
        tam_allow_non_strict_padding=True,
    )
    try:
        env.reset(seed=13)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        aid = env.agent_ids[0]
        actions[aid] = np.zeros((1, 4), dtype=np.float32)
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        warnings = info["tam_action_warnings"][aid]
        assert any("non_strict" in warning or "reshaped" in warning for warning in warnings)
    finally:
        env.close()

    env = _make_tam_env(
        max_steps=5,
        strict_action_shape=False,
        tam_allow_non_strict_padding=False,
    )
    try:
        env.reset(seed=14)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions[env.agent_ids[0]] = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="shape=\\(4,\\)"):
            env.step(actions)
    finally:
        env.close()


def test_tam_env_non_strict_padding_is_explicit():
    env = _make_tam_env(
        max_steps=5,
        strict_action_shape=False,
        tam_allow_non_strict_padding=True,
    )
    try:
        env.reset(seed=4)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions[env.agent_ids[0]] = np.array([0.5, 0.1], dtype=np.float32)
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        warnings = info["tam_action_warnings"][env.agent_ids[0]]
        assert any("padded" in warning for warning in warnings)
        assert len(info["tam_sanitized_actions"][env.agent_ids[0]]) == 4
    finally:
        env.close()


def test_tam_action_mapping_endpoints():
    env = _make_tam_env(max_steps=5)
    try:
        np.testing.assert_allclose(
            env._tam_action_to_fcs(np.array([-1.0, -2.0, 2.0, 0.0], dtype=np.float32)),
            (0.4, -1.0, 1.0, 0.0),
        )
        np.testing.assert_allclose(
            env._tam_action_to_fcs(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            (0.65, 0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            env._tam_action_to_fcs(np.array([1.0, 2.0, -2.0, 0.5], dtype=np.float32)),
            (0.9, 1.0, -1.0, 0.5),
        )
    finally:
        env.close()


def test_tam_action_nan_inf_sanitized():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=5)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        actions[env.agent_ids[0]] = np.array([np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_sanitized_actions"][env.agent_ids[0]] == [0.0, 1.0, -1.0, 0.25]
        warnings = info["tam_action_warnings"][env.agent_ids[0]]
        assert any("non-finite" in warning for warning in warnings)
    finally:
        env.close()


def test_tam_info_contains_direct_fcs_diagnostics():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=6)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_control_mode"] == "raw_direct_fcs"
        assert info["tam_action_order"] == ["throttle", "aileron", "elevator", "rudder"]
        assert "tam_fcs_commands" in info
        assert "tam_control_diagnostics" in info
        assert info["tam_parent_action_overrides_enabled"] is False
        assert set(info["tam_control_diagnostics"]) == set(env.agent_ids)
    finally:
        env.close()


def test_tam_control_diagnostics_g_load_total_fields():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=15)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        for diag in info["tam_control_diagnostics"].values():
            for key in ("g_load", "g_load_total", "g_load_x", "g_load_y", "g_load_z"):
                assert key in diag
            xyz = [diag["g_load_x"], diag["g_load_y"], diag["g_load_z"]]
            if all(value is not None for value in xyz):
                expected = float(np.sqrt(sum(float(value) ** 2 for value in xyz)))
                assert diag["g_load_total"] == pytest.approx(expected)
                assert diag["g_load"] == pytest.approx(expected)
    finally:
        env.close()


def test_tam_effective_actions_logged_as_box4():
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=7)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        env.step(actions)
        for aid in env.agent_ids:
            sim = env._get_sim(aid)
            if sim is not None and sim.is_alive:
                assert len(env._last_effective_actions[aid]) == 4
                assert env._last_action_trim_applied[aid] == [0.0, 0.0, 0.0, 0.0]
    finally:
        env.close()


def test_tam_no_pid_target_fields_required(monkeypatch):
    env = _make_tam_env(max_steps=5)
    try:
        env.reset(seed=8)

        def fail_compute_control(*_args, **_kwargs):
            raise AssertionError("PID compute_control should not be called by tam_env")

        for pid in env.pid_controllers.values():
            monkeypatch.setattr(pid, "compute_control", fail_compute_control)

        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_control_mode"] == "raw_direct_fcs"
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.1, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.0, 0.1, 0.0], dtype=np.float32),
        np.array([0.5, 0.0, 0.0, 0.1], dtype=np.float32),
    ],
)
def test_tam_env_direct_fcs_axis_smoke(action: np.ndarray):
    env = _make_tam_env(max_steps=8)
    try:
        env.reset(seed=1)
        actions = {aid: action.copy() for aid in env.agent_ids}
        for _ in range(3):
            _obs, _rewards, terminated, truncated, _info = env.step(actions)
            assert not all(terminated.values())
            assert not all(truncated.values())
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, -0.2, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, -0.2], dtype=np.float32),
    ],
)
def test_tam_direct_axis_smoke_longer(action: np.ndarray):
    env = _make_tam_env(max_steps=15)
    try:
        env.reset(seed=9)
        actions = {aid: action.copy() for aid in env.agent_ids}
        for _ in range(10):
            _obs, _rewards, _terminated, _truncated, info = env.step(actions)
            assert info["tam_control_mode"] == "raw_direct_fcs"
            assert "tam_applied_fcs" in info
    finally:
        env.close()


def test_tam_response_script_imports():
    from scripts import diagnose_tam_direct_fcs_response as diag

    parser = diag.build_parser()
    args = parser.parse_args(["--steps", "1", "--scenario", "neutral"])
    assert args.steps == 1
    assert args.scenario == "neutral"


def test_tam_response_script_short_no_missile_run(tmp_path):
    from scripts import diagnose_tam_direct_fcs_response as diag

    out_csv = tmp_path / "tam_response.csv"
    rc = diag.main([
        "--steps", "1",
        "--scenario", "neutral",
        "--output-csv", str(out_csv),
    ])
    assert rc == 0
    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert rows
    assert "initial_frame" in rows[0]
    assert "g_load_total" in rows[0]
    assert any(row["step"] == "-1" and row["initial_frame"] == "True" for row in rows)


def test_tam_maneuver_fcs_action_space_still_box4():
    env = _make_tam_env(max_steps=5, tam_control_mode="maneuver_fcs")
    try:
        for aid in env.agent_ids:
            assert env.action_space.spaces[aid].shape == (4,)
    finally:
        env.close()


def test_tam_maneuver_fcs_zero_action_semantics():
    env = _make_tam_env(max_steps=5, tam_control_mode="maneuver_fcs")
    try:
        commands = env._tam_maneuver_action_to_commands(
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        assert commands == pytest.approx((0.65, 0.0, 1.0, 0.0))
    finally:
        env.close()


def test_tam_maneuver_fcs_does_not_call_pid(monkeypatch):
    env = _make_tam_env(max_steps=5, tam_control_mode="maneuver_fcs")
    try:
        env.reset(seed=20)

        def fail_compute_control(*_args, **_kwargs):
            raise AssertionError("PID compute_control should not be called by maneuver_fcs")

        for pid in env.pid_controllers.values():
            monkeypatch.setattr(pid, "compute_control", fail_compute_control)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_control_mode"] == "maneuver_fcs"
    finally:
        env.close()


def test_tam_maneuver_fcs_info_contains_commands_and_terms():
    env = _make_tam_env(max_steps=5, tam_control_mode="maneuver_fcs")
    try:
        env.reset(seed=21)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
        assert info["tam_command_semantics"] == [
            "throttle", "bank_cmd_rad", "nz_cmd_g", "yaw_rate_cmd_rad_s"
        ]
        aid = env.agent_ids[0]
        assert len(info["tam_control_commands"][aid]) == 4
        terms = info["tam_controller_terms"][aid]
        for key in (
            "bank_error_rad",
            "roll_rate_rad_s",
            "nz_cmd_g",
            "nz_current_g",
            "nz_error_g",
            "pitch_rate_rad_s",
            "yaw_rate_cmd_rad_s",
            "yaw_rate_rad_s",
            "raw_aileron",
            "raw_elevator",
            "raw_rudder",
            "clipped_aileron",
            "clipped_elevator",
            "clipped_rudder",
        ):
            assert key in terms
    finally:
        env.close()


def test_tam_maneuver_fcs_short_neutral_smoke():
    env = _make_tam_env(max_steps=12, tam_control_mode="maneuver_fcs")
    try:
        env.reset(seed=22)
        actions = {aid: np.zeros(4, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(10):
            _obs, _rewards, _terminated, _truncated, info = env.step(actions)
            assert info["tam_control_mode"] == "maneuver_fcs"
            assert "tam_applied_fcs" in info
    finally:
        env.close()


def test_response_script_supports_maneuver_fcs(tmp_path):
    from scripts import diagnose_tam_direct_fcs_response as diag

    out_csv = tmp_path / "tam_maneuver_response.csv"
    rc = diag.main([
        "--steps", "1",
        "--scenario", "neutral",
        "--tam-control-mode", "maneuver_fcs",
        "--output-csv", str(out_csv),
    ])
    assert rc == 0
    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert rows
    assert rows[0]["tam_control_mode"] == "maneuver_fcs"
    assert "tam_command_semantics" in rows[0]


def test_tam_response_script_wrapped_angle_delta():
    from scripts import diagnose_tam_direct_fcs_response as diag

    assert diag._wrapped_angle_delta(-np.pi + 0.1, np.pi - 0.1) == pytest.approx(0.2)
    assert diag._wrapped_angle_delta(np.pi - 0.1, -np.pi + 0.1) == pytest.approx(-0.2)
