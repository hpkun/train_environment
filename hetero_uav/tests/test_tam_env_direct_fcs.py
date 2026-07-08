from __future__ import annotations

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


def test_tam_env_action_space_is_direct_fcs_box4():
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
        assert info["tam_control_mode"] == "direct_fcs_box4"
        assert info["tam_action_order"] == ["throttle", "aileron", "elevator", "rudder"]
        assert "tam_fcs_commands" in info
        assert "tam_control_diagnostics" in info
        assert info["tam_parent_action_overrides_enabled"] is False
        assert set(info["tam_control_diagnostics"]) == set(env.agent_ids)
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
        assert info["tam_control_mode"] == "direct_fcs_box4"
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
            assert info["tam_control_mode"] == "direct_fcs_box4"
            assert "tam_applied_fcs" in info
    finally:
        env.close()
