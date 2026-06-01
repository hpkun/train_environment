from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env
from scripts.diagnose_jsbsim_controller import run_diagnostic


def test_a4_level_10_seconds_no_crash():
    pytest.importorskip("jsbsim")
    row = run_diagnostic("A-4", "level", np.array([0.0, 0.0, 0.5], dtype=np.float32), 10.0)
    assert row["alive"]
    assert not row["crashed"]
    assert not row["nan_detected"]


def test_f16_level_10_seconds_no_crash():
    pytest.importorskip("jsbsim")
    row = run_diagnostic("F-16", "level", np.array([0.0, 0.0, 0.5], dtype=np.float32), 10.0)
    assert row["alive"]
    assert not row["crashed"]
    assert not row["nan_detected"]


def test_a4_turn_right_changes_heading():
    pytest.importorskip("jsbsim")
    row = run_diagnostic("A-4", "turn_right", np.array([0.0, 0.5, 0.5], dtype=np.float32), 10.0)
    assert abs(row["heading_delta"]) > 0.01
    assert not row["nan_detected"]


def test_f16_speed_up_increases_speed():
    pytest.importorskip("jsbsim")
    row = run_diagnostic("F-16", "speed_up", np.array([0.0, 0.0, 1.0], dtype=np.float32), 10.0)
    assert row["final_speed"] > row["initial_speed"]
    assert not row["nan_detected"]


def test_jsbsim_env_debug_steps_20():
    pytest.importorskip("jsbsim")
    env = make_env("uav_env/configs/hetero_2v2_jsbsim_debug.yaml", episode_limit=20)
    try:
        _obs, _info = env.reset(seed=0)
        for _ in range(20):
            actions = {aid: np.zeros(env.action_shape, dtype=np.float32) for aid in env.agent_ids}
            _obs, _rewards, terminated, truncated, _info = env.step(actions)
            for agent in env.task.agents:
                values = np.concatenate([
                    agent.position.astype(np.float64),
                    agent.velocity.astype(np.float64),
                    np.array([agent.pitch, agent.roll, agent.heading, agent.speed], dtype=np.float64),
                ])
                assert np.isfinite(values).all()
            if all(terminated.get(aid, False) or truncated.get(aid, False) for aid in env.agent_ids):
                break
    finally:
        env.close()
