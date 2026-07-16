import numpy as np
import pytest

from aircombat_env_v1.env import AirCombat1v1Env


pytestmark = pytest.mark.integration


def test_reset_seed_is_reproducible():
    env = AirCombat1v1Env(randomize=True, max_steps=2)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    assert np.allclose(first, second)
    env.close()


def test_random_actions_short_run_is_finite():
    env = AirCombat1v1Env(max_steps=5)
    observation, _ = env.reset(seed=0)
    for _ in range(5):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert np.isfinite(info["distance_m"])
        if terminated or truncated:
            break
    env.close()


def test_max_steps_sets_truncated():
    env = AirCombat1v1Env(max_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert info["event"] == "timeout"
    env.close()
