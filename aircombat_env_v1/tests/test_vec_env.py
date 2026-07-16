import numpy as np
import pytest

from aircombat_env_v1.vec_env import SubprocVecEnv


@pytest.mark.integration
def test_vec_worker_short_run():
    with SubprocVecEnv(
            num_envs=2, base_seed=3,
            env_kwargs={"max_steps": 2}, timeout=30.0) as env:
        observations, infos = env.reset()
        assert observations.shape == (2, 20)
        assert len(infos) == 2
        observations, rewards, terminated, truncated, infos = env.step(
            np.zeros((2, 3), dtype=np.float32))
        assert observations.shape == (2, 20)
        assert np.isfinite(rewards).all()
        observations, rewards, terminated, truncated, infos = env.step(
            np.zeros((2, 3), dtype=np.float32))
        assert truncated.all()
        assert all("terminal_observation" in info for info in infos)
