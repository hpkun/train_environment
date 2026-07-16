import numpy as np

from aircombat_env_v1.env import AirCombat1v1Env


def test_numerical_invalid_returns_previous_finite_observation_and_reward():
    env = AirCombat1v1Env(max_steps=2)
    observation, _ = env.reset(seed=1)
    original = env.red_sim.run

    def invalid_run():
        state = original()
        state["altitude"] = np.nan
        return state

    env.red_sim.run = invalid_run
    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32))
    env.close()
    np.testing.assert_array_equal(next_observation, observation)
    assert np.isfinite(next_observation).all()
    assert np.isfinite(reward)
    assert terminated and not truncated
    assert info["event"] == "red_numerical_invalid"
    assert info["numerical_invalid"] is True
    assert info["physics_exception"] is False
