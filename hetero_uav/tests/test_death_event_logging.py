import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from uav_env import make_env


def test_death_events_field_exists_and_has_stable_schema():
    env = make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
        env_type="jsbsim_hetero",
        max_steps=8,
    )
    try:
        obs, info = env.reset(seed=0)
        assert "death_events" in info
        assert isinstance(info["death_events"], list)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(4):
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert "death_events" in info
            assert isinstance(info["death_events"], list)
            for event in info["death_events"]:
                for key in [
                    "agent_id",
                    "death_reason",
                    "death_reason_source",
                    "altitude",
                    "roll_deg",
                    "pitch_deg",
                ]:
                    assert key in event
        assert env.action_space["red_0"].shape == (3,)
    finally:
        env.close()
