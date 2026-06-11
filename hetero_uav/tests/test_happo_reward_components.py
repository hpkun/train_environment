import numpy as np
import pytest


REQUIRED_KEYS = {
    "safety",
    "event",
    "death_penalty",
    "mav_survival",
    "mav_support",
    "mav_attack",
    "mav_dodge",
    "uav_attack_window",
    "uav_fire",
    "uav_hit",
    "uav_dodge",
}


def test_happo_ref_v0_reward_components_have_complete_keys():
    pytest.importorskip("gymnasium")
    from uav_env import make_env

    env = make_env(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
        env_type="jsbsim_hetero",
        hetero_reward_mode="happo_ref_v0",
        max_steps=8,
    )
    try:
        _obs, _info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _obs, _rewards, _terminated, _truncated, info = env.step(actions)
    finally:
        env.close()

    components = info.get("reward_components")
    assert isinstance(components, dict)
    for aid in ("red_0", "red_1"):
        assert aid in components
        assert REQUIRED_KEYS.issubset(components[aid].keys())
        for key in REQUIRED_KEYS:
            assert isinstance(float(components[aid][key]), float)
