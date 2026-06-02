from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env


CFG_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
CFG_5V4 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"
CFG_BRMA_SENSOR = "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml"
V2_KEYS = {
    "ego_geo_state",
    "ally_geo_states",
    "enemy_geo_states",
    "enemy_observed_mask",
    "enemy_track_source",
}


def _assert_no_nan(obs: dict) -> None:
    for aid, agent_obs in obs.items():
        for key, value in agent_obs.items():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"}:
                assert not np.isnan(arr).any(), f"NaN in {aid}/{key}"


def _step_smoke(env, policy: str, steps: int = 3) -> None:
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    for _ in range(steps):
        if policy == "zero":
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        else:
            actions = {
                aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
                for aid in env.agent_ids
            }
        obs, _rew, terminated, truncated, info = env.step(actions)
        _assert_no_nan(obs)
        if all(terminated.values()) or all(truncated.values()):
            break


@pytest.mark.parametrize(
    "config_path,ally_shape,enemy_shape",
    [
        (CFG_3V2, (2, 5), (2, 5)),
        (CFG_5V4, (4, 5), (4, 5)),
    ],
)
def test_mav_shared_geo_obs_shapes_and_steps(config_path, ally_shape, enemy_shape):
    env = make_env(config_path)
    try:
        obs, info = env.reset(seed=0)
        assert info["observation_mode"] == "mav_shared_geo"
        assert V2_KEYS.issubset(obs["red_0"])
        assert obs["red_0"]["ego_geo_state"].shape == (7,)
        assert obs["red_0"]["ally_geo_states"].shape == ally_shape
        assert obs["red_0"]["enemy_geo_states"].shape == enemy_shape
        assert obs["red_0"]["enemy_track_source"].shape[-1] == 2
        mask = obs["red_0"]["enemy_observed_mask"]
        assert np.all((mask == 0.0) | (mask == 1.0))
        _assert_no_nan(obs)

        _step_smoke(env, "zero", steps=3)
        _step_smoke(env, "bounded_random", steps=3)
    finally:
        env.close()


def test_brma_sensor_config_does_not_require_v2_keys():
    env = make_env(CFG_BRMA_SENSOR)
    try:
        obs, info = env.reset(seed=0)
        assert info["observation_mode"] == "brma_sensor"
        assert V2_KEYS.isdisjoint(obs["red_0"])
    finally:
        env.close()
