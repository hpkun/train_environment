from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.adapters import HeteroObsAdapterV2


CFG_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
CFG_5V4 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml"
CFG_BRMA_SENSOR = "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml"


def test_adapter_v2_constants():
    adapter = HeteroObsAdapterV2()
    assert adapter.flat_actor_obs_dim == 96
    assert adapter.critic_state_dim == 480
    assert adapter.ego_feature_dim == 12
    assert adapter.ally_entity_dim == 9
    assert adapter.enemy_entity_dim == 7


def test_adapter_v2_3v2_shapes():
    env = make_env(CFG_3V2)
    try:
        obs, info = env.reset(seed=0)
        adapter = HeteroObsAdapterV2()
        out = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        assert np.allclose(out["red_valid_mask"], [1, 1, 1, 0, 0])
        assert out["critic_state"].shape == (480,)
        for rid in env.red_ids:
            assert out["actor_obs"][rid].shape == (96,)
    finally:
        env.close()


def test_adapter_v2_5v4_shapes():
    env = make_env(CFG_5V4)
    try:
        obs, info = env.reset(seed=0)
        adapter = HeteroObsAdapterV2()
        out = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        assert np.allclose(out["red_valid_mask"], [1, 1, 1, 1, 1])
        assert out["critic_state"].shape == (480,)
        for rid in env.red_ids:
            assert out["actor_obs"][rid].shape == (96,)
    finally:
        env.close()


def _fake_obs(source, observed, alive=1.0):
    return {
        "ego_geo_state": np.ones(7, dtype=np.float32),
        "ego_role": np.array([0, 1, 0, 0], dtype=np.float32),
        "missile_warning": np.array([0], dtype=np.float32),
        "ally_geo_states": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        "ally_alive_mask": np.array([1], dtype=np.float32),
        "ally_roles": np.array([[0, 1, 0, 0]], dtype=np.float32),
        "enemy_geo_states": np.array([[0.2, 0.1, 0.5, 0.3, 0.4]], dtype=np.float32),
        "enemy_alive_mask": np.array([alive], dtype=np.float32),
        "enemy_track_source": np.array([source], dtype=np.float32),
        "enemy_observed_mask": np.array([observed], dtype=np.float32),
    }


@pytest.mark.parametrize("source", [[1, 0], [0, 1]])
def test_adapter_v2_source_preserved(source):
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs(source, 1.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    np.testing.assert_array_equal(out["enemy_entities"][0, -2:], np.asarray(source, dtype=np.float32))


def test_adapter_v2_alive_but_unobserved_enemy_keeps_alive_mask_and_zero_feature():
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs([0, 0], observed=0.0, alive=1.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    assert out["enemy_valid_mask"][0] == 1.0
    assert out["enemy_alive_mask"][0] == 1.0
    assert out["enemy_observed_mask"][0] == 0.0
    assert np.allclose(out["enemy_entities"][0], 0.0)


def test_adapter_v2_alive_observed_own_contains_geo_and_source():
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs([1, 0], observed=1.0, alive=1.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    assert out["enemy_alive_mask"][0] == 1.0
    assert out["enemy_observed_mask"][0] == 1.0
    assert not np.allclose(out["enemy_entities"][0, :5], 0.0)
    np.testing.assert_array_equal(out["enemy_entities"][0, -2:], np.array([1, 0], dtype=np.float32))


def test_adapter_v2_alive_observed_mav_shared_contains_geo_and_source():
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs([0, 1], observed=1.0, alive=1.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    assert out["enemy_alive_mask"][0] == 1.0
    assert out["enemy_observed_mask"][0] == 1.0
    assert not np.allclose(out["enemy_entities"][0, :5], 0.0)
    np.testing.assert_array_equal(out["enemy_entities"][0, -2:], np.array([0, 1], dtype=np.float32))


def test_adapter_v2_dead_real_enemy_zero_feature():
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs([1, 0], observed=0.0, alive=0.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    assert out["enemy_valid_mask"][0] == 1.0
    assert out["enemy_alive_mask"][0] == 0.0
    assert out["enemy_observed_mask"][0] == 0.0
    assert np.allclose(out["enemy_entities"][0], 0.0)


def test_adapter_v2_padding_enemy_zero_masks():
    adapter = HeteroObsAdapterV2()
    out = adapter.adapt_agent(
        "red_0", _fake_obs([1, 0], observed=1.0, alive=1.0),
        red_ids=["red_0", "red_1"], blue_ids=["blue_0"])
    assert out["enemy_valid_mask"][1] == 0.0
    assert out["enemy_alive_mask"][1] == 0.0
    assert out["enemy_observed_mask"][1] == 0.0
    assert np.allclose(out["enemy_entities"][1], 0.0)


def test_adapter_v2_rejects_brma_sensor_obs():
    env = make_env(CFG_BRMA_SENSOR)
    try:
        obs, info = env.reset(seed=0)
        adapter = HeteroObsAdapterV2()
        with pytest.raises(ValueError, match="observation_mode='mav_shared_geo'"):
            adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    finally:
        env.close()
