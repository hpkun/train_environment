"""Test paper-aligned 3v2 and 5v4 configs: types, missiles, masks, shapes."""
from __future__ import annotations

import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter

CFG_3V2 = "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml"
CFG_5V4 = "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml"


@pytest.fixture
def env_3v2():
    env = make_env(CFG_3V2, env_type="jsbsim_hetero", max_steps=10)
    yield env
    env.close()


@pytest.fixture
def env_5v4():
    env = make_env(CFG_5V4, env_type="jsbsim_hetero", max_steps=10)
    yield env
    env.close()


def test_configs_exist():
    from pathlib import Path
    assert Path(CFG_3V2).exists()
    assert Path(CFG_5V4).exists()


def test_3v2_reset(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    assert len(obs) == 5  # 3 red + 2 blue


def test_5v4_reset(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    assert len(obs) == 9


def test_3v2_red0_is_mav_a4(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    assert info["agent_types"]["red_0"] == "mav"
    assert info["agent_models"]["red_0"] == "A-4"


def test_3v2_mav_missiles_zero(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    m = env_3v2._num_missiles_for("red_0")
    assert m == 0


def test_3v2_uav_missiles_two(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    assert env_3v2._num_missiles_for("red_1") == 2
    assert env_3v2._num_missiles_for("red_2") == 2


def test_5v4_red0_is_mav_a4(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    assert info["agent_types"]["red_0"] == "mav"
    assert info["agent_models"]["red_0"] == "A-4"


def test_5v4_mav_missiles_zero(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    assert env_5v4._num_missiles_for("red_0") == 0


def test_5v4_uav_missiles_two(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    for i in range(1, 5):
        assert env_5v4._num_missiles_for(f"red_{i}") == 2


def test_blue_missiles_two(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    for bid in ["blue_0", "blue_1"]:
        assert env_3v2._num_missiles_for(bid) == 2


def test_adapter_shapes_3v2(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    a = HeteroObsAdapter()
    result = a.adapt_all(obs, info=info,
                         red_ids=env_3v2.red_ids,
                         blue_ids=env_3v2.blue_ids)
    assert result["actor_obs"]["red_0"].shape == (140,)
    assert result["critic_state"].shape == (700,)


def test_adapter_shapes_5v4(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    a = HeteroObsAdapter()
    result = a.adapt_all(obs, info=info,
                         red_ids=env_5v4.red_ids,
                         blue_ids=env_5v4.blue_ids)
    assert result["actor_obs"]["red_0"].shape == (140,)
    assert result["critic_state"].shape == (700,)


def test_3v2_red_valid_mask(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    a = HeteroObsAdapter()
    result = a.adapt_all(obs, info=info,
                         red_ids=env_3v2.red_ids,
                         blue_ids=env_3v2.blue_ids)
    assert np.allclose(result["red_valid_mask"], [1, 1, 1, 0, 0])


def test_5v4_red_valid_mask(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    a = HeteroObsAdapter()
    result = a.adapt_all(obs, info=info,
                         red_ids=env_5v4.red_ids,
                         blue_ids=env_5v4.blue_ids)
    assert np.allclose(result["red_valid_mask"], [1, 1, 1, 1, 1])


def test_zero_step_3v2_no_nan(env_3v2):
    obs, info = env_3v2.reset(seed=0)
    a = HeteroObsAdapter()
    for _ in range(3):
        actions = {aid: np.zeros(3, dtype=np.float32)
                   for aid in env_3v2.agent_ids}
        obs, _, _, _, info = env_3v2.step(actions)
        r = a.adapt_all(obs, info=info,
                        red_ids=env_3v2.red_ids,
                        blue_ids=env_3v2.blue_ids)
        assert not np.isnan(r["critic_state"]).any()


def test_zero_step_5v4_no_nan(env_5v4):
    obs, info = env_5v4.reset(seed=0)
    a = HeteroObsAdapter()
    for _ in range(3):
        actions = {aid: np.zeros(3, dtype=np.float32)
                   for aid in env_5v4.agent_ids}
        obs, _, _, _, info = env_5v4.step(actions)
        r = a.adapt_all(obs, info=info,
                        red_ids=env_5v4.red_ids,
                        blue_ids=env_5v4.blue_ids)
        assert not np.isnan(r["critic_state"]).any()
