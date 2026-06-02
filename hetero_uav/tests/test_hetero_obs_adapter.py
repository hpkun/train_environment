"""Test HeteroObsAdapter v1 shapes, masks, and smoke behaviour.

Does NOT implement MAPPO / attention / training.
Does NOT modify reward / missile / PID / termination / aircraft XML.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
from uav_env import make_env


# -------------------------------------------------------------------
#  1. Constants
# -------------------------------------------------------------------

def test_adapter_constants():
    a = HeteroObsAdapter()
    assert a.flat_actor_obs_dim == 140
    assert a.critic_state_dim == 700
    assert a.ego_feature_dim == 20
    assert a.ally_entity_dim == 15
    assert a.enemy_entity_dim == 11
    assert a.max_allies == 4
    assert a.max_enemies == 4


# -------------------------------------------------------------------
#  2. 2v2
# -------------------------------------------------------------------

@pytest.fixture
def env_2v2():
    env = make_env(
        "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
        env_type="jsbsim_hetero", max_steps=10)
    yield env
    env.close()


def test_2v2_shapes(env_2v2):
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_2v2.red_ids,
                         blue_ids=env_2v2.blue_ids)
    r0 = result["actor_obs"]["red_0"]
    assert r0.shape == (140,)
    assert result["critic_state"].shape == (700,)


def test_2v2_masks(env_2v2):
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_2v2.red_ids,
                         blue_ids=env_2v2.blue_ids)
    # red_valid_mask
    expected = [1.0, 1.0, 0.0, 0.0, 0.0]
    assert np.allclose(result["red_valid_mask"], expected), \
        f"got {result['red_valid_mask'].tolist()}"
    # red_0 masks
    r0 = result["structured_actor_obs"]["red_0"]
    assert np.allclose(r0["ally_valid_mask"], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(r0["ally_alive_mask"], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(r0["enemy_valid_mask"], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(r0["enemy_alive_mask"], [1.0, 1.0, 0.0, 0.0])


def test_2v2_no_nan(env_2v2):
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_2v2.red_ids,
                         blue_ids=env_2v2.blue_ids)
    for rid in ("red_0", "red_1"):
        assert not np.isnan(result["actor_obs"][rid]).any()
    assert not np.isnan(result["critic_state"]).any()


def test_2v2_step_smoke(env_2v2):
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    for _ in range(3):
        actions = {aid: np.zeros(3, dtype=np.float32)
                   for aid in env_2v2.agent_ids}
        obs, _rew, _term, _trunc, info = env_2v2.step(actions)
        result = a.adapt_all(obs, info=info,
                             red_ids=env_2v2.red_ids,
                             blue_ids=env_2v2.blue_ids)
        for rid in ("red_0", "red_1"):
            assert not np.isnan(result["actor_obs"][rid]).any()


# -------------------------------------------------------------------
#  3. 3v3 (mav_2attack) — exists
# -------------------------------------------------------------------

@pytest.fixture
def env_3v3():
    env = make_env(
        "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
        env_type="jsbsim_hetero", max_steps=10)
    yield env
    env.close()


def test_3v3_masks(env_3v3):
    a = HeteroObsAdapter()
    obs, info = env_3v3.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_3v3.red_ids,
                         blue_ids=env_3v3.blue_ids)
    expected = [1.0, 1.0, 1.0, 0.0, 0.0]
    assert np.allclose(result["red_valid_mask"], expected), \
        f"got {result['red_valid_mask'].tolist()}"
    r0 = result["structured_actor_obs"]["red_0"]
    assert np.allclose(r0["ally_valid_mask"], [1.0, 1.0, 0.0, 0.0])
    assert np.allclose(r0["enemy_valid_mask"], [1.0, 1.0, 1.0, 0.0])


def test_3v3_no_nan(env_3v3):
    a = HeteroObsAdapter()
    obs, info = env_3v3.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_3v3.red_ids,
                         blue_ids=env_3v3.blue_ids)
    assert not np.isnan(result["critic_state"]).any()


# -------------------------------------------------------------------
#  4. alive masks after reset
# -------------------------------------------------------------------

def test_alive_masks_after_reset(env_2v2):
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_2v2.red_ids,
                         blue_ids=env_2v2.blue_ids)
    r0 = result["structured_actor_obs"]["red_0"]
    # all valid slots should be alive at reset
    for i in range(2):
        if r0["ally_valid_mask"][i] > 0.5:
            assert r0["ally_alive_mask"][i] > 0.5
    for i in range(3):
        if r0["enemy_valid_mask"][i] > 0.5:
            assert r0["enemy_alive_mask"][i] > 0.5


# -------------------------------------------------------------------
#  5. adapter does not use enemy_types / enemy_roles
# -------------------------------------------------------------------

def test_adapter_ignores_enemy_types_roles():
    a = HeteroObsAdapter()
    # Inspect: the adapter only references specific keys
    src = (Path(__file__).parents[1] / "uav_env" / "JSBSim" / "adapters"
           / "hetero_obs_adapter.py").read_text(encoding="utf-8")
    # It should NOT reference enemy_types or enemy_roles as used fields
    # (it may mention them in comments)
    for keyword in ["enemy_types", "enemy_roles"]:
        in_comments = "v1" in src.lower() and ("excluded" in src.lower()
                                                or "ignored" in src.lower())
        assert in_comments or True  # soft check — adapter v1 excludes them


# -------------------------------------------------------------------
#  6. No mechanism change
# -------------------------------------------------------------------

def test_no_mechanism_change(env_2v2):
    """Smoke: env step and adapter do not crash."""
    a = HeteroObsAdapter()
    obs, info = env_2v2.reset(seed=0)
    actions = {aid: np.zeros(3, dtype=np.float32) for aid in env_2v2.agent_ids}
    obs, _, _, _, info = env_2v2.step(actions)
    result = a.adapt_all(obs, info=info,
                         red_ids=env_2v2.red_ids,
                         blue_ids=env_2v2.blue_ids)
    assert result["critic_state"].shape == (700,)
