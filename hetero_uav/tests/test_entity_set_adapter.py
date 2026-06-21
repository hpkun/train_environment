from __future__ import annotations

import numpy as np
import pytest

from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter


def _obs(agent_id: str, red_ids: list[str], blue_ids: list[str]) -> dict:
    own = red_ids if agent_id.startswith("red_") else blue_ids
    enemy = blue_ids if agent_id.startswith("red_") else red_ids
    allies = [aid for aid in own if aid != agent_id]
    role = np.array([1, 0, 0, 0], np.float32) if agent_id == "red_0" else np.array([0, 1, 0, 0], np.float32)
    return {
        "ego_geo_state": np.arange(7, dtype=np.float32) / 10,
        "ego_role": role,
        "missile_warning": np.array([0], np.float32),
        "ally_geo_states": np.ones((len(allies), 5), np.float32),
        "ally_roles": np.tile(np.array([0, 1, 0, 0], np.float32), (len(allies), 1)),
        "ally_alive_mask": np.ones(len(allies), np.float32),
        "enemy_geo_states": np.ones((len(enemy), 5), np.float32),
        "enemy_alive_mask": np.ones(len(enemy), np.float32),
        "enemy_observed_mask": np.ones(len(enemy), np.float32),
        "enemy_track_source": np.tile(np.array([1, 0], np.float32), (len(enemy), 1)),
    }


def _team_obs(red_count: int, blue_count: int):
    red_ids = [f"red_{i}" for i in range(red_count)]
    blue_ids = [f"blue_{i}" for i in range(blue_count)]
    ids = red_ids + blue_ids
    return {aid: _obs(aid, red_ids, blue_ids) for aid in ids}, red_ids, blue_ids


@pytest.mark.parametrize("red_count,blue_count,tokens", [(3, 2, 5), (5, 4, 9), (7, 6, 13)])
def test_entity_set_adapter_uses_actual_scale(red_count, blue_count, tokens):
    obs, red_ids, blue_ids = _team_obs(red_count, blue_count)
    out = HeteroEntitySetAdapter().adapt_all(obs, red_ids=red_ids, blue_ids=blue_ids)

    assert out["actor_entity_tokens"].shape == (red_count, tokens, out["entity_dim"])
    assert out["actor_keep_mask"].shape == (red_count, tokens)
    assert out["critic_entity_tokens"].shape == (red_count + blue_count, out["entity_dim"])
    assert out["critic_keep_mask"].shape == (red_count + blue_count,)
    assert out["role_ids"].shape == (red_count,)
    assert np.all(out["actor_keep_mask"][:, 0] == 1)


def test_entity_set_adapter_masks_dead_and_unobserved_without_masking_self():
    obs, red_ids, blue_ids = _team_obs(3, 2)
    obs["red_0"]["ally_alive_mask"][0] = 0
    obs["red_0"]["enemy_observed_mask"][1] = 0
    info = {aid: {"alive": aid != "blue_1"} for aid in red_ids + blue_ids}
    out = HeteroEntitySetAdapter().adapt_all(obs, info=info, red_ids=red_ids, blue_ids=blue_ids)

    assert out["actor_keep_mask"][0].tolist() == [1, 0, 1, 1, 0]
    assert out["critic_keep_mask"].tolist() == [1, 1, 1, 1, 0]


def test_entity_set_adapter_rejects_non_mav_shared_geo_observation():
    obs, red_ids, blue_ids = _team_obs(3, 2)
    del obs["red_1"]["enemy_observed_mask"]
    with pytest.raises(ValueError, match="observation_mode='mav_shared_geo'.*enemy_observed_mask"):
        HeteroEntitySetAdapter().adapt_all(obs, red_ids=red_ids, blue_ids=blue_ids)


def test_entity_set_adapter_rejects_blue_missing_global_critic_key():
    obs, red_ids, blue_ids = _team_obs(3, 2)
    del obs["blue_1"]["ego_geo_state"]
    with pytest.raises(ValueError, match="observation_mode='mav_shared_geo'.*blue_1.*ego_geo_state"):
        HeteroEntitySetAdapter().adapt_all(obs, red_ids=red_ids, blue_ids=blue_ids)


@pytest.mark.parametrize(
    "config,red_count,blue_count",
    [
        ("hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml", 3, 2),
        ("hetero_mav_shared_geo_5v4_f22_pid.yaml", 5, 4),
        ("hetero_mav_shared_geo_7v6_f22_pid.yaml", 7, 6),
    ],
)
def test_real_f22_pid_configs_reset_to_variable_entity_sets(config, red_count, blue_count):
    from uav_env.make_env import make_env

    env = make_env(f"uav_env/JSBSim/configs/{config}")
    try:
        obs, info = env.reset(seed=3)
        out = HeteroEntitySetAdapter().adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        assert out["actor_entity_tokens"].shape[:2] == (red_count, red_count + blue_count)
        assert out["critic_entity_tokens"].shape[0] == red_count + blue_count
    finally:
        env.close()
