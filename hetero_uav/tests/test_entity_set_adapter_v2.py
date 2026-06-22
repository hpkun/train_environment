"""Tests for HeteroEntitySetAdapter v2 (entity_dim=21, alive_flag, critic keeps dead)."""
import numpy as np
import pytest

from uav_env.JSBSim.adapters.hetero_entity_set_adapter import (
    HeteroEntitySetAdapter,
    ENTITY_DIM,
    FEATURE_SCHEMA_VERSION,
    V1_SCHEMA_VERSION,
)


def _make_obs(aid, geo, role, alive=True, enemies_observed=True):
    """Minimal mav_shared_geo observation for one agent."""
    n_allies = 2
    n_enemies = 4
    ally_geo = np.zeros((n_allies, 5), dtype=np.float32)
    ally_roles = np.zeros((n_allies, 4), dtype=np.float32)
    ally_alive = np.ones(n_allies, dtype=np.float32)
    enemy_geo = np.zeros((n_enemies, 5), dtype=np.float32)
    enemy_alive = np.ones(n_enemies, dtype=np.float32)
    enemy_obs = np.ones(n_enemies, dtype=np.float32) if enemies_observed else np.zeros(n_enemies, dtype=np.float32)
    enemy_track = np.zeros((n_enemies, 2), dtype=np.float32)
    enemy_track[:, 0] = 1.0  # own_sensor
    return {
        "ego_geo_state": np.array(geo, dtype=np.float32),
        "ego_role": np.array(role, dtype=np.float32),
        "missile_warning": np.array([0.0], dtype=np.float32),
        "ally_geo_states": ally_geo,
        "ally_roles": ally_roles,
        "ally_alive_mask": ally_alive,
        "enemy_geo_states": enemy_geo,
        "enemy_alive_mask": enemy_alive,
        "enemy_observed_mask": enemy_obs,
        "enemy_track_source": enemy_track,
    }


def _make_info(red_ids, blue_ids, red_alive=True, blue_alive=True):
    info = {}
    for aid in red_ids:
        info[aid] = {"alive": red_alive}
    for aid in blue_ids:
        info[aid] = {"alive": blue_alive}
    return info


class TestAdapterV2TokenLayout:
    """Verify v2 entity_dim=21 and alive/observed flags."""

    def test_entity_dim_is_21(self):
        assert ENTITY_DIM == 21
        adapter = HeteroEntitySetAdapter()
        assert adapter.entity_dim == 21

    def test_feature_schema_is_v2(self):
        assert FEATURE_SCHEMA_VERSION == "hetero_entity_set_v2"
        adapter = HeteroEntitySetAdapter()
        assert adapter.feature_schema_version == "hetero_entity_set_v2"

    def test_v1_schema_constant_exists(self):
        assert V1_SCHEMA_VERSION == "hetero_entity_set_v1"

    def test_self_token_has_alive_and_observed(self):
        adapter = HeteroEntitySetAdapter()
        obs = _make_obs("red_0", [0.1]*7, [1,0,0,0])
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1", "blue_2", "blue_3"]
        obs_dict = {"red_0": obs}
        for rid in red_ids[1:]:
            obs_dict[rid] = _make_obs(rid, [0.1]*7, [0,1,0,0])
        for bid in blue_ids:
            obs_dict[bid] = _make_obs(bid, [0.1]*7, [0,0,0,0])
        info = _make_info(red_ids, blue_ids)
        result = adapter.adapt_all(obs_dict, info=info, red_ids=red_ids, blue_ids=blue_ids)

        # Self token: alive=1, observed=1
        self_token = result["actor_entity_tokens"][0, 0]
        assert self_token[19] == 1.0  # alive_flag
        assert self_token[20] == 1.0  # observed_flag

    def test_dead_entities_in_critic(self):
        """Critic tokens should include dead agents with alive_flag=0."""
        adapter = HeteroEntitySetAdapter()
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1"]
        obs_dict = {}
        for rid in red_ids:
            obs_dict[rid] = _make_obs(rid, [0.1]*7, [1,0,0,0] if rid == "red_0" else [0,1,0,0])
        for bid in blue_ids:
            obs_dict[bid] = _make_obs(bid, [0.1]*7, [0,0,0,0])
        # blue_0 is dead
        info = _make_info(red_ids, blue_ids, blue_alive=True)
        info["blue_0"]["alive"] = False
        result = adapter.adapt_all(obs_dict, info=info, red_ids=red_ids, blue_ids=blue_ids)

        critic_tokens = result["critic_entity_tokens"]
        critic_keep = result["critic_keep_mask"]
        n_total = len(red_ids) + len(blue_ids)
        assert critic_tokens.shape == (n_total, 21)
        # All entities present in critic (keep_mask all 1)
        assert np.all(critic_keep == 1.0)

        # Find blue_0: it's after red_2 (index 2), so index 3 in global_ids = [red_0, red_1, red_2, blue_0, blue_1]
        blue0_idx = 3
        # alive_flag should be 0 for dead blue_0
        assert critic_tokens[blue0_idx, 19] == 0.0  # dead
        assert critic_tokens[blue0_idx, 20] == 0.0  # not observed (dead)

    def test_critic_counts_match(self):
        adapter = HeteroEntitySetAdapter()
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1"]
        obs_dict = {}
        for rid in red_ids:
            obs_dict[rid] = _make_obs(rid, [0.1]*7, [1,0,0,0] if rid == "red_0" else [0,1,0,0])
        for bid in blue_ids:
            obs_dict[bid] = _make_obs(bid, [0.1]*7, [0,0,0,0])
        # red_1 dead, blue_0 dead
        info = _make_info(red_ids, blue_ids)
        info["red_1"]["alive"] = False
        info["blue_0"]["alive"] = False
        result = adapter.adapt_all(obs_dict, info=info, red_ids=red_ids, blue_ids=blue_ids)

        counts = result["critic_counts"]
        # [red_alive, red_total, blue_alive, blue_total]
        assert counts[0] == 2.0  # red_alive: 2 out of 3
        assert counts[1] == 3.0  # red_total
        assert counts[2] == 1.0  # blue_alive: 1 out of 2
        assert counts[3] == 2.0  # blue_total

    def test_actor_mask_filters_unobserved(self):
        adapter = HeteroEntitySetAdapter()
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1"]
        obs_dict = {}
        for rid in red_ids:
            obs_dict[rid] = _make_obs(rid, [0.1]*7, [1,0,0,0] if rid == "red_0" else [0,1,0,0],
                                      enemies_observed=(rid != "red_2"))
        for bid in blue_ids:
            obs_dict[bid] = _make_obs(bid, [0.1]*7, [0,0,0,0])
        info = _make_info(red_ids, blue_ids)
        result = adapter.adapt_all(obs_dict, info=info, red_ids=red_ids, blue_ids=blue_ids)

        # Actor mask for red_2: enemies not observed -> mask=0
        actor_mask = result["actor_keep_mask"]
        # Layout: [self, ally0, ally1, enemy0, enemy1]
        # For red_2 (index 2), allies are red_0, red_1 (both alive), enemies are blue_0, blue_1 (alive but unobserved)
        # So enemy slots should be 0
        mask_r2 = actor_mask[2]
        assert mask_r2[0] == 1.0  # self
        assert mask_r2[1] == 1.0  # ally0 (red_0) alive
        assert mask_r2[2] == 1.0  # ally1 (red_1) alive
        assert mask_r2[3] == 0.0  # enemy0 unobserved
        assert mask_r2[4] == 0.0  # enemy1 unobserved


class TestCheckpointContract:
    """Verify v1 checkpoints are rejected."""

    def test_v1_schema_rejected(self):
        from algorithms.happo.hetero_entity_recurrent_policy import validate_entity_policy_meta
        with pytest.raises(ValueError, match="v1 checkpoint is incompatible"):
            validate_entity_policy_meta({"feature_schema_version": "hetero_entity_set_v1"})

    def test_v2_schema_accepted(self):
        from algorithms.happo.hetero_entity_recurrent_policy import validate_entity_policy_meta
        validate_entity_policy_meta({
            "action_dim": 3,
            "feature_schema_version": "hetero_entity_set_v2",
            "adapter_mode": "hetero_entity_set",
            "actor_obs_format": "entity_tokens_keep_mask",
            "critic_obs_format": "global_entity_tokens_keep_mask",
            "role_vocab": ["mav", "attack_uav", "scout_uav", "interceptor_uav"],
            "actor_arch": "entity_attention_grucell_role_heads",
            "critic_arch": "global_entity_attention_value_v2",
            "entity_dim": 21,
            "role_dim": 4,
            "observation_adapter": "HeteroEntitySetAdapter",
        })
