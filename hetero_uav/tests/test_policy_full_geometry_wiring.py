"""Tests for policy full-geometry wiring correctness with real trace verification."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helper: build a fake flat actor obs from HeteroObsAdapterV2
# ---------------------------------------------------------------------------
# Default to 5v4 so policies with default max_allies=4, max_enemies=4 can decode
def _build_fake_flat_obs(max_red=5, max_blue=4) -> np.ndarray:
    """Build a flat actor obs with non-zero values including enemy full-geometry.

    Defaults to max_red=5, max_blue=4 to match policy default max_allies=4, max_enemies=4.
    """
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    adapter = HeteroObsAdapterV2(max_red=max_red, max_blue=max_blue)
    red_ids = [f"red_{i}" for i in range(max_red)]
    blue_ids = [f"blue_{i}" for i in range(max_blue)]
    obs = {}
    for rid in red_ids:
        obs[rid] = {
            "ego_geo_state": np.random.randn(7).astype(np.float32),
            "ego_role": np.array([0, 1, 0, 0], dtype=np.float32),
            "missile_warning": np.array([0], dtype=np.float32),
            "ally_geo_states": np.random.randn(max_red - 1, 5).astype(np.float32),
            "ally_roles": np.tile(np.array([0, 1, 0, 0], dtype=np.float32), (max_red - 1, 1)),
            "ally_alive_mask": np.ones(max_red - 1, dtype=np.float32),
            "enemy_geo_states": np.random.randn(max_blue, 5).astype(np.float32),
            "enemy_alive_mask": np.ones(max_blue, dtype=np.float32),
            "enemy_observed_mask": np.ones(max_blue, dtype=np.float32),
            "enemy_track_source": np.random.randn(max_blue, 2).astype(np.float32),
            "enemy_relative_pos_xyz": np.random.randn(max_blue, 3).astype(np.float32),
            "enemy_relative_vel_xyz": np.random.randn(max_blue, 3).astype(np.float32),
            "enemy_bearing_elevation": np.random.randn(max_blue, 2).astype(np.float32),
            "enemy_speed_heading": np.random.randn(max_blue, 2).astype(np.float32),
            "enemy_full_geo_valid_mask": np.ones(max_blue, dtype=np.float32),
        }
    out = adapter.adapt_agent(red_ids[0], obs[red_ids[0]], red_ids=red_ids, blue_ids=blue_ids)
    return out["flat_actor_obs"]


# ---------------------------------------------------------------------------
# HeteroObsAdapterV2
# ---------------------------------------------------------------------------
class TestHeteroObsAdapterV2:
    def test_enemy_entity_dim_is_18(self):
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        assert adapter.enemy_entity_dim == 18

    def test_flat_actor_obs_dim_dynamic(self):
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        expected = 12 + 2 * 9 + 2 * 18 + 20  # 86
        assert adapter.flat_actor_obs_dim == expected

    def test_critic_state_dim_correct(self):
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        assert adapter.critic_state_dim == adapter.flat_actor_obs_dim * adapter.max_red

    def test_requires_mav_shared_geo_keys(self):
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        bad_obs = {"ego_geo_state": np.zeros(7)}
        with pytest.raises(ValueError, match="missing keys"):
            adapter.adapt_agent("red_0", bad_obs, red_ids=["red_0", "red_1", "red_2"],
                               blue_ids=["blue_0", "blue_1"])


# ---------------------------------------------------------------------------
# BRMA policy defaults
# ---------------------------------------------------------------------------
class TestBRMAPolicyDefaults:
    def test_brma_entity_default_entity_dim_30(self):
        from algorithms.happo.brma_entity_policy import (
            BRMAEntityHAPPOReferencePolicy, BRMAEntityObservationEncoder)
        encoder = BRMAEntityObservationEncoder()
        assert encoder.entity_dim == 30
        policy = BRMAEntityHAPPOReferencePolicy(critic_state_dim=480)
        assert policy.entity_dim == 30

    def test_brma_recurrent_default_entity_dim_30(self):
        from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy
        policy = BRMARecurrentHAPPOReferencePolicy(critic_state_dim=480, rnn_hidden_size=128)
        assert policy.entity_dim == 30

    def test_brma_masked_default_entity_dim_30(self):
        from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy
        policy = BRMARecurrentMaskedHAPPOReferencePolicy(
            critic_state_dim=480, rnn_hidden_size=128)
        assert policy.entity_dim == 30

    def test_brma_masked_correct_constructor_params(self):
        """Constructor uses random_scale_mask / biased_mask / random_mask_prob (not brma_ prefix)."""
        from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy
        policy = BRMARecurrentMaskedHAPPOReferencePolicy(
            critic_state_dim=480, rnn_hidden_size=128,
            random_scale_mask=False, biased_mask=False, random_mask_prob=0.0)
        assert policy.entity_dim == 30
        assert policy.random_scale_mask is False
        # Verify old parameter names would fail
        with pytest.raises(TypeError):
            BRMARecurrentMaskedHAPPOReferencePolicy(
                critic_state_dim=480,
                brma_random_scale_mask=False)


class TestBRMAEntityDim19Compat:
    def test_entity_dim_19_old_checkpoint_mode(self):
        from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
        policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480)
        assert policy.entity_dim == 19
        assert policy.enemy_flat_dim == 18

    def test_entity_dim_19_trace_shows_no_full_geometry(self):
        """With entity_dim=19, full-geometry region is empty (no slots >= 19)."""
        import torch
        from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
        flat_obs = _build_fake_flat_obs()
        policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
        entities, _keep = policy._flat_to_entities(
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0))
        enemy_start = 1 + policy.max_allies
        enemy_tokens = entities[:, enemy_start:enemy_start + policy.max_enemies, :]
        # entity_dim=19: indices 19:19 is empty slice
        full_geo = enemy_tokens[:, :, 19:19]
        assert full_geo.numel() == 0, "entity_dim=19 should have no full-geometry slots"


# ---------------------------------------------------------------------------
# Real trace: entity_dim=30 enemy token 19:30 is nonzero
# ---------------------------------------------------------------------------
class TestFullGeometryTrace:
    def test_brma_entity_token_19_30_nonzero(self):
        import torch
        from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
        flat_obs = _build_fake_flat_obs()
        policy = BRMAEntityHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480, action_dim=3)
        entities, _keep = policy._flat_to_entities(
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0))
        enemy_start = 1 + policy.max_allies
        enemy_tokens = entities[:, enemy_start:enemy_start + policy.max_enemies, :]
        region = enemy_tokens[:, :, 19:30]
        assert region.shape[-1] == 11
        assert torch.any(torch.abs(region) > 1e-8), "entity_dim=30 enemy token 19:30 should be nonzero"

    def test_brma_recurrent_token_19_30_nonzero(self):
        import torch
        from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy
        flat_obs = _build_fake_flat_obs()
        policy = BRMARecurrentHAPPOReferencePolicy(
            entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128)
        entities, _keep = policy._flat_to_entities(
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0))
        enemy_start = 1 + policy.max_allies
        enemy_tokens = entities[:, enemy_start:enemy_start + policy.max_enemies, :]
        region = enemy_tokens[:, :, 19:30]
        assert torch.any(torch.abs(region) > 1e-8)

    def test_brma_recurrent_masked_token_19_30_nonzero(self):
        import torch
        from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy
        flat_obs = _build_fake_flat_obs()
        policy = BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=30, critic_state_dim=480, action_dim=3, rnn_hidden_size=128,
            random_scale_mask=False, biased_mask=False, random_mask_prob=0.0)
        entities, _keep = policy._flat_to_entities(
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0))
        enemy_start = 1 + policy.max_allies
        enemy_tokens = entities[:, enemy_start:enemy_start + policy.max_enemies, :]
        region = enemy_tokens[:, :, 19:30]
        assert torch.any(torch.abs(region) > 1e-8)

    def test_entity_attention_token_19_30_nonzero(self):
        import torch
        from algorithms.happo.entity_policy import EntityHAPPOReferencePolicy
        flat_obs = _build_fake_flat_obs()
        policy = EntityHAPPOReferencePolicy(entity_dim=30, critic_state_dim=480)
        entities, _keep = policy._flat_to_entities(
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0))
        enemy_start = 1 + policy.max_allies
        enemy_tokens = entities[:, enemy_start:enemy_start + policy.max_enemies, :]
        region = enemy_tokens[:, :, 19:30]
        assert torch.any(torch.abs(region) > 1e-8)


# ---------------------------------------------------------------------------
# hetero_entity_recurrent: explicitly NOT full-geometry
# ---------------------------------------------------------------------------
class TestHeteroEntityRecurrentNotFullGeometry:
    def test_entity_dim_21_not_full_geo(self):
        from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
        policy = HeteroEntityRecurrentPolicy(
            entity_dim=21, action_dim=3, hidden_dim=128,
            rnn_hidden_size=128, num_attention_heads=4)
        assert policy.entity_dim == 21
        assert policy.entity_dim < 32

    def test_hetero_entity_set_adapter_entity_dim_is_21(self):
        from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
        assert HeteroEntitySetAdapter.entity_dim == 21

    def test_hetero_entity_recurrent_no_flat_to_entities(self):
        """HeteroEntityRecurrentPolicy does NOT have _flat_to_entities."""
        from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
        policy = HeteroEntityRecurrentPolicy(
            entity_dim=21, action_dim=3, hidden_dim=128,
            rnn_hidden_size=128, num_attention_heads=4)
        assert not hasattr(policy, "_flat_to_entities"), (
            "hetero_entity_recurrent should NOT have _flat_to_entities"
        )


# ---------------------------------------------------------------------------
# pure_happo / flat path
# ---------------------------------------------------------------------------
class TestPureHappoFlatPath:
    def test_pure_happo_no_flat_to_entities(self):
        from algorithms.pure_happo.policy import PureHAPPOPolicy
        policy = PureHAPPOPolicy(actor_obs_dim=86, critic_state_dim=258, action_dim=3, num_agents=3)
        assert not hasattr(policy, "_flat_to_entities")

    def test_flat_policy_no_flat_to_entities(self):
        from algorithms.happo.happo_policy import HAPPOReferencePolicy
        policy = HAPPOReferencePolicy(86, 258)
        assert not hasattr(policy, "_flat_to_entities")


# ---------------------------------------------------------------------------
# Observation mode validation
# ---------------------------------------------------------------------------
class TestObservationModeValidation:
    def test_mav_shared_geo_v2_rejected(self):
        from uav_env import make_env
        with pytest.raises(ValueError, match="unknown observation_mode"):
            make_env(None, env_type="jsbsim_hetero", observation_mode="mav_shared_geo_v2",
                    max_steps=10)

    def test_mav_shared_geo_accepted(self):
        from uav_env import make_env
        try:
            env = make_env(
                None, env_type="jsbsim_hetero",
                observation_mode="mav_shared_geo",
                max_num_red=2, max_num_blue=2,
                max_steps=10, suppress_jsbsim_output=True,
            )
            assert env.observation_mode == "mav_shared_geo"
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Train meta
# ---------------------------------------------------------------------------
class TestTrainMeta:
    def test_full_geometry_meta_brma_entity(self):
        from scripts.train_happo_reference import _full_geometry_meta
        meta = _full_geometry_meta("brma_entity", 30)
        assert meta["full_geometry_features_used"] is True
        assert meta["full_geometry_path"] == "flat_to_entity_token_19_30"
        assert meta["enemy_flat_dim"] == 18
        assert meta["entity_dim_required_for_full_geometry"] == 30

    def test_full_geometry_meta_pure_happo(self):
        from scripts.train_happo_reference import _full_geometry_meta
        meta = _full_geometry_meta("pure_happo", None)
        assert meta["full_geometry_features_used"] is True
        assert meta["full_geometry_path"] == "flat_actor_obs"
        assert meta["entity_dim"] == "n/a (flat policy)"

    def test_full_geometry_meta_old_checkpoint_entity_dim_19(self):
        from scripts.train_happo_reference import _full_geometry_meta
        meta = _full_geometry_meta("brma_entity", 19)
        assert meta["full_geometry_features_used"] is False
        assert "truncates" in meta.get("full_geometry_checkpoint_reason", "")

    def test_full_geometry_meta_hetero_entity_recurrent(self):
        from scripts.train_happo_reference import _full_geometry_meta
        meta = _full_geometry_meta("hetero_entity_recurrent", 21)
        assert meta["full_geometry_features_used"] is False
        assert meta["hetero_entity_recurrent_full_geometry"] is False
        assert meta["full_geometry_path"] == "unsupported_hetero_entity_set_adapter"

    def test_entity_policy_meta_marks_hetero_as_not_full_geo(self):
        from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
        from scripts.train_happo_reference import _entity_policy_meta
        policy = HeteroEntityRecurrentPolicy(
            entity_dim=21, action_dim=3, hidden_dim=128,
            rnn_hidden_size=128, num_attention_heads=4)
        meta = _entity_policy_meta(policy)
        assert meta["hetero_entity_recurrent_full_geometry"] is False
        assert meta["full_geometry_features_used"] is False
