"""Tests for policy full-geometry wiring correctness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestHeteroObsAdapterV2:
    def test_enemy_entity_dim_is_18(self):
        """HeteroObsAdapterV2 enemy_entity_dim must be 18."""
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        assert adapter.enemy_entity_dim == 18, (
            f"expected enemy_entity_dim=18, got {adapter.enemy_entity_dim}"
        )

    def test_flat_actor_obs_dim_correct(self):
        """flat_actor_obs_dim should match: ego(12) + allies((max_red-1)*9) + enemies(max_blue*18) + masks(20)."""
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        # max_allies = max_red - 1 = 2, max_enemies = max_blue = 2
        expected = 12 + 2 * 9 + 2 * 18 + 20  # 12 + 18 + 36 + 20 = 86
        assert adapter.flat_actor_obs_dim == expected, (
            f"expected {expected}, got {adapter.flat_actor_obs_dim}"
        )

    def test_critic_state_dim_correct(self):
        """critic_state_dim = flat_actor_obs_dim * max_red."""
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        expected = adapter.flat_actor_obs_dim * adapter.max_red
        assert adapter.critic_state_dim == expected

    def test_requires_mav_shared_geo_keys(self):
        """Missing full-geometry keys should raise ValueError."""
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        adapter = HeteroObsAdapterV2(max_red=3, max_blue=2)
        bad_obs = {"ego_geo_state": np.zeros(7)}
        with pytest.raises(ValueError, match="missing keys"):
            adapter.adapt_agent("red_0", bad_obs, red_ids=["red_0", "red_1", "red_2"],
                               blue_ids=["blue_0", "blue_1"])


class TestBRMAPolicyDefaults:
    def test_brma_entity_default_entity_dim_30(self):
        """New training default entity_dim must be 30 (>= 30 for full-geometry)."""
        from algorithms.happo.brma_entity_policy import (
            BRMAEntityHAPPOReferencePolicy, BRMAEntityObservationEncoder)
        encoder = BRMAEntityObservationEncoder()
        assert encoder.entity_dim == 30, f"encoder entity_dim={encoder.entity_dim}, expected 30"
        policy = BRMAEntityHAPPOReferencePolicy(critic_state_dim=480)
        assert policy.entity_dim == 30, f"policy entity_dim={policy.entity_dim}, expected 30"

    def test_brma_recurrent_default_entity_dim_30(self):
        from algorithms.happo.brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy
        policy = BRMARecurrentHAPPOReferencePolicy(critic_state_dim=480, rnn_hidden_size=128)
        assert policy.entity_dim == 30, f"entity_dim={policy.entity_dim}, expected 30"

    def test_brma_masked_default_entity_dim_30(self):
        from algorithms.happo.brma_masked_policy import (
            BRMARecurrentMaskedHAPPOReferencePolicy, BRMABiasedMaskGenerator)
        policy = BRMARecurrentMaskedHAPPOReferencePolicy(
            critic_state_dim=480, rnn_hidden_size=128)
        assert policy.entity_dim == 30, f"entity_dim={policy.entity_dim}, expected 30"
        mask_gen = BRMABiasedMaskGenerator()
        # BiasedMaskGenerator has its own entity_dim; it's used internally

    def test_entity_dim_19_old_checkpoint_mode(self):
        """When entity_dim=19, full-geometry should be explicitly NOT used."""
        from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
        policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480)
        assert policy.entity_dim == 19
        # With entity_dim=19, enemy_flat_dim=18, the extra 11 full-geometry dims are truncated
        assert policy.entity_dim < 30
        assert policy.enemy_flat_dim == 18

    def test_brma_entity_enemy_flat_dim_is_18(self):
        from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy
        policy = BRMAEntityHAPPOReferencePolicy(critic_state_dim=480)
        assert policy.enemy_flat_dim == 18


class TestHeteroEntityRecurrentNotFullGeometry:
    def test_hetero_entity_recurrent_entity_dim_21_not_full_geo(self):
        """HeteroEntitySetAdapter entity_dim=21 does NOT support full-geometry.
        Upgrading to entity_dim>=32 is needed."""
        from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy
        policy = HeteroEntityRecurrentPolicy(
            entity_dim=21, action_dim=3, hidden_dim=128,
            rnn_hidden_size=128, num_attention_heads=4)
        assert policy.entity_dim == 21
        # 21 < 32: full-geometry not supported
        assert policy.entity_dim < 32, (
            "hetero_entity_recurrent should be marked as not-full-geometry "
            "until HeteroEntitySetAdapter is upgraded"
        )

    def test_hetero_entity_set_adapter_entity_dim_is_21(self):
        from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
        assert HeteroEntitySetAdapter.entity_dim == 21


class TestObservationModeValidation:
    def test_mav_shared_geo_v2_rejected(self):
        """observation_mode='mav_shared_geo_v2' should raise ValueError."""
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
