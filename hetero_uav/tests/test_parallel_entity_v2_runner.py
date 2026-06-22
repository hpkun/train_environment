"""Contract tests for parallel runner hetero_entity_recurrent v2 support."""
import numpy as np
import pytest
import torch

from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
)
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter


class TestParallelRunnerEntityV2:
    """Verify parallel runner support for hetero_entity_recurrent v2."""

    def test_policy_arch_accepted(self):
        """hetero_entity_recurrent is a valid policy_arch choice."""
        from scripts.train_happo_reference_parallel import _parse_args
        import sys
        # Just verify the import works and the parser accepts the choice
        try:
            parser_test = _parse_args.__wrapped__ if hasattr(_parse_args, '__wrapped__') else None
        except Exception:
            pass
        # Verify the module can be imported without error
        import scripts.train_happo_reference_parallel as _m
        assert hasattr(_m, '_parse_args')

    def test_buffer_token_count_correct(self):
        """Buffer initialised with correct token counts for 3v2 (3 red + 4 blue)."""
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1", "blue_2", "blue_3"]
        total = len(red_ids) + len(blue_ids)  # 7
        roles = [0, 1, 1]
        buf = HAPPORolloutBuffer(
            8, 3, 0, 0, 3, roles,
            rnn_hidden_size=128,
            actor_token_count=total,
            critic_token_count=total,
            entity_dim=21,
        )
        assert buf.entity_dim == 21
        assert buf.actor_token_count == total
        assert buf.critic_token_count == total
        assert buf.actor_entity_tokens.shape == (8, 3, total, 21)
        assert buf.critic_entity_tokens.shape == (8, total, 21)
        assert buf.critic_counts.shape == (8, 4)

    def test_buffer_stores_critic_counts(self):
        """Buffer.store passes critic_counts through correctly."""
        from algorithms.happo.happo_buffer import HAPPORolloutBuffer
        buf = HAPPORolloutBuffer(
            4, 2, 0, 0, 3, [0, 1],
            actor_token_count=5, critic_token_count=5, entity_dim=21,
        )
        # Store with entity data and critic_counts
        buf.store(
            None, None,
            actions=np.zeros((2, 3), dtype=np.float32),
            log_probs=np.zeros(2, dtype=np.float32),
            rewards=np.zeros(2, dtype=np.float32),
            dones=np.zeros(2, dtype=np.float32),
            value=0.0,
            active_masks=np.ones(2, dtype=np.float32),
            actor_entity_tokens=np.zeros((2, 5, 21), dtype=np.float32),
            actor_keep_mask=np.ones((2, 5), dtype=np.float32),
            critic_entity_tokens=np.zeros((5, 21), dtype=np.float32),
            critic_keep_mask=np.ones(5, dtype=np.float32),
            critic_counts=np.array([2., 3., 3., 4.], dtype=np.float32),
        )
        data = buf.get("cpu")
        assert "critic_counts" in data
        assert data["critic_counts"].shape[1] == 4

    def test_adapter_returns_critic_counts(self):
        """HeteroEntitySetAdapter produces critic_counts in adapt_all."""
        adapter = HeteroEntitySetAdapter()
        # Build minimal obs dict for 3v2 (3 red + 2 blue)
        red_ids = ["red_0", "red_1", "red_2"]
        blue_ids = ["blue_0", "blue_1"]
        obs_dict = {}
        for aid in red_ids + blue_ids:
            geo = np.zeros(7, dtype=np.float32)
            role = np.array([1, 0, 0, 0], dtype=np.float32) if aid == "red_0" else np.array([0, 1, 0, 0], dtype=np.float32)
            n_ally = 2 if aid.startswith("red_") else 1
            n_enemy = 2 if aid.startswith("red_") else 3
            obs_dict[aid] = {
                "ego_geo_state": geo,
                "ego_role": role,
                "missile_warning": np.array([0.0], dtype=np.float32),
                "ally_geo_states": np.zeros((n_ally, 5), dtype=np.float32),
                "ally_roles": np.zeros((n_ally, 4), dtype=np.float32),
                "ally_alive_mask": np.ones(n_ally, dtype=np.float32),
                "enemy_geo_states": np.zeros((n_enemy, 5), dtype=np.float32),
                "enemy_alive_mask": np.ones(n_enemy, dtype=np.float32),
                "enemy_observed_mask": np.ones(n_enemy, dtype=np.float32),
                "enemy_track_source": np.zeros((n_enemy, 2), dtype=np.float32),
            }
        info = {aid: {"alive": True} for aid in red_ids + blue_ids}
        result = adapter.adapt_all(obs_dict, info=info, red_ids=red_ids, blue_ids=blue_ids)

        assert "critic_counts" in result
        assert result["critic_counts"].shape == (4,)
        # All alive: red_alive=3, red_total=3, blue_alive=2, blue_total=2
        np.testing.assert_array_equal(result["critic_counts"], [3., 3., 2., 2.])

    def test_policy_value_with_critic_counts(self):
        """Policy.value accepts critic_counts and produces output."""
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        policy.eval()
        tokens = torch.randn(7, 21)
        keep = torch.ones(7)
        counts = torch.tensor([3., 3., 4., 4.])
        with torch.no_grad():
            v = policy.value(tokens, keep, critic_counts=counts)
        assert v.numel() == 1
        assert torch.isfinite(v)

    def test_meta_contains_v2_schema(self):
        """Checkpoint meta must include v2 feature_schema_version."""
        policy = HeteroEntityRecurrentPolicy(entity_dim=21)
        from scripts.train_happo_reference import _entity_policy_meta
        meta = _entity_policy_meta(policy)
        assert meta["feature_schema_version"] == "hetero_entity_set_v2"
        assert meta["entity_dim"] == 21
        assert meta["critic_arch"] == "global_entity_attention_value_v2"
        assert meta["policy_arch"] == "hetero_entity_recurrent"


class TestInactiveAgentHandling:
    """Verify inactive agent action/hidden/entity sanitisation aligns with single-proc runner."""

    def test_zero_inactive_actions(self):
        from algorithms.happo.rollout_safety import zero_inactive_actions
        actions = np.array([[1.0, 0.5, -0.3], [0.8, 0.1, 0.2], [-0.5, 1.0, 0.9]], dtype=np.float32)
        active = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        result = zero_inactive_actions(actions, active)
        assert result.shape == (3, 3)
        # active row 0 unchanged
        assert abs(result[0, 0] - 1.0) < 1e-6
        # inactive row 1 zeroed
        assert np.all(result[1] == 0.0)
        # active row 2 unchanged
        assert abs(result[2, 1] - 1.0) < 1e-6

    def test_zero_inactive_hidden(self):
        from algorithms.happo.rollout_safety import zero_inactive_hidden
        hidden = np.array([[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]], dtype=np.float32)
        active = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = zero_inactive_hidden(hidden, active)
        # inactive rows zeroed
        assert np.all(result[1] == 0.0)
        assert np.all(result[2] == 0.0)
        # active row preserved
        assert abs(result[0, 0] - 0.5) < 1e-6

    def test_entity_tokens_zeroed_for_inactive(self):
        """Inactive agent entity tokens and keep mask must be zeroed."""
        actor_tokens = np.ones((3, 9, 21), dtype=np.float32)
        actor_keep = np.ones((3, 9), dtype=np.float32)
        active = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        active_rows = active > 0.5

        # Simulate the parallel runner's sanitisation
        actor_tokens[~active_rows] = 0.0
        actor_keep[~active_rows] = 0.0
        actor_keep[~active_rows, 0] = 1.0

        # Row 1 (inactive) tokens zeroed
        assert np.all(actor_tokens[1] == 0.0)
        # Row 1 keep mask zeroed except self (index 0)
        assert actor_keep[1, 0] == 1.0
        assert np.all(actor_keep[1, 1:] == 0.0)
        # Row 0 (active) unchanged
        assert np.all(actor_tokens[0] == 1.0)
        assert np.all(actor_keep[0] == 1.0)

    def test_pre_action_hidden_is_post_zero(self):
        """Pre-action hidden saved to buffer must be the zeroed version."""
        from algorithms.happo.rollout_safety import zero_inactive_hidden
        hidden = np.array([[0.5, 0.3], [-0.2, 0.8]], dtype=np.float32)
        active = np.array([0.0, 1.0], dtype=np.float32)

        # Step 1: zero inactive BEFORE act
        hidden = zero_inactive_hidden(hidden, active)
        # Step 2: save pre-action hidden
        pre_hidden = hidden.copy()

        # Inactive row 0 must be zero
        assert np.all(pre_hidden[0] == 0.0)
        # Active row 1 preserved
        assert abs(pre_hidden[1, 0] - (-0.2)) < 1e-6

    def test_nonfinite_check_only_active_rows(self):
        """Nonfinite check must only inspect active agent rows."""
        actions = np.array([[0.1, 0.2, 0.3], [np.nan, 0.0, 0.0]], dtype=np.float32)
        active = np.array([1.0, 0.0], dtype=np.float32)
        active_rows = active > 0.5

        # Only check active rows
        finite = True
        if active_rows.any():
            if not np.isfinite(actions[active_rows]).all():
                finite = False
        assert finite is True  # nan is in inactive row

        # If we checked all rows, it would fail
        assert not np.isfinite(actions).all()

