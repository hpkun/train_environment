"""Tests for reward contribution diagnostics. Static only, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestContributionStats:
    def test_signed_and_abs_separate(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 1.0, "b": -2.0}, {"a": 3.0, "b": -4.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert stats["a"]["signed_mean"] == 2.0
        assert stats["a"]["abs_mean"] == 2.0
        assert stats["b"]["signed_mean"] == -3.0
        assert stats["b"]["abs_mean"] == 3.0

    def test_positive_negative_sum_separate(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 5.0}, {"a": -3.0}]
        stats = _compute_contribution_stats(rows, ["a"])
        assert stats["a"]["positive_sum"] == 5.0
        assert stats["a"]["negative_sum"] == -3.0
        assert stats["a"]["negative_abs_sum"] == 3.0

    def test_cancellation_shows_in_abs_share(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 100.0, "b": -100.0}, {"a": 100.0, "b": -100.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert abs(stats["a"]["signed_sum"] + stats["b"]["signed_sum"]) < 1e-6
        assert stats["a"]["abs_share_of_total_abs"] == pytest.approx(0.5, abs=1e-4)
        assert stats["b"]["abs_share_of_total_abs"] == pytest.approx(0.5, abs=1e-4)

    def test_positive_share_uses_positive_sum(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 10.0, "b": -1.0}, {"a": 1.0, "b": -4.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        # a positive_sum=11, b positive_sum=0 → a gets 100% positive_share
        assert stats["a"]["positive_share"] == 1.0
        assert stats["b"]["positive_share"] == 0.0

    def test_negative_share_uses_negative_abs_sum(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 1.0, "b": -10.0}, {"a": 1.0, "b": -5.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        # b negative_abs_sum=15, a negative_abs_sum=0 → b gets 100% negative_share
        assert stats["b"]["negative_share"] == 1.0
        assert stats["a"]["negative_share"] == 0.0

    def test_zero_total_no_division_error(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 0.0, "b": 0.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert stats["a"]["signed_share_of_total"] is None

    def test_missing_fields_default_to_zero(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 1.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert stats["b"]["signed_mean"] == 0.0

    def test_dense_event_ratio_per_role(self):
        from scripts.diagnose_reward_control_failure import (_compute_contribution_stats,
            MAV_ACTIVE_GROUP, UAV_ACTIVE_GROUP, MAV_DETAIL_KEYS, UAV_DETAIL_KEYS)
        rows = [{"tam_v2_mav_safety": -0.3, "tam_v2_mav_support": 0.05, "tam_v2_mav_event": -0.5}]
        stats = _compute_contribution_stats(rows, MAV_DETAIL_KEYS, active_group=MAV_ACTIVE_GROUP)
        assert stats.get("dense_event_ratio") is not None
        # dense=0.35, event=0.5 → ratio=0.7
        assert stats["dense_event_ratio"] == pytest.approx(0.7, abs=0.01)

    def test_dense_event_ratio_null_when_no_events(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats, MAV_ACTIVE_GROUP, MAV_DETAIL_KEYS
        rows = [{"tam_v2_mav_safety": -0.3, "tam_v2_mav_event": 0.0}]
        stats = _compute_contribution_stats(rows, MAV_DETAIL_KEYS, active_group=MAV_ACTIVE_GROUP)
        assert stats["dense_event_ratio"] is None


class TestPhaseClassification:
    def test_alive_all_excludes_dead(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        flight = [{"episode": 0, "agent_id": "red_0", "step": 10, "alive": 1, "missile_warning": 0}]
        reward = [{"episode": 0, "agent_id": "red_0", "step": 10}]
        phases = _classify_phase(flight, reward)
        assert len(phases["alive_all"]) == 1
        flight_dead = [{"episode": 0, "agent_id": "red_0", "step": 5, "alive": 0, "missile_warning": 0}]
        reward_dead = [{"episode": 0, "agent_id": "red_0", "step": 5}]
        phases2 = _classify_phase(flight_dead, reward_dead)
        assert len(phases2["alive_all"]) == 0

    def test_no_missile_warning_excludes_dead(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        flight = [{"episode": 0, "agent_id": "red_0", "step": 5, "alive": 0, "missile_warning": 0}]
        reward = [{"episode": 0, "agent_id": "red_0", "step": 5}]
        phases = _classify_phase(flight, reward)
        assert len(phases["no_missile_warning"]) == 0

    def test_missile_warning_uses_flight_field_first(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        flight = [{"episode": 0, "agent_id": "red_0", "step": 10, "alive": 1, "missile_warning": 1}]
        reward = [{"episode": 0, "agent_id": "red_0", "step": 10,
                    "tam_v2_uav_dodge": 0.0, "tam_v2_uav_dodge_angle": 0.0}]
        phases = _classify_phase(flight, reward)
        assert len(phases["missile_warning"]) == 1
        assert len(phases["no_missile_warning"]) == 0

    def test_missile_warning_fallback_dodge(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        flight = [{"episode": 0, "agent_id": "red_0", "step": 10, "alive": 1, "missile_warning": -1}]
        reward = [{"episode": 0, "agent_id": "red_0", "step": 10,
                    "tam_v2_uav_dodge": -30.0, "tam_v2_uav_dodge_angle": -1.0}]
        phases = _classify_phase(flight, reward)
        assert len(phases["missile_warning"]) == 1


class TestEventTo100StepRatio:
    def test_atomic_event_keys_no_double_count(self):
        from scripts.diagnose_reward_control_failure import (MAV_ATOMIC_EVENT_KEYS,
            UAV_ATOMIC_EVENT_KEYS)
        # Aggregate keys must NOT be in atomic sets
        assert "tam_v2_mav_event" not in MAV_ATOMIC_EVENT_KEYS
        assert "tam_v2_uav_event" not in UAV_ATOMIC_EVENT_KEYS
        # Atomic keys are subsets of detail keys
        from scripts.diagnose_reward_control_failure import MAV_DETAIL_KEYS, UAV_DETAIL_KEYS
        for k in MAV_ATOMIC_EVENT_KEYS:
            assert k in MAV_DETAIL_KEYS
        for k in UAV_ATOMIC_EVENT_KEYS:
            assert k in UAV_DETAIL_KEYS


class TestScaleAnalysis:
    def test_reads_config(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        config = {"tam_paper_reward_v2": {"global_scale": 0.05}}
        rows = [{"tam_v2_total": -0.05, "tam_v2_uav_height": 8.0}]
        result = _compute_scale_analysis(rows, "uav", config=config)
        assert result["current_scale"] == 0.05
        assert result["current_scale_source"] == "config"

    def test_fallback_when_no_config(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        rows = [{"tam_v2_total": -0.02}]
        result = _compute_scale_analysis(rows, "uav")
        assert result["current_scale"] == 0.02
        assert result["current_scale_source"] == "fallback"

    def test_raw_vs_scaled_distinction(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        rows = [{"tam_v2_total": -0.05, "tam_v2_uav_height": 8.0}] * 10
        result = _compute_scale_analysis(rows, "uav")
        assert "raw_total_signed_mean" in result
        assert "raw_total_abs_mean" in result
        sim = result["simulated_scales"]["0.05"]
        assert "raw_component_means" in sim
        assert "estimated_scaled_component_means" in sim
        assert "estimated_events" in sim

    def test_abs_share_invariant_across_scales(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        rows = [{"tam_v2_total": -0.03, "tam_v2_uav_height": 8.0,
                 "tam_v2_uav_angle": -5.0, "tam_v2_uav_dodge": -5.0}] * 20
        result = _compute_scale_analysis(rows, "uav")
        s02 = result["simulated_scales"]["0.02"]["abs_shares"]
        s10 = result["simulated_scales"]["0.10"]["abs_shares"]
        h02 = s02.get("tam_v2_uav_height_abs_share")
        h10 = s10.get("tam_v2_uav_height_abs_share")
        if h02 is not None and h10 is not None:
            assert abs(h02 - h10) < 1e-6


class TestV1Compatibility:
    def test_v1_keys_still_defined(self):
        from scripts.diagnose_reward_control_failure import BRMA_KEYS, MAV_TAM_KEYS, EVENT_KEYS
        assert len(BRMA_KEYS) > 0
        assert len(MAV_TAM_KEYS) > 0
        assert len(EVENT_KEYS) > 0
