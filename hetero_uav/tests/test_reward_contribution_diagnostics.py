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
        rows = [
            {"a": 1.0, "b": -2.0, "tam_v2_total": -1.0},
            {"a": 3.0, "b": -4.0, "tam_v2_total": -1.0},
        ]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert stats["a"]["signed_mean"] == 2.0
        assert stats["a"]["abs_mean"] == 2.0
        assert stats["b"]["signed_mean"] == -3.0
        assert stats["b"]["abs_mean"] == 3.0

    def test_cancellation_shows_in_abs_share(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [
            {"a": 100.0, "b": -100.0, "tam_v2_total": 0.0},
            {"a": 100.0, "b": -100.0, "tam_v2_total": 0.0},
        ]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        # Signed sums cancel → signed_share should be None since total ≈ 0
        assert abs(stats["a"]["signed_sum"] + stats["b"]["signed_sum"]) < 1e-6
        # But abs_share should show equal contribution
        assert stats["a"]["abs_share_of_total_abs"] == pytest.approx(0.5, abs=1e-4)
        assert stats["b"]["abs_share_of_total_abs"] == pytest.approx(0.5, abs=1e-4)

    def test_zero_total_no_division_error(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 0.0, "b": 0.0, "tam_v2_total": 0.0}]
        stats = _compute_contribution_stats(rows, ["a", "b"])
        # Should not raise
        assert stats["a"]["signed_share_of_total"] is None
        assert stats["a"]["abs_share_of_total_abs"] is None

    def test_positive_negative_separate(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [
            {"a": 5.0, "tam_v2_total": 5.0},
            {"a": -3.0, "tam_v2_total": -3.0},
        ]
        stats = _compute_contribution_stats(rows, ["a"])
        assert stats["a"]["positive_mean"] == 5.0
        assert stats["a"]["negative_mean"] == -3.0
        assert stats["a"]["positive_rate"] == 0.5
        assert stats["a"]["negative_rate"] == 0.5

    def test_missing_fields_default_to_zero(self):
        from scripts.diagnose_reward_control_failure import _compute_contribution_stats
        rows = [{"a": 1.0}]  # 'b' missing
        stats = _compute_contribution_stats(rows, ["a", "b"])
        assert stats["b"]["signed_mean"] == 0.0
        assert stats["b"]["abs_mean"] == 0.0


class TestActiveGroupIdentification:
    def test_mav_group_keys(self):
        from scripts.diagnose_reward_control_failure import MAV_ACTIVE_GROUP
        assert "dense_active" in MAV_ACTIVE_GROUP
        assert "event_active" in MAV_ACTIVE_GROUP
        assert "tam_v2_mav_safety" in MAV_ACTIVE_GROUP["dense_active"]
        assert "tam_v2_mav_event" in MAV_ACTIVE_GROUP["event_active"]

    def test_uav_group_keys(self):
        from scripts.diagnose_reward_control_failure import UAV_ACTIVE_GROUP
        assert "dense_active" in UAV_ACTIVE_GROUP
        assert "tam_v2_uav_height" in UAV_ACTIVE_GROUP["dense_active"]
        assert "tam_v2_uav_dodge" in UAV_ACTIVE_GROUP["dense_active"]


class TestPhaseClassification:
    def test_phase_keys_exist(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        phases = _classify_phase([], [])
        for p in ["alive_all", "pre_crash_100", "missile_warning", "no_missile_warning"]:
            assert p in phases

    def test_dodge_nonzero_is_missile_warning(self):
        from scripts.diagnose_reward_control_failure import _classify_phase
        reward = [{"episode": 0, "agent_id": "red_1", "step": 10,
                    "tam_v2_uav_dodge": -30.0, "tam_v2_uav_dodge_angle": -1.0}]
        flight = []
        phases = _classify_phase(flight, reward)
        assert len(phases["missile_warning"]) == 1
        assert len(phases["no_missile_warning"]) == 0


class TestScaleAnalysis:
    def test_scale_simulates_all_levels(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        rows = [
            {"tam_v2_uav_height": 8.0, "tam_v2_uav_dodge": -5.0,
             "tam_v2_uav_event": -0.5, "tam_v2_total": -0.05},
        ] * 10
        result = _compute_scale_analysis(rows, "uav")
        for scale_str in ["0.02", "0.05", "0.10"]:
            assert scale_str in result["simulated_scales"], f"missing scale {scale_str}"
            assert result["simulated_scales"][scale_str]["dense_event_ratio_preserved"] is True

    def test_scale_does_not_change_abs_shares(self):
        from scripts.diagnose_reward_control_failure import _compute_scale_analysis
        rows = [
            {"tam_v2_uav_height": 8.0, "tam_v2_uav_angle": -5.0,
             "tam_v2_uav_distance": -3.0, "tam_v2_uav_dodge": -5.0,
             "tam_v2_uav_speed": 1.0, "tam_v2_uav_event": -0.5,
             "tam_v2_total": -0.09},
        ] * 50
        result = _compute_scale_analysis(rows, "uav")
        s02 = result["simulated_scales"]["0.02"]["abs_shares"]
        s10 = result["simulated_scales"]["0.10"]["abs_shares"]
        h02 = s02.get("tam_v2_uav_height_abs_share")
        h10 = s10.get("tam_v2_uav_height_abs_share")
        if h02 is not None and h10 is not None:
            assert abs(h02 - h10) < 1e-6, f"scale changed proportions: {h02} vs {h10}"


class TestV1Compatibility:
    def test_v1_keys_still_defined(self):
        from scripts.diagnose_reward_control_failure import BRMA_KEYS, MAV_TAM_KEYS, EVENT_KEYS
        assert len(BRMA_KEYS) > 0
        assert len(MAV_TAM_KEYS) > 0
        assert len(EVENT_KEYS) > 0
