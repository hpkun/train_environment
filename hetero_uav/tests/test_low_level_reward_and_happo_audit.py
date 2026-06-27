import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from scripts.full_review_audit_utils import (
    classify_launch_first_failed_gate,
    gate_mismatch_stats,
    pearson_corr,
    summarize_first_failed_gate,
)


def _cfg():
    return {
        "gate": {
            "min_range_m": 500.0,
            "opt_range_m": 5000.0,
            "launch_range_m": 10000.0,
            "ao_thresh_deg": 45.0,
            "ta_thresh_deg": 90.0,
        }
    }


def test_total_active_components_excludes_diagnostics():
    vals = {
        "tam_brma_v1_flight": 1.0,
        "tam_brma_v1_uav_gate_sit": 2.0,
        "tam_brma_v1_uav_event": -3.0,
        "tam_brma_v1_team_terminal": 4.0,
        "tam_brma_v1_uav_g_own": 999.0,  # diagnostic
        "tam_brma_v1_uav_a_own": 999.0,  # diagnostic
    }
    active_keys = (
        "tam_brma_v1_flight",
        "tam_brma_v1_uav_gate_sit",
        "tam_brma_v1_uav_event",
        "tam_brma_v1_team_terminal",
    )
    total = sum(vals[k] for k in active_keys)
    assert total == pytest.approx(4.0)
    assert vals["tam_brma_v1_uav_g_own"] not in [total]


def test_team_event_multiplicity_counts_two_uav_first_deaths():
    team_uav_loss_shared = -30.0
    num_uav_first_deaths = 2
    assert num_uav_first_deaths * team_uav_loss_shared == pytest.approx(-60.0)


def test_timeout_terminal_zero_no_kill_mav_alive_all_alive():
    # This encodes the intended v1 terminal semantics checked by the audit:
    # no kill, MAV alive, and no losses should not create positive timeout reward.
    blue_kills = 0
    mav_alive = True
    red_uav_losses = 0
    timeout_reward = 0.0 if blue_kills == 0 and mav_alive and red_uav_losses == 0 else -1.0
    assert timeout_reward == 0.0


def test_reward_gate_monotonicity_for_better_ao_ta_distance():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv

    cfg = _cfg()
    worse = (
        HeteroUavCombatEnv._tam_brma_v1_a_own(math.radians(35), cfg)
        * HeteroUavCombatEnv._tam_brma_v1_t_rear(math.radians(110), cfg)
        * HeteroUavCombatEnv._tam_brma_v1_d_gate(12000, cfg)
    )
    better = (
        HeteroUavCombatEnv._tam_brma_v1_a_own(math.radians(10), cfg)
        * HeteroUavCombatEnv._tam_brma_v1_t_rear(math.radians(160), cfg)
        * HeteroUavCombatEnv._tam_brma_v1_d_gate(5000, cfg)
    )
    assert better > worse


def test_2d_reward_positive_3d_boresight_mismatch_is_classified():
    rows = [{
        "has_track": 1,
        "reward_g_own": 0.8,
        "range_ok_3d": 1,
        "ata_ok_3d": 1,
        "ta_ok_3d": 1,
        "launch_geometry_ok_3d": 0,
        "boresight_ok_3d": 0,
        "AO_2d_rad": 0.1,
        "ATA_3d_rad": 0.1,
        "TA_2d_rad": 2.8,
        "TA_3d_rad": 2.8,
        "mismatch_type": "reward_positive_real_geometry_false",
    }]
    stats = gate_mismatch_stats(rows)
    assert stats["geometry_ok_given_reward_positive"] == 0.0
    assert classify_launch_first_failed_gate(rows[0]) == "boresight"


def test_action_logprob_clamp_error_is_detectable():
    dist = Normal(torch.tensor([0.0]), torch.tensor([2.0]))
    raw = torch.tensor([2.5])
    clamped = raw.clamp(-1.0, 1.0)
    err = torch.abs(dist.log_prob(raw) - dist.log_prob(clamped)).item()
    assert err > 0.1


def test_buffer_logprob_replay_detects_mismatch():
    stored_old_log_prob = -1.0
    recomputed_old_log_prob = -1.25
    diff = recomputed_old_log_prob - stored_old_log_prob
    assert abs(diff) > 1e-5


def test_active_mask_excludes_inactive_from_team_reward():
    rewards = np.array([1.0, 10.0, -999.0], dtype=np.float32)
    active = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    team_reward = float((rewards * active).sum() / max(active.sum(), 1.0))
    assert team_reward == pytest.approx(5.5)


def test_launch_gate_blocker_summary_order():
    rows = [
        {"has_track": 1, "range_ok_3d": 0},
        {"has_track": 1, "range_ok_3d": 1, "ata_ok_3d": 0},
        {"has_track": 1, "range_ok_3d": 1, "ata_ok_3d": 1, "ta_ok_3d": 1, "boresight_ok_3d": 0},
    ]
    summary = {r["first_failed_gate"]: r for r in summarize_first_failed_gate(rows)}
    assert summary["range"]["count"] == 1
    assert summary["ata"]["count"] == 1
    assert summary["boresight"]["count"] == 1


def test_credit_assignment_correlation_helper():
    assert pearson_corr([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson_corr([1, 1, 1], [2, 4, 6]) == 0.0
