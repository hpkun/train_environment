from pathlib import Path

import pytest

from scripts.full_review_audit_utils import (
    action_clamp_stats,
    explained_variance,
    gate_mismatch_stats,
    phase_summary,
    reward_component_stats,
    select_checkpoints_from_train_log,
    source_line_hits,
)


def test_action_clamp_rate_stats_from_train_rows():
    rows = [
        {
            "mav_action_saturation_rate": "0.1",
            "uav_action_saturation_rate": "0.2",
            "action_log_std_mav_mean": "-1.0",
            "action_log_std_uav_mean": "-0.5",
        },
        {
            "mav_action_saturation_rate": "0.3",
            "uav_action_saturation_rate": "0.4",
            "action_log_std_mav_mean": "-0.8",
            "action_log_std_uav_mean": "-0.2",
        },
    ]
    stats = action_clamp_stats(rows)
    assert stats["mav_saturation_mean"] == pytest.approx(0.2)
    assert stats["uav_saturation_max"] == pytest.approx(0.4)
    assert stats["mav_log_std_final"] == pytest.approx(-0.8)


def test_explained_variance_basic_cases():
    assert explained_variance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert explained_variance([1, 1, 1], [0, 0, 0]) == 0.0


def test_reward_active_component_classification_by_role():
    rows = [
        {"role": "mav", "tam_brma_v1_mav_support": "0.4", "tam_brma_v1_uav_gate_sit": "0.0", "reward_total": "0.5"},
        {"role": "attack_uav", "tam_brma_v1_uav_gate_sit": "-0.2", "reward_total": "-0.1"},
    ]
    stats = reward_component_stats(rows)
    by_key = {(r["role"], r["component"]): r for r in stats}
    assert by_key[("mav", "tam_brma_v1_mav_support")]["mean"] == pytest.approx(0.4)
    assert by_key[("attack_uav", "tam_brma_v1_uav_gate_sit")]["negative_rate"] == pytest.approx(1.0)


def test_launch_gate_mismatch_classification():
    rows = [
        {
            "has_track": "1",
            "reward_g_own": "0.5",
            "launch_geometry_ok_3d": "0",
            "boresight_ok_3d": "0",
            "AO_2d_rad": "0.1",
            "ATA_3d_rad": "0.4",
            "TA_2d_rad": "1.0",
            "TA_3d_rad": "0.7",
            "mismatch_type": "reward_positive_real_geometry_false",
        },
        {
            "has_track": "1",
            "reward_g_own": "0.0",
            "launch_geometry_ok_3d": "1",
            "boresight_ok_3d": "1",
            "AO_2d_rad": "0.2",
            "ATA_3d_rad": "0.2",
            "TA_2d_rad": "1.0",
            "TA_3d_rad": "1.0",
            "mismatch_type": "reward_zero_real_geometry_true",
        },
    ]
    stats = gate_mismatch_stats(rows)
    assert stats["track_ok_rate"] == pytest.approx(1.0)
    assert stats["reward_g_own_positive_rate"] == pytest.approx(0.5)
    assert stats["geometry_ok_given_reward_positive"] == pytest.approx(0.0)
    assert stats["reward_positive_given_geometry_ok"] == pytest.approx(0.0)


def test_checkpoint_sweep_selector_and_phase_summary():
    rows = [
        {"total_steps": "1000", "avg_return": "-10", "red_win": "0.0", "mav_survival": "0.1", "red_episode_missiles_fired_mean": "0.0"},
        {"total_steps": "100000", "avg_return": "-5", "red_win": "0.2", "mav_survival": "0.8", "red_episode_missiles_fired_mean": "0.3"},
        {"total_steps": "200000", "avg_return": "-20", "red_win": "0.1", "mav_survival": "0.2", "red_episode_missiles_fired_mean": "0.1"},
    ]
    selected = {r["selector"]: r for r in select_checkpoints_from_train_log(rows)}
    assert selected["best_return"]["total_steps"] == 100000
    assert selected["red_fire_peak"]["total_steps"] == 100000
    assert selected["latest"]["total_steps"] == 200000

    phases = phase_summary(rows, phase_size=100000)
    assert len(phases) == 2
    assert phases[0]["rows"] == 2


def test_source_line_hits(tmp_path: Path):
    p = tmp_path / "sample.py"
    p.write_text("a = 1\nteam_reward = rewards.mean()\n", encoding="utf-8")
    rows = source_line_hits(p, {"team_reward": "team_reward"})
    assert rows[0]["line"] == 2
    assert rows[0]["symbol"] == "team_reward"

