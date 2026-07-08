from __future__ import annotations

import csv
import json

from scripts.analyze_marl_training_dynamics import analyze


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_marl_training_dynamics_reads_mock_logs(tmp_path):
    _write_csv(tmp_path / "train_log.csv", [
        {
            "iteration": 1, "total_steps": 100, "avg_return": 1.0,
            "red_win": 0.0, "blue_win": 1.0, "draw": 0.0, "timeout": 0.0,
            "mav_survival": 0.0, "critic_loss": 10.0,
            "critic_epochs": 3, "critic_loss_first_epoch": 10.0,
            "critic_loss_last_epoch": 6.0,
            "critic_grad_norm_mean_over_epochs": 2.0,
            "critic_grad_norm_max_over_epochs": 3.0,
            "return_std": 4.0, "value_pred_new_std": 0.05,
            "value_pred_old_mean": 0.0, "value_pred_new_mean": 0.1,
            "value_explained_variance_old": -0.5,
            "value_explained_variance_new": -0.2,
            "approx_kl_uav": 0.001, "clip_fraction_uav": 0.0,
            "entropy_uav": 0.7, "action_log_std_uav_mean": -1.2,
            "uav_action_saturation_rate": 0.1, "rollout_transitions": 256,
        },
        {
            "iteration": 2, "total_steps": 200, "avg_return": -1.0,
            "red_win": 0.0, "blue_win": 0.0, "draw": 1.0, "timeout": 1.0,
            "mav_survival": 1.0, "critic_loss": 1.0,
            "critic_epochs": 3, "critic_loss_first_epoch": 8.0,
            "critic_loss_last_epoch": 5.0,
            "critic_grad_norm_mean_over_epochs": 2.5,
            "critic_grad_norm_max_over_epochs": 3.5,
            "return_std": 5.0, "value_pred_new_std": 0.02,
            "value_pred_old_mean": 0.0, "value_pred_new_mean": 0.1,
            "value_explained_variance_old": -0.6,
            "value_explained_variance_new": -0.1,
            "approx_kl_uav": 0.2, "clip_fraction_uav": 0.4,
            "entropy_uav": 0.2, "action_log_std_uav_mean": -2.4,
            "uav_action_saturation_rate": 0.5, "rollout_transitions": 256,
        },
    ])
    _write_csv(tmp_path / "terminal_episode_audit.csv", [
        {
            "total_steps": 100, "winner": "blue", "end_reason": "red_eliminated",
            "episode_length": 80,
        },
        {
            "total_steps": 200, "winner": "draw", "end_reason": "timeout",
            "episode_length": 200,
        },
    ])
    (tmp_path / "update_diagnostics.jsonl").write_text(
        json.dumps({
            "iteration": 1,
            "total_steps": 100,
            "valid_sample_count_per_agent": [10, 20, 20],
            "active_sample_ratio_per_agent": [1.0, 1.0, 1.0],
            "clip_fraction_per_agent": [0.0, 0.1, 0.2],
            "approx_kl_per_agent": [0.0, 0.01, 0.02],
            "entropy_per_agent": [0.6, 0.7, 0.7],
            "m_abs_max_after_each_agent": [1.0, 1.2, 1.5],
        }) + "\n",
        encoding="utf-8",
    )

    payload = analyze(tmp_path, phase_bins="0,150,250")

    assert (tmp_path / "learning_dynamics_summary.csv").exists()
    assert (tmp_path / "learning_dynamics_summary.json").exists()
    assert (tmp_path / "learning_dynamics_report.md").exists()
    assert payload["update_diagnostics_rows"] == 1
    assert len(payload["phases"]) == 2
    assert "ppo_clip_high" in payload["diagnostic_flags"]
    assert "critic_underfit" in payload["diagnostic_flags"]
    assert "value_collapse" in payload["diagnostic_flags"]
    assert "critic_improved_by_epochs" in payload["diagnostic_flags"]
    assert payload["phases"][0]["value_explained_variance_old"] == -0.5
    assert payload["phases"][0]["value_explained_variance_new"] == -0.2
    assert payload["phases"][0]["update_diagnostics_summary"]["clip_fraction_per_agent_mean"] == [0.0, 0.1, 0.2]


def test_analyze_marl_training_dynamics_accepts_old_missing_fields(tmp_path):
    _write_csv(tmp_path / "train_log.csv", [
        {"iteration": 1, "total_steps": 100, "avg_return": 0.0},
    ])

    payload = analyze(tmp_path, phase_bins="0,200")

    assert payload["train_rows"] == 1
    assert payload["phases"][0]["clip_fraction_uav"] == 0.0
