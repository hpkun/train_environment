import copy

import numpy as np
import torch
import csv
import json

from algorithms.happo import TAMCategoricalHAPPOTrainer
from scripts.train_tam_happo_direct import (
    _new_rollout_agent_missile_counts,
    _record_rollout_agent_missiles,
    _summarize_recent_mav_deaths,
)
from scripts.analyze_tam_mav_policy_drift import analyze_run, write_report
from test_tam_categorical_happo_trainer import _buffer, _policy


def test_neutral_prior_probabilities_are_normalized_and_role_specific():
    policy = _policy()
    mav = policy.neutral_prior_probabilities(0)
    uav = policy.neutral_prior_probabilities(1)
    assert mav.shape == uav.shape == (4, 40)
    torch.testing.assert_close(mav.sum(-1), torch.ones(4))
    torch.testing.assert_close(uav.sum(-1), torch.ones(4))
    assert mav.argmax(-1).tolist() == [39, 20, 20, 20]
    assert uav.argmax(-1).tolist() == [39, 20, 4, 20]
    assert mav.requires_grad is False


def test_policy_drift_telemetry_is_reported_without_affecting_loss_updates():
    torch.manual_seed(91)
    policy_a = _policy()
    policy_b = copy.deepcopy(policy_a)
    buffer_a = _buffer(policy_a, steps=6)
    buffer_b = copy.deepcopy(buffer_a)
    trainer_a = TAMCategoricalHAPPOTrainer(policy_a, ppo_epochs=1)
    trainer_b = TAMCategoricalHAPPOTrainer(policy_b, ppo_epochs=1)
    trainer_b._distribution_metrics = lambda sequences: {
        **trainer_a._distribution_metrics(sequences),
        "kl_to_neutral_mav": 999.0,
    }
    metrics = trainer_a.update(buffer_a)
    trainer_b.update(buffer_b)
    required = {
        "neutral_prior_probs_mav", "neutral_prior_probs_uav",
        "kl_to_neutral_mav", "kl_to_neutral_uav",
        "per_axis_kl_to_neutral_mav", "per_axis_kl_to_neutral_uav",
        "dominant_bin_mav_throttle", "dominant_bin_mav_aileron",
        "dominant_bin_mav_elevator", "dominant_bin_mav_rudder",
        "dominant_bin_uav_throttle", "dominant_bin_uav_aileron",
        "dominant_bin_uav_elevator", "dominant_bin_uav_rudder",
    }
    assert required <= metrics.keys()
    assert np.isfinite(metrics["kl_to_neutral_mav"])
    assert len(metrics["per_axis_kl_to_neutral_mav"]) == 4
    for left, right in zip(policy_a.parameters(), policy_b.parameters()):
        torch.testing.assert_close(left, right)


def test_mav_death_and_per_agent_missile_telemetry_are_aggregated():
    recent = [
        {"mav_death_step": 310, "mav_death_reason": "Crash_LowAlt"},
        {"mav_death_step": 450, "mav_death_reason": "Missile_Kill"},
        {"mav_death_step": None, "mav_death_reason": "alive"},
    ]
    summary = _summarize_recent_mav_deaths(recent)
    assert summary["mav_death_step_mean_recent"] == 380.0
    assert summary["mav_death_step_median_recent"] == 380.0
    assert summary["mav_death_reason_top_recent"] == "Crash_LowAlt"
    assert summary["mav_crash_lowalt_rate_recent"] == 0.5
    assert summary["mav_missile_kill_rate_recent"] == 0.5

    counts = _new_rollout_agent_missile_counts(["red_0", "red_1", "red_2"])
    info = {
        "red_0": {"missiles_fired_this_step": 0},
        "red_1": {"missiles_fired_this_step": 1},
        "red_2": {"missiles_fired_this_step": 2},
        "__launch_quality_done__": [
            {"owner_id": "red_1", "is_success": True},
            {"owner_id": "red_2", "termination_reason": "hit"},
        ],
    }
    _record_rollout_agent_missiles(counts, info, ["red_0", "red_1", "red_2"])
    assert counts["fired"] == {"red_0": 0, "red_1": 1, "red_2": 2}
    assert counts["hits"] == {"red_0": 0, "red_1": 1, "red_2": 1}


def test_drift_analyzer_prefers_new_death_and_agent_firing_fields(tmp_path):
    fields = [
        "total_steps", "mav_death_step_mean_recent",
        "mav_death_reason_top_recent", "red_uav_fired_rollout",
        "red_uav_hits_rollout",
    ]
    with (tmp_path / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "total_steps": 256, "mav_death_step_mean_recent": 310,
            "mav_death_reason_top_recent": "Crash_LowAlt",
            "red_uav_fired_rollout": 2, "red_uav_hits_rollout": 1,
        })
    (tmp_path / "runner_status.json").write_text(json.dumps({
        "status": "normal", "total_env_steps_actual": 256,
    }), encoding="utf-8")
    result = analyze_run(tmp_path)
    assert result["mav_death_time"]["available"] is True
    assert result["mav_death_time"]["end"] == 310.0
    assert result["mav_death_reason"]["top_recent"] == "Crash_LowAlt"
    assert result["per_agent_missiles"]["red_uav_fired_total"] == 2
    assert result["per_agent_missiles"]["red_uav_hits_total"] == 1
    output = tmp_path / "reports"
    write_report(result, None, output)
    assert (output / f"{tmp_path.name}_mav_policy_drift_50k.json").exists()
