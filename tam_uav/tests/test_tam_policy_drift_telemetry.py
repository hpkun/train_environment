import copy

import numpy as np
import torch

from algorithms.happo import TAMCategoricalHAPPOTrainer
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
