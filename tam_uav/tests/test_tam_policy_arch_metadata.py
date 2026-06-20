from __future__ import annotations

import json

import pytest
import torch

from algorithms.happo import (
    HAPPOReferencePolicy,
    TAMCategoricalRecurrentHAPPOPolicy,
)
from scripts.eval_tam_happo_direct import _build_policy_from_meta
from scripts.train_tam_happo_direct import (
    POLICY_ARCH_CHOICES,
    _build_policy,
    _eval_checkpoint_extra,
    _resolve_policy_arch,
)


def _categorical_policy(requested="tam_categorical_recurrent", meta=None):
    return _build_policy(
        requested, 96, 480, 4, torch.device("cpu"),
        init_checkpoint_meta=meta,
        action_distribution="multidiscrete_categorical", action_levels=40,
    )


def test_categorical_policy_arch_supports_explicit_name_and_legacy_alias():
    assert "tam_categorical_recurrent" in POLICY_ARCH_CHOICES
    assert _resolve_policy_arch(
        "tam_categorical_recurrent", "multidiscrete_categorical"
    ) == ("tam_categorical_recurrent", "tam_categorical_recurrent", False)
    assert _resolve_policy_arch(
        "brma_recurrent_masked", "multidiscrete_categorical"
    ) == ("brma_recurrent_masked", "tam_categorical_recurrent", True)
    assert isinstance(_categorical_policy(), TAMCategoricalRecurrentHAPPOPolicy)
    assert isinstance(
        _categorical_policy("brma_recurrent_masked"),
        TAMCategoricalRecurrentHAPPOPolicy,
    )


def test_categorical_metadata_records_requested_and_effective_architecture():
    class Args:
        reward_mode = "happo_ref_v0"
        opponent_policy = "tam_direct_fsm"
        policy_arch = "brma_recurrent_masked"
        requested_policy_arch = "brma_recurrent_masked"
        effective_policy_arch = "tam_categorical_recurrent"
        policy_arch_alias_used = True
        num_envs = 1
        rollout_length = 256
        init_checkpoint = None

    meta = _eval_checkpoint_extra(
        Args(), _categorical_policy("brma_recurrent_masked"), 96, 480, 4, 256
    )
    required = {
        "requested_policy_arch", "effective_policy_arch", "policy_arch",
        "policy_arch_alias_used", "policy_class", "trainer_class",
        "tam_action_distribution", "tam_action_levels", "action_distribution",
        "action_levels", "action_space", "critic_arch", "recurrent_update",
        "happo_correction", "neutral_action_init", "neutral_action_centers",
    }
    assert required <= meta.keys()
    assert meta["requested_policy_arch"] == "brma_recurrent_masked"
    assert meta["effective_policy_arch"] == "tam_categorical_recurrent"
    assert meta["policy_arch"] == "tam_categorical_recurrent"
    assert meta["policy_arch_alias_used"] is True
    assert meta["critic_arch"] == "centralized_attention"


def test_legacy_continuous_flat_policy_route_is_unchanged():
    requested, effective, alias = _resolve_policy_arch(
        "flat", "continuous_quantized"
    )
    assert (requested, effective, alias) == ("flat", "flat", False)
    policy = _build_policy(
        effective, 96, 480, 4, torch.device("cpu"),
        action_distribution="continuous_quantized",
    )
    assert isinstance(policy, HAPPOReferencePolicy)


@pytest.mark.parametrize("invalid_meta", [
    {"tam_action_distribution": "continuous_quantized", "tam_action_levels": 40},
    {
        "tam_action_distribution": "multidiscrete_categorical",
        "tam_action_levels": 40,
        "effective_policy_arch": "brma_recurrent_masked",
    },
])
def test_categorical_checkpoint_rejects_incompatible_metadata(tmp_path, invalid_meta):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(invalid_meta), encoding="utf-8")
    with pytest.raises(ValueError):
        _categorical_policy(meta=meta)


def test_eval_rejects_categorical_checkpoint_with_conflicting_effective_architecture():
    with pytest.raises(ValueError, match="effective_policy_arch"):
        _build_policy_from_meta({
            "tam_action_distribution": "multidiscrete_categorical",
            "tam_action_levels": 40,
            "effective_policy_arch": "brma_recurrent_masked",
        }, torch.device("cpu"), 4)
