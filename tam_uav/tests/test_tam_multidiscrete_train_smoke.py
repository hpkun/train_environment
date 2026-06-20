from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from scripts.train_tam_happo_direct import _build_policy, _eval_checkpoint_extra


def test_formal_route_builds_categorical_policy_and_int64_buffer():
    policy = _build_policy(
        "brma_recurrent_masked", 96, 480, 4, torch.device("cpu"),
        action_distribution="multidiscrete_categorical", action_levels=40,
    )
    assert isinstance(policy, TAMCategoricalRecurrentHAPPOPolicy)
    buffer = HAPPORolloutBuffer(
        2, 3, 96, 480, 4, [0, 1, 1],
        rnn_hidden_size=policy.rnn_hidden_size, action_dtype="int64",
    )
    assert buffer.actions.dtype == np.int64


def test_formal_route_rejects_legacy_checkpoint_meta(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "action_distribution": "continuous_quantized",
        "action_levels": 40,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="rejects legacy continuous"):
        _build_policy(
            "brma_recurrent_masked", 96, 480, 4, torch.device("cpu"),
            init_checkpoint_meta=meta,
            action_distribution="multidiscrete_categorical", action_levels=40,
        )


def test_checkpoint_metadata_contains_formal_action_contract():
    class Args:
        reward_mode = "happo_ref_v0"
        opponent_policy = "tam_direct_fsm"
        policy_arch = "brma_recurrent_masked"
        num_envs = 1
        rollout_length = 256
        init_checkpoint = None

    policy = TAMCategoricalRecurrentHAPPOPolicy(hidden_dim=32, rnn_hidden_size=32)
    meta = _eval_checkpoint_extra(Args(), policy, 96, 480, 4, 256)
    assert meta["tam_action_distribution"] == "multidiscrete_categorical"
    assert meta["tam_action_levels"] == 40
    assert meta["action_space"] == "MultiDiscrete"
    assert meta["policy_class"] == "TAMCategoricalRecurrentHAPPOPolicy"
