from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from algorithms.happo.vanilla_happo import (
    VanillaHAPPOPolicy, VanillaHAPPOTrainer, clipped_value_loss, huber_loss)
from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, read_vanilla_happo_checkpoint_metadata,
    save_vanilla_happo_checkpoint)
from scripts.train_tam_paper_vanilla_happo import parse_args
from uav_env.JSBSim.paper.protocol import PAPER_NOMINAL_PROTOCOL, protocol_metadata


ROOT = Path(__file__).parents[1]


def make_policy(*, legacy=False):
    kwargs = ({"hidden_dim": 128} if legacy else {
        "hidden_dim": 128,
        "actor_hidden_sizes": [256, 128],
        "critic_hidden_sizes": [256, 128],
    })
    return VanillaHAPPOPolicy(
        ["red_0", "red_1"], {"red_0": "attack_uav", "red_1": "attack_uav"},
        7, 14, **kwargs)


def formal_config(**extra):
    return {"scenario": "2v2", **protocol_metadata(
        "2v2", "none", "jsbsim", PAPER_NOMINAL_PROTOCOL), **extra}


def test_table4_actor_and_critic_widths_are_256_128_feedforward():
    policy = make_policy()
    actor = policy.actors["red_0"]
    assert [(actor[index].in_features, actor[index].out_features)
            for index in (0, 2, 4)] == [(7, 256), (256, 128), (128, 160)]
    backbone = policy.critic.backbone
    assert [(backbone[index].in_features, backbone[index].out_features)
            for index in (0, 2)] == [(14, 256), (256, 128)]
    assert policy.critic.heads["red_0"].in_features == 128
    assert not any(isinstance(module, (nn.GRU, nn.GRUCell, nn.LSTM, nn.LSTMCell,
                                      nn.MultiheadAttention))
                   for module in policy.modules())
    logits = policy.logits("red_0", torch.zeros(2, 7))
    assert logits.shape == (2, 4, 40)


@pytest.mark.parametrize(("error", "expected"), [
    (0.0, 0.0), (5.0, 12.5), (10.0, 50.0), (20.0, 150.0), (-20.0, 150.0),
])
def test_explicit_huber_formula(error, expected):
    actual = huber_loss(torch.tensor(error), delta=10.0)
    assert float(actual) == pytest.approx(expected)


def test_clipped_huber_takes_maximum_and_active_mask_excludes_dead_sample():
    predicted = torch.tensor([0.0, 1000.0])
    old = torch.tensor([20.0, 0.0])
    returns = torch.zeros(2)
    losses = clipped_value_loss(
        predicted, old, returns, 0.2, "clipped_huber", 10.0)
    assert float(losses[0]) == pytest.approx(148.0)
    assert float(losses[1]) == pytest.approx(9950.0)
    active = torch.tensor([True, False])
    assert float(losses[active].mean()) == pytest.approx(148.0)


def test_new_training_cli_defaults_record_published_architecture_and_huber():
    args = parse_args([])
    assert args.actor_hidden_sizes == [256, 128]
    assert args.critic_hidden_sizes == [256, 128]
    assert args.value_loss_type == "clipped_huber"
    assert args.huber_delta == 10.0


def test_new_checkpoint_records_table4_metadata(tmp_path):
    policy = make_policy()
    trainer = VanillaHAPPOTrainer(
        policy, value_loss_type="clipped_huber", huber_delta=10.0)
    path = tmp_path / "new.pt"
    save_vanilla_happo_checkpoint(
        path, policy, trainer, environment_steps=128, episodes=1,
        config=formal_config(
            actor_hidden_sizes=[256, 128], critic_hidden_sizes=[256, 128],
            value_loss_type="clipped_huber", huber_delta=10.0),
        numpy_rng=np.random.default_rng(1))
    metadata = read_vanilla_happo_checkpoint_metadata(path)
    assert metadata["actor_hidden_sizes"] == [256, 128]
    assert metadata["critic_hidden_sizes"] == [256, 128]
    assert metadata["value_loss_type"] == "clipped_huber"
    assert metadata["huber_delta"] == 10.0
    assert metadata["paper_table_4_feedforward_alignment"] is True


def test_legacy_hidden_dim_only_checkpoint_loads_but_new_architecture_rejects_it(
        tmp_path):
    legacy_policy = make_policy(legacy=True)
    legacy_trainer = VanillaHAPPOTrainer(
        legacy_policy, value_loss_type="legacy_clipped_mse")
    path = tmp_path / "legacy.pt"
    save_vanilla_happo_checkpoint(
        path, legacy_policy, legacy_trainer, environment_steps=8, episodes=1,
        config=formal_config(hidden_dim=128), numpy_rng=np.random.default_rng(1))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("actor_hidden_sizes", "critic_hidden_sizes", "value_loss_type",
                "huber_delta", "paper_table_4_feedforward_alignment"):
        payload.pop(key, None)
    torch.save(payload, path)

    restored_legacy = make_policy(legacy=True)
    load_vanilla_happo_checkpoint(path, restored_legacy, restore_rng=False)
    new_policy = make_policy()
    with pytest.raises(ValueError, match="actor architecture mismatch"):
        load_vanilla_happo_checkpoint(path, new_policy, restore_rng=False)


def test_public_fidelity_document_has_required_closure_flags():
    text = (ROOT / "docs/tam_paper_public_fidelity_closure.md").read_text(
        encoding="utf-8")
    assert "PUBLICLY_SPECIFIED_ENVIRONMENT_COMPONENTS_ALIGNED=true" in text
    assert "EXACT_PRIVATE_ENVIRONMENT_REPRODUCED=false" in text
    assert "PAPER_SILENT_ASSUMPTIONS_PRESENT=true" in text
    assert "PURE_FEEDFORWARD_HAPPO_BASELINE=true" in text
