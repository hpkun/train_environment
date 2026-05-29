"""Static smoke test for attention BRMA train-mode integration.

No env, no JSBSim, no train_attention_mappo.py execution.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_models import AttentionActor  # noqa: E402
from brma.losses import BRMALossConfig  # noqa: E402
from brma.mask_generator import BRMAMaskGenerator, BRMAMaskGeneratorConfig  # noqa: E402
from train_attention_mappo import (  # noqa: E402
    AttentionRolloutBuffer,
    brma_update_attention_mask_generator,
    parse_args_attention,
)


def _parse_with(argv):
    old = sys.argv
    try:
        sys.argv = argv
        return parse_args_attention()
    finally:
        sys.argv = old


def test_parse_args() -> None:
    args = _parse_with(["train_attention_mappo.py"])
    assert args.brma_mode == "off"
    args = _parse_with([
        "train_attention_mappo.py",
        "--brma-mode",
        "train",
        "--brma-lr",
        "0.001",
    ])
    assert args.brma_mode == "train"
    assert args.brma_lr == 0.001
    try:
        _parse_with(["train_attention_mappo.py", "--brma-mode", "bad"])
        assert False, "invalid mode should fail"
    except SystemExit:
        pass


def test_preset_exists() -> None:
    from configs.experiment_presets import get_preset, list_presets
    name = "attention_1v1_strict_eq33_attncritic_brma_train_smoke"
    assert name in list_presets()
    preset = get_preset(name)
    assert preset["brma_mode"] == "train"
    assert preset["obs_adapter"] == "strict"


def _make_buffer(alive: bool = True):
    torch.manual_seed(5)
    np.random.seed(5)
    buffer = AttentionRolloutBuffer(
        num_steps=4,
        num_envs=1,
        num_red=1,
        action_dim=3,
        rnn_hidden_size=128,
    )
    for step in range(4):
        ent = np.random.randn(2, 10).astype(np.float32)
        mask = np.zeros(2, dtype=np.int64)
        action = np.random.randn(3).astype(np.float32) * 0.2
        buffer.store_step(
            step,
            0,
            0,
            ent,
            mask,
            np.zeros(1, dtype=np.float32),
            action,
            reward=0.0,
            value=0.0,
            log_prob=0.0,
            done=0.0,
            alive=alive,
        )
    return buffer


def test_brma_update_changes_mask_generator_only() -> None:
    torch.manual_seed(1)
    actor = AttentionActor(entity_dim=10, action_dim=3, encoder_mode="paper_eq33")
    mask_gen = BRMAMaskGenerator(
        BRMAMaskGeneratorConfig(
            entity_feature_dim=10,
            max_mask_allies=0,
            max_mask_enemies=1,
        )
    )
    optimizer = torch.optim.Adam(mask_gen.parameters(), lr=1e-2)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    mask_before = [p.detach().clone() for p in mask_gen.parameters()]
    config = SimpleNamespace(
        num_red=1,
        num_blue=1,
        brma_update_minibatch_size=2,
        brma_max_grad_norm=0.5,
    )
    loss_config = BRMALossConfig(
        entropy_coef=0.0,
        kl_mode="gaussian",
        detach_unmasked_policy=True,
        detach_masked_policy=False,
        detach_actor_terms=None,
    )
    stats = brma_update_attention_mask_generator(
        actor=actor,
        mask_generator=mask_gen,
        optimizer=optimizer,
        buffer=_make_buffer(alive=True),
        config=config,
        device=torch.device("cpu"),
        loss_config=loss_config,
    )
    assert stats["brma_num_samples"] > 0
    assert stats["brma_num_updates"] > 0
    assert np.isfinite(stats["brma_grad_norm"])
    mask_changed = any(
        not torch.allclose(a, b.detach())
        for a, b in zip(mask_before, mask_gen.parameters())
    )
    assert mask_changed
    for old, new in zip(actor_before, actor.parameters()):
        assert torch.allclose(old, new.detach())


def test_zero_alive_returns_zero_stats() -> None:
    actor = AttentionActor(entity_dim=10, action_dim=3, encoder_mode="paper_eq33")
    mask_gen = BRMAMaskGenerator(BRMAMaskGeneratorConfig(entity_feature_dim=10))
    optimizer = torch.optim.Adam(mask_gen.parameters(), lr=1e-3)
    config = SimpleNamespace(
        num_red=1,
        num_blue=1,
        brma_update_minibatch_size=2,
        brma_max_grad_norm=0.5,
    )
    stats = brma_update_attention_mask_generator(
        actor=actor,
        mask_generator=mask_gen,
        optimizer=optimizer,
        buffer=_make_buffer(alive=False),
        config=config,
        device=torch.device("cpu"),
        loss_config=BRMALossConfig(),
    )
    assert stats["brma_num_samples"] == 0
    assert stats["brma_num_updates"] == 0
    assert stats["brma_loss"] == 0.0


def main() -> None:
    test_parse_args()
    test_preset_exists()
    test_brma_update_changes_mask_generator_only()
    test_zero_alive_returns_zero_stats()
    print("brma train mode static smoke test passed")


if __name__ == "__main__":
    main()
