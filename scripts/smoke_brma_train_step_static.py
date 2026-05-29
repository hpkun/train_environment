"""Static smoke test for standalone BRMA mask-generator train step.

No env, no JSBSim, no PPO wiring.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_models import AttentionActor  # noqa: E402
from brma.losses import BRMALossConfig  # noqa: E402
from brma.mask_generator import BRMAMaskGenerator, BRMAMaskGeneratorConfig  # noqa: E402
from brma.train_step import (  # noqa: E402
    brma_mask_generator_train_step,
    compute_brma_mask_generator_loss_batch,
)


def _setup(seed: int = 13):
    torch.manual_seed(seed)
    actor = AttentionActor(
        entity_dim=10,
        action_dim=3,
        hidden_size=128,
        rnn_hidden=128,
        encoder_mode="paper_eq33",
    )
    mask_gen = BRMAMaskGenerator(BRMAMaskGeneratorConfig(entity_feature_dim=10))
    B, N, D = 4, 5, 10
    entities = torch.randn(B, N, D)
    entity_mask = torch.zeros(B, N, dtype=torch.long)
    rnn_hidden = torch.zeros(B, 128)
    actions = torch.randn(B, 3).clamp(-0.8, 0.8)
    mR_count = torch.ones(B, dtype=torch.long)
    mB_count = torch.ones(B, dtype=torch.long)
    return actor, mask_gen, entities, entity_mask, rnn_hidden, actions, mR_count, mB_count


def _train_config(entropy_coef: float = 0.05, detach_masked_policy: bool = False):
    return BRMALossConfig(
        kl_mode="gaussian",
        entropy_coef=entropy_coef,
        detach_unmasked_policy=True,
        detach_masked_policy=detach_masked_policy,
    )


def _nonzero_grad(module) -> bool:
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def test_compute_loss_batch_and_grad() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup()
    out = compute_brma_mask_generator_loss_batch(
        actor=actor,
        mask_generator=mask_gen,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_h,
        actions=actions,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        loss_config=_train_config(),
        mR_count=mR,
        mB_count=mB,
    )
    assert torch.isfinite(out["loss"])
    assert out["maskable_count_mean"].item() > 0
    out["loss"].backward()
    assert _nonzero_grad(mask_gen), "mask generator did not receive gradient"
    assert not _nonzero_grad(actor), "actor should remain frozen/no grad"


def test_kl_only_gradient_reaches_mask_generator() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup(17)
    out = compute_brma_mask_generator_loss_batch(
        actor=actor,
        mask_generator=mask_gen,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_h,
        actions=actions,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        loss_config=_train_config(entropy_coef=0.0),
        mR_count=mR,
        mB_count=mB,
    )
    out["loss"].backward()
    assert _nonzero_grad(mask_gen), "KL-only loss did not reach mask generator"
    assert not _nonzero_grad(actor)


def test_optimizer_step_changes_mask_generator_only() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup(19)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    optimizer = torch.optim.Adam(mask_gen.parameters(), lr=1e-3)
    stats = brma_mask_generator_train_step(
        actor=actor,
        mask_generator=mask_gen,
        optimizer=optimizer,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_h,
        actions=actions,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        loss_config=_train_config(entropy_coef=0.0),
        mR_count=mR,
        mB_count=mB,
    )
    assert stats["params_changed"] is True
    assert stats["mask_generator_grad_norm"] > 0
    assert stats["actor_grad_norm_after"] == 0.0
    for old, new in zip(actor_before, actor.parameters()):
        assert torch.allclose(old, new.detach())


def test_detach_masked_policy_blocks_kl_gradient() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup(23)
    out = compute_brma_mask_generator_loss_batch(
        actor=actor,
        mask_generator=mask_gen,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_h,
        actions=actions,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        loss_config=_train_config(entropy_coef=0.0, detach_masked_policy=True),
        mR_count=mR,
        mB_count=mB,
    )
    try:
        out["loss"].backward()
    except RuntimeError:
        # Fully detached KL has no grad_fn; this is acceptable diagnostic proof.
        return
    assert not _nonzero_grad(mask_gen)


def test_legacy_sample_proxy_compute_only() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup(29)
    cfg = BRMALossConfig(
        kl_mode="sample_logprob_proxy",
        entropy_coef=0.05,
        detach_unmasked_policy=True,
        detach_masked_policy=False,
    )
    out = compute_brma_mask_generator_loss_batch(
        actor=actor,
        mask_generator=mask_gen,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_h,
        actions=actions,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        loss_config=cfg,
        mR_count=mR,
        mB_count=mB,
    )
    assert torch.isfinite(out["loss"])
    assert out["kl_mode"] == "sample_logprob_proxy"


def test_shape_mismatch() -> None:
    actor, mask_gen, entities, entity_mask, rnn_h, actions, mR, mB = _setup(31)
    try:
        compute_brma_mask_generator_loss_batch(
            actor=actor,
            mask_generator=mask_gen,
            entities=entities[:, :-1],
            entity_mask=entity_mask,
            rnn_hidden=rnn_h,
            actions=actions,
            n_ego=1,
            n_allies=1,
            n_enemies=3,
            loss_config=_train_config(),
            mR_count=mR,
            mB_count=mB,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def main() -> None:
    test_compute_loss_batch_and_grad()
    test_kl_only_gradient_reaches_mask_generator()
    test_optimizer_step_changes_mask_generator_only()
    test_detach_masked_policy_blocks_kl_gradient()
    test_legacy_sample_proxy_compute_only()
    test_shape_mismatch()
    print("brma train-step static smoke test passed")


if __name__ == "__main__":
    main()
