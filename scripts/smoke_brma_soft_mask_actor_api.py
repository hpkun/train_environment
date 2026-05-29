"""Static smoke test for differentiable BRMA soft-mask actor API.

No env, no JSBSim, no training, no evaluation.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_models import AttentionActor  # noqa: E402


def _make_inputs(batch: int = 4, entities_n: int = 5, entity_dim: int = 10):
    entities = torch.randn(batch, entities_n, entity_dim)
    entity_mask = torch.zeros(batch, entities_n, dtype=torch.long)
    rnn_h = torch.zeros(batch, 128)
    actions = torch.randn(batch, 3).clamp(-0.8, 0.8)
    return entities, entity_mask, rnn_h, actions


def _assert_grad(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0


def test_forward_default() -> None:
    B, N, D = 4, 5, 10
    entities, entity_mask, rnn_h, _actions = _make_inputs(B, N, D)
    actor = AttentionActor(entity_dim=D, hidden_size=128, rnn_hidden=128)
    dist, new_h, attn = actor(entities, entity_mask, rnn_h)
    assert dist.mean.shape == (B, 3)
    assert new_h.shape == (B, 128)
    assert attn.shape == (B, 4, N, N)
    assert torch.isfinite(dist.mean).all()


def test_evaluate_actions_soft_mask_grad() -> None:
    entities, entity_mask, rnn_h, actions = _make_inputs()
    actor = AttentionActor(entity_dim=10, hidden_size=128, rnn_hidden=128)
    soft_keep = torch.full((4, 5), 0.5, requires_grad=True)
    out = actor.evaluate_actions(
        entities,
        entity_mask,
        rnn_h,
        actions,
        soft_keep_mask=soft_keep,
    )
    loss = out["mu"].sum()
    loss.backward()
    _assert_grad(soft_keep)


def test_self_keep_and_hard_invalid_no_crash() -> None:
    entities, entity_mask, rnn_h, actions = _make_inputs()
    actor = AttentionActor(entity_dim=10, hidden_size=128, rnn_hidden=128)
    entity_mask[:, -1] = 1
    soft_keep = torch.full((4, 5), 0.5, requires_grad=True)
    soft_keep.data[:, 0] = 0.0
    soft_keep.data[:, -1] = 1.0
    out = actor.evaluate_actions(
        entities,
        entity_mask,
        rnn_h,
        actions,
        soft_keep_mask=soft_keep,
    )
    assert torch.isfinite(out["log_prob"]).all()
    assert out["attn_weights"].shape == (4, 4, 5, 5)


def test_evaluate_dual_actions_soft_mask_grad() -> None:
    entities, zero_mask, rnn_h, actions = _make_inputs()
    actor = AttentionActor(entity_dim=10, hidden_size=128, rnn_hidden=128)
    hard_mask = zero_mask.clone()
    hard_mask[:, -1] = 1
    soft_keep = torch.full((4, 5), 0.5, requires_grad=True)
    dual = actor.evaluate_dual_actions(
        entities,
        zero_mask,
        hard_mask,
        rnn_h,
        actions,
        masked_soft_keep_mask=soft_keep,
    )
    assert torch.isfinite(dual["mu_masked"]).all()
    loss = (dual["mu_masked"] - dual["mu_unmasked"].detach()).square().sum()
    loss.backward()
    _assert_grad(soft_keep)


def test_paper_eq33_soft_mask() -> None:
    entities, entity_mask, rnn_h, actions = _make_inputs()
    actor = AttentionActor(
        entity_dim=10,
        hidden_size=128,
        rnn_hidden=128,
        encoder_mode="paper_eq33",
    )
    soft_keep = torch.full((4, 5), 0.6, requires_grad=True)
    out = actor.evaluate_actions(
        entities,
        entity_mask,
        rnn_h,
        actions,
        soft_keep_mask=soft_keep,
    )
    assert out["mu"].shape == (4, 3)
    assert torch.isfinite(out["log_prob"]).all()
    out["mu"].sum().backward()
    _assert_grad(soft_keep)


def test_hard_mask_compatibility() -> None:
    entities, zero_mask, rnn_h, actions = _make_inputs()
    actor = AttentionActor(entity_dim=10, hidden_size=128, rnn_hidden=128)
    hard_mask = zero_mask.clone()
    hard_mask[:, 2] = 1
    dual = actor.evaluate_dual_actions(
        entities,
        zero_mask,
        hard_mask,
        rnn_h,
        actions,
    )
    assert dual["log_prob_unmasked"].shape == (4,)
    assert dual["log_prob_masked"].shape == (4,)
    assert torch.isfinite(dual["log_prob_masked"]).all()


def main() -> None:
    torch.manual_seed(7)
    test_forward_default()
    test_evaluate_actions_soft_mask_grad()
    test_self_keep_and_hard_invalid_no_crash()
    test_evaluate_dual_actions_soft_mask_grad()
    test_paper_eq33_soft_mask()
    test_hard_mask_compatibility()
    print("brma soft-mask actor api smoke test passed")


if __name__ == "__main__":
    main()
