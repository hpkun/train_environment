"""Smoke test for BRMA dual actor evaluation API. No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from attention_models import AttentionActor


def main() -> None:
    B, N, D = 4, 5, 10
    entities = torch.randn(B, N, D)
    zero_mask = torch.zeros(B, N, dtype=torch.long)
    rnn_h = torch.zeros(B, 128)
    actions = torch.randn(B, 3)

    # ---- 1. forward backward compatibility ----
    actor = AttentionActor(entity_dim=D, hidden_size=128, rnn_hidden=128,
                           encoder_mode="current")
    dist, new_h, attn = actor(entities, zero_mask, rnn_h)
    assert dist.mean.shape == (B, 3)
    assert new_h.shape == (B, 128)
    assert attn.shape == (B, 4, N, N)  # 4 heads

    # ---- 2. evaluate_actions ----
    ev = actor.evaluate_actions(entities, zero_mask, rnn_h, actions)
    assert ev["log_prob"].shape == (B,)
    assert ev["entropy_mean"].shape == (B,)
    assert ev["entropy_sum"].shape == (B,)
    assert ev["mu"].shape == (B, 3)
    assert ev["sigma"].shape == (B, 3)
    assert ev["new_rnn_hidden"].shape == (B, 128)
    assert torch.isfinite(ev["log_prob"]).all()
    assert torch.isfinite(ev["entropy_mean"]).all()

    # ---- 3. evaluate_dual_actions with same masks ----
    dual_same = actor.evaluate_dual_actions(
        entities, zero_mask, zero_mask, rnn_h, actions)
    assert torch.allclose(dual_same["log_prob_unmasked"],
                          dual_same["log_prob_masked"])
    assert torch.allclose(dual_same["mu_unmasked"],
                          dual_same["mu_masked"])

    # ---- 4. evaluate_dual_actions with different masks ----
    masked_mask = zero_mask.clone()
    masked_mask[:, 2] = 1  # drop entity 2 (not self)
    dual_diff = actor.evaluate_dual_actions(
        entities, zero_mask, masked_mask, rnn_h, actions)
    assert torch.isfinite(dual_diff["log_prob_unmasked"]).all()
    assert torch.isfinite(dual_diff["log_prob_masked"]).all()
    assert dual_diff["unmasked"]["attn_weights"].shape == (B, 4, N, N)
    assert dual_diff["masked"]["attn_weights"].shape == (B, 4, N, N)

    # ---- 5. self mask protection (encoder forces self visible) ----
    self_masked = zero_mask.clone()
    self_masked[:, 0] = 1  # try to mask self
    dual_self = actor.evaluate_dual_actions(
        entities, zero_mask, self_masked, rnn_h, actions)
    assert torch.isfinite(dual_self["log_prob_masked"]).all()
    # Different log-probs expected because entity 2 is masked differently
    # from the unmasked path - we just check no crash.

    # ---- 6. paper_eq33 mode ----
    actor_eq33 = AttentionActor(entity_dim=D, hidden_size=128, rnn_hidden=128,
                                encoder_mode="paper_eq33")
    dual_eq33 = actor_eq33.evaluate_dual_actions(
        entities, zero_mask, masked_mask, rnn_h, actions)
    assert dual_eq33["log_prob_unmasked"].shape == (B,)
    assert dual_eq33["log_prob_masked"].shape == (B,)
    assert torch.isfinite(dual_eq33["log_prob_unmasked"]).all()

    # ---- 7. gradient through masked path ----
    actor_grad = AttentionActor(entity_dim=D, hidden_size=128, rnn_hidden=128)
    dual_g = actor_grad.evaluate_dual_actions(
        entities, zero_mask, masked_mask, rnn_h, actions)
    loss = dual_g["log_prob_masked"].mean() + dual_g["entropy_masked_mean"].mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any()
        for p in actor_grad.parameters())
    assert has_grad, "no parameter received gradient"

    print("brma dual actor api smoke test passed")


if __name__ == "__main__":
    main()
