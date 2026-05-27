"""Static smoke test for EntityObservationEncoder eq.33 mode.

No env, no JSBSim.  Tests shape compatibility for both encoder modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from attention_models import (
    AttentionActor,
    EntityObservationEncoder,
)


def main() -> None:
    B, N, D = 3, 5, 10
    entities = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.long)

    # ---- 1. current mode ----
    enc_cur = EntityObservationEncoder(entity_dim=D, hidden_size=128,
                                       num_heads=4, encoder_mode="current")
    out_cur, attn = enc_cur(entities, mask)
    assert out_cur.shape == (B, 128), f"current shape: {out_cur.shape}"
    assert enc_cur.output_dim == 128

    # ---- 2. paper_eq33 mode ----
    enc_eq33 = EntityObservationEncoder(entity_dim=D, hidden_size=128,
                                        num_heads=4, encoder_mode="paper_eq33")
    out_eq33, attn2 = enc_eq33(entities, mask)
    assert out_eq33.shape == (B, 256), f"eq33 shape: {out_eq33.shape}"
    assert enc_eq33.output_dim == 256

    # ---- 3. AttentionActor current ----
    actor_cur = AttentionActor(entity_dim=D, hidden_size=128,
                               rnn_hidden=128, encoder_mode="current")
    rnn_h = torch.randn(B, 128)
    dist_cur, new_h, _attn = actor_cur(entities, mask, rnn_h)
    assert dist_cur.mean.shape == (B, 3)

    # ---- 4. AttentionActor paper_eq33 ----
    actor_eq33 = AttentionActor(entity_dim=D, hidden_size=128,
                                rnn_hidden=128, encoder_mode="paper_eq33")
    rnn_h2 = torch.randn(B, 128)
    dist_eq33, new_h2, _attn2 = actor_eq33(entities, mask, rnn_h2)
    assert dist_eq33.mean.shape == (B, 3)

    # ---- 5. masked entity works ----
    mask2 = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long).expand(B, N)
    out_masked, _ = enc_eq33(entities, mask2)
    assert out_masked.shape == (B, 256)
    assert torch.isfinite(out_masked).all()

    # ---- 6. invalid encoder_mode ----
    try:
        EntityObservationEncoder(entity_dim=D, encoder_mode="invalid")
        assert False, "should have raised ValueError"
    except ValueError:
        pass

    print("attention eq33 encoder static smoke test passed")


if __name__ == "__main__":
    main()
