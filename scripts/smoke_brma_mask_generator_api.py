"""Smoke test for BRMA mask generator API. No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from brma.mask_generator import (
    BRMAMaskGenerator,
    BRMAMaskGeneratorConfig,
    fuse_brma_masks,
    generate_brma_masks,
    gumbel_sigmoid_straight_through,
    make_type_masks_torch,
    sample_random_friendly_drop_mask,
    sample_train_mask_counts,
    select_biased_enemy_drop_mask,
)


def main() -> None:
    B, N, F = 3, 6, 16
    n_ego, n_ally, n_enemy = 1, 2, 3

    # ---- 1. BRMAMaskGenerator forward ----
    cfg = BRMAMaskGeneratorConfig(entity_feature_dim=F, temperature=0.1)
    gen = BRMAMaskGenerator(cfg)
    feats = torch.randn(B, N, F)
    emask = torch.zeros(B, N, dtype=torch.long)
    emask[0, -1] = 1  # one invalid entity
    out = gen(feats, emask)
    assert out["logits"].shape == (B, N)
    assert out["p"].shape == (B, N)
    assert out["valid_mask"].shape == (B, N)
    assert (out["p"] >= 0).all() and (out["p"] <= 1).all()
    assert torch.isfinite(out["p"]).all()
    assert not out["valid_mask"][0, -1].item()  # invalid entity marked

    # ---- 2. sample_train_mask_counts ----
    mR, mB = sample_train_mask_counts(100, device=torch.device("cpu"))
    assert mR.shape == (100,)
    assert mB.shape == (100,)
    assert (mR >= 0).all() and (mR <= 2).all()
    assert (mB >= 0).all() and (mB <= 2).all()
    assert mR.dtype == torch.int64

    # ---- 3. type masks ----
    tmsk = make_type_masks_torch(B, n_ego, n_ally, n_enemy)
    assert tmsk["self"].shape == (B, N)
    assert tmsk["self"][0, 0].item()
    assert not tmsk["self"][0, 1].item()
    assert tmsk["ally"][0, 1].item() and tmsk["ally"][0, 2].item()
    assert not tmsk["ally"][0, 3].item()
    assert tmsk["enemy"][0, 3].item() and tmsk["enemy"][0, 5].item()

    # ---- 4. random friendly drop mask ----
    valid = torch.ones(B, N, dtype=torch.bool)
    mR_count = torch.tensor([0, 1, 2], dtype=torch.int64)
    fdrop = sample_random_friendly_drop_mask(valid, tmsk["ally"], mR_count)
    assert fdrop.shape == (B, N)
    assert not fdrop[0].any()  # mR=0 → no drops
    assert fdrop[1].sum().item() <= 1
    assert fdrop[2].sum().item() <= 2
    assert not (fdrop & tmsk["self"]).any()   # never drops self
    assert not (fdrop & tmsk["enemy"]).any()  # never drops enemy

    # ---- 5. biased enemy drop mask ----
    p = torch.full((B, N), 0.5)
    p[0, 3] = 0.1  # enemy 0 — lowest retention, should be dropped first
    p[0, 4] = 0.9
    p[0, 5] = 0.3
    mB_count = torch.tensor([1, 0, 2], dtype=torch.int64)
    edrop = select_biased_enemy_drop_mask(p, valid, tmsk["enemy"], mB_count)
    assert edrop.shape == (B, N)
    assert edrop[0, 3].item()   # lowest p = 0.1 → dropped first
    assert not edrop[1].any()   # mB=0
    assert edrop[2].sum().item() <= 2
    assert not (edrop & tmsk["self"]).any()
    assert not (edrop & tmsk["ally"]).any()

    # ---- 6. mask fusion ----
    emask_fuse = torch.zeros(B, N, dtype=torch.int64)
    emask_fuse[0, 4] = 1  # one dead enemy
    fused = fuse_brma_masks(emask_fuse, tmsk["self"], fdrop, edrop)
    assert fused["death_or_padding_mask"][0, 4].item()  # marked invalid
    assert fused["key_padding_mask"][0, 4].item()
    assert fused["keep_mask"][0, 0].item()  # self always kept

    # ---- 7. Gumbel ST ----
    logits = torch.randn(B, N, requires_grad=True)
    msoft, mhard = gumbel_sigmoid_straight_through(logits, temperature=0.1)
    assert msoft.shape == (B, N)
    assert (msoft >= 0).all() and (msoft <= 1).all()
    assert torch.isfinite(msoft).all()
    assert mhard.shape == (B, N)
    assert torch.all((mhard == 0) | (mhard == 1))
    loss = msoft.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    # ---- 8. generate_brma_masks high-level API ----
    result = generate_brma_masks(
        gen, feats, emask, n_ego, n_ally, n_enemy,
        mR_count=mR_count[:B], mB_count=mB_count[:B],
    )
    for key in ("logits", "p", "msoft", "mhard", "key_padding_mask",
                "keep_mask", "mR_count", "mB_count", "enemy_drop_mask",
                "friendly_drop_mask", "meta"):
        assert key in result, f"missing key: {key}"
    assert result["key_padding_mask"].shape == (B, N)
    assert result["keep_mask"][:, 0].all()  # self kept
    assert (result["enemy_drop_mask"].sum(dim=1) <= result["mB_count"]).all()
    assert (result["friendly_drop_mask"].sum(dim=1) <= result["mR_count"]).all()

    # ---- 9. no shared mutable state ----
    result2 = generate_brma_masks(
        gen, feats, emask, n_ego, n_ally, n_enemy,
    )
    assert not torch.equal(result["msoft"], result2["msoft"])  # new random draws

    # ---- 10. config validation ----
    try:
        BRMAMaskGeneratorConfig(entity_feature_dim=1, temperature=0)
        assert False
    except ValueError:
        pass
    try:
        BRMAMaskGeneratorConfig(entity_feature_dim=1, max_mask_allies=-1)
        assert False
    except ValueError:
        pass

    print("brma mask generator api smoke test passed")


if __name__ == "__main__":
    main()
