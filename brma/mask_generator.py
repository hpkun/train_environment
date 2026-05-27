"""Mask-vector utilities for BRMA-MAPPO.

Two generations of mask tools are provided:

- ``MaskGeneratorConfig`` / ``MaskVectorGenerator`` — legacy deterministic
  type-aware and uniform random masking (numpy).  Retained for earlier
  smoke tests and as a simple baseline.
- ``BRMAMaskGeneratorConfig`` / ``BRMAMaskGenerator`` — paper-candidate
  learned mask generator (torch) with count-constrained random/biased
  masks, mask fusion, and Gumbel-Softmax / straight-through utilities.
  **Not wired into training or rollout.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskGeneratorConfig:
    """Configuration for candidate mask-generation infrastructure.

    These fields are engineering controls for the standalone module.  They are
    not a claim that the paper used the same probability or keep-rule names.
    """

    random_mask_prob: float = 0.0
    keep_self: bool = True
    keep_allies: bool = True
    keep_enemies: bool = True
    force_keep_at_least_one_enemy: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.random_mask_prob <= 1.0:
            raise ValueError("random_mask_prob must be in [0, 1]")


class MaskVectorGenerator:
    """Generate entity keep masks without touching model or env behavior."""

    def __init__(self, config: MaskGeneratorConfig | None = None):
        self.config = config or MaskGeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)

    @staticmethod
    def make_valid_entity_mask(entity_mask: np.ndarray) -> np.ndarray:
        """Convert entity mask semantics from 0-valid/1-invalid to bool valid."""

        mask = np.asarray(entity_mask)
        if mask.ndim != 1:
            raise ValueError("entity_mask must have shape (N,)")
        return mask == 0

    @staticmethod
    def make_type_masks(n_ego: int, n_allies: int, n_enemies: int) -> dict[str, np.ndarray]:
        """Return entity type masks for order: self, allies, enemies."""

        if n_ego < 0 or n_allies < 0 or n_enemies < 0:
            raise ValueError("entity counts must be non-negative")
        n_entities = n_ego + n_allies + n_enemies
        self_mask = np.zeros(n_entities, dtype=bool)
        ally_mask = np.zeros(n_entities, dtype=bool)
        enemy_mask = np.zeros(n_entities, dtype=bool)

        self_mask[:n_ego] = True
        ally_start = n_ego
        ally_end = ally_start + n_allies
        ally_mask[ally_start:ally_end] = True
        enemy_mask[ally_end:] = True

        return {"self": self_mask, "ally": ally_mask, "enemy": enemy_mask}

    def generate_random_keep_mask(
        self,
        entity_mask: np.ndarray,
        n_ego: int,
        n_allies: int,
        n_enemies: int,
        rng=None,
    ) -> np.ndarray:
        """Generate a bool keep mask.

        ``True`` means the entity is visible to attention. ``False`` means it
        should be masked. Invalid/padded entities are always ``False``.
        """

        valid_mask = self.make_valid_entity_mask(entity_mask)
        type_masks = self.make_type_masks(n_ego, n_allies, n_enemies)
        if valid_mask.shape[0] != n_ego + n_allies + n_enemies:
            raise ValueError("entity_mask length does not match entity counts")

        keep_mask = valid_mask.copy()
        if not self.config.keep_self:
            keep_mask[type_masks["self"]] = False
        if not self.config.keep_allies:
            keep_mask[type_masks["ally"]] = False
        if not self.config.keep_enemies:
            keep_mask[type_masks["enemy"]] = False

        random_candidates = (
            valid_mask
            & keep_mask
            & ~type_masks["self"]
        )
        if self.config.random_mask_prob > 0.0 and np.any(random_candidates):
            active_rng = rng if rng is not None else self._rng
            draws = active_rng.random(int(random_candidates.sum()))
            keep_mask[random_candidates] = draws >= self.config.random_mask_prob

        if self.config.keep_self:
            keep_mask[type_masks["self"] & valid_mask] = True
        keep_mask[~valid_mask] = False

        if self.config.force_keep_at_least_one_enemy and self.config.keep_enemies:
            valid_enemies = np.flatnonzero(valid_mask & type_masks["enemy"])
            if valid_enemies.size and not np.any(keep_mask[valid_enemies]):
                active_rng = rng if rng is not None else self._rng
                keep_mask[int(active_rng.choice(valid_enemies))] = True

        return keep_mask.astype(bool)

    @staticmethod
    def convert_keep_mask_to_attention_key_padding_mask(
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """Convert keep-mask semantics to PyTorch key_padding_mask semantics.

        PyTorch ``nn.MultiheadAttention`` uses ``True`` for keys that should be
        ignored and ``False`` for visible keys.
        """

        keep = np.asarray(keep_mask, dtype=bool)
        if keep.ndim != 1:
            raise ValueError("keep_mask must have shape (N,)")
        return ~keep

    def generate(
        self,
        entity_mask: np.ndarray,
        n_ego: int,
        n_allies: int,
        n_enemies: int,
        rng=None,
    ) -> dict:
        """Generate keep and key-padding masks with diagnostic metadata."""

        keep_mask = self.generate_random_keep_mask(
            entity_mask,
            n_ego,
            n_allies,
            n_enemies,
            rng=rng,
        )
        key_padding_mask = self.convert_keep_mask_to_attention_key_padding_mask(
            keep_mask)
        valid_mask = self.make_valid_entity_mask(entity_mask)
        return {
            "keep_mask": keep_mask,
            "key_padding_mask": key_padding_mask,
            "meta": {
                "random_mask_prob": self.config.random_mask_prob,
                "keep_self": self.config.keep_self,
                "keep_allies": self.config.keep_allies,
                "keep_enemies": self.config.keep_enemies,
                "force_keep_at_least_one_enemy": (
                    self.config.force_keep_at_least_one_enemy),
                "n_valid": int(valid_mask.sum()),
                "n_kept": int(keep_mask.sum()),
                "paper_bias_rule": "NEEDS PAPER TEXT VERIFICATION",
            },
        }

    def generate_biased_random_mask(self, *args, **kwargs) -> np.ndarray:
        """Placeholder for the paper-specific biased random mask.

        NEEDS PAPER TEXT VERIFICATION: the local PDF extraction did not expose
        a reliable mask probability or bias-ranking equation.  This method is
        intentionally not implemented to avoid inventing paper behavior.
        """

        raise NotImplementedError(
            "NEEDS PAPER TEXT VERIFICATION before implementing biased random mask")


# ============================================================================
#  BRMA learned mask generator (candidate API — not wired into training)
# ============================================================================

@dataclass
class BRMAMaskGeneratorConfig:
    """Configuration for the learned BRMA mask generator."""

    entity_feature_dim: int
    hidden_size: int = 128
    temperature: float = 0.1
    max_mask_allies: int = 2
    max_mask_enemies: int = 2
    keep_self: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.max_mask_allies < 0:
            raise ValueError("max_mask_allies must be >= 0")
        if self.max_mask_enemies < 0:
            raise ValueError("max_mask_enemies must be >= 0")
        if self.entity_feature_dim <= 0:
            raise ValueError("entity_feature_dim must be > 0")


class BRMAMaskGenerator(nn.Module):
    """Candidate learned mask generator for BRMA-MAPPO.

    Maps per-entity features to a retention probability ``p`` via a
    two-layer MLP.  Lower ``p`` means the entity is more likely to be
    masked.  This module is **not** wired into training — it provides
    ``logits`` and ``p`` only; mask sampling is done by external helpers.
    """

    def __init__(self, config: BRMAMaskGeneratorConfig):
        super().__init__()
        self.cfg = config
        self.mlp = nn.Sequential(
            nn.Linear(config.entity_feature_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, entity_features: torch.Tensor,
                entity_mask: torch.Tensor | None = None) -> dict:
        """Compute retention probabilities.

        Args:
            entity_features: (B, N, F) per-entity features.
            entity_mask:     (B, N), 0 = valid, 1 = invalid.  Optional.

        Returns dict with logits, p (retention probability), valid_mask.
        """
        logits = self.mlp(entity_features).squeeze(-1)  # (B, N)
        p = torch.sigmoid(logits)
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
        if entity_mask is not None:
            valid_mask = entity_mask == 0
        return {"logits": logits, "p": p, "valid_mask": valid_mask}


# ---------------------------------------------------------------------------
#  Count sampling (paper MR,train / MB,train from Uniform(0, max_*_mask))
# ---------------------------------------------------------------------------

def sample_train_mask_counts(
    batch_size: int,
    max_mask_allies: int = 2,
    max_mask_enemies: int = 2,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample mR_count and mB_count per batch element.

    Returns (mR_count, mB_count), each (B,) int64, uniformly sampled
    from [0, max_mask_*] inclusive.
    """
    mR = torch.randint(0, max_mask_allies + 1, (batch_size,),
                       device=device, generator=generator, dtype=torch.int64)
    mB = torch.randint(0, max_mask_enemies + 1, (batch_size,),
                       device=device, generator=generator, dtype=torch.int64)
    return mR, mB


# ---------------------------------------------------------------------------
#  Type masks (torch)
# ---------------------------------------------------------------------------

def make_type_masks_torch(
    batch_size: int,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Return bool type masks of shape (B, N).

    Entity order: self, allies, enemies.
    """
    n_total = n_ego + n_allies + n_enemies
    self_msk = torch.zeros(batch_size, n_total, dtype=torch.bool, device=device)
    ally_msk = torch.zeros(batch_size, n_total, dtype=torch.bool, device=device)
    enmy_msk = torch.zeros(batch_size, n_total, dtype=torch.bool, device=device)
    self_msk[:, :n_ego] = True
    ally_msk[:, n_ego:n_ego + n_allies] = True
    enmy_msk[:, n_ego + n_allies:] = True
    return {"self": self_msk, "ally": ally_msk, "enemy": enmy_msk}


# ---------------------------------------------------------------------------
#  Count-constrained random friendly drop mask (mR)
# ---------------------------------------------------------------------------

def sample_random_friendly_drop_mask(
    valid_mask: torch.Tensor,
    ally_mask: torch.Tensor,
    mR_count: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a random friendly drop mask.

    Returns bool (B, N), True = dropped / masked.
    Only drops from valid allies; never drops self or enemy.
    At most ``mR_count[b]`` allies are dropped per batch element.
    """
    B, N = valid_mask.shape
    # eligible candidates: valid AND ally
    eligible = valid_mask & ally_mask  # (B, N)
    # Each row: randomly pick up to mR_count[b] entries
    drop = torch.zeros(B, N, dtype=torch.bool, device=valid_mask.device)
    for b in range(B):
        idx = torch.where(eligible[b])[0]
        if idx.numel() == 0:
            continue
        k = min(mR_count[b].item(), int(idx.numel()))
        if k > 0:
            perm = torch.randperm(int(idx.numel()), device=valid_mask.device,
                                  generator=generator)[:k]
            drop[b, idx[perm]] = True
    return drop


# ---------------------------------------------------------------------------
#  Biased enemy drop mask (mB) — select lowest retention-probability enemies
# ---------------------------------------------------------------------------

def select_biased_enemy_drop_mask(
    p: torch.Tensor,
    valid_mask: torch.Tensor,
    enemy_mask: torch.Tensor,
    mB_count: torch.Tensor,
) -> torch.Tensor:
    """Select enemies with lowest retention probability for dropping.

    Returns bool (B, N), True = dropped / masked.

    This follows the current paper-audit interpretation that **lower**
    retention probability means a more mask-worthy entity.  If visual PDF
    verification later shows the opposite Top-M convention, this function
    must be adjusted.
    """
    B, N = p.shape
    eligible = valid_mask & enemy_mask  # (B, N)
    drop = torch.zeros(B, N, dtype=torch.bool, device=p.device)
    for b in range(B):
        idx = torch.where(eligible[b])[0]
        if idx.numel() == 0:
            continue
        k = min(mB_count[b].item(), int(idx.numel()))
        if k > 0:
            # lowest p → highest priority to drop
            _, topk_idx = torch.topk(-p[b, idx], k)
            drop[b, idx[topk_idx]] = True
    return drop


# ---------------------------------------------------------------------------
#  Gumbel-Sigmoid straight-through (msoft)
# ---------------------------------------------------------------------------

def gumbel_sigmoid_straight_through(
    logits: torch.Tensor,
    temperature: float = 0.1,
    hard: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-Sigmoid with optional straight-through hard sample.

    Returns (msoft, mhard):
    - msoft: differentiable soft sample in (0, 1).
    - mhard: binary hard sample if hard=True, else None.
    """
    u = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(u.clamp(min=1e-8) + 1e-8))
    y = logits + gumbel_noise
    msoft = torch.sigmoid(y / temperature)
    if hard:
        mhard = (msoft > 0.5).float()
        mhard = mhard + msoft - msoft.detach()  # straight-through
        return msoft, mhard
    return msoft, None


# ---------------------------------------------------------------------------
#  Mask fusion (death mask + random/biased BRMA masks)
# ---------------------------------------------------------------------------

def fuse_brma_masks(
    entity_mask: torch.Tensor,
    self_mask: torch.Tensor,
    friendly_drop_mask: torch.Tensor,
    enemy_drop_mask: torch.Tensor,
    keep_self: bool = True,
) -> dict:
    """Fuse death/padding mask with BRMA random/biased drop masks.

    Args:
        entity_mask:        (B, N) 0=valid, 1=invalid.
        self_mask:          (B, N) bool for self entity positions.
        friendly_drop_mask: (B, N) bool, True = drop this friendly.
        enemy_drop_mask:    (B, N) bool, True = drop this enemy.
        keep_self:          always keep the self entity if valid.

    Returns dict:
        drop_mask, key_padding_mask, keep_mask, death_or_padding_mask.
    """
    invalid = entity_mask != 0  # True = dead / padded
    drop = invalid | friendly_drop_mask | enemy_drop_mask
    if keep_self:
        drop[self_mask] = False
        drop[invalid & self_mask] = True  # self is still invalid if dead
    keep = ~drop

    return {
        "drop_mask": drop,
        "key_padding_mask": drop,
        "keep_mask": keep,
        "death_or_padding_mask": invalid,
    }


# ---------------------------------------------------------------------------
#  High-level BRMA mask generation API
# ---------------------------------------------------------------------------

def generate_brma_masks(
    generator_model: BRMAMaskGenerator,
    entity_features: torch.Tensor,
    entity_mask: torch.Tensor,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
    mR_count: torch.Tensor | None = None,
    mB_count: torch.Tensor | None = None,
    torch_generator: torch.Generator | None = None,
) -> dict:
    """Generate a full BRMA mask set for one timestep.

    This is a candidate API; it is **not** called by any training script.
    """
    B = entity_features.shape[0]
    forward_out = generator_model(entity_features, entity_mask)
    type_masks = make_type_masks_torch(B, n_ego, n_allies, n_enemies,
                                       device=entity_features.device)

    if mR_count is None or mB_count is None:
        mR_count, mB_count = sample_train_mask_counts(
            B, generator_model.cfg.max_mask_allies,
            generator_model.cfg.max_mask_enemies,
            device=entity_features.device, generator=torch_generator)

    friendly_drop = sample_random_friendly_drop_mask(
        forward_out["valid_mask"], type_masks["ally"], mR_count,
        generator=torch_generator)

    enemy_drop = select_biased_enemy_drop_mask(
        forward_out["p"], forward_out["valid_mask"],
        type_masks["enemy"], mB_count)

    fused = fuse_brma_masks(
        entity_mask, type_masks["self"],
        friendly_drop, enemy_drop,
        keep_self=generator_model.cfg.keep_self)

    msoft, mhard = gumbel_sigmoid_straight_through(
        forward_out["logits"],
        temperature=generator_model.cfg.temperature,
        hard=True, generator=torch_generator)

    return {
        "logits": forward_out["logits"],
        "p": forward_out["p"],
        "msoft": msoft,
        "mhard": mhard,
        "mR_count": mR_count,
        "mB_count": mB_count,
        "friendly_drop_mask": friendly_drop,
        "enemy_drop_mask": enemy_drop,
        **fused,
        "meta": {
            "n_ego": n_ego,
            "n_allies": n_allies,
            "n_enemies": n_enemies,
            "temperature": generator_model.cfg.temperature,
            "mask_type": "brma_candidate_no_verification",
        },
    }
