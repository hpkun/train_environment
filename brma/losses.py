"""Standalone BRMA mask-generator loss helpers.

This module is intentionally not wired into PPO or training.  The paper states
the mask-generator objective as KL[p(a|e) || p(a|emask)] - beta * H(mask).  The
API below implements a sampled-log-prob proxy for the KL term because the
current dry-run storage exposes dual action log-probs, not full Gaussian
parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BRMALossConfig:
    entropy_coef: float = 0.01
    eps: float = 1e-8
    detach_actor_terms: bool = True
    maskable_entropy_only: bool = True

    def __post_init__(self) -> None:
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


def _require_same_shape(name: str, *tensors: torch.Tensor) -> None:
    if not tensors:
        return
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {tensor.shape}")


def compute_maskable_set(
    self_mask: torch.Tensor,
    ally_mask: torch.Tensor,
    enemy_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    include_allies: bool = True,
    include_enemies: bool = True,
    keep_self: bool = True,
) -> torch.Tensor:
    """Return the BRMA mask/entropy-active set S.

    Args:
        *_mask: bool tensors with shape (B, N).
        valid_mask: bool tensor with shape (B, N), True for alive/valid entities.
        include_allies/include_enemies: include those entity types in S.
        keep_self: when True, self entities are excluded from S.  When False,
            self entities are included if valid.

    Returns:
        Bool tensor (B, N), True where BRMA mask loss/entropy may act.
    """

    _require_same_shape("compute_maskable_set", self_mask, ally_mask, enemy_mask, valid_mask)
    self_b = self_mask.bool()
    ally_b = ally_mask.bool()
    enemy_b = enemy_mask.bool()
    valid_b = valid_mask.bool()

    maskable = torch.zeros_like(valid_b, dtype=torch.bool)
    if include_allies:
        maskable |= ally_b
    if include_enemies:
        maskable |= enemy_b
    if not keep_self:
        maskable |= self_b
    else:
        maskable &= ~self_b
    return maskable & valid_b


def masked_entropy_loss(
    msoft: torch.Tensor,
    maskable_set: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return batch-mean Bernoulli entropy over the maskable set.

    This returns entropy itself, not a negated loss.  `compute_brma_mask_loss`
    applies the paper-style `- beta * entropy` sign.
    """

    if eps <= 0:
        raise ValueError("eps must be > 0")
    _require_same_shape("masked_entropy_loss", msoft, maskable_set)
    if msoft.ndim != 2:
        raise ValueError("msoft and maskable_set must have shape (B, N)")

    p = msoft.clamp(min=eps, max=1.0 - eps)
    entropy_each = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    mask = maskable_set.bool()
    masked = entropy_each * mask.to(dtype=entropy_each.dtype)
    counts = mask.sum(dim=1).to(dtype=entropy_each.dtype)
    per_batch = torch.where(
        counts > 0,
        masked.sum(dim=1) / counts.clamp_min(1.0),
        torch.zeros_like(counts),
    )
    return per_batch.mean()


def compute_brma_mask_loss(
    *,
    log_prob_unmasked: torch.Tensor,
    log_prob_masked: torch.Tensor,
    msoft: torch.Tensor,
    maskable_set: torch.Tensor,
    config: BRMALossConfig,
) -> dict:
    """Compute a standalone BRMA mask loss candidate.

    Formula status:
    - Paper-confirmed objective: minimize KL[p(a|e) || p(a|emask)] - beta H.
    - Current standalone implementation: sampled log-prob proxy
      `log_prob_unmasked - log_prob_masked` for the KL term.
    - With `detach_actor_terms=True`, actor log-prob tensors are detached so
      this helper cannot update actor parameters through the proxy term.
    """

    if log_prob_unmasked.shape != log_prob_masked.shape:
        raise ValueError(
            "log_prob_unmasked and log_prob_masked must have the same shape")
    if log_prob_unmasked.ndim != 1:
        raise ValueError("log_prob tensors must have shape (B,)")
    if msoft.ndim != 2:
        raise ValueError("msoft must have shape (B, N)")
    if msoft.shape[0] != log_prob_unmasked.shape[0]:
        raise ValueError("msoft batch size must match log_prob batch size")
    _require_same_shape("compute_brma_mask_loss", msoft, maskable_set)

    lp_unmasked = log_prob_unmasked
    lp_masked = log_prob_masked
    if config.detach_actor_terms:
        lp_unmasked = lp_unmasked.detach()
        lp_masked = lp_masked.detach()

    discrepancy = lp_unmasked - lp_masked
    entropy = masked_entropy_loss(
        msoft,
        maskable_set if config.maskable_entropy_only else torch.ones_like(maskable_set),
        eps=config.eps,
    )
    loss = discrepancy.mean() - config.entropy_coef * entropy
    maskable_count_mean = maskable_set.bool().sum(dim=1).float().mean()
    return {
        "loss": loss,
        "discrepancy_mean": discrepancy.mean(),
        "entropy": entropy,
        "formula_status": (
            "PROJECT_INTERPRETATION_LOGPROB_KL_PROXY_AND_BERNOULLI_ENTROPY_FOR_"
            "CONFIRMED_KL_MINUS_ENTROPY_OBJECTIVE"
        ),
        "maskable_count_mean": maskable_count_mean,
    }
