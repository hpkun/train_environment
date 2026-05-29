"""Standalone BRMA mask-generator loss helpers.

This module is intentionally not wired into PPO or training.  The paper states
the mask-generator objective as KL[p(a|e) || p(a|emask)] - beta * H(mask).  The
default API below implements the paper's diagonal-Gaussian closed-form KL term.
The older sampled-log-prob proxy remains available for static compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BRMALossConfig:
    entropy_coef: float = 0.05
    eps: float = 1e-8
    detach_actor_terms: bool | None = None
    detach_unmasked_policy: bool = True
    detach_masked_policy: bool = False
    maskable_entropy_only: bool = True
    kl_mode: str = "gaussian"

    def __post_init__(self) -> None:
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be >= 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.kl_mode not in {"gaussian", "sample_logprob_proxy"}:
            raise ValueError(
                "kl_mode must be one of {'gaussian', 'sample_logprob_proxy'}")


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

    dtype_eps = torch.finfo(msoft.dtype).eps if msoft.is_floating_point() else eps
    clamp_eps = max(float(eps), float(dtype_eps))
    p = msoft.clamp(min=clamp_eps, max=1.0 - clamp_eps)
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


def diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-batch KL(N_p || N_q) for diagonal Gaussian policies.

    Inputs must have shape (B, A).  The result has shape (B,).
    Sigma values are clamped to `eps` for numerical stability.
    """

    if eps <= 0:
        raise ValueError("eps must be > 0")
    _require_same_shape("diagonal_gaussian_kl", mu_p, sigma_p, mu_q, sigma_q)
    if mu_p.ndim != 2:
        raise ValueError("Gaussian parameters must have shape (B, A)")

    sp = sigma_p.clamp_min(eps)
    sq = sigma_q.clamp_min(eps)
    mean_delta = mu_p - mu_q
    kl_each = torch.log(sq / sp) + (sp.square() + mean_delta.square()) / (
        2.0 * sq.square()
    ) - 0.5
    return kl_each.sum(dim=1)


def compute_brma_mask_loss(
    *,
    log_prob_unmasked: torch.Tensor,
    log_prob_masked: torch.Tensor,
    msoft: torch.Tensor,
    maskable_set: torch.Tensor,
    config: BRMALossConfig,
    mu_unmasked: torch.Tensor | None = None,
    sigma_unmasked: torch.Tensor | None = None,
    mu_masked: torch.Tensor | None = None,
    sigma_masked: torch.Tensor | None = None,
) -> dict:
    """Compute a standalone BRMA mask loss candidate.

    Formula status:
    - Paper-confirmed objective: minimize KL[p(a|e) || p(a|emask)] - beta H.
    - `kl_mode="gaussian"` uses the exact closed-form KL for diagonal Gaussian
      policies.
    - `kl_mode="sample_logprob_proxy"` preserves the earlier sampled proxy
      `log_prob_unmasked - log_prob_masked`.
    - `detach_unmasked_policy=True` treats p(a|e) as the fixed reference.
    - `detach_masked_policy=False` preserves gradients from KL into the masked
      policy path, allowing future gradients into msoft / mask generator.
    - Legacy `detach_actor_terms=True` overrides the split policy and detaches
      both sides.  That mode is conservative diagnostics only because it cuts
      KL gradients to the mask generator.

    Exact Gaussian KL is paper-aligned for distribution divergence, but future
    BRMA integration still needs a differentiable masked encoder path for mask
    generator gradients.
    """

    if msoft.ndim != 2:
        raise ValueError("msoft must have shape (B, N)")
    _require_same_shape("compute_brma_mask_loss", msoft, maskable_set)

    if config.kl_mode == "gaussian":
        if (
            mu_unmasked is None
            or sigma_unmasked is None
            or mu_masked is None
            or sigma_masked is None
        ):
            raise ValueError("gaussian kl_mode requires mu/sigma tensors")
        if mu_unmasked.ndim != 2:
            raise ValueError("mu/sigma tensors must have shape (B, A)")
        if msoft.shape[0] != mu_unmasked.shape[0]:
            raise ValueError("msoft batch size must match Gaussian batch size")

        mu_p = mu_unmasked
        sigma_p = sigma_unmasked
        mu_q = mu_masked
        sigma_q = sigma_masked
        if config.detach_actor_terms is True:
            mu_p = mu_p.detach()
            sigma_p = sigma_p.detach()
            mu_q = mu_q.detach()
            sigma_q = sigma_q.detach()
        elif config.detach_actor_terms is None:
            if config.detach_unmasked_policy:
                mu_p = mu_p.detach()
                sigma_p = sigma_p.detach()
            if config.detach_masked_policy:
                mu_q = mu_q.detach()
                sigma_q = sigma_q.detach()
        discrepancy = diagonal_gaussian_kl(
            mu_p,
            sigma_p,
            mu_q,
            sigma_q,
            eps=config.eps,
        )
        formula_status = (
            "PAPER_ALIGNED_DIAGONAL_GAUSSIAN_KL_MINUS_ENTROPY_CANDIDATE"
        )
    else:
        if log_prob_unmasked.shape != log_prob_masked.shape:
            raise ValueError(
                "log_prob_unmasked and log_prob_masked must have the same shape")
        if log_prob_unmasked.ndim != 1:
            raise ValueError("log_prob tensors must have shape (B,)")
        if msoft.shape[0] != log_prob_unmasked.shape[0]:
            raise ValueError("msoft batch size must match log_prob batch size")

        lp_unmasked = log_prob_unmasked
        lp_masked = log_prob_masked
        if config.detach_actor_terms is True:
            lp_unmasked = lp_unmasked.detach()
            lp_masked = lp_masked.detach()
        elif config.detach_actor_terms is None:
            if config.detach_unmasked_policy:
                lp_unmasked = lp_unmasked.detach()
            if config.detach_masked_policy:
                lp_masked = lp_masked.detach()
        discrepancy = lp_unmasked - lp_masked
        formula_status = (
            "PROJECT_INTERPRETATION_SAMPLE_LOGPROB_PROXY_AND_BERNOULLI_ENTROPY_FOR_"
            "CONFIRMED_KL_MINUS_ENTROPY_OBJECTIVE"
        )

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
        "kl_mode": config.kl_mode,
        "kl_per_batch_mean": discrepancy.mean(),
        "entropy": entropy,
        "formula_status": formula_status,
        "maskable_count_mean": maskable_count_mean,
    }
