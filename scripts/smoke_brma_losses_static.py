"""Pure static smoke tests for standalone BRMA loss helpers.

This script does not create the environment, reset JSBSim, train, or evaluate.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brma.losses import (  # noqa: E402
    BRMALossConfig,
    compute_brma_mask_loss,
    compute_maskable_set,
    diagonal_gaussian_kl,
    masked_entropy_loss,
)


def _assert_raises_value_error(fn) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_config_validation() -> None:
    BRMALossConfig()
    _assert_raises_value_error(lambda: BRMALossConfig(entropy_coef=-0.1))
    _assert_raises_value_error(lambda: BRMALossConfig(eps=0.0))
    _assert_raises_value_error(lambda: BRMALossConfig(kl_mode="bad"))


def test_compute_maskable_set() -> None:
    self_mask = torch.tensor([[True, False, False, False]])
    ally_mask = torch.tensor([[False, True, False, False]])
    enemy_mask = torch.tensor([[False, False, True, True]])
    valid_mask = torch.tensor([[True, True, True, False]])
    maskable = compute_maskable_set(self_mask, ally_mask, enemy_mask, valid_mask)
    expected = torch.tensor([[False, True, True, False]])
    assert torch.equal(maskable, expected)


def test_masked_entropy_loss() -> None:
    msoft = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.9, 0.4]])
    maskable = torch.tensor([[True, False, False], [False, False, True]])
    entropy = masked_entropy_loss(msoft, maskable)
    manual = torch.stack([
        -(msoft[0, 0] * torch.log(msoft[0, 0]) + (1 - msoft[0, 0]) * torch.log(1 - msoft[0, 0])),
        -(msoft[1, 2] * torch.log(msoft[1, 2]) + (1 - msoft[1, 2]) * torch.log(1 - msoft[1, 2])),
    ]).mean()
    assert entropy.ndim == 0
    assert torch.isfinite(entropy)
    assert torch.allclose(entropy, manual)


def test_diagonal_gaussian_kl() -> None:
    mu = torch.zeros(2, 3)
    sigma = torch.ones(2, 3)
    identical = diagonal_gaussian_kl(mu, sigma, mu, sigma)
    assert identical.shape == (2,)
    assert torch.allclose(identical, torch.zeros(2), atol=1e-6)

    shifted = diagonal_gaussian_kl(mu, sigma, torch.ones(2, 3), sigma)
    assert shifted.shape == (2,)
    assert (shifted > 0).all()
    _assert_raises_value_error(
        lambda: diagonal_gaussian_kl(torch.zeros(2, 3), sigma, torch.zeros(2, 4), sigma)
    )


def test_compute_brma_mask_loss_gaussian_detached() -> None:
    log_prob_unmasked = torch.tensor([-1.0, -2.0], requires_grad=True)
    log_prob_masked = torch.tensor([-1.5, -1.0], requires_grad=True)
    mu_unmasked = torch.tensor([[0.0, 0.2, -0.3], [0.4, -0.1, 0.2]], requires_grad=True)
    sigma_unmasked = torch.ones(2, 3, requires_grad=True)
    mu_masked = torch.tensor([[0.1, 0.0, -0.1], [0.3, 0.2, 0.0]], requires_grad=True)
    sigma_masked = torch.full((2, 3), 1.2, requires_grad=True)
    msoft = torch.tensor([[0.2, 0.7, 0.4], [0.3, 0.8, 0.6]], requires_grad=True)
    maskable = torch.tensor([[True, False, True], [False, True, True]])
    out = compute_brma_mask_loss(
        log_prob_unmasked=log_prob_unmasked,
        log_prob_masked=log_prob_masked,
        mu_unmasked=mu_unmasked,
        sigma_unmasked=sigma_unmasked,
        mu_masked=mu_masked,
        sigma_masked=sigma_masked,
        msoft=msoft,
        maskable_set=maskable,
        config=BRMALossConfig(detach_actor_terms=True, kl_mode="gaussian"),
    )
    expected_keys = {
        "loss",
        "discrepancy_mean",
        "kl_mode",
        "kl_per_batch_mean",
        "entropy",
        "formula_status",
        "maskable_count_mean",
    }
    assert set(out) == expected_keys
    assert out["kl_mode"] == "gaussian"
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert msoft.grad is not None
    assert torch.isfinite(msoft.grad).all()
    assert mu_unmasked.grad is None
    assert sigma_unmasked.grad is None
    assert mu_masked.grad is None
    assert sigma_masked.grad is None
    assert log_prob_unmasked.grad is None
    assert log_prob_masked.grad is None


def test_compute_brma_mask_loss_gaussian_not_detached() -> None:
    log_prob_unmasked = torch.tensor([-1.0, -2.0], requires_grad=True)
    log_prob_masked = torch.tensor([-1.5, -1.0], requires_grad=True)
    mu_unmasked = torch.tensor([[0.0, 0.2], [0.4, -0.1]], requires_grad=True)
    sigma_unmasked = torch.ones(2, 2, requires_grad=True)
    mu_masked = torch.tensor([[0.1, 0.0], [0.3, 0.2]], requires_grad=True)
    sigma_masked = torch.full((2, 2), 1.2, requires_grad=True)
    msoft = torch.tensor([[0.2, 0.7], [0.3, 0.8]], requires_grad=True)
    maskable = torch.tensor([[True, False], [False, True]])
    out = compute_brma_mask_loss(
        log_prob_unmasked=log_prob_unmasked,
        log_prob_masked=log_prob_masked,
        mu_unmasked=mu_unmasked,
        sigma_unmasked=sigma_unmasked,
        mu_masked=mu_masked,
        sigma_masked=sigma_masked,
        msoft=msoft,
        maskable_set=maskable,
        config=BRMALossConfig(detach_actor_terms=False, kl_mode="gaussian"),
    )
    out["loss"].backward()
    assert mu_unmasked.grad is not None
    assert sigma_unmasked.grad is not None
    assert mu_masked.grad is not None
    assert sigma_masked.grad is not None
    assert torch.isfinite(mu_unmasked.grad).all()
    assert torch.isfinite(sigma_unmasked.grad).all()


def test_compute_brma_mask_loss_sample_proxy_and_backward() -> None:
    log_prob_unmasked = torch.tensor([-1.0, -2.0], requires_grad=True)
    log_prob_masked = torch.tensor([-1.5, -1.0], requires_grad=True)
    msoft = torch.tensor([[0.2, 0.7, 0.4], [0.3, 0.8, 0.6]], requires_grad=True)
    maskable = torch.tensor([[True, False, True], [False, True, True]])
    out = compute_brma_mask_loss(
        log_prob_unmasked=log_prob_unmasked,
        log_prob_masked=log_prob_masked,
        msoft=msoft,
        maskable_set=maskable,
        config=BRMALossConfig(detach_actor_terms=True, kl_mode="sample_logprob_proxy"),
    )
    expected_keys = {
        "loss",
        "discrepancy_mean",
        "kl_mode",
        "kl_per_batch_mean",
        "entropy",
        "formula_status",
        "maskable_count_mean",
    }
    assert set(out) == expected_keys
    assert out["kl_mode"] == "sample_logprob_proxy"
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert msoft.grad is not None
    assert torch.isfinite(msoft.grad).all()
    assert log_prob_unmasked.grad is None
    assert log_prob_masked.grad is None


def test_empty_maskable_set() -> None:
    msoft = torch.tensor([[0.2, 0.7], [0.3, 0.8]])
    maskable = torch.zeros_like(msoft, dtype=torch.bool)
    entropy = masked_entropy_loss(msoft, maskable)
    assert torch.isfinite(entropy)
    assert entropy.item() == 0.0


def test_shape_mismatch() -> None:
    _assert_raises_value_error(
        lambda: compute_brma_mask_loss(
            log_prob_unmasked=torch.zeros(2),
            log_prob_masked=torch.zeros(2),
            msoft=torch.zeros(2, 4),
            maskable_set=torch.ones(2, 4, dtype=torch.bool),
            config=BRMALossConfig(kl_mode="gaussian"),
        )
    )
    _assert_raises_value_error(
        lambda: compute_brma_mask_loss(
            log_prob_unmasked=torch.zeros(2),
            log_prob_masked=torch.zeros(3),
            msoft=torch.zeros(2, 4),
            maskable_set=torch.ones(2, 4, dtype=torch.bool),
            config=BRMALossConfig(kl_mode="sample_logprob_proxy"),
        )
    )
    _assert_raises_value_error(
        lambda: masked_entropy_loss(torch.zeros(2, 4), torch.ones(2, 5, dtype=torch.bool))
    )


def main() -> None:
    test_config_validation()
    test_compute_maskable_set()
    test_masked_entropy_loss()
    test_diagonal_gaussian_kl()
    test_compute_brma_mask_loss_gaussian_detached()
    test_compute_brma_mask_loss_gaussian_not_detached()
    test_compute_brma_mask_loss_sample_proxy_and_backward()
    test_empty_maskable_set()
    test_shape_mismatch()
    print("brma losses static smoke test passed")


if __name__ == "__main__":
    main()
