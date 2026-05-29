"""Standalone BRMA mask-generator train-step helpers.

These helpers are pure PyTorch infrastructure.  They are not wired into PPO,
do not create an optimizer internally, and do not touch the environment.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .collection import build_selected_soft_keep_mask
from .losses import BRMALossConfig, compute_brma_mask_loss
from .mask_generator import generate_brma_masks


@contextmanager
def temporary_freeze_module(module):
    """Temporarily set all module parameters to requires_grad=False."""

    params = list(module.parameters())
    old_flags = [p.requires_grad for p in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield module
    finally:
        for param, old_flag in zip(params, old_flags):
            param.requires_grad_(old_flag)


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def _validate_batch_shapes(
    entities: torch.Tensor,
    entity_mask: torch.Tensor,
    rnn_hidden: torch.Tensor,
    actions: torch.Tensor,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
) -> None:
    if entities.ndim != 3:
        raise ValueError("entities must have shape (B, N, D)")
    if entity_mask.shape != entities.shape[:2]:
        raise ValueError("entity_mask must have shape (B, N)")
    if rnn_hidden.ndim != 2 or rnn_hidden.shape[0] != entities.shape[0]:
        raise ValueError("rnn_hidden must have shape (B, H)")
    if actions.ndim != 2 or actions.shape[0] != entities.shape[0]:
        raise ValueError("actions must have shape (B, A)")
    if entities.shape[1] != n_ego + n_allies + n_enemies:
        raise ValueError("entity count does not match n_ego+n_allies+n_enemies")


def compute_brma_mask_generator_loss_batch(
    *,
    actor,
    mask_generator,
    entities: torch.Tensor,
    entity_mask: torch.Tensor,
    rnn_hidden: torch.Tensor,
    actions: torch.Tensor,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
    loss_config: BRMALossConfig,
    mR_count: torch.Tensor | None = None,
    mB_count: torch.Tensor | None = None,
    torch_generator=None,
) -> dict:
    """Compute standalone BRMA mask-generator loss for one batch.

    Actor parameters are frozen while the forward graph remains differentiable
    with respect to the selected soft keep mask and mask-generator outputs.
    This function does not call backward or step an optimizer.
    """

    _validate_batch_shapes(
        entities, entity_mask, rnn_hidden, actions, n_ego, n_allies, n_enemies)
    brma_out = generate_brma_masks(
        mask_generator,
        entities,
        entity_mask,
        n_ego,
        n_allies,
        n_enemies,
        mR_count=mR_count,
        mB_count=mB_count,
        torch_generator=torch_generator,
    )
    soft_keep_mask = build_selected_soft_keep_mask(
        msoft=brma_out["msoft"],
        entity_mask=entity_mask,
        friendly_drop_mask=brma_out["friendly_drop_mask"],
        enemy_drop_mask=brma_out["enemy_drop_mask"],
        n_ego=n_ego,
    )
    valid = ~entity_mask.bool()
    selected_mask = (
        (brma_out["friendly_drop_mask"].bool() | brma_out["enemy_drop_mask"].bool())
        & valid
    )
    if n_ego > 0:
        selected_mask[:, :n_ego] = False

    with temporary_freeze_module(actor):
        unmasked = actor.evaluate_actions(
            entities,
            entity_mask,
            rnn_hidden,
            actions,
            soft_keep_mask=None,
        )
        masked = actor.evaluate_actions(
            entities,
            entity_mask,
            rnn_hidden,
            actions,
            soft_keep_mask=soft_keep_mask,
        )

    loss_out = compute_brma_mask_loss(
        log_prob_unmasked=unmasked["log_prob"],
        log_prob_masked=masked["log_prob"],
        mu_unmasked=unmasked["mu"],
        sigma_unmasked=unmasked["sigma"],
        mu_masked=masked["mu"],
        sigma_masked=masked["sigma"],
        msoft=brma_out["msoft"],
        maskable_set=selected_mask,
        config=loss_config,
    )
    return {
        **loss_out,
        "kl": loss_out["kl_per_batch_mean"],
        "brma_out": brma_out,
        "soft_keep_mask": soft_keep_mask,
        "selected_mask": selected_mask,
        "mu_unmasked": unmasked["mu"],
        "mu_masked": masked["mu"],
        "sigma_unmasked": unmasked["sigma"],
        "sigma_masked": masked["sigma"],
    }


def brma_mask_generator_train_step(
    *,
    actor,
    mask_generator,
    optimizer,
    entities: torch.Tensor,
    entity_mask: torch.Tensor,
    rnn_hidden: torch.Tensor,
    actions: torch.Tensor,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
    loss_config: BRMALossConfig,
    mR_count: torch.Tensor | None = None,
    mB_count: torch.Tensor | None = None,
    torch_generator=None,
    max_grad_norm: float | None = 0.5,
) -> dict:
    """Run one standalone optimizer step for the BRMA mask generator only."""

    before = [p.detach().clone() for p in mask_generator.parameters()]
    optimizer.zero_grad()
    out = compute_brma_mask_generator_loss_batch(
        actor=actor,
        mask_generator=mask_generator,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_hidden,
        actions=actions,
        n_ego=n_ego,
        n_allies=n_allies,
        n_enemies=n_enemies,
        loss_config=loss_config,
        mR_count=mR_count,
        mB_count=mB_count,
        torch_generator=torch_generator,
    )
    out["loss"].backward()

    for param in mask_generator.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError("non-finite mask generator gradient")

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(mask_generator.parameters(), max_grad_norm)
    mask_grad_norm = _grad_norm(mask_generator.parameters())
    actor_grad_norm = _grad_norm(actor.parameters())
    optimizer.step()
    params_changed = any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, mask_generator.parameters())
    )

    return {
        "loss": float(out["loss"].detach().item()),
        "kl": float(out["kl"].detach().item()),
        "entropy": float(out["entropy"].detach().item()),
        "mask_generator_grad_norm": mask_grad_norm,
        "actor_grad_norm_after": actor_grad_norm,
        "params_changed": params_changed,
        "formula_status": out["formula_status"],
    }
