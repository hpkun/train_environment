"""BRMA offline dry-run collection — validates the full mask → dual-eval → store
pipeline without touching training or environment.

``collect_brma_dry_run_step`` wires ``BRMAMaskGenerator``,
``AttentionActor.evaluate_dual_actions``, and ``BRMARolloutStorage``
together for a single offline timestep.
"""
from __future__ import annotations

import numpy as np
import torch

from .mask_generator import generate_brma_masks
from .rollout_schema import BRMARolloutStorage


def _to_torch(x, dtype, device):
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)


def _normalize_entities(t: torch.Tensor) -> torch.Tensor:
    """Accept (N,D) or (1,N,D); reject (B,N,D) with B != 1."""
    if t.dim() == 2:
        return t.unsqueeze(0)
    if t.dim() == 3 and t.shape[0] != 1:
        raise ValueError(
            f"entities batch size must be 1, got shape {t.shape}")
    if t.dim() != 3:
        raise ValueError(
            f"entities must be (N,D) or (1,N,D), got shape {t.shape}")
    return t


def _normalize_vector_or_batch(t: torch.Tensor, name: str) -> torch.Tensor:
    """Accept (D,) or (1,D); reject (B,D) with B != 1."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2 and t.shape[0] != 1:
        raise ValueError(
            f"{name} batch size must be 1, got shape {t.shape}")
    if t.dim() != 2:
        raise ValueError(
            f"{name} must be (D,) or (1,D), got shape {t.shape}")
    return t


def collect_brma_dry_run_step(
    *,
    actor,
    mask_generator,
    storage: BRMARolloutStorage,
    step: int,
    env_idx: int,
    agent_idx: int,
    entities,
    entity_mask,
    rnn_hidden,
    action,
    n_ego: int,
    n_allies: int,
    n_enemies: int,
    next_entities=None,
    next_entity_mask=None,
    mR_count=None,
    mB_count=None,
    torch_generator=None,
    device=None,
    use_soft_mask_path: bool = True,
) -> dict:
    """Dry-run one BRMA collection step (offline, no env, no training).

    Args:
        entities:      (N, D) or (1, N, D)
        entity_mask:   (N,) or (1, N), 0=valid, 1=invalid
        rnn_hidden:    (H,) or (1, H)
        action:        (A,) or (1, A)
        n_ego/n_allies/n_enemies: entity layout counts.

    Returns a summary dict with per-agent diagnostics.
    """
    if not storage.has_storage:
        raise RuntimeError(
            "BRMARolloutStorage is disabled; cannot collect BRMA fields")

    N = n_ego + n_allies + n_enemies
    if device is None:
        device = next(actor.parameters()).device

    # ---- shape validation and batch-unsqueeze ----
    ents = _to_torch(entities, torch.float32, device)
    emask = _to_torch(entity_mask, torch.long, device)
    rnn_h = _to_torch(rnn_hidden, torch.float32, device)
    act = _to_torch(action, torch.float32, device)

    ents = _normalize_entities(ents)
    emask = _normalize_vector_or_batch(emask, "entity_mask")
    rnn_h = _normalize_vector_or_batch(rnn_h, "rnn_hidden")
    act = _normalize_vector_or_batch(act, "action")

    if ents.shape[1] != N:
        raise ValueError(
            f"entities N={ents.shape[1]} but n_ego+n_allies+n_enemies={N}")
    if emask.shape[1] != N:
        raise ValueError(
            f"entity_mask N={emask.shape[1]} but expected {N}")

    if mR_count is not None:
        mR_count = _to_torch(mR_count, torch.int64, device)
        mR_count = mR_count.view(-1)
        if mR_count.shape[0] != 1:
            mR_count = mR_count[:1]
    if mB_count is not None:
        mB_count = _to_torch(mB_count, torch.int64, device)
        mB_count = mB_count.view(-1)
        if mB_count.shape[0] != 1:
            mB_count = mB_count[:1]

    # ---- generate BRMA masks ----
    brma_out = generate_brma_masks(
        mask_generator, ents, emask, n_ego, n_allies, n_enemies,
        mR_count=mR_count, mB_count=mB_count,
        torch_generator=torch_generator,
    )

    # ---- dual actor evaluation ----
    soft_keep_mask = None
    hard_masked_entity_mask = brma_out["key_padding_mask"]
    if use_soft_mask_path:
        soft_keep_mask = torch.ones_like(brma_out["msoft"])
        valid = ~emask.bool()
        maskable = valid.clone()
        maskable[:, :n_ego] = False
        soft_keep_mask = torch.where(maskable, brma_out["msoft"], soft_keep_mask)
        soft_keep_mask[:, :n_ego] = 1.0
        hard_masked_entity_mask = emask

    dual = actor.evaluate_dual_actions(
        ents,
        unmasked_entity_mask=emask,
        masked_entity_mask=hard_masked_entity_mask,
        rnn_hidden=rnn_h,
        actions=act,
        masked_soft_keep_mask=soft_keep_mask,
    )

    # ---- store into rollout storage ----
    def _to_np(t, dtype_=np.float32, squeeze=True):
        arr = t.detach().cpu().numpy().astype(dtype_)
        if squeeze and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    store_kwargs = dict(
        p=_to_np(brma_out["p"]),
        msoft=_to_np(brma_out["msoft"]),
        mhard=_to_np(brma_out["mhard"]),
        mR_count=int(brma_out["mR_count"].item()),
        mB_count=int(brma_out["mB_count"].item()),
        friendly_drop_mask=_to_np(brma_out["friendly_drop_mask"], bool),
        enemy_drop_mask=_to_np(brma_out["enemy_drop_mask"], bool),
        key_padding_mask=_to_np(brma_out["key_padding_mask"], bool),
        keep_mask=_to_np(brma_out["keep_mask"], bool),
        log_prob_unmasked=float(dual["log_prob_unmasked"].item()),
        log_prob_masked=float(dual["log_prob_masked"].item()),
        entropy_unmasked=float(dual["entropy_unmasked_mean"].item()),
        entropy_masked=float(dual["entropy_masked_mean"].item()),
        mu_unmasked=_to_np(dual["mu_unmasked"]),
        mu_masked=_to_np(dual["mu_masked"]),
        sigma_unmasked=_to_np(dual["sigma_unmasked"]),
        sigma_masked=_to_np(dual["sigma_masked"]),
    )
    if next_entities is not None:
        ne = np.asarray(next_entities, dtype=np.float32)
        if ne.ndim == 3:
            ne = ne[0]
        if ne.shape != (N, storage._cfg.entity_dim):
            raise ValueError(
                f"next_entities shape {ne.shape} != ({N},{storage._cfg.entity_dim})")
        store_kwargs["next_entities"] = ne
    if next_entity_mask is not None:
        nm = np.asarray(next_entity_mask, dtype=np.int64)
        if nm.ndim == 2:
            nm = nm[0]
        if nm.shape != (N,):
            raise ValueError(
                f"next_entity_mask shape {nm.shape} != ({N},)")
        store_kwargs["next_entity_masks"] = nm

    storage.store_step(step, env_idx, agent_idx, **store_kwargs)

    return {
        "log_prob_unmasked": store_kwargs["log_prob_unmasked"],
        "log_prob_masked": store_kwargs["log_prob_masked"],
        "entropy_unmasked_mean": store_kwargs["entropy_unmasked"],
        "entropy_masked_mean": store_kwargs["entropy_masked"],
        "mR_count": store_kwargs["mR_count"],
        "mB_count": store_kwargs["mB_count"],
        "enemy_drop_count": int(store_kwargs["enemy_drop_mask"].sum()),
        "friendly_drop_count": int(store_kwargs["friendly_drop_mask"].sum()),
        "key_padding_count": int(store_kwargs["key_padding_mask"].sum()),
        "use_soft_mask_path": bool(use_soft_mask_path),
        "soft_keep_mean": (
            float(soft_keep_mask.detach().mean().item())
            if soft_keep_mask is not None else 0.0
        ),
        "hard_key_padding_count": int(store_kwargs["key_padding_mask"].sum()),
        "storage_summary": storage.summary(),
    }
