"""Opt-in BRMA recurrent policy with random and biased entity masks.

This module adds the P4/P5 paper-aligned mask path on top of the existing
BRMA recurrent actor.  It intentionally does not add mask loss, reward changes,
missile changes, or strict HAPPO correction.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .brma_recurrent_policy import BRMARecurrentHAPPOReferencePolicy


def _stats_from_keep_mask(base_keep: torch.Tensor, masked_keep: torch.Tensor,
                          entropy: torch.Tensor | None = None) -> dict[str, float]:
    valid = base_keep.bool()
    kept = masked_keep.bool() & valid
    valid_count = valid.sum().clamp(min=1).to(dtype=torch.float32)
    masked_count = (valid & ~kept).sum().to(dtype=torch.float32)
    keep_ratio = kept.sum().to(dtype=torch.float32) / valid_count
    out = {
        "mask_keep_ratio": float(keep_ratio.detach().item()),
        "masked_entity_count": float(masked_count.detach().item()),
        "mask_entropy": 0.0,
    }
    if entropy is not None:
        out["mask_entropy"] = float(entropy.detach().mean().item())
    return out


def apply_random_scale_mask(
    keep_mask: torch.Tensor,
    drop_prob: float = 0.25,
    training: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Randomly drop valid non-self entities while preserving self/padding masks."""

    keep = keep_mask.bool()
    if not training or drop_prob <= 0.0:
        return keep.clone(), _stats_from_keep_mask(keep, keep)
    candidate = keep.clone()
    candidate[:, 0] = False
    draw = torch.rand(keep.shape, device=keep.device)
    drop = (draw < float(drop_prob)) & candidate
    masked = keep & ~drop
    masked[:, 0] = keep[:, 0]
    masked[~keep] = False
    return masked, _stats_from_keep_mask(keep, masked)


class BRMABiasedMaskGenerator(nn.Module):
    """Learned keep-probability generator adapted from the parent BRMA module."""

    def __init__(self, entity_dim: int = 30, hidden_dim: int = 128,
                 temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.mlp = nn.Sequential(
            nn.Linear(int(entity_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, entities: torch.Tensor, keep_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.mlp(entities).squeeze(-1)
        keep_prob = torch.sigmoid(logits)
        valid = keep_mask.bool()
        entropy = F.binary_cross_entropy(
            keep_prob.clamp(1e-6, 1.0 - 1e-6),
            keep_prob.detach().clamp(1e-6, 1.0 - 1e-6),
            reduction="none",
        )
        entropy = torch.where(valid, entropy, torch.zeros_like(entropy)).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
        return {"logits": logits, "keep_prob": keep_prob, "valid_mask": valid, "entropy": entropy}


def _type_slices(max_allies: int, max_enemies: int) -> tuple[slice, slice]:
    ally_slice = slice(1, 1 + int(max_allies))
    enemy_slice = slice(1 + int(max_allies), 1 + int(max_allies) + int(max_enemies))
    return ally_slice, enemy_slice


def apply_biased_mask(
    generator: BRMABiasedMaskGenerator,
    entities: torch.Tensor,
    keep_mask: torch.Tensor,
    max_mask_allies: int = 2,
    max_mask_enemies: int = 2,
    training: bool = True,
    max_allies: int = 4,
    max_enemies: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Drop low keep-probability allies/enemies with Top-M style selection."""

    keep = keep_mask.bool()
    out = generator(entities, keep)
    if not training:
        return keep.clone(), _stats_from_keep_mask(keep, keep, out["entropy"])

    keep_prob = out["keep_prob"]
    masked = keep.clone()
    ally_slice, enemy_slice = _type_slices(max_allies, max_enemies)
    for row in range(keep.shape[0]):
        for span, max_drop in ((ally_slice, max_mask_allies), (enemy_slice, max_mask_enemies)):
            idx = torch.arange(keep.shape[1], device=keep.device)[span]
            valid_idx = idx[keep[row, idx]]
            if int(max_drop) <= 0 or valid_idx.numel() == 0:
                continue
            k = min(int(max_drop), int(valid_idx.numel()))
            _, selected = torch.topk(-keep_prob[row, valid_idx], k=k)
            masked[row, valid_idx[selected]] = False
    masked[:, 0] = keep[:, 0]
    masked[~keep] = False
    return masked, _stats_from_keep_mask(keep, masked, out["entropy"])


class BRMARecurrentMaskedHAPPOReferencePolicy(BRMARecurrentHAPPOReferencePolicy):
    """BRMA entity encoder + GRU actor with opt-in random/biased masks."""

    def __init__(
        self,
        entity_dim: int = 30,
        critic_state_dim: int = 480,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        rnn_hidden_size: int = 128,
        max_allies: int = 4,
        max_enemies: int = 4,
        random_scale_mask: bool = False,
        random_mask_prob: float = 0.25,
        biased_mask: bool = False,
        max_mask_allies: int = 2,
        max_mask_enemies: int = 2,
    ):
        super().__init__(
            entity_dim=entity_dim,
            critic_state_dim=critic_state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            rnn_hidden_size=rnn_hidden_size,
            max_allies=max_allies,
            max_enemies=max_enemies,
        )
        self.random_scale_mask = bool(random_scale_mask)
        self.random_mask_prob = float(random_mask_prob)
        self.biased_mask = bool(biased_mask)
        self.max_mask_allies = int(max_mask_allies)
        self.max_mask_enemies = int(max_mask_enemies)
        self.mask_generator = BRMABiasedMaskGenerator(
            entity_dim=self.entity_dim,
            hidden_dim=self.hidden_dim,
        )
        self.last_mask_stats = {
            "mask_keep_ratio": 1.0,
            "mask_entropy": 0.0,
            "masked_entity_count": 0.0,
        }

    def actor_shared_parameters(self):
        params = list(self.encoder.parameters()) + list(self.rnn.parameters())
        if self.biased_mask:
            params += list(self.mask_generator.parameters())
        return params

    def _apply_masks(self, entities: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        stats = {"mask_keep_ratio": 1.0, "mask_entropy": 0.0, "masked_entity_count": 0.0}
        original_keep = keep_mask.bool()
        masked = original_keep.clone()
        biased_entropy = None
        if self.random_scale_mask:
            masked, stats = apply_random_scale_mask(
                masked,
                drop_prob=self.random_mask_prob,
                training=self.training,
            )
        if self.biased_mask:
            masked, biased_stats = apply_biased_mask(
                self.mask_generator,
                entities,
                masked,
                max_mask_allies=self.max_mask_allies,
                max_mask_enemies=self.max_mask_enemies,
                training=self.training,
                max_allies=self.max_allies,
                max_enemies=self.max_enemies,
            )
            biased_entropy = torch.as_tensor(biased_stats.get("mask_entropy", 0.0), device=entities.device)
            stats = biased_stats
        # Report final keep/drop statistics against the original alive/padding mask.
        # When random and biased masks are combined, this keeps the log fields from
        # under-reporting entities dropped by the first masking stage.
        stats = _stats_from_keep_mask(original_keep, masked, biased_entropy)
        self.last_mask_stats = stats
        return masked

    def encode(self, actor_obs) -> tuple[torch.Tensor, tuple[int, ...]]:
        raw_t = torch.as_tensor(actor_obs, dtype=torch.float32, device=next(self.parameters()).device)
        if raw_t.shape[-1] == self.flat_actor_obs_dim:
            leading_shape = tuple(raw_t.shape[:-1])
            entities_t, keep_mask = self._flat_to_entities(raw_t)
        elif raw_t.ndim >= 3 and raw_t.shape[-1] == self.entity_dim:
            leading_shape = tuple(raw_t.shape[:-2])
            entities_t = raw_t.reshape(-1, raw_t.shape[-2], raw_t.shape[-1])
            keep_mask = torch.ones(entities_t.shape[:2], dtype=torch.bool, device=entities_t.device)
        else:
            raise ValueError(
                f"expected flat actor obs dim {self.flat_actor_obs_dim} or entity dim {self.entity_dim}, "
                f"got shape {tuple(raw_t.shape)}"
            )
        keep_mask = self._apply_masks(entities_t, keep_mask)
        pooled, _attn = self.encoder(entities_t, keep_mask)
        return pooled, leading_shape

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
