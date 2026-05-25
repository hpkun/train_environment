"""Mask-vector utilities for future BRMA-MAPPO work.

The paper motivates biased random entity masking for zero-shot scale
generalization, but the exact bias formula was not reliably extracted from the
local PDF in this pass.  This module therefore implements only deterministic
type-aware masking and uniform random masking infrastructure.  Paper-specific
bias rules should be added only after the original formula is verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
