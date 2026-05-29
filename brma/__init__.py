"""BRMA preparation utilities.

This package contains pure infrastructure only.  It is not wired into
the training loop or attention actor yet.
"""

from .mask_generator import (
    BRMAMaskGenerator,
    BRMAMaskGeneratorConfig,
    MaskGeneratorConfig,
    MaskVectorGenerator,
)
from .collection import collect_brma_dry_run_step
from .losses import (
    BRMALossConfig,
    compute_brma_mask_loss,
    compute_maskable_set,
    diagonal_gaussian_kl,
    masked_entropy_loss,
)
from .rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage
from .train_step import (
    brma_mask_generator_train_step,
    compute_brma_mask_generator_loss_batch,
    temporary_freeze_module,
)

__all__ = [
    "BRMAMaskGenerator",
    "BRMAMaskGeneratorConfig",
    "BRMALossConfig",
    "BRMARolloutSchemaConfig",
    "BRMARolloutStorage",
    "MaskGeneratorConfig",
    "MaskVectorGenerator",
    "collect_brma_dry_run_step",
    "compute_brma_mask_loss",
    "compute_maskable_set",
    "diagonal_gaussian_kl",
    "masked_entropy_loss",
    "brma_mask_generator_train_step",
    "compute_brma_mask_generator_loss_batch",
    "temporary_freeze_module",
]
