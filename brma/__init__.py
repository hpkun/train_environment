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
from .rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage

__all__ = [
    "BRMAMaskGenerator",
    "BRMAMaskGeneratorConfig",
    "BRMARolloutSchemaConfig",
    "BRMARolloutStorage",
    "MaskGeneratorConfig",
    "MaskVectorGenerator",
    "collect_brma_dry_run_step",
]
