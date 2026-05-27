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
from .rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage

__all__ = [
    "BRMAMaskGenerator",
    "BRMAMaskGeneratorConfig",
    "BRMARolloutSchemaConfig",
    "BRMARolloutStorage",
    "MaskGeneratorConfig",
    "MaskVectorGenerator",
]
