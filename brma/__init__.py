"""BRMA preparation utilities.

This package currently contains pure infrastructure only.  It is not wired into
the training loop or attention actor yet.
"""

from .mask_generator import MaskGeneratorConfig, MaskVectorGenerator

__all__ = ["MaskGeneratorConfig", "MaskVectorGenerator"]
