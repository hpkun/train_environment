"""Pure HAPPO baseline: paper-aligned ICLR 2022 HAPPO implementation."""

from .policy import PureHAPPOPolicy
from .trainer import PureHAPPOTrainer, _compute_grouped_gae

__all__ = [
    "PureHAPPOPolicy",
    "PureHAPPOTrainer",
    "_compute_grouped_gae",
]
