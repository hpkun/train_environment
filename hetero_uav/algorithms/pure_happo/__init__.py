"""Pure HAPPO baseline: paper-aligned ICLR 2022 HAPPO implementation."""

from .policy import PureHAPPOPolicy, PureHAPPOTanhPolicy
from .trainer import PureHAPPOTrainer, _compute_grouped_gae

__all__ = [
    "PureHAPPOPolicy",
    "PureHAPPOTanhPolicy",
    "PureHAPPOTrainer",
    "_compute_grouped_gae",
]
