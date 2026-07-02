"""Pure HAPPO baseline: paper-aligned ICLR 2022 HAPPO implementation."""

from .policy import LegacyClampPureHAPPOPolicy, PureHAPPOPolicy
from .trainer import PureHAPPOTrainer, _compute_grouped_gae

# Backward-compat alias: old checkpoints may reference PureHAPPOTanhPolicy.
PureHAPPOTanhPolicy = PureHAPPOPolicy

__all__ = [
    "PureHAPPOPolicy",
    "PureHAPPOTrainer",
    "_compute_grouped_gae",
    "LegacyClampPureHAPPOPolicy",
]
