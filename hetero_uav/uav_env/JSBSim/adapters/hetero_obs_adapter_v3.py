"""Compatibility alias for the canonical MAV-shared full-geometry adapter.

New code should use :class:`HeteroObsAdapterV2`.  This class is kept only so
older diagnostic imports do not fail while the config-level ``mav_shared_geo_v2``
mode is retired.
"""
from __future__ import annotations

from .hetero_obs_adapter_v2 import HeteroObsAdapterV2


class HeteroObsAdapterV3(HeteroObsAdapterV2):
    """Deprecated alias of the canonical full-geometry V2 adapter."""

    schema_version = "hetero_obs_adapter_canonical_mav_shared_full_geo"
