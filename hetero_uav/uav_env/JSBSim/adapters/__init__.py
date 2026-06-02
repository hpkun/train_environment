"""Observation adapters for heterogeneous UAV/MAV environments."""

from .hetero_obs_adapter import HeteroObsAdapter
from .hetero_obs_adapter_v2 import HeteroObsAdapterV2

__all__ = ["HeteroObsAdapter", "HeteroObsAdapterV2"]
