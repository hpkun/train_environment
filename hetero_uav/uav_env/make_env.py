"""Factory helpers for environment construction."""

from __future__ import annotations

from .JSBSim.envs.hetero_uav_env import HeteroUAVEnv
from .JSBSim.core.utils import load_yaml


def make_env(config_path: str, **kwargs) -> HeteroUAVEnv:
    """Create a HeteroUAVEnv from a YAML config path."""

    config = load_yaml(config_path)
    config.update(kwargs)
    return HeteroUAVEnv(config=config, config_path=config_path)
