"""Factory helpers for environment construction."""

from __future__ import annotations

from pathlib import Path

from .JSBSim.envs.hetero_uav_env import HeteroUAVEnv
from .JSBSim.core.utils import load_yaml


def make_env(config_path: str | None = None, **kwargs):
    """Create a hetero or copied BRMA-compatible environment."""

    config = load_yaml(config_path) if config_path else {}
    config.update(kwargs)
    env_type = str(config.pop("env_type", "hetero"))
    if env_type == "brma_original":
        from .brma_env.make_brma_env import make_brma_env

        brma_kwargs = {k: v for k, v in config.items() if k != "config_path"}
        return make_brma_env(**brma_kwargs)
    if env_type != "hetero":
        raise ValueError(f"Unknown env_type: {env_type}")
    if config_path is not None:
        config_path = str(Path(config_path))
    return HeteroUAVEnv(config=config, config_path=config_path)
