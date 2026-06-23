"""Factory helpers for environment construction."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_env(config_path: str | None = None, **kwargs):
    """Create a hetero or copied BRMA-compatible environment."""

    config = _load_yaml(config_path)
    config.update(kwargs)
    env_type = str(config.pop("env_type", "hetero"))
    # Pop visual-only metadata fields that env constructors don't accept
    config.pop("acmi_visual_by_role", None)
    if env_type == "jsbsim_brma":
        from .JSBSim.envs.uav_combat_env import UavCombatEnv

        return UavCombatEnv(**config)
    if env_type == "jsbsim_hetero":
        from .JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv

        return HeteroUavCombatEnv(**config)
    if env_type == "brma_original":
        from .brma_env.make_brma_env import make_brma_env

        brma_kwargs = {k: v for k, v in config.items() if k != "config_path"}
        return make_brma_env(**brma_kwargs)
    if env_type != "hetero":
        raise ValueError(f"Unknown env_type: {env_type}")
    from .JSBSim.envs.hetero_uav_env import HeteroUAVEnv

    if config_path is not None:
        config_path = str(Path(config_path))
    return HeteroUAVEnv(config=config, config_path=config_path)
