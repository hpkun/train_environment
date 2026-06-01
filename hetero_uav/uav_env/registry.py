"""Minimal environment registry."""

from __future__ import annotations

from .make_env import make_env

_REGISTRY = {
    "HeteroUAVEnv": make_env,
    "hetero_uav": make_env,
}


def make(name: str, config_path: str, **kwargs):
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown environment {name!r}; known: {known}")
    return _REGISTRY[name](config_path=config_path, **kwargs)
