"""Factory for the copied BRMA-compatible UAV combat environment."""

from __future__ import annotations

from .env import UavCombatEnv


def make_brma_env(**kwargs) -> UavCombatEnv:
    """Create the original BRMA-style UavCombatEnv from package-local code."""

    return UavCombatEnv(**kwargs)


__all__ = ["make_brma_env", "UavCombatEnv"]
