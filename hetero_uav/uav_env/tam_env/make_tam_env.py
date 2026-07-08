"""Factory for TAM-HAPPO direct-FCS diagnostic environments."""

from __future__ import annotations

from .env import TamCombatEnv


def make_tam_env(**kwargs) -> TamCombatEnv:
    return TamCombatEnv(**kwargs)


__all__ = ["make_tam_env", "TamCombatEnv"]
