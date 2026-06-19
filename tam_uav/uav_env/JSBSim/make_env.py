"""Factories for BRMA-style JSBSim environments."""

from __future__ import annotations

from .envs.hetero_uav_combat_env import HeteroUavCombatEnv
from .envs.uav_combat_env import UavCombatEnv


def make_jsbsim_brma_env(**kwargs) -> UavCombatEnv:
    return UavCombatEnv(**kwargs)


def make_jsbsim_hetero_env(**kwargs) -> HeteroUavCombatEnv:
    return HeteroUavCombatEnv(**kwargs)


__all__ = ["make_jsbsim_brma_env", "make_jsbsim_hetero_env", "UavCombatEnv", "HeteroUavCombatEnv"]
