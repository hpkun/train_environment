"""BRMA-style JSBSim UAV combat environments.

This package is the formal environment implementation path for hetero_uav.
The code was ported from the package-local BRMA backup and is self-contained.
"""

from .env import UavCombatEnv

__all__ = ["UavCombatEnv"]
