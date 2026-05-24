"""
my_uav_env: Pure physics-simulation multi-agent UAV combat environment.

Zero torch/nn dependencies, Dict observation spaces for zero-shot scale
generalization, PID-based hierarchical control.
"""
from .env import UavCombatEnv

__all__ = ["UavCombatEnv"]
