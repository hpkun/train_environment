"""
my_uav_env: Pure physics-simulation multi-agent UAV combat environment.

Zero torch/nn dependencies, Dict observation spaces for zero-shot scale
generalization, PID-based hierarchical control.
"""

__all__ = ["UavCombatEnv"]


def __getattr__(name):
    if name == "UavCombatEnv":
        from .env import UavCombatEnv
        return UavCombatEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__ + list(globals().keys()))
