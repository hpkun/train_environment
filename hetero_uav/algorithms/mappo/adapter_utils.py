"""Shared adapter/model helpers for v1/v2 MAPPO baseline scripts."""
from __future__ import annotations

import json
from pathlib import Path

from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from .policy import MAPPOActorCritic, RoleConditionedMAPPOActorCritic


def load_model_meta(model_path: str | Path) -> dict:
    """Load meta.json from the model directory. Returns {} if missing."""
    meta_path = Path(model_path).parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def make_obs_adapter(version: str):
    """Return HeteroObsAdapter(v1) or HeteroObsAdapterV2(v2)."""
    if version == "v1":
        return HeteroObsAdapter()
    if version == "v2":
        return HeteroObsAdapterV2()
    raise ValueError(f"Unknown obs_adapter_version: {version}")


def resolve_obs_adapter_version(cli_version: str | None, meta: dict) -> str:
    """Resolve adapter version: CLI takes priority, then meta, then v1."""
    if cli_version is not None:
        return cli_version
    return meta.get("obs_adapter_version", "v1")


def validate_model_dims(adapter, meta: dict) -> None:
    """Raise if meta actor_obs_dim/critic_state_dim don't match adapter."""
    expected_actor = meta.get("actor_obs_dim")
    expected_critic = meta.get("critic_state_dim")
    if expected_actor is not None and expected_actor != adapter.flat_actor_obs_dim:
        raise ValueError(
            f"actor_obs_dim mismatch: meta={expected_actor} "
            f"adapter={adapter.flat_actor_obs_dim}")
    if expected_critic is not None and expected_critic != adapter.critic_state_dim:
        raise ValueError(
            f"critic_state_dim mismatch: meta={expected_critic} "
            f"adapter={adapter.critic_state_dim}")


def make_mappo_model_for_adapter(adapter, device, actor_arch: str = "mlp"):
    """Create model with dimensions from the adapter.

    actor_arch:
      "mlp"              → MAPPOActorCritic (baseline)
      "role_conditioned" → RoleConditionedMAPPOActorCritic
    """
    kwargs = dict(
        actor_obs_dim=adapter.flat_actor_obs_dim,
        critic_state_dim=adapter.critic_state_dim,
    )
    if actor_arch == "role_conditioned":
        return RoleConditionedMAPPOActorCritic(**kwargs).to(device)
    return MAPPOActorCritic(**kwargs).to(device)
