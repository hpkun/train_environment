"""Test adapter_utils functions."""
from __future__ import annotations

from algorithms.mappo.adapter_utils import (
    make_obs_adapter,
    resolve_obs_adapter_version,
    validate_model_dims,
)


def test_make_adapter_v1():
    a = make_obs_adapter("v1")
    from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
    assert isinstance(a, HeteroObsAdapter)


def test_make_adapter_v2():
    a = make_obs_adapter("v2")
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    assert isinstance(a, HeteroObsAdapterV2)


def test_resolve_none_from_meta():
    assert resolve_obs_adapter_version(None, {"obs_adapter_version": "v2"}) == "v2"


def test_resolve_cli_overrides_meta():
    assert resolve_obs_adapter_version("v1", {"obs_adapter_version": "v2"}) == "v1"


def test_resolve_default():
    assert resolve_obs_adapter_version(None, {}) == "v1"


def test_validate_mismatch_raises():
    a = make_obs_adapter("v2")
    try:
        validate_model_dims(a, {"actor_obs_dim": 999})
        assert False, "should have raised"
    except ValueError:
        pass
