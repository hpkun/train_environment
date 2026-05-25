"""Pure smoke test for strict team global state utilities.

No env, no JSBSim.  Uses fake strict team observations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.global_state import (
    build_strict_team_global_state,
    describe_strict_global_state_layout,
    flatten_strict_team_observations,
    infer_strict_team_global_state_dim,
)


def _make_fake_team_obs(num_agents: int, n_entities: int,
                        entity_dim: int = 10) -> dict:
    """Return a fake team_obs dict."""
    return {
        f"red_{i}": (
            np.arange(i * 100 + 1, i * 100 + 1 + n_entities * entity_dim,
                      dtype=np.float32).reshape(n_entities, entity_dim),
            np.ones(n_entities, dtype=np.int64),
            {"schema": "test"},
        )
        for i in range(num_agents)
    }


def main() -> None:
    num_red, num_blue = 2, 2
    n_entities = num_red + num_blue  # 4

    # ---- infer dims ----
    dim_with = infer_strict_team_global_state_dim(num_red, num_blue,
                                                  entity_dim=10, include_masks=True)
    assert dim_with == 88, f"expected 88, got {dim_with}"
    dim_without = infer_strict_team_global_state_dim(num_red, num_blue,
                                                     entity_dim=10, include_masks=False)
    assert dim_without == 80, f"expected 80, got {dim_without}"

    # ---- build global state ----
    team_obs = _make_fake_team_obs(num_red, n_entities)
    gs = build_strict_team_global_state(team_obs, num_red, num_blue)
    assert gs.shape == (88,), f"expected (88,), got {gs.shape}"
    assert gs.dtype == np.float32

    gs_nomask = build_strict_team_global_state(team_obs, num_red, num_blue,
                                               include_masks=False)
    assert gs_nomask.shape == (80,)

    # ---- flatten with explicit agent_ids ordering ----
    flat_default = flatten_strict_team_observations(team_obs, include_masks=True)
    flat_reversed = flatten_strict_team_observations(
        team_obs, agent_ids=["red_1", "red_0"], include_masks=True)
    assert flat_default.shape == flat_reversed.shape
    # The order should differ; check first entity block value
    assert not np.allclose(flat_default[:40], flat_reversed[:40]), \
        "reversed ordering should produce different concatenation"

    # ---- missing agent raises KeyError ----
    partial = {"red_0": team_obs["red_0"]}
    try:
        flatten_strict_team_observations(partial, agent_ids=["red_0", "red_1"])
        assert False, "should have raised KeyError"
    except KeyError:
        pass

    # ---- describe layout ----
    layout = describe_strict_global_state_layout(num_red, num_blue)
    assert layout["global_state_dim"] == 88
    assert layout["n_entities_per_red_agent"] == 4
    assert layout["per_agent_dim"] == 44

    print("Global state layout:")
    for k, v in layout.items():
        print(f"  {k}: {v}")

    print("global state utils smoke test passed")


if __name__ == "__main__":
    main()
