"""Pure numpy smoke test for the paper observation adapter."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from my_uav_env.alignment.obs_adapter import (
    build_paper_entity_observation_from_env_obs,
    compare_current_and_paper_adapter_shapes,
    infer_paper_entity_layout,
)


def main():
    obs = {
        "ego_state": np.ones(11, dtype=np.float32),
        "ally_states": np.zeros((1, 11), dtype=np.float32),
        "enemy_states": np.ones((2, 11), dtype=np.float32),
        "death_mask": np.ones(4, dtype=np.int64),
        "missile_warning": np.zeros(1, dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([300.0], dtype=np.float32),
    }

    entities, mask = build_paper_entity_observation_from_env_obs(obs)
    assert entities.shape == (4, 10)
    assert mask.shape == (4,)
    assert mask.tolist() == [0, 1, 0, 0]

    layout = infer_paper_entity_layout(obs)
    assert layout["entity_dim"] == 10
    assert layout["n_entities"] == 4
    assert layout["adapter"] == "placeholder_11_to_10"

    shapes = compare_current_and_paper_adapter_shapes(obs)
    assert shapes["current_entities_shape"] == (4, 11)
    assert shapes["paper_entities_shape"] == (4, 10)
    assert shapes["current_mask_shape"] == (4,)
    assert shapes["paper_mask_shape"] == (4,)

    print("paper obs adapter smoke test passed")


if __name__ == "__main__":
    main()
