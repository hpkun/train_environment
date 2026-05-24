"""Pure shape smoke test for train_attention_mappo observation adapters."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_attention_mappo import _build_attention_entities


def main():
    obs = {
        "ego_state": np.ones(11, dtype=np.float32),
        "ally_states": np.zeros((1, 11), dtype=np.float32),
        "enemy_states": np.ones((2, 11), dtype=np.float32),
    }

    current_entities, current_mask = _build_attention_entities(obs, "current")
    paper_entities, paper_mask = _build_attention_entities(obs, "paper-placeholder")

    assert current_entities.shape == (4, 11)
    assert paper_entities.shape == (4, 10)
    assert current_mask.shape == (4,)
    assert paper_mask.shape == (4,)

    print("attention adapter shape smoke test passed")


if __name__ == "__main__":
    main()
