"""Static smoke test for train_attention_mappo strict adapter helpers.

No env is created here; this must not trigger JSBSim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_attention_mappo import _build_attention_entities, _zero_entity_like


def main() -> None:
    fake_obs = {
        "ego_state": np.zeros(11, dtype=np.float32),
        "ally_states": np.zeros((1, 11), dtype=np.float32),
        "enemy_states": np.zeros((2, 11), dtype=np.float32),
    }

    entities, mask = _zero_entity_like(fake_obs, "strict")
    assert entities.shape == (4, 10)
    assert mask.shape == (4,)
    assert np.allclose(entities, 0.0)
    assert np.all(mask == 1)

    try:
        _build_attention_entities(fake_obs, "strict")
        raise AssertionError("_build_attention_entities should reject strict")
    except ValueError as exc:
        assert "strict adapter must use env.get_strict_team_observations" in str(exc)

    print("attention strict adapter static smoke test passed")


if __name__ == "__main__":
    main()
