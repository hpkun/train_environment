"""Static smoke test for --critic-state attention critic global obs helpers.

No env, no JSBSim.  Tests _compute_attention_global_obs_dim,
_build_global_obs_for_env, and AttentionRolloutBuffer.global_obs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_attention_mappo import (
    AttentionRolloutBuffer,
    _build_global_obs_for_env,
    _compute_attention_global_obs_dim,
)
from train_vanilla_mappo import Config, _compute_obs_dim, _flatten_obs


def main() -> None:
    obs_dim = _compute_obs_dim(2, 2, is_red=True)

    # ---- engineering mode ----
    config_eng = Config()
    config_eng.num_red = 2
    config_eng.num_blue = 2
    config_eng.obs_adapter = "strict"
    config_eng.critic_state = "engineering"

    god_eng = _compute_attention_global_obs_dim(config_eng, obs_dim)
    assert god_eng == obs_dim * 2

    # ---- strict-global mode ----
    config_sg = Config()
    config_sg.num_red = 2
    config_sg.num_blue = 2
    config_sg.obs_adapter = "strict"
    config_sg.critic_state = "strict-global"

    god_sg = _compute_attention_global_obs_dim(config_sg, obs_dim)
    assert god_sg == 88, f"expected 88, got {god_sg}"

    # strict-global + non-strict obs_adapter → ValueError
    config_bad = Config()
    config_bad.num_red = 2
    config_bad.num_blue = 2
    config_bad.obs_adapter = "current"
    config_bad.critic_state = "strict-global"
    try:
        _compute_attention_global_obs_dim(config_bad, obs_dim)
        assert False, "should have raised ValueError"
    except ValueError:
        pass

    # ---- _build_global_obs_for_env ----
    red_ids = ["red_0", "red_1"]

    # engineering: build fake env_obs
    fake_env_obs = {}
    for i, rid in enumerate(red_ids):
        dummy = {
            "ego_state": np.ones(11, dtype=np.float32) * (i + 1),
            "ally_states": np.zeros((1, 11), dtype=np.float32),
            "enemy_states": np.ones((2, 11), dtype=np.float32),
            "death_mask": np.ones(4, dtype=np.float32),
            "missile_warning": np.zeros(1, dtype=np.float32),
            "altitude": np.array([6000.0], dtype=np.float32),
            "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
        }
        fake_env_obs[rid] = dummy

    g_eng = _build_global_obs_for_env(fake_env_obs, None, red_ids, obs_dim, config_eng)
    assert g_eng.shape == (obs_dim * 2,), f"eng shape: {g_eng.shape}"

    # strict-global: build fake strict_env_obs
    fake_strict = {}
    for i, rid in enumerate(red_ids):
        entities = np.full((4, 10), float(i + 1), dtype=np.float32)
        mask = np.zeros(4, dtype=np.int64)
        fake_strict[rid] = (entities, mask, {"schema": "test"})

    g_sg = _build_global_obs_for_env({}, fake_strict, red_ids, obs_dim, config_sg)
    assert g_sg.shape == (88,), f"strict-global shape: {g_sg.shape}"

    # strict-global with missing env → zeros
    g_empty = _build_global_obs_for_env({}, {}, red_ids, obs_dim, config_sg)
    assert g_empty.shape == (88,)
    assert np.all(g_empty == 0.0)

    # ---- AttentionRolloutBuffer.global_obs ----
    buf = AttentionRolloutBuffer(
        num_steps=5, num_envs=2, num_red=2,
        action_dim=3, rnn_hidden_size=128,
        global_obs_dim=88,
    )
    assert buf.global_obs.shape == (5, 2, 88)

    print("attention critic state static smoke test passed")


if __name__ == "__main__":
    main()
