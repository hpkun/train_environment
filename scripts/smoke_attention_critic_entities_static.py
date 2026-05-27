"""Static smoke test for attention-entities critic path.

No env, no JSBSim.  Tests CentralizedAttentionCritic and the
_build_attention_critic_entities_for_env helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from attention_models import CentralizedAttentionCritic
from train_vanilla_mappo import Config
from train_attention_mappo import _build_attention_critic_entities_for_env


def _fake_strict_env_obs(num_red: int) -> dict:
    n_entities = num_red + 2  # 2 blue
    obs = {}
    for i in range(num_red):
        entities = np.full((n_entities, 10), float(i + 1), dtype=np.float32)
        mask = np.zeros(n_entities, dtype=np.int64)
        obs[f"red_{i}"] = (entities, mask, {"schema": "test"})
    return obs


def main() -> None:
    B, A, N, D = 4, 2, 5, 10

    # ---- 1. CentralizedAttentionCritic current ----
    critic_cur = CentralizedAttentionCritic(
        entity_dim=D, hidden_size=128, num_heads=4,
        num_agents=A, encoder_mode="current")
    ent = torch.randn(B, A, N, D)
    msk = torch.zeros(B, A, N, dtype=torch.long)
    out_cur = critic_cur(ent, msk)
    assert out_cur.shape == (B, A), f"current shape: {out_cur.shape}"
    assert torch.isfinite(out_cur).all()

    # ---- 2. CentralizedAttentionCritic paper_eq33 ----
    critic_eq33 = CentralizedAttentionCritic(
        entity_dim=D, hidden_size=128, num_heads=4,
        num_agents=A, encoder_mode="paper_eq33")
    out_eq33 = critic_eq33(ent, msk)
    assert out_eq33.shape == (B, A)
    assert torch.isfinite(out_eq33).all()
    assert critic_eq33.encoder.output_dim == 256

    # ---- 3. _build_attention_critic_entities_for_env ----
    config = Config()
    config.num_red = 2
    config.num_blue = 2
    strict = _fake_strict_env_obs(2)
    team_ent, team_msk = _build_attention_critic_entities_for_env(strict, config)
    assert team_ent.shape == (2, 4, 10)
    assert team_msk.shape == (2, 4)
    assert np.isfinite(team_ent).all()

    # ---- 4. missing agent ----
    partial = {"red_0": strict["red_0"]}
    team_ent2, team_msk2 = _build_attention_critic_entities_for_env(partial, config)
    assert team_ent2.shape == (2, 4, 10)
    assert np.all(team_msk2[1] == 1)  # missing agent → all-ones mask
    assert np.all(team_ent2[1] == 0)  # zeros

    # ---- 5. empty strict_obs ----
    team_ent3, team_msk3 = _build_attention_critic_entities_for_env(None, config)
    assert team_ent3.shape == (2, 4, 10)
    assert team_msk3.shape == (2, 4)

    # ---- 6. Invalid obs_adapter for attention-entities ----
    config_bad = Config()
    config_bad.num_red = 2
    config_bad.num_blue = 2
    config_bad.obs_adapter = "current"
    config_bad.critic_state = "attention-entities"
    # The config validation happens at CLI parsing level; here we test that
    # the helper still produces correct shapes even with a bad config
    team_ent4, team_msk4 = _build_attention_critic_entities_for_env(None, config_bad)
    assert team_ent4.shape == (2, 4, 10)

    print("attention critic entities static smoke test passed")


if __name__ == "__main__":
    main()
