"""Smoke test for BRMA rollout schema/storage. No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brma.rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage


def main() -> None:
    # ---- 1. disabled storage ----
    cfg_off = BRMARolloutSchemaConfig(
        num_steps=4, num_envs=2, num_agents=3,
        n_entities=6, entity_dim=10, enabled=False)
    store_off = BRMARolloutStorage(cfg_off)
    assert not store_off.has_storage
    try:
        store_off.store_step(0, 0, 0, p=np.zeros(6, dtype=np.float32),
                             msoft=np.zeros(6, dtype=np.float32),
                             mhard=np.zeros(6, dtype=np.float32),
                             mR_count=0, mB_count=0,
                             friendly_drop_mask=np.zeros(6, dtype=bool),
                             enemy_drop_mask=np.zeros(6, dtype=bool),
                             key_padding_mask=np.zeros(6, dtype=bool),
                             keep_mask=np.ones(6, dtype=bool))
        assert False, "should have raised RuntimeError"
    except RuntimeError:
        pass
    s = store_off.summary()
    assert s["enabled"] == False

    # ---- 2. enabled storage shape ----
    cfg_on = BRMARolloutSchemaConfig(
        num_steps=4, num_envs=2, num_agents=3,
        n_entities=6, entity_dim=10, enabled=True)
    store = BRMARolloutStorage(cfg_on)
    assert store.has_storage
    # valid all False
    store._valid is not None
    assert not store._valid.any()

    # ---- 3. store_step ----
    store.store_step(
        1, 0, 2,
        p=np.full(6, 0.5, dtype=np.float32),
        msoft=np.full(6, 0.3, dtype=np.float32),
        mhard=np.array([0, 0, 0, 1, 1, 0], dtype=np.float32),
        mR_count=1, mB_count=2,
        friendly_drop_mask=np.array([0, 0, 0, 1, 0, 0], dtype=bool),
        enemy_drop_mask=np.array([0, 0, 0, 0, 1, 1], dtype=bool),
        key_padding_mask=np.array([0, 0, 0, 1, 1, 0], dtype=bool),
        keep_mask=np.array([1, 1, 1, 0, 0, 1], dtype=bool),
        log_prob_unmasked=-1.5, log_prob_masked=-2.0,
        entropy_unmasked=0.5, entropy_masked=0.3,
    )
    step_data = store.get_step(1, 0, 2)
    assert step_data["mR_count"] == 1
    assert step_data["mB_count"] == 2
    assert step_data["log_prob_unmasked"] == -1.5
    assert step_data["p"][0] == 0.5

    # ---- 4. get_step returns copy ----
    original = step_data["p"][0]
    step_data["p"][0] = 999.0
    step_data2 = store.get_step(1, 0, 2)
    assert step_data2["p"][0] == original, "get_step must return a copy"

    # ---- 5. invalid shapes ----
    try:
        store.store_step(
            0, 0, 0,
            p=np.zeros(5, dtype=np.float32),  # wrong shape
            msoft=np.zeros(6, dtype=np.float32),
            mhard=np.zeros(6, dtype=np.float32),
            mR_count=0, mB_count=0,
            friendly_drop_mask=np.zeros(6, dtype=bool),
            enemy_drop_mask=np.zeros(6, dtype=bool),
            key_padding_mask=np.zeros(6, dtype=bool),
            keep_mask=np.ones(6, dtype=bool),
        )
        assert False, "should have raised ValueError for wrong shape"
    except ValueError:
        pass

    try:
        store.store_step(
            0, 0, 0,
            p=np.zeros(6, dtype=np.float32),
            msoft=np.zeros(6, dtype=np.float32),
            mhard=np.zeros(6, dtype=np.float32),
            mR_count=0, mB_count=0,
            friendly_drop_mask=np.zeros(6, dtype=bool),
            enemy_drop_mask=np.zeros(6, dtype=bool),
            key_padding_mask=np.zeros(6, dtype=bool),
            keep_mask=np.ones(6, dtype=bool),
            next_entities=np.zeros((5, 10), dtype=np.float32),  # wrong
        )
        assert False
    except ValueError:
        pass

    # ---- 6. summary ----
    store.store_step(
        2, 1, 1,
        p=np.full(6, 0.2, dtype=np.float32),
        msoft=np.full(6, 0.1, dtype=np.float32),
        mhard=np.ones(6, dtype=np.float32),
        mR_count=2, mB_count=1,
        friendly_drop_mask=np.array([0, 1, 1, 0, 0, 0], dtype=bool),
        enemy_drop_mask=np.array([0, 0, 0, 1, 0, 0], dtype=bool),
        key_padding_mask=np.array([0, 1, 1, 1, 0, 1], dtype=bool),
        keep_mask=np.array([1, 0, 0, 0, 1, 0], dtype=bool),
    )
    sm = store.summary()
    assert sm["enabled"] == True
    assert sm["valid_count"] == 2
    assert sm["total_slots"] == 4 * 2 * 3
    assert sm["mean_mR_count"] == 1.5  # (1+2)/2
    assert sm["mean_mB_count"] == 1.5  # (2+1)/2
    assert sm["mean_friendly_drop_count"] == (1 + 2) / 2  # 1 drop [0,0,0,1,0,0] + 2 drops
    assert sm["mean_enemy_drop_count"] == (2 + 1) / 2   # 2 drops + 1 drop

    # ---- 7. AttentionRolloutBuffer default ----
    from train_attention_mappo import AttentionRolloutBuffer
    buf = AttentionRolloutBuffer(
        num_steps=3, num_envs=2, num_red=1,
        action_dim=3, rnn_hidden_size=128)
    assert buf.brma_storage is None
    assert buf.actions.shape == (3, 2, 1, 3)

    print("brma rollout schema static smoke test passed")


if __name__ == "__main__":
    main()
