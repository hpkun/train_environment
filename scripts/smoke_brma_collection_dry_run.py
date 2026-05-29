"""Smoke test for BRMA collection dry-run API. No env, no JSBSim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from attention_models import AttentionActor
from brma.collection import collect_brma_dry_run_step
from brma.mask_generator import BRMAMaskGenerator, BRMAMaskGeneratorConfig
from brma.rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage


def main() -> None:
    # ---- setup ----
    actor = AttentionActor(entity_dim=10, action_dim=3, hidden_size=128,
                           rnn_hidden=128, encoder_mode="paper_eq33")
    mg_cfg = BRMAMaskGeneratorConfig(entity_feature_dim=10)
    mask_gen = BRMAMaskGenerator(mg_cfg)
    sc_cfg = BRMARolloutSchemaConfig(
        num_steps=4, num_envs=2, num_agents=2,
        n_entities=5, entity_dim=10, enabled=True)
    storage = BRMARolloutStorage(sc_cfg)

    N, D = 5, 10
    n_ego, n_ally, n_enemy = 1, 1, 3
    entities_np = np.random.randn(N, D).astype(np.float32)
    emask_np = np.zeros(N, dtype=np.int64)
    rnn_np = np.zeros(128, dtype=np.float32)
    action_np = np.zeros(3, dtype=np.float32)

    # ---- 1. basic dry-run ----
    summary = collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen, storage=storage,
        step=0, env_idx=0, agent_idx=0,
        entities=entities_np, entity_mask=emask_np,
        rnn_hidden=rnn_np, action=action_np,
        n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
        mR_count=torch.tensor([1]), mB_count=torch.tensor([2]),
    )
    assert storage._valid[0, 0, 0]
    assert np.isfinite(summary["log_prob_unmasked"])
    assert np.isfinite(summary["log_prob_masked"])
    assert summary["enemy_drop_count"] <= 2
    assert summary["friendly_drop_count"] <= 1
    assert summary["use_soft_mask_path"] is True
    assert summary["soft_keep_mean"] > 0.0

    # ---- 2. same mask count 0 (keys identical) ----
    storage2 = BRMARolloutStorage(sc_cfg)
    summary2 = collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen, storage=storage2,
        step=1, env_idx=0, agent_idx=0,
        entities=entities_np, entity_mask=emask_np,
        rnn_hidden=rnn_np, action=action_np,
        n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
        mR_count=torch.tensor([0]), mB_count=torch.tensor([0]),
        use_soft_mask_path=False,
    )
    assert summary2["enemy_drop_count"] == 0
    assert summary2["friendly_drop_count"] == 0
    assert np.allclose(summary2["log_prob_unmasked"],
                       summary2["log_prob_masked"], atol=1e-5)

    # ---- 3. invalid/death mask ----
    emask_dead = emask_np.copy()
    emask_dead[-1] = 1
    storage3 = BRMARolloutStorage(sc_cfg)
    summary3 = collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen, storage=storage3,
        step=0, env_idx=1, agent_idx=1,
        entities=entities_np, entity_mask=emask_dead,
        rnn_hidden=rnn_np, action=action_np,
        n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
        mR_count=torch.tensor([0]), mB_count=torch.tensor([0]),
        next_entities=entities_np, next_entity_mask=emask_dead,
    )
    assert summary3["key_padding_count"] >= 1

    # ---- 4. storage disabled ----
    sc_off = BRMARolloutSchemaConfig(
        num_steps=4, num_envs=2, num_agents=2, enabled=False)
    store_off = BRMARolloutStorage(sc_off)
    try:
        collect_brma_dry_run_step(
            actor=actor, mask_generator=mask_gen, storage=store_off,
            step=0, env_idx=0, agent_idx=0,
            entities=entities_np, entity_mask=emask_np,
            rnn_hidden=rnn_np, action=action_np,
            n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
        )
        assert False, "should have raised RuntimeError for disabled storage"
    except RuntimeError:
        pass

    # ---- 5. shape validation ----
    try:
        collect_brma_dry_run_step(
            actor=actor, mask_generator=mask_gen, storage=storage,
            step=0, env_idx=0, agent_idx=0,
            entities=np.random.randn(3, D).astype(np.float32),  # wrong N
            entity_mask=emask_np,
            rnn_hidden=rnn_np, action=action_np,
            n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
        )
        assert False, "should have raised ValueError for wrong N"
    except ValueError:
        pass

    for bad_inputs, desc in [
        (dict(entities=np.random.randn(2, N, D).astype(np.float32),
              entity_mask=np.zeros((2, N), dtype=np.int64),
              rnn_hidden=np.stack([rnn_np, rnn_np]),
              action=np.stack([action_np, action_np])),
         "entities batch=2"),
        (dict(entities=entities_np,
              entity_mask=np.zeros((2, N), dtype=np.int64),
              rnn_hidden=rnn_np, action=action_np),
         "entity_mask batch=2"),
        (dict(entities=entities_np, entity_mask=emask_np,
              rnn_hidden=np.stack([rnn_np, rnn_np]),
              action=action_np),
         "rnn_hidden batch=2"),
        (dict(entities=entities_np, entity_mask=emask_np,
              rnn_hidden=rnn_np,
              action=np.stack([action_np, action_np])),
         "action batch=2"),
    ]:
        try:
            collect_brma_dry_run_step(
                actor=actor, mask_generator=mask_gen, storage=storage,
                step=0, env_idx=0, agent_idx=0,
                n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
                **bad_inputs,
            )
            assert False, f"should have raised ValueError for {desc}"
        except ValueError:
            pass

    # ---- 5b. valid (1,…) inputs pass ----
    collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen, storage=storage,
        step=0, env_idx=0, agent_idx=1,
        entities=entities_np.reshape(1, N, D),
        entity_mask=emask_np.reshape(1, N),
        rnn_hidden=rnn_np.reshape(1, -1),
        action=action_np.reshape(1, -1),
        n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
    )

    # ---- 6. no parameter mutation ----
    params_before = {n: p.clone() for n, p in actor.named_parameters()}
    storage6 = BRMARolloutStorage(sc_cfg)
    collect_brma_dry_run_step(
        actor=actor, mask_generator=mask_gen, storage=storage6,
        step=0, env_idx=0, agent_idx=0,
        entities=entities_np, entity_mask=emask_np,
        rnn_hidden=rnn_np, action=action_np,
        n_ego=n_ego, n_allies=n_ally, n_enemies=n_enemy,
    )
    for n, p in actor.named_parameters():
        assert torch.allclose(params_before[n], p), f"actor.{n} mutated"

    # ---- 7. get_step copy ----
    step_data = storage3.get_step(0, 1, 1)
    assert step_data["p"].shape == (N,)
    assert step_data["key_padding_mask"].shape == (N,)
    assert step_data["mu_unmasked"].shape == (3,)
    assert step_data["sigma_masked"].shape == (3,)
    orig_p = step_data["p"][0]
    step_data["p"][0] = 999.0
    step_data2 = storage3.get_step(0, 1, 1)
    assert step_data2["p"][0] == orig_p

    print("brma collection dry-run smoke test passed")


if __name__ == "__main__":
    main()
