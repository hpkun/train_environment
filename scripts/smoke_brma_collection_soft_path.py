"""Smoke test for BRMA soft-mask collection path. No env, no JSBSim."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention_models import AttentionActor  # noqa: E402
from brma.collection import collect_brma_dry_run_step  # noqa: E402
from brma.losses import diagonal_gaussian_kl  # noqa: E402
from brma.mask_generator import (  # noqa: E402
    BRMAMaskGenerator,
    BRMAMaskGeneratorConfig,
    generate_brma_masks,
)
from brma.rollout_schema import BRMARolloutSchemaConfig, BRMARolloutStorage  # noqa: E402


def _setup():
    torch.manual_seed(11)
    np.random.seed(11)
    actor = AttentionActor(
        entity_dim=10,
        action_dim=3,
        hidden_size=128,
        rnn_hidden=128,
        encoder_mode="paper_eq33",
    )
    mask_gen = BRMAMaskGenerator(BRMAMaskGeneratorConfig(entity_feature_dim=10))
    cfg = BRMARolloutSchemaConfig(
        num_steps=4,
        num_envs=2,
        num_agents=2,
        n_entities=5,
        entity_dim=10,
        action_dim=3,
        enabled=True,
    )
    entities = np.random.randn(5, 10).astype(np.float32)
    entity_mask = np.zeros(5, dtype=np.int64)
    rnn_hidden = np.zeros(128, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    return actor, mask_gen, cfg, entities, entity_mask, rnn_hidden, action


def test_soft_path_basic() -> None:
    actor, mask_gen, cfg, entities, entity_mask, rnn_hidden, action = _setup()
    storage = BRMARolloutStorage(cfg)
    summary = collect_brma_dry_run_step(
        actor=actor,
        mask_generator=mask_gen,
        storage=storage,
        step=0,
        env_idx=0,
        agent_idx=0,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_hidden,
        action=action,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        mR_count=torch.tensor([1]),
        mB_count=torch.tensor([2]),
        use_soft_mask_path=True,
    )
    assert summary["use_soft_mask_path"] is True
    assert summary["storage_summary"]["valid_count"] == 1
    assert summary["soft_keep_mean"] > 0.0
    stored = storage.get_step(0, 0, 0)
    for key in ("mu_unmasked", "mu_masked", "sigma_unmasked", "sigma_masked"):
        assert stored[key].shape == (3,)
        assert np.isfinite(stored[key]).all()


def test_hard_fallback() -> None:
    actor, mask_gen, cfg, entities, entity_mask, rnn_hidden, action = _setup()
    storage = BRMARolloutStorage(cfg)
    summary = collect_brma_dry_run_step(
        actor=actor,
        mask_generator=mask_gen,
        storage=storage,
        step=0,
        env_idx=0,
        agent_idx=0,
        entities=entities,
        entity_mask=entity_mask,
        rnn_hidden=rnn_hidden,
        action=action,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        mR_count=torch.tensor([0]),
        mB_count=torch.tensor([1]),
        use_soft_mask_path=False,
    )
    assert summary["use_soft_mask_path"] is False
    assert summary["soft_keep_mean"] == 0.0
    assert storage.summary()["valid_count"] == 1


def test_soft_path_gradient() -> None:
    actor, mask_gen, _cfg, _entities, _entity_mask, _rnn_hidden, _action = _setup()
    entities = torch.randn(1, 5, 10)
    entity_mask = torch.zeros(1, 5, dtype=torch.long)
    rnn_hidden = torch.zeros(1, 128)
    action = torch.zeros(1, 3)
    brma_out = generate_brma_masks(
        mask_gen,
        entities,
        entity_mask,
        n_ego=1,
        n_allies=1,
        n_enemies=3,
        mR_count=torch.tensor([1]),
        mB_count=torch.tensor([2]),
    )
    msoft = brma_out["msoft"].detach().clone().requires_grad_(True)
    soft_keep = torch.ones_like(msoft)
    soft_keep[:, 1:] = msoft[:, 1:]
    dual = actor.evaluate_dual_actions(
        entities,
        unmasked_entity_mask=entity_mask,
        masked_entity_mask=entity_mask,
        rnn_hidden=rnn_hidden,
        actions=action,
        masked_soft_keep_mask=soft_keep,
    )
    kl = diagonal_gaussian_kl(
        dual["mu_unmasked"].detach(),
        dual["sigma_unmasked"].detach(),
        dual["mu_masked"],
        dual["sigma_masked"],
    )
    kl.mean().backward()
    assert msoft.grad is not None
    assert torch.isfinite(msoft.grad).all()
    assert msoft.grad.abs().sum() > 0


def test_hard_path_does_not_depend_on_msoft() -> None:
    actor, _mask_gen, _cfg, _entities, _entity_mask, _rnn_hidden, _action = _setup()
    entities = torch.randn(1, 5, 10)
    entity_mask = torch.zeros(1, 5, dtype=torch.long)
    hard_mask = entity_mask.clone()
    hard_mask[:, 2] = 1
    msoft = torch.full((1, 5), 0.5, requires_grad=True)
    dual = actor.evaluate_dual_actions(
        entities,
        unmasked_entity_mask=entity_mask,
        masked_entity_mask=hard_mask,
        rnn_hidden=torch.zeros(1, 128),
        actions=torch.zeros(1, 3),
    )
    loss = dual["mu_masked"].sum()
    loss.backward()
    assert msoft.grad is None


def test_storage_shape_validation() -> None:
    cfg = BRMARolloutSchemaConfig(
        num_steps=1,
        num_envs=1,
        num_agents=1,
        n_entities=5,
        entity_dim=10,
        action_dim=3,
        enabled=True,
    )
    storage = BRMARolloutStorage(cfg)
    kwargs = dict(
        p=np.zeros(5, dtype=np.float32),
        msoft=np.zeros(5, dtype=np.float32),
        mhard=np.zeros(5, dtype=np.float32),
        mR_count=0,
        mB_count=0,
        friendly_drop_mask=np.zeros(5, dtype=bool),
        enemy_drop_mask=np.zeros(5, dtype=bool),
        key_padding_mask=np.zeros(5, dtype=bool),
        keep_mask=np.ones(5, dtype=bool),
        mu_unmasked=np.zeros(2, dtype=np.float32),
    )
    try:
        storage.store_step(0, 0, 0, **kwargs)
        assert False, "expected ValueError for action_dim mismatch"
    except ValueError:
        pass


def main() -> None:
    test_soft_path_basic()
    test_hard_fallback()
    test_soft_path_gradient()
    test_hard_path_does_not_depend_on_msoft()
    test_storage_shape_validation()
    print("brma collection soft path smoke test passed")


if __name__ == "__main__":
    main()
