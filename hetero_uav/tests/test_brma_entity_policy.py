from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_brma_entity_policy_forward_shapes():
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy

    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    actor_obs = torch.zeros((3, 96), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)

    assert out["action"].shape == (3, 3)
    assert out["log_prob"].shape == (3,)
    assert out["entropy"].shape == (3,)
    assert out["mean"].shape == (3, 3)
    assert torch.isfinite(out["action"]).all()


def test_brma_entity_policy_5v4_forward_shape():
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy

    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    actor_obs = torch.zeros((5, 96), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1, 1, 1], deterministic=True)

    assert out["action"].shape == (5, 3)
    assert out["role_mask"].tolist() == [0, 1, 1, 1, 1]


def test_brma_entity_policy_evaluate_actions_batch():
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy

    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    actor_obs = torch.zeros((4, 3, 96), dtype=torch.float32)
    roles = torch.tensor([[0, 1, 1]] * 4)
    critic = torch.zeros((4, 480), dtype=torch.float32)
    actions = torch.zeros((4, 3, 3), dtype=torch.float32)

    log_prob, entropy, value, mean, role_ids = policy.evaluate_actions(
        actor_obs, roles, critic, actions)

    assert log_prob.shape == (4, 3)
    assert entropy.shape == (4, 3)
    assert value.shape == (4,)
    assert mean.shape == (4, 3, 3)
    assert role_ids.shape == (4, 3)
    assert torch.isfinite(log_prob).all()


def test_brma_entity_attention_mask_changes_output():
    from algorithms.happo.brma_entity_policy import BRMAEntityObservationEncoder

    encoder = BRMAEntityObservationEncoder(entity_dim=19, hidden_size=32, num_heads=4)
    entities = torch.randn((2, 9, 19), dtype=torch.float32)
    keep_all = torch.ones((2, 9), dtype=torch.bool)
    keep_self_only = torch.zeros((2, 9), dtype=torch.bool)
    keep_self_only[:, 0] = True

    out_all, _ = encoder(entities, keep_all)
    out_self, _ = encoder(entities, keep_self_only)

    assert out_all.shape == (2, 64)
    assert out_self.shape == (2, 64)
    assert not torch.allclose(out_all, out_self)


def test_brma_entity_policy_save_load_roundtrip(tmp_path):
    from algorithms.happo.brma_entity_policy import BRMAEntityHAPPOReferencePolicy

    policy = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    path = tmp_path / "model.pt"
    policy.save(path)
    loaded = BRMAEntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    loaded.load(path, map_location="cpu")
    out = loaded.act(torch.zeros((3, 96)), roles=[0, 1, 1], deterministic=True)
    assert out["action"].shape == (3, 3)


def test_train_and_eval_help_include_brma_entity():
    train = subprocess.run(
        [sys.executable, "scripts/train_happo_reference.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "brma_entity" in train.stdout


def test_train_policy_factory_rejects_flat_checkpoint_for_brma_entity(tmp_path):
    from scripts.train_happo_reference import _build_policy

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"policy_arch": "flat"}), encoding="utf-8")
    try:
        _build_policy("brma_entity", 96, 480, torch.device("cpu"), init_checkpoint_meta=meta)
    except ValueError as exc:
        assert "brma_entity cannot load" in str(exc)
    else:
        raise AssertionError("brma_entity should reject flat checkpoint metadata")


def test_run_brma_entity_smoke_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_brma_entity_smoke.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch" in result.stdout
    assert "brma_entity" in result.stdout
    assert "debug_brma_entity_smoke" in result.stdout
