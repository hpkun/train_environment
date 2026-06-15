from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_train_help_exposes_policy_arch():
    result = subprocess.run(
        [sys.executable, "scripts/train_happo_reference.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch" in result.stdout
    assert "entity_attention" in result.stdout


def test_entity_policy_evaluate_actions_and_checkpoint_roundtrip(tmp_path):
    from algorithms.happo.entity_policy import EntityHAPPOReferencePolicy

    policy = EntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    actor_obs = torch.zeros((4, 3, 96), dtype=torch.float32)
    roles = torch.tensor([[0, 1, 1]] * 4)
    critic = torch.zeros((4, 480), dtype=torch.float32)
    action = torch.zeros((4, 3, 3), dtype=torch.float32)

    log_prob, entropy, value, mean, inferred_roles = policy.evaluate_actions(
        actor_obs, roles, critic, action)

    assert log_prob.shape == (4, 3)
    assert entropy.shape == (4, 3)
    assert value.shape == (4,)
    assert mean.shape == (4, 3, 3)
    assert inferred_roles.shape == (4, 3)
    assert torch.isfinite(log_prob).all()

    path = tmp_path / "entity_model.pt"
    policy.save(path)
    loaded = EntityHAPPOReferencePolicy(entity_dim=19, critic_state_dim=480, action_dim=3)
    loaded.load(path, map_location="cpu")
    loaded_out = loaded.act(torch.zeros((3, 96)), roles=[0, 1, 1], deterministic=True)
    assert loaded_out["action"].shape == (3, 3)


def test_policy_factory_rejects_flat_checkpoint_for_entity_policy(tmp_path):
    from scripts.train_happo_reference import _build_policy

    flat_meta = tmp_path / "meta.json"
    flat_meta.write_text(json.dumps({"policy_arch": "flat"}), encoding="utf-8")
    try:
        _build_policy("entity_attention", 96, 480, torch.device("cpu"), init_checkpoint_meta=flat_meta)
    except ValueError as exc:
        assert "flat checkpoint" in str(exc)
    else:
        raise AssertionError("entity policy should reject flat checkpoint metadata")


def test_run_entity_policy_smoke_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_entity_policy_smoke.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch" in result.stdout
    assert "entity_attention" in result.stdout
    assert "debug_entity_policy_smoke" in result.stdout
