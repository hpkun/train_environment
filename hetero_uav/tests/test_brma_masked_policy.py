from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_random_scale_mask_never_masks_self_and_keeps_padding_masked():
    from algorithms.happo.brma_masked_policy import apply_random_scale_mask

    keep = torch.ones((4, 9), dtype=torch.bool)
    keep[:, -1] = False
    masked, stats = apply_random_scale_mask(keep, drop_prob=1.0, training=True)

    assert masked[:, 0].all()
    assert not masked[:, -1].any()
    assert 0.0 <= stats["mask_keep_ratio"] <= 1.0
    assert stats["masked_entity_count"] > 0


def test_biased_mask_generator_forward_shape_and_range():
    from algorithms.happo.brma_masked_policy import BRMABiasedMaskGenerator

    generator = BRMABiasedMaskGenerator(entity_dim=19, hidden_dim=32)
    entities = torch.randn((2, 9, 19), dtype=torch.float32)
    keep = torch.ones((2, 9), dtype=torch.bool)
    out = generator(entities, keep)

    assert out["keep_prob"].shape == (2, 9)
    assert out["logits"].shape == (2, 9)
    assert out["entropy"].shape == (2,)
    assert torch.all((out["keep_prob"] >= 0.0) & (out["keep_prob"] <= 1.0))


def test_biased_mask_application_keeps_self_and_masks_padding():
    from algorithms.happo.brma_masked_policy import BRMABiasedMaskGenerator, apply_biased_mask

    generator = BRMABiasedMaskGenerator(entity_dim=19, hidden_dim=32)
    entities = torch.randn((3, 9, 19), dtype=torch.float32)
    keep = torch.ones((3, 9), dtype=torch.bool)
    keep[:, -2:] = False
    masked, stats = apply_biased_mask(generator, entities, keep, max_mask_allies=2, max_mask_enemies=2, training=True)

    assert masked[:, 0].all()
    assert not masked[:, -1].any()
    assert not masked[:, -2].any()
    assert 0.0 <= stats["mask_keep_ratio"] <= 1.0
    assert stats["mask_entropy"] >= 0.0


def test_brma_recurrent_masked_policy_forward_and_stats():
    from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy

    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=True,
        random_mask_prob=0.5,
        biased_mask=True,
    )
    actor_obs = torch.zeros((3, 96), dtype=torch.float32)
    out = policy.act(actor_obs, roles=[0, 1, 1], deterministic=True)

    assert out["action"].shape == (3, 3)
    assert out["rnn_hidden"].shape == (3, 128)
    assert policy.last_mask_stats["mask_keep_ratio"] >= 0.0
    assert policy.last_mask_stats["masked_entity_count"] >= 0.0


def test_brma_recurrent_masked_policy_save_load_roundtrip(tmp_path):
    from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy

    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=True,
        biased_mask=True,
    )
    path = tmp_path / "model.pt"
    policy.save(path)
    loaded = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=True,
        biased_mask=True,
    )
    loaded.load(path, map_location="cpu")
    out = loaded.act(torch.zeros((3, 96)), roles=[0, 1, 1], deterministic=True)
    assert out["action"].shape == (3, 3)


def test_eval_loader_supports_brma_recurrent_masked(tmp_path):
    from algorithms.happo.brma_masked_policy import BRMARecurrentMaskedHAPPOReferencePolicy
    from scripts.eval_happo_reference import _build_policy_from_meta

    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=19,
        critic_state_dim=480,
        action_dim=3,
        random_scale_mask=True,
        biased_mask=True,
    )
    model = tmp_path / "model.pt"
    policy.save(model)
    (tmp_path / "meta.json").write_text(json.dumps({
        "policy_arch": "brma_recurrent_masked",
        "entity_dim": 19,
        "critic_state_dim": 480,
        "random_scale_mask": True,
        "biased_mask": True,
    }), encoding="utf-8")

    loaded = _build_policy_from_meta(json.loads((tmp_path / "meta.json").read_text()), torch.device("cpu"))
    loaded.load(model, map_location="cpu")
    assert loaded.act(torch.zeros((3, 96)), roles=[0, 1, 1], deterministic=True)["action"].shape == (3, 3)


def test_run_brma_masked_smoke_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_brma_masked_smoke.py", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--policy-arch" in result.stdout
    assert "brma_recurrent_masked" in result.stdout
    assert "debug_brma_random_mask_smoke" in result.stdout
