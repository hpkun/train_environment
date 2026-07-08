from __future__ import annotations

import json

import numpy as np
import torch

from algorithms.pure_happo.trainer import _compute_grouped_gae
from scripts.audit_pure_happo_update_invariants import run_invariants


def test_pure_happo_update_invariants_summary(tmp_path):
    summary = run_invariants(device="cpu", output_dir=tmp_path)

    assert summary["overall_pass"] is True
    assert summary["logprob_replay_pass"] is True
    assert summary["tanh_action_bounds_pass"] is True
    assert summary["boundary_logprob_finite_pass"] is True
    assert summary["zero_lr_noop_pass"] is True
    assert summary["inactive_mask_no_nan_pass"] is True
    assert summary["grouped_gae_by_env_id_pass"] is True
    assert "actor_loss_mav" in summary["update_metric_keys"]
    assert "actor_loss_uav" in summary["update_metric_keys"]
    assert "approx_kl_mav" in summary["update_metric_keys"]
    assert "approx_kl_uav" in summary["update_metric_keys"]

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved["overall_pass"] is True


def test_grouped_gae_does_not_leak_between_interleaved_env_ids():
    rewards = torch.tensor([1.0, 10.0, 1.0, 10.0])
    values = torch.zeros(4)
    next_values = torch.zeros(4)
    dones = torch.tensor([0.0, 0.0, 1.0, 1.0])
    env_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    advantages, returns = _compute_grouped_gae(
        rewards, values, next_values, dones, env_ids, gamma=1.0, lam=1.0)

    np.testing.assert_allclose(advantages.numpy(), np.array([2.0, 20.0, 1.0, 10.0]))
    np.testing.assert_allclose(returns.numpy(), np.array([2.0, 20.0, 1.0, 10.0]))
