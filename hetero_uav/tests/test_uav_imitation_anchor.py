from __future__ import annotations

import torch
import numpy as np


def test_uav_imitation_loss_wraps_heading_and_uses_uav_actor_only():
    from algorithms.happo import HAPPOReferencePolicy
    from algorithms.happo.happo_trainer import _uav_imitation_loss

    policy = HAPPOReferencePolicy(96, 480)
    obs = torch.zeros((2, 96), dtype=torch.float32)
    target = torch.zeros((2, 3), dtype=torch.float32)
    with torch.no_grad():
        target[:, 1] = -0.99
    loss = _uav_imitation_loss(policy, obs, target)
    assert loss.ndim == 0
    loss.backward()
    uav_grad = sum(
        float(p.grad.abs().sum().item())
        for p in policy.uav_actor.parameters()
        if p.grad is not None
    )
    mav_grad = sum(
        float(p.grad.abs().sum().item())
        for p in policy.mav_actor.parameters()
        if p.grad is not None
    )
    critic_grad = sum(
        float(p.grad.abs().sum().item())
        for p in policy.critic.parameters()
        if p.grad is not None
    )
    assert uav_grad > 0.0
    assert mav_grad == 0.0
    assert critic_grad == 0.0


def test_train_script_loads_and_samples_uav_imitation_dataset(tmp_path):
    from scripts.train_happo_reference import (
        _load_uav_imitation_dataset,
        _sample_uav_imitation_batch,
    )

    dataset = tmp_path / "oracle.npz"
    np.savez_compressed(
        dataset,
        actor_obs=np.ones((6, 96), dtype=np.float32),
        oracle_action=np.zeros((6, 3), dtype=np.float32),
    )
    data = _load_uav_imitation_dataset(str(dataset))
    obs, act = _sample_uav_imitation_batch(data, batch_size=4, device=torch.device("cpu"))
    assert obs.shape == (4, 96)
    assert act.shape == (4, 3)
    assert obs.dtype == torch.float32
    assert act.dtype == torch.float32
