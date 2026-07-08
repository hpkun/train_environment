from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from scripts.train_happo_reference import (
    MARL_DYNAMICS_TRAIN_FIELDS,
    UPDATE_DIAGNOSTIC_ARRAY_FIELDS,
)


def _fake_buffer(policy: PureHAPPOPolicy, steps: int = 10) -> HAPPORolloutBuffer:
    rng = np.random.default_rng(123)
    buf = HAPPORolloutBuffer(steps, policy.num_agents, 96, 480, policy.action_dim, [0, 1, 1])
    for t in range(steps):
        obs = rng.normal(0.0, 0.2, size=(policy.num_agents, 96)).astype(np.float32)
        critic = rng.normal(0.0, 0.2, size=(480,)).astype(np.float32)
        with torch.no_grad():
            out = policy.act(torch.as_tensor(obs), critic_state=torch.as_tensor(critic))
        active = np.ones(policy.num_agents, dtype=np.float32)
        if t % 3 == 0:
            active[2] = 0.0
        buf.store(
            obs,
            critic,
            out["action"].numpy(),
            out["log_prob"].numpy(),
            rng.normal(0.0, 0.1, size=(policy.num_agents,)).astype(np.float32),
            np.zeros(policy.num_agents, dtype=np.float32),
            float(out["value"].item()),
            active,
            next_value=float(out["value"].item()),
            env_id=t % 2,
        )
    return buf


def test_pure_happo_update_returns_training_dynamics_metrics():
    policy = PureHAPPOPolicy(num_agents=3)
    trainer = PureHAPPOTrainer(policy, ppo_epochs=1, seed=7)
    metrics = trainer.update(_fake_buffer(policy))

    required = [
        "ratio_mean_per_agent", "ratio_std_per_agent", "ratio_p95_per_agent",
        "ratio_p99_per_agent", "clip_fraction_per_agent", "approx_kl_abs_per_agent",
        "clip_fraction_mav", "clip_fraction_uav", "approx_kl_abs_mav",
        "approx_kl_abs_uav", "critic_loss_unscaled", "critic_loss_scaled",
        "critic_grad_norm", "critic_update_norm", "policy_update_norm_per_agent",
        "value_explained_variance", "advantage_raw_mean", "advantage_norm_std",
        "m_mean_after_each_agent", "m_std_after_each_agent", "m_abs_max_after_each_agent",
        "mav_action_mean_pitch", "uav_action_saturation_speed",
        "active_sample_ratio_per_agent",
    ]
    for key in required:
        assert key in metrics

    scalar_values = [
        value for value in metrics.values()
        if isinstance(value, (float, int, np.floating, np.integer))
    ]
    assert scalar_values
    assert all(np.isfinite(float(value)) for value in scalar_values)
    assert len(metrics["valid_sample_count_per_agent"]) == 3
    assert len(metrics["active_sample_ratio_per_agent"]) == 3


def test_train_log_field_contract_contains_new_marl_diagnostics():
    for field in [
        "clip_fraction_mav", "approx_kl_abs_uav", "critic_loss_unscaled",
        "value_explained_variance", "advantage_raw_std",
        "uav_action_saturation_heading", "rollout_transitions",
        "actor_obs_abs_max", "critic_state_nan_count",
    ]:
        assert field in MARL_DYNAMICS_TRAIN_FIELDS
    for field in [
        "actor_loss_per_agent", "clip_fraction_per_agent",
        "valid_sample_count_per_agent", "m_abs_max_after_each_agent",
    ]:
        assert field in UPDATE_DIAGNOSTIC_ARRAY_FIELDS
    source = Path("scripts/train_happo_reference.py").read_text(encoding="utf-8")
    assert "*MARL_DYNAMICS_TRAIN_FIELDS" in source
    assert "update_diagnostics.jsonl" in source


def test_update_diagnostics_jsonl_payload_roundtrip(tmp_path):
    payload = {
        "iteration": 1,
        "total_steps": 10,
        "actor_loss_per_agent": [0.1, 0.2, 0.3],
        "clip_fraction_per_agent": [0.0, 0.1, 0.2],
    }
    path = tmp_path / "update_diagnostics.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["actor_loss_per_agent"] == [0.1, 0.2, 0.3]
