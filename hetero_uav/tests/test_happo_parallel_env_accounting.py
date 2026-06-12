from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.happo.happo_trainer import _compute_grouped_gae
from scripts.train_happo_reference import NUM_ENVS


def test_happo_buffer_records_env_id_and_next_value():
    buffer = HAPPORolloutBuffer(4, 3, 96, 480, 3, role_ids=[0, 1, 1])
    for env_id in range(4):
        buffer.store(
            actor_obs=np.zeros((3, 96), dtype=np.float32),
            critic_state=np.zeros(480, dtype=np.float32),
            actions=np.zeros((3, 3), dtype=np.float32),
            log_probs=np.zeros(3, dtype=np.float32),
            rewards=np.ones(3, dtype=np.float32),
            dones=np.zeros(3, dtype=np.float32),
            value=float(env_id),
            active_masks=np.ones(3, dtype=np.float32),
            next_value=float(env_id + 10),
            env_id=env_id,
        )

    data = buffer.get(torch.device("cpu"))
    assert data["env_ids"].tolist() == [0, 1, 2, 3]
    assert data["next_values"].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_grouped_gae_done_does_not_cross_envs():
    rewards = torch.tensor([1.0, 1.0, 10.0, 10.0])
    values = torch.zeros(4)
    next_values = torch.zeros(4)
    dones = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env_ids = torch.tensor([0, 0, 1, 1])

    adv, returns = _compute_grouped_gae(
        rewards,
        values,
        next_values,
        dones,
        env_ids,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert adv.tolist() == [1.0, 1.0, 20.0, 10.0]
    assert returns.tolist() == adv.tolist()


def test_rollout_length_accounting_constants():
    rollout_length_per_env = 64
    assert NUM_ENVS == 4
    assert rollout_length_per_env * NUM_ENVS == 256
