"""Checkpoint contract for the vanilla HAPPO baseline."""

from __future__ import annotations

import random
import numpy as np
import torch


def save_vanilla_happo_checkpoint(path, policy, trainer, *, environment_steps,
                                  episodes, config, numpy_rng):
    payload = {
        "format": "tam_paper_vanilla_happo_v1",
        "policy": policy.state_dict(),
        "actor_optimizers": {key: value.state_dict()
                             for key, value in trainer.actor_optimizers.items()},
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "trainer_update_count": trainer.update_count,
        "environment_steps": int(environment_steps),
        "episodes": int(episodes),
        "python_random_state": random.getstate(),
        "numpy_random_state": numpy_rng.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": dict(config),
        "agent_ids": list(policy.agent_ids),
        "agent_actor_mapping": {aid: policy.actor_key(aid) for aid in policy.agent_ids},
        "actor_obs_dim": policy.actor_obs_dim,
        "critic_state_dim": policy.critic_state_dim,
        "action_spec": [policy.action_levels] * policy.action_dim,
        "actor_sharing": policy.actor_sharing,
    }
    torch.save(payload, path)
    return payload


def load_vanilla_happo_checkpoint(path, policy, trainer=None, *, numpy_rng=None,
                                  restore_rng=True):
    payload = torch.load(path, map_location=next(policy.parameters()).device,
                         weights_only=False)
    expected = {
        "agent_ids": list(policy.agent_ids),
        "actor_obs_dim": policy.actor_obs_dim,
        "critic_state_dim": policy.critic_state_dim,
        "action_spec": [policy.action_levels] * policy.action_dim,
        "actor_sharing": policy.actor_sharing,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch: {payload.get(key)!r} != {value!r}")
    policy.load_state_dict(payload["policy"])
    if trainer is not None:
        for key, state in payload["actor_optimizers"].items():
            trainer.actor_optimizers[key].load_state_dict(state)
        trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        trainer.update_count = int(payload["trainer_update_count"])
    if restore_rng:
        random.setstate(payload["python_random_state"])
        if numpy_rng is not None:
            numpy_rng.bit_generator.state = payload["numpy_random_state"]
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["torch_cuda_rng_state"]])
    return payload
