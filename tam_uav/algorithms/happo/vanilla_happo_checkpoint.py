"""Checkpoint and deterministic episode-boundary resume contract."""

from __future__ import annotations

import random
import numpy as np
import torch


FORMAT = "tam_paper_heterogeneous_reward_vanilla_happo_v2"


def read_vanilla_happo_checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    return {key: value for key, value in payload.items()
            if key not in {"policy", "actor_optimizers", "critic_optimizer",
                           "python_random_state", "numpy_generator_state",
                           "numpy_global_state", "trainer_rng_state",
                           "torch_cpu_rng_state", "torch_cuda_rng_state"}}


def save_vanilla_happo_checkpoint(
        path, policy, trainer, *, environment_steps, episodes, config, numpy_rng,
        checkpoint_type="evaluation_weights", at_episode_boundary=False,
        policy_version=None, seed_schedule=None):
    if checkpoint_type not in {"evaluation_weights", "resumable"}:
        raise ValueError("checkpoint_type must be evaluation_weights or resumable")
    if checkpoint_type == "resumable" and not at_episode_boundary:
        raise ValueError("resumable checkpoints require at_episode_boundary=true")
    scenario = config.get("scenario")
    payload = {
        "format": FORMAT,
        "checkpoint_type": checkpoint_type,
        "at_episode_boundary": bool(at_episode_boundary),
        "resume_semantics": ("episode_boundary" if checkpoint_type == "resumable"
                             else "evaluation_only"),
        "algorithm_mode": trainer.algorithm_mode,
        "policy": policy.state_dict(),
        "actor_optimizers": {key: value.state_dict()
                             for key, value in trainer.actor_optimizers.items()},
        "critic_optimizer": (trainer.critic_optimizer.state_dict()
                             if hasattr(trainer, "critic_optimizer") else None),
        "trainer_update_count": int(getattr(trainer, "update_count", 0)),
        "policy_version": int(policy_version if policy_version is not None
                              else getattr(trainer, "update_count", 0)),
        "environment_steps": int(environment_steps),
        "episodes": int(episodes),
        "python_random_state": random.getstate(),
        "numpy_generator_state": numpy_rng.bit_generator.state,
        "numpy_global_state": np.random.get_state(),
        "trainer_rng_state": getattr(trainer, "rng", numpy_rng).bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (torch.cuda.get_rng_state_all()
                                 if torch.cuda.is_available() else None),
        "config": dict(config),
        "scenario": scenario,
        "seed_schedule": dict(seed_schedule or {}),
        "agent_ids": list(policy.agent_ids),
        "agent_roles": dict(policy.role_by_agent),
        "agent_actor_mapping": {aid: policy.actor_key(aid) for aid in policy.agent_ids},
        "agent_critic_mapping": {aid: aid for aid in policy.agent_ids},
        "actor_obs_dim": policy.actor_obs_dim,
        "critic_state_dim": policy.critic_state_dim,
        "action_spec": [policy.action_levels] * policy.action_dim,
        "actor_sharing": policy.actor_sharing,
        "hidden_dim": policy.hidden_dim,
        "critic_architecture": "shared_centralized_backbone_independent_agent_heads",
        "critic_head_ids": list(policy.critic.heads.keys()),
    }
    torch.save(payload, path)
    return payload


def load_vanilla_happo_checkpoint(
        path, policy, trainer=None, *, numpy_rng=None, restore_rng=True,
        for_resume=False, allow_episode_restart=False, expected_scenario=None):
    payload = torch.load(path, map_location=next(policy.parameters()).device,
                         weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    if for_resume:
        strict = (payload.get("checkpoint_type") == "resumable"
                  and payload.get("at_episode_boundary") is True)
        if not strict and not allow_episode_restart:
            raise ValueError(
                "strict resume requires a resumable checkpoint saved at an episode boundary")
        payload["resume_semantics"] = "episode_boundary" if strict else "episode_restart"
    expected = {
        "agent_ids": list(policy.agent_ids),
        "agent_roles": dict(policy.role_by_agent),
        "actor_obs_dim": policy.actor_obs_dim,
        "critic_state_dim": policy.critic_state_dim,
        "action_spec": [policy.action_levels] * policy.action_dim,
        "actor_sharing": policy.actor_sharing,
        "hidden_dim": policy.hidden_dim,
        "critic_head_ids": list(policy.critic.heads.keys()),
    }
    if expected_scenario is not None:
        expected["scenario"] = expected_scenario
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch: {payload.get(key)!r} != {value!r}")
    policy.load_state_dict(payload["policy"])
    if trainer is not None:
        if payload.get("algorithm_mode") != trainer.algorithm_mode:
            raise ValueError("checkpoint algorithm mode does not match trainer")
        for key, state in payload["actor_optimizers"].items():
            trainer.actor_optimizers[key].load_state_dict(state)
        if payload.get("critic_optimizer") is not None:
            trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        trainer.update_count = int(payload["trainer_update_count"])
        trainer.rng.bit_generator.state = payload["trainer_rng_state"]
    if restore_rng:
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_global_state"])
        if numpy_rng is not None:
            numpy_rng.bit_generator.state = payload["numpy_generator_state"]
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda_rng_state"]])
    return payload
