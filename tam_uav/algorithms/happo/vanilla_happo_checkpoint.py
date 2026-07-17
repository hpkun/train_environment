"""Checkpoint and deterministic episode-boundary resume contract."""

from __future__ import annotations

import random
import numpy as np
import torch

from uav_env.JSBSim.paper.protocol import (
    BLUE_POLICY_FIDELITY, ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION,
    PAPER_NOMINAL_PROTOCOL, PAPER_SILENT_ASSUMPTIONS_PRESENT,
    REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED)
from uav_env.JSBSim.paper.action_semantics import NEUTRAL_ACTION_SEMANTICS


FORMAT = "tam_paper_heterogeneous_reward_vanilla_happo_v4"
REQUIRED_ENVIRONMENT_FIELDS = (
    "environment_fidelity_revision", "experiment_protocol", "initial_perturbation",
    "dynamics_backend", "paper_silent_assumptions_present", "scenario",
    "neutral_action_semantics", "blue_policy_fidelity",
    "reference_8_exact_blue_fsm_reproduced")


def read_vanilla_happo_checkpoint_metadata(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {key: value for key, value in payload.items()
            if key not in {"policy", "actor_optimizers", "critic_optimizer",
                           "python_random_state", "numpy_generator_state",
                           "numpy_global_state", "trainer_rng_state",
                           "torch_cpu_rng_state", "torch_cuda_rng_state"}}


def _validate_formal_config(config):
    missing = [key for key in REQUIRED_ENVIRONMENT_FIELDS if key not in config]
    if missing:
        raise ValueError(f"formal checkpoint config missing required fields: {missing}")
    expected = {
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "experiment_protocol": PAPER_NOMINAL_PROTOCOL,
        "initial_perturbation": NOMINAL_PERTURBATION,
        "dynamics_backend": "jsbsim",
        "paper_silent_assumptions_present": PAPER_SILENT_ASSUMPTIONS_PRESENT,
        "neutral_action_semantics": NEUTRAL_ACTION_SEMANTICS,
        "blue_policy_fidelity": BLUE_POLICY_FIDELITY,
        "reference_8_exact_blue_fsm_reproduced": (
            REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"formal checkpoint config {key} mismatch: {config.get(key)!r} != {value!r}")


def save_vanilla_happo_checkpoint(
        path, policy, trainer, *, environment_steps, episodes, config, numpy_rng,
        checkpoint_type="evaluation_weights", at_episode_boundary=False,
        policy_version=None, seed_schedule=None, extra_metadata=None,
        saved_at_update_boundary=False, discarded_partial_rollout_steps=0):
    _validate_formal_config(config)
    allowed = {"evaluation_weights", "resumable",
               "exact_update_and_episode_boundary", "episode_boundary_restart"}
    if checkpoint_type not in allowed:
        raise ValueError(f"unsupported checkpoint_type {checkpoint_type!r}")
    if checkpoint_type != "evaluation_weights" and not at_episode_boundary:
        raise ValueError("training checkpoints require episode_boundary=true")
    if (checkpoint_type == "exact_update_and_episode_boundary"
            and not saved_at_update_boundary):
        raise ValueError("exact checkpoint requires an update boundary")
    if checkpoint_type == "exact_update_and_episode_boundary":
        resume_semantics, exact_continuation = checkpoint_type, True
    elif checkpoint_type == "episode_boundary_restart":
        resume_semantics, exact_continuation = checkpoint_type, False
    elif checkpoint_type == "resumable":  # Legacy v4 compatibility.
        resume_semantics, exact_continuation = "episode_boundary", True
    else:
        resume_semantics, exact_continuation = "evaluation_only", False
    scenario = config.get("scenario")
    payload = {
        "format": FORMAT,
        "checkpoint_type": checkpoint_type,
        "at_episode_boundary": bool(at_episode_boundary),
        "resume_semantics": resume_semantics,
        "saved_at_episode_boundary": bool(at_episode_boundary),
        "saved_at_update_boundary": bool(saved_at_update_boundary),
        "discarded_partial_rollout_steps": int(discarded_partial_rollout_steps),
        "exact_training_continuation": bool(exact_continuation),
        "algorithm_mode": trainer.algorithm_mode,
        **{key: config[key] for key in REQUIRED_ENVIRONMENT_FIELDS},
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
    if extra_metadata:
        collisions = set(extra_metadata) & set(payload)
        if collisions:
            raise ValueError(
                f"extra checkpoint metadata collides with formal fields: {sorted(collisions)}")
        payload.update(dict(extra_metadata))
    torch.save(payload, path)
    return payload


def load_vanilla_happo_checkpoint(
        path, policy, trainer=None, *, numpy_rng=None, restore_rng=True,
        for_resume=False, allow_episode_restart=False, expected_scenario=None,
        expected_environment_fidelity_revision=None,
        expected_experiment_protocol=None, expected_initial_perturbation=None,
        expected_dynamics_backend=None,
        expected_paper_silent_assumptions_present=None):
    payload = torch.load(path, map_location=next(policy.parameters()).device,
                         weights_only=False)
    revision = payload.get("environment_fidelity_revision")
    if revision is None:
        raise ValueError("pre-fidelity checkpoint missing environment_fidelity_revision")
    if revision != ENVIRONMENT_FIDELITY_REVISION:
        raise ValueError(
            f"environment_fidelity_revision mismatch: {revision!r} != "
            f"{ENVIRONMENT_FIDELITY_REVISION!r}")
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    if payload.get("paper_silent_assumptions_present") is not True:
        raise ValueError(
            "checkpoint paper_silent_assumptions_present must be explicitly True")
    if for_resume:
        legacy_strict = (payload.get("checkpoint_type") == "resumable"
                         and payload.get("at_episode_boundary") is True)
        strict = (payload.get("checkpoint_type") ==
                  "exact_update_and_episode_boundary"
                  and payload.get("saved_at_episode_boundary") is True
                  and payload.get("saved_at_update_boundary") is True
                  and payload.get("exact_training_continuation") is True)
        restart = (payload.get("checkpoint_type") == "episode_boundary_restart"
                   and payload.get("saved_at_episode_boundary") is True)
        legacy_restart = (payload.get("checkpoint_type") == "evaluation_weights"
                          and allow_episode_restart)
        if not (strict or legacy_strict) and not (
                (restart and allow_episode_restart) or legacy_restart):
            raise ValueError(
                "strict resume requires an exact dual-boundary checkpoint or explicit "
                "--allow-episode-restart-resume for episode_boundary_restart")
        payload["resume_semantics"] = (
            "exact_update_and_episode_boundary" if strict else
            "episode_boundary" if legacy_strict else
            "episode_boundary_restart" if restart else "episode_restart")
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
    expected_lineage = {
        "environment_fidelity_revision": expected_environment_fidelity_revision,
        "experiment_protocol": expected_experiment_protocol,
        "initial_perturbation": expected_initial_perturbation,
        "dynamics_backend": expected_dynamics_backend,
        "paper_silent_assumptions_present": (
            expected_paper_silent_assumptions_present),
    }
    expected.update({key: value for key, value in expected_lineage.items()
                     if value is not None})
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
