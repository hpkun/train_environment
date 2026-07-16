"""Strict Pure HAPPO runner for the isolated formal 3v2 contract."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
from collections import defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy, PureHAPPOTrainer
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ACTION_DIM, ENV_TYPE
from uav_env.JSBSim.formal_v1.reward import (
    EVENT_REWARDS, GLOBAL_REWARD_SCALE, MAV_WEIGHTS, REWARD_CONTRACT_VERSION, UAV_WEIGHTS,
)
from uav_env.JSBSim.formal_v2.contract import (
    ENV_TYPE as V2_ENV_TYPE,
    OBSERVATION_CONTRACT as V2_OBSERVATION_CONTRACT,
    REWARD_CONTRACT_VERSION as V2_REWARD_CONTRACT_VERSION,
)
from uav_env.JSBSim.formal_v2.reward import (
    MAV_SAFETY_WEIGHTS as V2_MAV_SAFETY_WEIGHTS,
    MAV_SUPPORT_WEIGHTS as V2_MAV_SUPPORT_WEIGHTS,
    UAV_WEIGHTS as V2_UAV_WEIGHTS,
)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    parser.add_argument("--output-dir", default="outputs/formal_v1_smoke")
    parser.add_argument("--total-env-steps", type=int, default=2048)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--resume-from")
    return parser.parse_args(argv)


def _flat(obs, red_ids):
    return np.stack([obs[aid]["flat"] for aid in red_ids]).astype(np.float32)


def _save(policy, directory: Path, meta: dict):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), directory / "model.pt")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _training_hyperparameters(args) -> dict:
    return {
        "actor_lr": float(args.actor_lr),
        "critic_lr": float(args.critic_lr),
        "ppo_epochs": int(args.ppo_epochs),
        "critic_epochs": int(args.critic_epochs),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "entropy_coef": float(args.entropy_coef),
        "clip_param": float(args.clip_param),
        "max_grad_norm": float(args.max_grad_norm),
        "rollout_length": int(args.rollout_length),
    }


def _planned_rollout_count(requested_steps: int, rollout_length: int) -> int:
    if requested_steps <= 0 or rollout_length <= 0:
        raise ValueError("requested steps and rollout length must be positive")
    return int(math.ceil(requested_steps / rollout_length))


def _prepare_output_dir(path: Path, resume_from: Path | None) -> None:
    if resume_from is None:
        if path.exists() and any(path.iterdir()):
            raise ValueError(
                f"output directory is non-empty: {path}; use a new directory or --resume-from")
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.exists():
        raise ValueError(f"resume output directory does not exist: {path}")
    if not resume_from.is_file():
        raise ValueError(f"resume state does not exist: {resume_from}")


def _capture_rng_state(trainer) -> dict:
    return {
        "trainer_rng_state": copy.deepcopy(trainer.rng.bit_generator.state),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict, trainer) -> None:
    trainer.rng.bit_generator.state = copy.deepcopy(state["trainer_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    cuda_state = state.get("torch_cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _buffer_state(buffer: HAPPORolloutBuffer | None) -> dict | None:
    if buffer is None or len(buffer) == 0:
        return None
    return {key: copy.deepcopy(value) for key, value in buffer.__dict__.items()}


def _restore_buffer(state: dict | None) -> HAPPORolloutBuffer | None:
    if state is None:
        return None
    buffer = HAPPORolloutBuffer(
        state["max_len"], state["num_red"], state["actor_obs"].shape[-1],
        state["critic_state"].shape[-1], state["actions"].shape[-1], state["role_ids"],
        value_dim=state["value_dim"])
    for key, value in state.items():
        setattr(buffer, key, copy.deepcopy(value))
    return buffer


def _save_train_state(path: Path, policy, trainer, *, meta: dict, total_steps: int,
                      iteration: int, completed_episode_count: int,
                      requested_checkpoint_step: int | None,
                      buffer: HAPPORolloutBuffer | None,
                      rollout_stats: dict | None, rollout_counts: dict | None,
                      rollout_completed: list | None, episode_reset_seed: int,
                      saved_eval_steps: set[int], saved_resume_steps: set[int],
                      resumed: bool, extra: dict | None = None) -> None:
    state = {
        "policy_state_dict": policy.state_dict(),
        "actor_optimizer_state_dicts": [optimizer.state_dict() for optimizer in trainer.actor_opts],
        "critic_optimizer_state_dict": trainer.critic_opt.state_dict(),
        **_capture_rng_state(trainer),
        "total_env_steps": int(total_steps),
        "iteration": int(iteration),
        "completed_episode_count": int(completed_episode_count),
        "seed": int(meta["seed"]),
        "training_hyperparameters": dict(meta["training_hyperparameters"]),
        "formal_contract": meta["formal_contract"],
        "reward_contract": meta["reward_contract"],
        "observation_contract": meta.get("observation_contract"),
        "algorithm_contract": meta["algorithm_contract"],
        "actor_obs_dim": int(meta["actor_obs_dim"]),
        "critic_state_dim": int(meta["critic_state_dim"]),
        "action_dim": int(meta["action_dim"]),
        "num_agents": int(meta["num_agents"]),
        "run_identity": meta["run_identity"],
        "requested_checkpoint_step": requested_checkpoint_step,
        "checkpoint_actual_step": int(total_steps),
        "resume_semantics": "episode_boundary_exact_training_state",
        "episode_reset_seed": int(episode_reset_seed),
        "partial_rollout_buffer": _buffer_state(buffer),
        "partial_rollout_stats": copy.deepcopy(rollout_stats or {}),
        "partial_rollout_counts": dict(rollout_counts or {}),
        "partial_rollout_completed": copy.deepcopy(rollout_completed or []),
        "saved_evaluation_checkpoint_steps": sorted(saved_eval_steps),
        "saved_resume_checkpoint_steps": sorted(saved_resume_steps),
        "resumed": bool(resumed),
    }
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    state_meta = {
        **meta,
        "checkpoint_stage": "resume",
        "requested_checkpoint_step": requested_checkpoint_step,
        "total_env_steps_actual": int(total_steps),
        "iteration": int(iteration),
        "completed_episode_count": int(completed_episode_count),
        "resume_semantics": "episode_boundary_exact_training_state",
        "resumed": bool(resumed),
    }
    if extra:
        state_meta.update(extra)
    (path.parent / "meta.json").write_text(json.dumps(state_meta, indent=2), encoding="utf-8")


def _validate_resume_state(state: dict, meta: dict, output: Path) -> None:
    expected = {
        "formal_contract": meta["formal_contract"],
        "reward_contract": meta["reward_contract"],
        "algorithm_contract": meta["algorithm_contract"],
        "actor_obs_dim": meta["actor_obs_dim"],
        "critic_state_dim": meta["critic_state_dim"],
        "action_dim": meta["action_dim"],
        "num_agents": meta["num_agents"],
        "seed": meta["seed"],
        "training_hyperparameters": meta["training_hyperparameters"],
        "run_identity": str(output.resolve()),
    }
    if meta.get("observation_contract") is not None:
        expected["observation_contract"] = meta["observation_contract"]
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"incompatible resume state {key}: {state.get(key)!r} != {value!r}")
    if state.get("resume_semantics") != "episode_boundary_exact_training_state":
        raise ValueError("resume state is not an episode-boundary state")


def _last_record_step(path: Path, *, csv_file: bool) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"resume log is missing or empty: {path}")
    if csv_file:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return int(rows[-1]["total_steps"])
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return int(json.loads(lines[-1])["total_steps"])


def _write_resume_marker(writer, handle, fields: list[str], detail_log: Path,
                         iteration: int, total_steps: int) -> None:
    row = {key: 0 for key in fields}
    row.update({"record_type": "resume_boundary", "iteration": int(iteration),
                "total_steps": int(total_steps), "finite": 1})
    writer.writerow(row)
    handle.flush()
    with detail_log.open("a", encoding="utf-8") as detail_handle:
        detail_handle.write(json.dumps({
            "record_type": "resume_boundary",
            "iteration": int(iteration),
            "total_steps": int(total_steps),
        }) + "\n")


def _finite_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def _contract_meta(env) -> dict:
    common = {
        "formal_contract": env.formal_contract,
        "actor_obs_dim": int(env.actor_obs_dim),
        "critic_state_dim": int(env.critic_state_dim),
        "action_dim": int(env.action_dim),
        "num_agents": len(env.red_ids),
    }
    if env.formal_contract == V2_ENV_TYPE:
        return {
            **common,
            "observation_contract": V2_OBSERVATION_CONTRACT,
            "reward_contract": V2_REWARD_CONTRACT_VERSION,
            "reward_contract_details": {
                "uav_weights": V2_UAV_WEIGHTS,
                "mav_safety_weights": V2_MAV_SAFETY_WEIGHTS,
                "mav_support_weights": V2_MAV_SUPPORT_WEIGHTS,
                "event_rewards": EVENT_REWARDS,
            },
        }
    return {
        **common,
        "reward_contract": {
            "version": REWARD_CONTRACT_VERSION,
            "global_reward_scale": GLOBAL_REWARD_SCALE,
            "uav_weights": UAV_WEIGHTS,
            "mav_weights": MAV_WEIGHTS,
            "event_rewards": EVENT_REWARDS,
        },
    }


def main():
    args = _args()
    if args.rollout_length <= 0 or args.total_env_steps <= 0:
        raise ValueError("rollout length and total environment steps must be positive")
    if args.ppo_epochs <= 0 or args.critic_epochs <= 0:
        raise ValueError("PPO and critic epochs must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = Path(args.config)
    if not config.is_absolute(): config = ROOT / config
    output = Path(args.output_dir)
    if not output.is_absolute(): output = ROOT / output
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    _prepare_output_dir(output, resume_path)
    env = make_env(str(config))
    if env.formal_contract not in {ENV_TYPE, V2_ENV_TYPE} or env.action_dim != ACTION_DIM:
        raise ValueError("formal runner accepts only formal V1/V2 Box(3) contracts")
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA requested but unavailable: {requested_device}")
    device = torch.device(requested_device)
    policy = PureHAPPOPolicy(env.actor_obs_dim, env.critic_state_dim, ACTION_DIM,
                             len(env.red_ids), credit_mode="shared_alive_team_mean").to(device)
    trainer = PureHAPPOTrainer(
        policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        clip_param=args.clip_param, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
        critic_epochs=args.critic_epochs, gamma=args.gamma,
        gae_lambda=args.gae_lambda, seed=args.seed)
    requested_rollouts = _planned_rollout_count(args.total_env_steps, args.rollout_length)
    planned_actual_steps = requested_rollouts * args.rollout_length
    meta = {**_contract_meta(env), "policy_arch": "pure_happo",
            "credit_mode": "shared_alive_team_mean", "config": str(config),
            "seed": int(args.seed), "run_identity": str(output.resolve()),
            "requested_device": requested_device, "actual_device": str(device),
            "requested_total_env_steps": int(args.total_env_steps),
            "planned_actual_total_env_steps": int(planned_actual_steps),
            "training_hyperparameters": _training_hyperparameters(args),
            "algorithm_contract": ALGORITHM_CONTRACT,
            "policy_distribution": "tanh_squashed_gaussian_raw_action",
            "critic_contract": "centralized_shared_scalar_v",
            "gae_contract": "separated_termination_truncation"}
    is_v2 = env.formal_contract == V2_ENV_TYPE
    log_path = output / "train_log.csv"
    fields = [
        "record_type", "iteration", "total_steps", "episodes_completed", "avg_role_reward_mav",
        "avg_role_reward_uav", "mav_dense", "mav_safety", "mav_support_position",
        "mav_shared_information", "uav_dense", "uav_flight", "uav_speed", "uav_angle",
        "uav_distance", "uav_dodge", "team_event_reward", "red_launches", "blue_launches",
        "red_hits", "blue_hits", "red_kills", "blue_kills", "red_win", "blue_win",
        "mutual_elimination", "timeout", "mav_survival", "red_alive_final",
        "blue_alive_final", "flight_failures", "out_of_zone_deaths", "missile_deaths",
        "actor_loss", "entropy", "approx_kl", "critic_loss", "value_explained_variance",
        "approx_kl_abs", "approx_kl_mav", "approx_kl_uav",
        "approx_kl_abs_mav", "approx_kl_abs_uav",
        "final_approx_kl_abs_mav", "final_approx_kl_abs_uav",
        "clip_fraction_mav", "clip_fraction_uav", "ratio_p95_mav", "ratio_p95_uav",
        "ratio_p99_mav", "ratio_p99_uav", "policy_update_norm_mav",
        "policy_update_norm_uav", "actor_grad_norm_mav", "actor_grad_norm_uav",
        "agent_update_order", "final_ratio_p95_mav", "final_ratio_p95_uav",
        "final_ratio_p99_mav", "final_ratio_p99_uav", "factor_final_mean",
        "factor_final_std", "factor_final_min", "factor_final_max",
        "critic_grad_norm", "critic_update_norm", "value_explained_variance_old",
        "advantage_raw_mean", "advantage_raw_std", "advantage_norm_mean",
        "advantage_norm_std", "return_mean", "return_std", "terminated_count",
        "truncation_count", "episode_boundary_count", "action_log_std_mav_mean",
        "action_log_std_uav_mean",
        "red_geometry_samples", "red_range_rate", "red_ata_rate", "red_ta_rate",
        "red_geometry_rate", "action_saturation", "finite",
    ]
    if is_v2:
        fields.extend([
            "mav_safety_distance", "mav_safety_threat", "mav_safety_aspect",
            "mav_support", "mav_awareness", "mav_shared_information_metric",
            "mav_event", "mav_total", "uav_height", "uav_dodge_angle",
            "uav_dodge_speed", "uav_event", "uav_total",
        ])
    for role in ("mav", "uav"):
        for dimension in ("pitch", "heading", "speed"):
            fields.extend((f"{role}_action_mean_{dimension}",
                           f"{role}_action_std_{dimension}",
                           f"{role}_action_saturation_{dimension}"))
    for agent_idx in range(len(env.red_ids)):
        for position in ("before", "after"):
            for stat in ("mean", "std", "min", "max"):
                fields.append(f"factor_{position}_{stat}_agent{agent_idx}")
    detail_log = output / "update_metrics.jsonl"
    checkpoint_steps = sorted(set(step for step in args.checkpoint_steps if step > 0))
    resumed = resume_path is not None
    restored_buffer = None
    restored_stats = None
    restored_counts = None
    restored_completed = None
    if resumed:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        _validate_resume_state(state, meta, output)
        policy.load_state_dict(state["policy_state_dict"])
        for optimizer, optimizer_state in zip(
                trainer.actor_opts, state["actor_optimizer_state_dicts"]):
            optimizer.load_state_dict(optimizer_state)
        trainer.critic_opt.load_state_dict(state["critic_optimizer_state_dict"])
        total_steps = int(state["total_env_steps"])
        iteration = int(state["iteration"])
        completed_episode_count = int(state["completed_episode_count"])
        saved_checkpoint_steps = set(state.get("saved_evaluation_checkpoint_steps", []))
        saved_resume_steps = set(state.get("saved_resume_checkpoint_steps", []))
        restored_buffer = _restore_buffer(state.get("partial_rollout_buffer"))
        restored_stats = defaultdict(list, state.get("partial_rollout_stats", {}))
        restored_counts = defaultdict(int, state.get("partial_rollout_counts", {}))
        restored_completed = list(state.get("partial_rollout_completed", []))
        reset_seed = int(state["episode_reset_seed"])
        obs, info = env.reset(seed=reset_seed)
        _restore_rng_state(state, trainer)
        at_episode_start = True
        if _last_record_step(log_path, csv_file=True) != total_steps:
            raise ValueError("train_log.csv last step does not match resume state")
        if _last_record_step(detail_log, csv_file=False) != total_steps:
            raise ValueError("update_metrics.jsonl last step does not match resume state")
    else:
        _save(policy, output / "initial", {
            **meta, "checkpoint_stage": "initial", "total_env_steps_actual": 0,
            "iteration": 0, "resumed": False})
        obs, info = env.reset(seed=args.seed)
        at_episode_start = True
        total_steps = 0
        iteration = 0
        completed_episode_count = 0
        saved_checkpoint_steps = set()
        saved_resume_steps = set()

    log_mode = "a" if resumed else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not resumed:
            writer.writeheader()
        while iteration < requested_rollouts:
            length = args.rollout_length
            if restored_buffer is not None:
                buffer = restored_buffer
                stats = restored_stats
                counts = restored_counts
                completed = restored_completed
                restored_buffer = restored_stats = restored_counts = restored_completed = None
            else:
                buffer = HAPPORolloutBuffer(length, len(env.red_ids), env.actor_obs_dim,
                                            env.critic_state_dim, env.action_dim, [0, 1, 1])
                stats = defaultdict(list)
                counts = defaultdict(int)
                completed = []
            for _ in range(len(buffer), length):
                actor_obs = _flat(obs, env.red_ids)
                critic = np.asarray(info["critic_state"], np.float32)
                active = np.asarray(info["active_mask"], np.float32)
                with torch.no_grad():
                    result = policy.act(actor_obs, critic_state=critic)
                actions = result["action"].detach().cpu().numpy().astype(np.float32)
                actions *= active[:, None]
                next_obs, rewards, terms, truncs, next_info = env.step(
                    {aid: actions[i] for i, aid in enumerate(env.red_ids)})
                at_episode_start = False
                terminated = float(any(terms.values()))
                episode_done = float(terminated or any(truncs.values()))
                with torch.no_grad():
                    next_value = policy.value(next_info["critic_state"]).detach().cpu().numpy()
                reward_vec = np.asarray([rewards[aid] for aid in env.red_ids], np.float32)
                buffer.store(actor_obs, critic, actions,
                             result["log_prob"].detach().cpu().numpy(), reward_vec,
                             np.full(len(env.red_ids), episode_done, np.float32),
                             result["value"].detach().cpu().numpy(), active,
                             next_value=next_value,
                             raw_actions=result["raw_action"].detach().cpu().numpy(),
                             terminated=terminated, episode_done=episode_done)
                stats["role_mav"].append(float(reward_vec[0]))
                stats["role_uav"].append(float(reward_vec[1:].mean()))
                components = next_info["reward_components"]["per_agent"]
                mav = components["red_0"]
                for key in ("dense", "safety", "support_position", "shared_information"):
                    stats[f"mav_{key}"].append(float(mav.get(key, 0.0)))
                for key in ("dense", "flight", "speed", "angle", "distance", "dodge"):
                    stats[f"uav_{key}"].append(float(np.mean([
                        components[aid].get(key, 0.0) for aid in ("red_1", "red_2")])))
                if is_v2:
                    for key in (
                            "mav_safety_distance", "mav_safety_threat",
                            "mav_safety_aspect", "mav_support", "mav_awareness",
                            "mav_shared_information_metric", "mav_event", "mav_total"):
                        stats[key].append(float(mav.get(key, 0.0)))
                    for key in (
                            "uav_height", "uav_dodge_angle", "uav_dodge_speed",
                            "uav_event", "uav_total"):
                        stats[key].append(float(np.mean([
                            components[aid].get(key, 0.0)
                            for aid in ("red_1", "red_2")])))
                event_values = np.asarray([components[aid].get("event", 0.0) for aid in env.red_ids])
                stats["team_event"].append(float((event_values * active).sum() / max(active.sum(), 1.0)))
                stats["saturation"].append(float(np.mean(np.abs(actions[active > 0.5]) > 0.95))
                                            if np.any(active > 0.5) else 0.0)
                for event in next_info["step_events"]:
                    side = "red" if str(event.get("shooter_id", "")).startswith("red") else "blue"
                    if event.get("event") == "launch": counts[f"{side}_launches"] += 1
                    if event.get("event") == "hit":
                        counts[f"{side}_hits"] += 1; counts[f"{side}_kills"] += 1
                next_active = np.asarray(next_info["active_mask"], np.float32)
                for i, aid in enumerate(env.red_ids):
                    if active[i] > 0.5 and next_active[i] < 0.5:
                        reason = next_info["death_reasons"].get(aid, "")
                        counts["out_of_zone_deaths" if reason == "out_of_zone" else
                               ("missile_deaths" if reason == "missile_hit" else "flight_failures")] += 1
                for i, aid in enumerate(("red_1", "red_2"), start=1):
                    gate = next_info.get("fire_gates", {}).get(aid, {})
                    if active[i] > 0.5 and gate.get("observable", False):
                        counts["red_geometry_samples"] += 1
                        counts["red_range_ok"] += int(gate.get("range_ok", False))
                        counts["red_ata_ok"] += int(gate.get("ata_ok", False))
                        counts["red_ta_ok"] += int(gate.get("ta_ok", False))
                        counts["red_geometry_ok"] += int(gate.get("geometry_ok", False))
                obs, info = next_obs, next_info
                total_steps += 1
                if episode_done:
                    completed.append({"outcome": next_info["outcome"],
                                      "mav": float(next_info["mav_alive"]),
                                      "red_alive": float(next_info["red_alive"]),
                                      "blue_alive": float(next_info["blue_alive"])})
                    completed_episode_count += 1
                    episode_reset_seed = args.seed + total_steps
                    obs, info = env.reset(seed=episode_reset_seed)
                    at_episode_start = True
                    crossed_resume = [step for step in checkpoint_steps
                                      if step <= total_steps and step not in saved_resume_steps]
                    if crossed_resume:
                        _write_resume_marker(
                            writer, handle, fields, detail_log, iteration, total_steps)
                        for requested_step in crossed_resume:
                            saved_resume_steps.add(requested_step)
                            state_args = dict(
                                meta=meta, total_steps=total_steps, iteration=iteration,
                                completed_episode_count=completed_episode_count,
                                requested_checkpoint_step=requested_step, buffer=buffer,
                                rollout_stats=dict(stats), rollout_counts=dict(counts),
                                rollout_completed=completed,
                                episode_reset_seed=episode_reset_seed,
                                saved_eval_steps=saved_checkpoint_steps,
                                saved_resume_steps=saved_resume_steps, resumed=resumed)
                            _save_train_state(
                                output / "resume_checkpoints" / f"step_{requested_step:06d}"
                                / "train_state.pt", policy, trainer, **state_args)
                            _save_train_state(
                                output / "latest_resume" / "train_state.pt",
                                policy, trainer, **state_args)
            metrics = trainer.update(buffer)
            iteration += 1
            mean = lambda key: float(np.mean(stats[key])) if stats[key] else 0.0
            episode_mean = lambda key: (float(np.mean([x[key] for x in completed]))
                                        if completed else 0.0)
            metric = lambda key: _finite_float(metrics.get(key, 0.0))
            geometry_samples = max(counts["red_geometry_samples"], 1)
            kl_abs_values = metrics.get("approx_kl_abs_per_agent", [])
            row = {"record_type": "update", "iteration": iteration,
                   "total_steps": total_steps,
                   "episodes_completed": len(completed),
                   "avg_role_reward_mav": mean("role_mav"), "avg_role_reward_uav": mean("role_uav"),
                   "mav_dense": mean("mav_dense"), "mav_safety": mean("mav_safety"),
                   "mav_support_position": mean("mav_support_position"),
                   "mav_shared_information": mean("mav_shared_information"),
                   "uav_dense": mean("uav_dense"), "uav_flight": mean("uav_flight"),
                   "uav_speed": mean("uav_speed"), "uav_angle": mean("uav_angle"),
                   "uav_distance": mean("uav_distance"), "uav_dodge": mean("uav_dodge"),
                   "team_event_reward": mean("team_event"),
                   **{key: counts[key] for key in ("red_launches", "blue_launches", "red_hits",
                       "blue_hits", "red_kills", "blue_kills", "flight_failures",
                       "out_of_zone_deaths", "missile_deaths")},
                   "red_win": sum(x["outcome"] == "red_win" for x in completed),
                   "blue_win": sum(x["outcome"] == "blue_win" for x in completed),
                   "mutual_elimination": sum(x["outcome"] == "mutual_elimination" for x in completed),
                   "timeout": sum(x["outcome"] == "draw" for x in completed),
                   "mav_survival": episode_mean("mav"), "red_alive_final": episode_mean("red_alive"),
                   "blue_alive_final": episode_mean("blue_alive"),
                   "actor_loss": float(metrics.get("actor_loss_mean", 0.0)),
                   "entropy": float(metrics.get("entropy_mean", 0.0)),
                   "approx_kl": float(metrics.get("approx_kl_mean", 0.0)),
                   "critic_loss": float(metrics.get("critic_loss", 0.0)),
                   "value_explained_variance": float(metrics.get("value_explained_variance", 0.0)),
                   "approx_kl_abs": _finite_float(np.mean(kl_abs_values) if kl_abs_values else 0.0),
                   **{key: metric(key) for key in (
                       "approx_kl_mav", "approx_kl_uav", "approx_kl_abs_mav",
                       "approx_kl_abs_uav", "final_approx_kl_abs_mav",
                       "final_approx_kl_abs_uav", "clip_fraction_mav",
                       "clip_fraction_uav", "ratio_p95_mav", "ratio_p95_uav",
                       "ratio_p99_mav", "ratio_p99_uav", "policy_update_norm_mav",
                       "policy_update_norm_uav", "actor_grad_norm_mav", "actor_grad_norm_uav")},
                   "agent_update_order": int("".join(
                       str(value) for value in metrics["agent_update_order"])),
                   **{key: metric(key) for key in (
                       "final_ratio_p95_mav", "final_ratio_p95_uav",
                       "final_ratio_p99_mav", "final_ratio_p99_uav",
                       "critic_grad_norm", "critic_update_norm",
                       "value_explained_variance_old", "advantage_raw_mean",
                       "advantage_raw_std", "advantage_norm_mean", "advantage_norm_std",
                       "return_mean", "return_std", "terminated_count", "truncation_count",
                       "episode_boundary_count", "action_log_std_mav_mean",
                       "action_log_std_uav_mean")},
                   "factor_final_mean": _finite_float(metrics["final_factor"]["mean"]),
                   "factor_final_std": _finite_float(metrics["final_factor"]["std"]),
                   "factor_final_min": _finite_float(metrics["final_factor"]["min"]),
                   "factor_final_max": _finite_float(metrics["final_factor"]["max"]),
                   "red_geometry_samples": counts["red_geometry_samples"],
                   "red_range_rate": counts["red_range_ok"] / geometry_samples,
                   "red_ata_rate": counts["red_ata_ok"] / geometry_samples,
                   "red_ta_rate": counts["red_ta_ok"] / geometry_samples,
                   "red_geometry_rate": counts["red_geometry_ok"] / geometry_samples,
                   "action_saturation": mean("saturation"),
                   "finite": int(all(torch.isfinite(parameter).all().item()
                                     for parameter in policy.parameters()))}
            if is_v2:
                row.update({
                    key: mean(key) for key in (
                        "mav_safety_distance", "mav_safety_threat",
                        "mav_safety_aspect", "mav_support", "mav_awareness",
                        "mav_shared_information_metric", "mav_event", "mav_total",
                        "uav_height", "uav_dodge_angle", "uav_dodge_speed",
                        "uav_event", "uav_total",
                    )
                })
            for role in ("mav", "uav"):
                for dimension in ("pitch", "heading", "speed"):
                    row[f"{role}_action_mean_{dimension}"] = metric(
                        f"{role}_action_mean_{dimension}_active")
                    row[f"{role}_action_std_{dimension}"] = metric(
                        f"{role}_action_std_{dimension}_active")
                    row[f"{role}_action_saturation_{dimension}"] = metric(
                        f"{role}_action_saturation_{dimension}_active")
            for agent_idx in range(len(env.red_ids)):
                for position in ("before", "after"):
                    factor_stats = metrics[f"factor_{position}_per_agent"][agent_idx]
                    for stat in ("mean", "std", "min", "max"):
                        row[f"factor_{position}_{stat}_agent{agent_idx}"] = _finite_float(
                            factor_stats.get(stat, 0.0))
            if not all(np.isfinite(float(value)) for key, value in row.items()
                       if key != "record_type"):
                raise ValueError(f"non-finite formal training log row at step {total_steps}")
            writer.writerow(row); handle.flush()
            with detail_log.open("a", encoding="utf-8") as detail_handle:
                detail_handle.write(json.dumps({
                    "record_type": "update", "iteration": iteration,
                    "total_steps": total_steps,
                    "agent_update_order": [int(value) for value in metrics["agent_update_order"]],
                    "actor_update_trace": metrics["actor_update_trace"],
                    "actor_loss_epochs_per_agent": metrics["actor_loss_epochs_per_agent"],
                    "entropy_epochs_per_agent": metrics["entropy_epochs_per_agent"],
                    "approx_kl_epochs_per_agent": metrics["approx_kl_epochs_per_agent"],
                    "approx_kl_abs_epochs_per_agent": metrics["approx_kl_abs_epochs_per_agent"],
                    "clip_fraction_epochs_per_agent": metrics["clip_fraction_epochs_per_agent"],
                    "actor_loss_epoch_summary_per_agent": metrics[
                        "actor_loss_epoch_summary_per_agent"],
                    "entropy_epoch_summary_per_agent": metrics[
                        "entropy_epoch_summary_per_agent"],
                    "approx_kl_epoch_summary_per_agent": metrics[
                        "approx_kl_epoch_summary_per_agent"],
                    "approx_kl_abs_epoch_summary_per_agent": metrics[
                        "approx_kl_abs_epoch_summary_per_agent"],
                    "clip_fraction_epoch_summary_per_agent": metrics[
                        "clip_fraction_epoch_summary_per_agent"],
                    "factor_before_per_agent": metrics["factor_before_per_agent"],
                    "factor_after_per_agent": metrics["factor_after_per_agent"],
                    "final_factor": metrics["final_factor"],
                    "final_ratio_mean_per_agent": metrics["final_ratio_mean_per_agent"],
                    "final_ratio_std_per_agent": metrics["final_ratio_std_per_agent"],
                    "final_ratio_p95_per_agent": metrics["final_ratio_p95_per_agent"],
                    "final_ratio_p99_per_agent": metrics["final_ratio_p99_per_agent"],
                }) + "\n")
            for requested_step in checkpoint_steps:
                if requested_step <= total_steps and requested_step not in saved_checkpoint_steps:
                    checkpoint_meta = {
                        **meta, "checkpoint_stage": "periodic",
                        "requested_checkpoint_step": requested_step,
                        "total_env_steps_actual": total_steps,
                        "iteration": iteration,
                        "resumed": resumed,
                    }
                    _save(policy, output / "checkpoints" / f"step_{requested_step:06d}",
                          checkpoint_meta)
                    saved_checkpoint_steps.add(requested_step)
            print(f"[{env.formal_contract}] it={iteration:04d} "
                  f"steps={total_steps}/{args.total_env_steps} "
                  f"reward:M/U={row['avg_role_reward_mav']:+.3f}/{row['avg_role_reward_uav']:+.3f} "
                  f"launch:R/B={row['red_launches']}/{row['blue_launches']}", flush=True)
    _save(policy, output / "latest", {
        **meta, "checkpoint_stage": "latest",
        "total_env_steps_actual": total_steps, "iteration": iteration,
        "completed_episode_count": completed_episode_count,
        "resumed": resumed,
    })

    discarded_steps = 0
    if not at_episode_start:
        while True:
            actor_obs = _flat(obs, env.red_ids)
            with torch.no_grad():
                result = policy.act(actor_obs, critic_state=info["critic_state"],
                                    deterministic=True)
            actions = result["action"].detach().cpu().numpy().astype(np.float32)
            actions *= np.asarray(info["active_mask"], np.float32)[:, None]
            obs, _reward, terms, truncs, info = env.step(
                {aid: actions[i] for i, aid in enumerate(env.red_ids)})
            discarded_steps += 1
            if any(terms.values()) or any(truncs.values()):
                episode_reset_seed = args.seed + total_steps + discarded_steps
                obs, info = env.reset(seed=episode_reset_seed)
                at_episode_start = True
                break
    else:
        episode_reset_seed = args.seed + total_steps

    with log_path.open("a", newline="", encoding="utf-8") as final_handle:
        final_writer = csv.DictWriter(final_handle, fieldnames=fields)
        _write_resume_marker(
            final_writer, final_handle, fields, detail_log, iteration, total_steps)
    final_resume_steps = [step for step in checkpoint_steps
                          if step <= total_steps and step not in saved_resume_steps]
    if final_resume_steps:
        saved_resume_steps.update(final_resume_steps)
    final_requested = max(final_resume_steps or saved_resume_steps or [0])
    final_extra = {
        "discarded_post_training_steps_to_boundary": int(discarded_steps),
        "resume_boundary_interaction_step": int(total_steps + discarded_steps),
        "checkpoint_actual_step": int(total_steps + discarded_steps),
    }
    final_state_args = dict(
        meta=meta, total_steps=total_steps, iteration=iteration,
        completed_episode_count=completed_episode_count,
        requested_checkpoint_step=final_requested, buffer=None,
        rollout_stats=None, rollout_counts=None, rollout_completed=None,
        episode_reset_seed=episode_reset_seed,
        saved_eval_steps=saved_checkpoint_steps,
        saved_resume_steps=saved_resume_steps, resumed=resumed, extra=final_extra)
    for requested_step in final_resume_steps:
        _save_train_state(
            output / "resume_checkpoints" / f"step_{requested_step:06d}" / "train_state.pt",
            policy, trainer, **{**final_state_args, "requested_checkpoint_step": requested_step})
    _save_train_state(
        output / "latest_resume" / "train_state.pt", policy, trainer, **final_state_args)
    env.close()


if __name__ == "__main__":
    main()
