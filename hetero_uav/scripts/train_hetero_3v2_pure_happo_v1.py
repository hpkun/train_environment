"""Strict Pure HAPPO runner for the isolated formal 3v2 contract."""
from __future__ import annotations

import argparse
import csv
import json
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
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ACTION_DIM, ENV_TYPE
from uav_env.JSBSim.formal_v1.reward import (
    EVENT_REWARDS, GLOBAL_REWARD_SCALE, MAV_WEIGHTS, REWARD_CONTRACT_VERSION, UAV_WEIGHTS,
)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    parser.add_argument("--output-dir", default="outputs/formal_v1_smoke")
    parser.add_argument("--total-env-steps", type=int, default=2048)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[])
    return parser.parse_args()


def _flat(obs, red_ids):
    return np.stack([obs[aid]["flat"] for aid in red_ids]).astype(np.float32)


def _save(policy, directory: Path, meta: dict):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), directory / "model.pt")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _finite_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


def main():
    args = _args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = Path(args.config)
    if not config.is_absolute(): config = ROOT / config
    env = make_env(str(config))
    if env.config.get("env_type") != ENV_TYPE or env.action_dim != ACTION_DIM:
        raise ValueError("formal runner accepts only hetero_3v2_pure_happo_v1 Box(3)")
    device = torch.device(args.device)
    policy = PureHAPPOPolicy(env.actor_obs_dim, env.critic_state_dim, ACTION_DIM,
                             len(env.red_ids), credit_mode="shared_alive_team_mean").to(device)
    trainer = PureHAPPOTrainer(policy, actor_lr=5e-4, critic_lr=5e-4, clip_param=0.2,
                               entropy_coef=0.01, max_grad_norm=10.0, ppo_epochs=5,
                               gamma=0.99, gae_lambda=0.95, seed=args.seed)
    output = Path(args.output_dir)
    if not output.is_absolute(): output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    meta = {"formal_contract": ENV_TYPE, "policy_arch": "pure_happo",
            "credit_mode": "shared_alive_team_mean", "actor_obs_dim": env.actor_obs_dim,
            "critic_state_dim": env.critic_state_dim, "action_dim": ACTION_DIM,
            "num_agents": len(env.red_ids), "config": str(config)}
    meta["reward_contract"] = {
        "version": REWARD_CONTRACT_VERSION,
        "global_reward_scale": GLOBAL_REWARD_SCALE,
        "uav_weights": UAV_WEIGHTS, "mav_weights": MAV_WEIGHTS,
        "event_rewards": EVENT_REWARDS,
    }
    _save(policy, output / "initial", {**meta, "total_env_steps_actual": 0})
    log_path = output / "train_log.csv"
    fields = [
        "iteration", "total_steps", "episodes_completed", "avg_role_reward_mav",
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
        "red_geometry_samples", "red_range_rate", "red_ata_rate", "red_ta_rate",
        "red_geometry_rate", "action_saturation", "finite",
    ]
    for role in ("mav", "uav"):
        for dimension in ("pitch", "heading", "speed"):
            fields.extend((f"{role}_action_mean_{dimension}",
                           f"{role}_action_std_{dimension}",
                           f"{role}_action_saturation_{dimension}"))
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        obs, info = env.reset(seed=args.seed)
        total_steps = 0; iteration = 0
        pending_checkpoints = sorted(set(
            step for step in args.checkpoint_steps if 0 < step <= args.total_env_steps))
        saved_checkpoint_steps: set[int] = set()
        while total_steps < args.total_env_steps:
            length = min(args.rollout_length, args.total_env_steps - total_steps)
            buffer = HAPPORolloutBuffer(length, len(env.red_ids), env.actor_obs_dim,
                                        env.critic_state_dim, ACTION_DIM, [0, 1, 1])
            stats = defaultdict(list); counts = defaultdict(int); completed = []
            for _ in range(length):
                actor_obs = _flat(obs, env.red_ids)
                critic = np.asarray(info["critic_state"], np.float32)
                active = np.asarray(info["active_mask"], np.float32)
                with torch.no_grad():
                    result = policy.act(actor_obs, critic_state=critic)
                actions = result["action"].detach().cpu().numpy().astype(np.float32)
                actions *= active[:, None]
                next_obs, rewards, terms, truncs, next_info = env.step(
                    {aid: actions[i] for i, aid in enumerate(env.red_ids)})
                done = float(next_info["team_done"])
                with torch.no_grad():
                    next_value = policy.value(next_info["critic_state"]).detach().cpu().numpy()
                reward_vec = np.asarray([rewards[aid] for aid in env.red_ids], np.float32)
                buffer.store(actor_obs, critic, actions,
                             result["log_prob"].detach().cpu().numpy(), reward_vec,
                             np.full(len(env.red_ids), done, np.float32),
                             result["value"].detach().cpu().numpy(), active,
                             next_value=next_value)
                stats["role_mav"].append(float(reward_vec[0]))
                stats["role_uav"].append(float(reward_vec[1:].mean()))
                components = next_info["reward_components"]["per_agent"]
                mav = components["red_0"]
                for key in ("dense", "safety", "support_position", "shared_information"):
                    stats[f"mav_{key}"].append(float(mav.get(key, 0.0)))
                for key in ("dense", "flight", "speed", "angle", "distance", "dodge"):
                    stats[f"uav_{key}"].append(float(np.mean([
                        components[aid].get(key, 0.0) for aid in ("red_1", "red_2")])))
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
                if done:
                    completed.append({"outcome": next_info["outcome"],
                                      "mav": float(next_info["mav_alive"]),
                                      "red_alive": float(next_info["red_alive"]),
                                      "blue_alive": float(next_info["blue_alive"])})
                    obs, info = env.reset(seed=args.seed + total_steps)
            metrics = trainer.update(buffer)
            iteration += 1
            mean = lambda key: float(np.mean(stats[key])) if stats[key] else 0.0
            episode_mean = lambda key: (float(np.mean([x[key] for x in completed]))
                                        if completed else 0.0)
            metric = lambda key: _finite_float(metrics.get(key, 0.0))
            geometry_samples = max(counts["red_geometry_samples"], 1)
            kl_abs_values = metrics.get("approx_kl_abs_per_agent", [])
            row = {"iteration": iteration, "total_steps": total_steps,
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
                   "red_geometry_samples": counts["red_geometry_samples"],
                   "red_range_rate": counts["red_range_ok"] / geometry_samples,
                   "red_ata_rate": counts["red_ata_ok"] / geometry_samples,
                   "red_ta_rate": counts["red_ta_ok"] / geometry_samples,
                   "red_geometry_rate": counts["red_geometry_ok"] / geometry_samples,
                   "action_saturation": mean("saturation"),
                   "finite": int(all(torch.isfinite(parameter).all().item()
                                     for parameter in policy.parameters()))}
            for role in ("mav", "uav"):
                for dimension in ("pitch", "heading", "speed"):
                    row[f"{role}_action_mean_{dimension}"] = metric(
                        f"{role}_action_mean_{dimension}_active")
                    row[f"{role}_action_std_{dimension}"] = metric(
                        f"{role}_action_std_{dimension}_active")
                    row[f"{role}_action_saturation_{dimension}"] = metric(
                        f"{role}_action_saturation_{dimension}_active")
            if not all(np.isfinite(float(value)) for value in row.values()):
                raise ValueError(f"non-finite formal training log row at step {total_steps}")
            writer.writerow(row); handle.flush()
            for requested_step in pending_checkpoints:
                if requested_step <= total_steps and requested_step not in saved_checkpoint_steps:
                    checkpoint_meta = {
                        **meta, "checkpoint_stage": "periodic",
                        "requested_checkpoint_step": requested_step,
                        "total_env_steps_actual": total_steps,
                        "iteration": iteration,
                    }
                    _save(policy, output / "checkpoints" / f"step_{requested_step:06d}",
                          checkpoint_meta)
                    saved_checkpoint_steps.add(requested_step)
            print(f"[formal-v1] it={iteration:04d} steps={total_steps}/{args.total_env_steps} "
                  f"reward:M/U={row['avg_role_reward_mav']:+.3f}/{row['avg_role_reward_uav']:+.3f} "
                  f"launch:R/B={row['red_launches']}/{row['blue_launches']}", flush=True)
    _save(policy, output / "latest", {**meta, "total_env_steps_actual": total_steps})
    env.close()


if __name__ == "__main__":
    main()
