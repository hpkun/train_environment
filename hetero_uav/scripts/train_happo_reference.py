"""Train minimal HAPPO reference v0 on the heterogeneous JSBSim env."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy, HAPPORolloutBuffer, HAPPOReferenceTrainer
from algorithms.mappo.opponent_policy import OpponentPolicy


DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
DEFAULT_EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
NUM_ENVS = 4


def _build_red_alive_mask(info: dict, env, red_ids: list[str]) -> np.ndarray:
    mask = np.zeros(len(red_ids), dtype=np.float32)
    for i, rid in enumerate(red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = env.red_planes.get(rid)
            alive = bool(sim is not None and sim.is_alive)
        mask[i] = 1.0 if alive else 0.0
    return mask


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def _alive_counts(env) -> tuple[int, int]:
    return (
        sum(1 for sim in env.red_planes.values() if sim.is_alive),
        sum(1 for sim in env.blue_planes.values() if sim.is_alive),
    )


def _mav_alive(env) -> bool:
    sim = env.red_planes.get("red_0")
    return bool(sim is not None and sim.is_alive)


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _episode_outcome(env, truncated: dict, length: int) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    timeout = bool(all(truncated.values()) or length >= getattr(env, "max_steps", 0))
    if blue_alive == 0 and red_alive > 0:
        winner, reason = "red", "blue_eliminated"
    elif red_alive == 0 and blue_alive > 0:
        winner, reason = "blue", "red_eliminated"
    elif red_alive == 0 and blue_alive == 0:
        winner, reason = "draw", "mutual_elimination"
    elif timeout:
        reason = "timeout"
        if red_alive > blue_alive:
            winner = "red"
        elif blue_alive > red_alive:
            winner = "blue"
        else:
            winner = "draw"
    else:
        winner, reason = "none", "ongoing"
    return {"winner": winner, "end_reason": reason}


def _run_eval(model_path: str, args, summary_json: str) -> list[dict] | None:
    cmd = [
        "python", "-u", str(ROOT / "scripts" / "eval_happo_reference.py"),
        "--model", model_path,
        "--episodes", str(args.train_eval_episodes),
        "--device", str(args.device),
        "--opponent-policy", args.opponent_policy,
        "--summary-json", summary_json,
        "--configs", *(args.eval_configs or DEFAULT_EVAL_CONFIGS),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=1200)
    if result.returncode != 0:
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        return None
    try:
        return json.loads((ROOT / summary_json).read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_eval(records: list[dict]) -> float:
    for record in records:
        if "3v2" in record.get("config", ""):
            return (
                record.get("red_win_rate", 0.0)
                + 0.1 * record.get("mav_survival_rate", 0.0)
                + 0.05 * record.get("blue_dead_mean", 0.0)
                + 0.05 * record.get("red_missile_hits_mean", 0.0)
            )
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="outputs/happo_reference")
    parser.add_argument("--total-env-steps", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule"])
    parser.add_argument("--reward-mode", default="happo_ref_v0")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eval-during-training", action="store_true")
    parser.add_argument("--eval-interval-steps", type=int, default=25000)
    parser.add_argument("--train-eval-episodes", type=int, default=1)
    parser.add_argument("--eval-configs", nargs="*", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()

    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / args.output_dir
    (out_dir / "latest").mkdir(parents=True, exist_ok=True)
    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    envs = [
        make_env(args.config, env_type="jsbsim_hetero",
                 hetero_reward_mode=args.reward_mode, max_steps=args.max_steps)
        for _ in range(NUM_ENVS)
    ]
    env = envs[0]
    adapter = HeteroObsAdapterV2()
    actor_dim = adapter.flat_actor_obs_dim
    critic_dim = adapter.critic_state_dim
    policy = HAPPOReferencePolicy(actor_dim, critic_dim).to(device)
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        policy.load(init_path, map_location=device)
        print(f"Loaded init_checkpoint: {init_path}", flush=True)
    trainer = HAPPOReferenceTrainer(
        policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        clip_param=args.clip_param, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
    )
    opponents = [
        OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17 + i)
        for i in range(NUM_ENVS)
    ]
    env_states = [e.reset(seed=args.seed + i) for i, e in enumerate(envs)]
    obs_list = [state[0] for state in env_states]
    info_list = [state[1] for state in env_states]
    roles = _role_ids(env)
    iterations = int(math.ceil(args.total_env_steps / args.rollout_length))
    total_steps = 0
    episodes = 0
    current_ep_return = [np.zeros(len(env.red_ids), dtype=np.float32) for _ in range(NUM_ENVS)]
    current_ep_len = [0 for _ in range(NUM_ENVS)]
    prev_hit_totals = [{"red": 0, "blue": 0} for _ in range(NUM_ENVS)]
    recent = deque(maxlen=100)
    best_score = -float("inf")
    nan_detected = False

    train_log = out_dir / "train_log.csv"
    with train_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "total_steps", "avg_return", "red_win", "blue_win",
            "draw", "timeout", "mav_survival", "red_alive_final",
            "blue_alive_final", "red_missiles_fired", "blue_missiles_fired",
            "missile_hits", "actor_loss_mav", "actor_loss_uav",
            "critic_loss", "entropy_mav", "entropy_uav",
            "mav_action_saturation_rate", "uav_action_saturation_rate",
            "nan_detected",
        ])
        eval_writer = None
        eval_f = None
        if args.eval_during_training:
            eval_f = (out_dir / "eval_log.csv").open("w", newline="", encoding="utf-8")
            eval_writer = csv.writer(eval_f)
            eval_writer.writerow([
                "total_steps", "iteration", "config", "red_win_rate",
                "blue_win_rate", "draw_rate", "timeout_rate",
                "mav_survival_rate", "blue_dead_mean", "red_missile_hits_mean",
            ])
        last_eval = -999999
        for iteration in range(1, iterations + 1):
            rollout_len = min(args.rollout_length, args.total_env_steps - total_steps)
            if rollout_len <= 0:
                break
            buffer = HAPPORolloutBuffer(rollout_len, len(env.red_ids), actor_dim,
                                        critic_dim, 3, roles)
            red_fired = blue_fired = hits = 0
            while len(buffer) < rollout_len and total_steps < args.total_env_steps:
                for env_idx, rollout_env in enumerate(envs):
                    if len(buffer) >= rollout_len or total_steps >= args.total_env_steps:
                        break
                    obs = obs_list[env_idx]
                    info = info_list[env_idx]
                    adapted = adapter.adapt_all(
                        obs, info=info, red_ids=rollout_env.red_ids, blue_ids=rollout_env.blue_ids)
                    actor_obs = np.stack([
                        adapted["actor_obs"].get(rid, np.zeros(actor_dim, dtype=np.float32))
                        for rid in rollout_env.red_ids
                    ])
                    critic = adapted["critic_state"]
                    active = _build_red_alive_mask(info, rollout_env, rollout_env.red_ids)
                    with torch.no_grad():
                        out = policy.act(
                            torch.as_tensor(actor_obs, device=device),
                            roles=roles,
                            critic_state=torch.as_tensor(critic, device=device),
                            deterministic=False,
                        )
                    actions = out["action"].cpu().numpy()
                    log_probs = out["log_prob"].cpu().numpy()
                    value = float(out["value"].item())
                    if np.isnan(actions).any() or np.isnan(value):
                        nan_detected = True
                        break
                    action_dict = {rid: actions[i].astype(np.float32)
                                   for i, rid in enumerate(rollout_env.red_ids)}
                    action_dict.update(opponents[env_idx].act(obs, rollout_env.blue_ids, env=rollout_env))
                    next_obs, rewards, terminated, truncated, next_info = rollout_env.step(action_dict)
                    reward_np = np.array([float(rewards.get(rid, 0.0)) for rid in rollout_env.red_ids], dtype=np.float32)
                    done = _team_done(terminated, truncated)
                    done_np = np.full((len(rollout_env.red_ids),), float(done), dtype=np.float32)
                    if done:
                        next_value = 0.0
                    else:
                        next_adapted = adapter.adapt_all(
                            next_obs, info=next_info, red_ids=rollout_env.red_ids, blue_ids=rollout_env.blue_ids)
                        with torch.no_grad():
                            next_value = float(policy.value(
                                torch.as_tensor(next_adapted["critic_state"], device=device).unsqueeze(0)
                            ).item())
                    buffer.store(
                        actor_obs, critic, actions, log_probs, reward_np, done_np,
                        value, active, next_value=next_value, env_id=env_idx)
                    current_ep_return[env_idx] += reward_np
                    current_ep_len[env_idx] += 1
                    total_steps += 1
                    for aid in rollout_env.agent_ids:
                        fired = int(next_info.get(aid, {}).get("missiles_fired_this_step", 0))
                        if aid.startswith("red_"):
                            red_fired += fired
                        else:
                            blue_fired += fired
                    mt = next_info.get("__missile_term__", {})
                    if isinstance(mt, dict):
                        red_hit_total = int(mt.get("red", {}).get("hit", 0))
                        blue_hit_total = int(mt.get("blue", {}).get("hit", 0))
                        hits += max(red_hit_total - prev_hit_totals[env_idx]["red"], 0)
                        hits += max(blue_hit_total - prev_hit_totals[env_idx]["blue"], 0)
                        prev_hit_totals[env_idx]["red"] = red_hit_total
                        prev_hit_totals[env_idx]["blue"] = blue_hit_total
                    if done:
                        outcome = _episode_outcome(rollout_env, truncated, current_ep_len[env_idx])
                        ra, ba = _alive_counts(rollout_env)
                        recent.append({
                            "return": float(current_ep_return[env_idx].mean()),
                            "winner": outcome["winner"],
                            "end_reason": outcome["end_reason"],
                            "mav": _mav_alive(rollout_env),
                            "red_alive": ra,
                            "blue_alive": ba,
                        })
                        episodes += 1
                        current_ep_return[env_idx][:] = 0.0
                        current_ep_len[env_idx] = 0
                        next_obs, next_info = rollout_env.reset(seed=args.seed + total_steps + env_idx)
                        prev_hit_totals[env_idx] = {"red": 0, "blue": 0}
                    obs_list[env_idx] = next_obs
                    info_list[env_idx] = next_info
                if nan_detected:
                    break
            if nan_detected:
                break
            stats = trainer.update(buffer)
            rec = list(recent)
            n = max(len(rec), 1)
            avg_return = float(np.mean([r["return"] for r in rec])) if rec else 0.0
            red_win = sum(1 for r in rec if r["winner"] == "red") / n
            blue_win = sum(1 for r in rec if r["winner"] == "blue") / n
            draw = sum(1 for r in rec if r["winner"] == "draw") / n
            timeout = sum(1 for r in rec if r["end_reason"] == "timeout") / n
            mav_surv = sum(1 for r in rec if r["mav"]) / n
            red_alive = float(np.mean([r["red_alive"] for r in rec])) if rec else 0.0
            blue_alive = float(np.mean([r["blue_alive"] for r in rec])) if rec else 0.0
            writer.writerow([
                iteration, total_steps, f"{avg_return:.4f}", f"{red_win:.4f}",
                f"{blue_win:.4f}", f"{draw:.4f}", f"{timeout:.4f}",
                f"{mav_surv:.4f}", f"{red_alive:.2f}", f"{blue_alive:.2f}",
                red_fired, blue_fired, hits, f"{stats['actor_loss_mav']:.6f}",
                f"{stats['actor_loss_uav']:.6f}", f"{stats['critic_loss']:.6f}",
                f"{stats['entropy_mav']:.6f}", f"{stats['entropy_uav']:.6f}",
                f"{stats['mav_action_saturation_rate']:.6f}",
                f"{stats['uav_action_saturation_rate']:.6f}", int(nan_detected),
            ])
            f.flush()
            print(
                f"[happo] iter={iteration:04d} steps={total_steps}/{args.total_env_steps} "
                f"ret={avg_return:+.2f} red_win={red_win:.2f} blue_win={blue_win:.2f} "
                f"mav_surv={mav_surv:.2f} blue_alive={blue_alive:.1f} "
                f"loss_mav={stats['actor_loss_mav']:.4f} loss_uav={stats['actor_loss_uav']:.4f}",
                flush=True,
            )
            if total_steps - last_eval >= args.eval_interval_steps and args.eval_during_training:
                last_eval = total_steps
                tmp_model = out_dir / "_tmp_eval.pt"
                policy.save(tmp_model)
                tmp_json = str((out_dir / "_tmp_eval.json").relative_to(ROOT))
                records = _run_eval(str(tmp_model), args, tmp_json)
                if records and eval_writer is not None:
                    for r in records:
                        eval_writer.writerow([
                            total_steps, iteration, r["config"], r["red_win_rate"],
                            r["blue_win_rate"], r["draw_rate"], r["timeout_rate"],
                            r["mav_survival_rate"], r["blue_dead_mean"],
                            r["red_missile_hits_mean"],
                        ])
                    eval_f.flush()
                    score = _score_eval(records)
                    if score > best_score:
                        best_score = score
                        policy.save(out_dir / "best" / "model.pt")
                        (out_dir / "best" / "meta.json").write_text(json.dumps({
                            "algorithm": "happo_reference_v0",
                            "reward_mode": args.reward_mode,
                            "opponent_policy": args.opponent_policy,
                            "best_score": best_score,
                            "actor_obs_dim": actor_dim,
                            "critic_state_dim": critic_dim,
                            "separate_actors": True,
                            "centralized_critic": True,
                            "sequential_update": True,
                            "attention": False,
                            "recurrent": False,
                            "num_envs": NUM_ENVS,
                            "init_checkpoint": args.init_checkpoint,
                        }, indent=2), encoding="utf-8")
                tmp_model.unlink(missing_ok=True)
                (out_dir / "_tmp_eval.json").unlink(missing_ok=True)
        if eval_f is not None:
            eval_f.close()

    latest_model = out_dir / "latest" / "model.pt"
    policy.save(latest_model)
    meta = {
        "algorithm": "happo_reference_v0",
        "config": args.config,
        "reward_mode": args.reward_mode,
        "opponent_policy": args.opponent_policy,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": critic_dim,
        "separate_actors": True,
        "centralized_critic": True,
        "sequential_update": True,
        "sequential_update_detail": "simplified HAPPO-style v0 role-wise PPO",
        "attention": False,
        "recurrent": False,
        "missile_scripted": True,
        "evasion_scripted": True,
        "num_envs": NUM_ENVS,
        "init_checkpoint": args.init_checkpoint,
        "total_env_steps_actual": total_steps,
        "episodes": episodes,
        "nan_detected": nan_detected,
    }
    (out_dir / "latest" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "main_experiment_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for rollout_env in envs:
        rollout_env.close()
    print(f"Saved {latest_model}", flush=True)


if __name__ == "__main__":
    main()
