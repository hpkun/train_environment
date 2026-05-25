"""Batch evaluation for MAPPO-Attention actors without Tacview output."""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from attention_models import AttentionActor
from my_uav_env import UavCombatEnv
from my_uav_env.alignment.reward_utils import REWARD_VERSION
from rule_based_agent import blue_coordinated_actions
from train_attention_mappo import _build_attention_entities
from train_vanilla_mappo import (
    _classify_death_reason,
    _episode_outcome,
    _safe_div,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO-Attention actor over multiple episodes.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num-red", type=int, default=2)
    parser.add_argument("--num-blue", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--output", type=str,
                        default="results/eval_attention_mappo.csv")
    parser.add_argument("--obs-adapter", type=str,
                        choices=("current", "paper-placeholder"),
                        default="current")
    parser.add_argument("--enable-blue-gcas", action="store_true", default=False)
    return parser.parse_args()


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] --device cuda requested but CUDA is unavailable; "
              "falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expected_entity_dim(obs_adapter: str) -> int:
    if obs_adapter == "current":
        return 11
    if obs_adapter == "paper-placeholder":
        return 10
    raise ValueError(f"Unknown obs_adapter: {obs_adapter}")


def _resolve_checkpoint(path: str | None) -> str | None:
    if path:
        return path
    candidates = [
        os.path.join("checkpoints_attention", "attention_actor_best.pt"),
        os.path.join("checkpoints_attention", "attention_actor_final.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_actor_shapes(state: dict):
    entity_dim = None
    hidden_size = 128
    rnn_hidden = 128
    for key, tensor in state.items():
        if key == "encoder.entity_mlp.0.weight":
            hidden_size = int(tensor.shape[0])
            entity_dim = int(tensor.shape[1])
        elif key == "rnn.weight_ih":
            rnn_hidden = int(tensor.shape[0] // 3)
    return entity_dim, hidden_size, rnn_hidden


def _load_actor(args, device: torch.device):
    expected_entity_dim = _expected_entity_dim(args.obs_adapter)
    if args.random:
        print("[INFO] --random set; red team uses random actions.", flush=True)
        return None, 128, expected_entity_dim, None

    checkpoint = _resolve_checkpoint(args.checkpoint)
    if checkpoint is None:
        print("[WARN] No attention checkpoint found; red team uses random actions.",
              flush=True)
        return None, 128, expected_entity_dim, None

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_entity_dim, hidden_size, rnn_hidden = _infer_actor_shapes(state)
    if ckpt_entity_dim != expected_entity_dim:
        raise SystemExit(
            "ERROR: attention checkpoint entity_dim does not match obs_adapter.\n"
            f"  checkpoint entity_dim: {ckpt_entity_dim}\n"
            f"  obs_adapter:           {args.obs_adapter}\n"
            f"  expected entity_dim:   {expected_entity_dim}"
        )

    actor = AttentionActor(
        entity_dim=expected_entity_dim,
        action_dim=3,
        hidden_size=hidden_size,
        rnn_hidden=rnn_hidden,
    ).to(device)
    actor.load_state_dict(state)
    actor.eval()
    print(f"[INFO] Loaded attention actor checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Actor shape: entity_dim={expected_entity_dim}, "
          f"hidden={hidden_size}, rnn_hidden={rnn_hidden}", flush=True)
    return actor, rnn_hidden, expected_entity_dim, checkpoint


def _death_counts(death_reasons: dict[str, str], ids: list[str]) -> Counter:
    counts = Counter()
    for aid in ids:
        reason = death_reasons.get(aid)
        if reason:
            counts[_classify_death_reason(reason)] += 1
    return counts


def run_one_episode(actor, rnn_hidden_size: int, num_red: int, num_blue: int,
                    max_steps: int, device: torch.device, episode_idx: int,
                    obs_adapter: str, entity_dim: int,
                    enable_blue_gcas: bool):
    env = UavCombatEnv(
        max_num_blue=num_blue,
        max_num_red=num_red,
        max_steps=max_steps,
        enable_gcas_for_blue=enable_blue_gcas,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _ = env.reset()
        red_ids = [f"red_{i}" for i in range(num_red)]
        blue_ids = [f"blue_{i}" for i in range(num_blue)]
        rnn_a = np.zeros((num_red, rnn_hidden_size), dtype=np.float32)
        death_reasons: dict[str, str] = {}
        red_missiles_fired = 0.0
        blue_missiles_fired = 0.0
        info = {}
        steps = 0
        done = False

        while not done:
            actions = {}
            blue_obs_dict = {bid: obs[bid] for bid in blue_ids}
            engaged = env.refresh_engaged_targets()
            kin = env.get_blue_own_kinematics()
            blue_own_positions = {
                bid: data["position"] for bid, data in kin.items()
                if "position" in data
            }
            blue_own_headings = {
                bid: data["heading"] for bid, data in kin.items()
                if "heading" in data
            }
            actions.update(blue_coordinated_actions(
                blue_obs_dict, num_blue, num_red, engaged_targets=engaged,
                own_positions=blue_own_positions,
                own_headings=blue_own_headings))

            if actor is not None:
                alive_indices = []
                entity_batch = []
                mask_batch = []
                for i, rid in enumerate(red_ids):
                    obs_np = obs[rid]
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        entities_np, entity_mask_np = _build_attention_entities(
                            obs_np, obs_adapter)
                        entity_batch.append(entities_np)
                        mask_batch.append(entity_mask_np)
                        alive_indices.append(i)
                    else:
                        actions[rid] = np.zeros(3, dtype=np.float32)

                if alive_indices:
                    entities_t = torch.as_tensor(np.stack(entity_batch),
                                                 dtype=torch.float32, device=device)
                    masks_t = torch.as_tensor(np.stack(mask_batch),
                                              dtype=torch.long, device=device)
                    rnn_t = torch.as_tensor(rnn_a[alive_indices],
                                            dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action_dist, new_rnn, _attn = actor(entities_t, masks_t, rnn_t)
                        act = action_dist.mean.clamp(-1.0, 1.0)
                    for k, i in enumerate(alive_indices):
                        actions[red_ids[i]] = act[k].cpu().numpy().astype(np.float32)
                        rnn_a[i] = new_rnn[k].cpu().numpy()
            else:
                for rid in red_ids:
                    actions[rid] = np.random.uniform(-1, 1, 3).astype(np.float32)

            obs, _rewards, terminated, truncated, info = env.step(actions)
            steps += 1

            for rid in red_ids:
                red_missiles_fired += info.get(rid, {}).get(
                    "missiles_fired_this_step", 0)
            for bid in blue_ids:
                blue_missiles_fired += info.get(bid, {}).get(
                    "missiles_fired_this_step", 0)

            for aid in red_ids + blue_ids:
                if aid not in death_reasons:
                    reason = info.get(aid, {}).get("death_reason")
                    if reason:
                        death_reasons[aid] = reason

            if actor is not None:
                for i, rid in enumerate(red_ids):
                    if terminated.get(rid, False) or truncated.get(rid, False):
                        rnn_a[i] = np.zeros(rnn_hidden_size, dtype=np.float32)

            done = all(bool(terminated.get(aid, False) or truncated.get(aid, False))
                       for aid in red_ids + blue_ids)

        red_alive = sum(1 for rid in red_ids if info.get(rid, {}).get("alive", False))
        blue_alive = sum(1 for bid in blue_ids if info.get(bid, {}).get("alive", False))
        outcome = _episode_outcome(red_alive, blue_alive)

        red_deaths = _death_counts(death_reasons, red_ids)
        blue_deaths = _death_counts(death_reasons, blue_ids)
        red_deaths_missile = red_deaths["missile"]
        red_deaths_crash = red_deaths["crash"]
        blue_deaths_missile = blue_deaths["missile"]
        blue_deaths_crash = blue_deaths["crash"]
        red_missile_hits = blue_deaths_missile
        blue_missile_hits = red_deaths_missile

        return {
            "Episode": episode_idx,
            "Outcome": outcome,
            "RedWin": 1 if outcome == "red" else 0,
            "BlueWin": 1 if outcome == "blue" else 0,
            "Draw": 1 if outcome == "draw" else 0,
            "Steps": steps,
            "RedAlive": red_alive,
            "BlueAlive": blue_alive,
            "RedMissilesFired": red_missiles_fired,
            "BlueMissilesFired": blue_missiles_fired,
            "RedMissileHits": red_missile_hits,
            "BlueMissileHits": blue_missile_hits,
            "RedMissileHitRate": _safe_div(red_missile_hits, red_missiles_fired),
            "BlueMissileHitRate": _safe_div(blue_missile_hits, blue_missiles_fired),
            "RedDeathsMissile": red_deaths_missile,
            "RedDeathsCrash": red_deaths_crash,
            "BlueDeathsMissile": blue_deaths_missile,
            "BlueDeathsCrash": blue_deaths_crash,
            "KD_Red": _safe_div(
                blue_deaths_missile + blue_deaths_crash,
                red_deaths_missile + red_deaths_crash,
            ),
            "ObsAdapter": obs_adapter,
            "EntityDim": entity_dim,
            "RewardVersion": REWARD_VERSION,
        }
    finally:
        env.close()


def _print_summary(rows: list[dict], output_path: str,
                   obs_adapter: str, entity_dim: int):
    episodes = len(rows)
    print("=" * 70)
    print("Summary")
    print(f"Episodes: {episodes}")
    print(f"Obs adapter: {obs_adapter}")
    print(f"Entity dim: {entity_dim}")
    print(f"Reward version: {REWARD_VERSION}")
    print(f"Red win rate: {_safe_div(sum(r['RedWin'] for r in rows), episodes):.6f}")
    print(f"Blue win rate: {_safe_div(sum(r['BlueWin'] for r in rows), episodes):.6f}")
    print(f"Draw rate: {_safe_div(sum(r['Draw'] for r in rows), episodes):.6f}")
    print(f"Mean red alive: {np.mean([r['RedAlive'] for r in rows]) if rows else 0.0:.4f}")
    print(f"Mean blue alive: {np.mean([r['BlueAlive'] for r in rows]) if rows else 0.0:.4f}")
    print("Mean red missiles fired: "
          f"{np.mean([r['RedMissilesFired'] for r in rows]) if rows else 0.0:.4f}")
    print("Mean blue missiles fired: "
          f"{np.mean([r['BlueMissilesFired'] for r in rows]) if rows else 0.0:.4f}")
    print("Mean red missile hit rate: "
          f"{np.mean([r['RedMissileHitRate'] for r in rows]) if rows else 0.0:.6f}")
    print("Mean blue missile hit rate: "
          f"{np.mean([r['BlueMissileHitRate'] for r in rows]) if rows else 0.0:.6f}")
    print(f"Mean KD_Red: {np.mean([r['KD_Red'] for r in rows]) if rows else 0.0:.6f}")
    print(f"Output path: {output_path}")


def main():
    args = parse_args()
    _set_seed(args.seed)
    device = _select_device(args.device)
    actor, rnn_hidden_size, entity_dim, _checkpoint = _load_actor(args, device)
    print(f"obs_adapter: {args.obs_adapter}", flush=True)
    print(f"entity_dim: {entity_dim}", flush=True)
    print(f"enable_blue_gcas: {args.enable_blue_gcas}", flush=True)
    print(f"reward_version: {REWARD_VERSION}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "Episode", "Outcome", "RedWin", "BlueWin", "Draw", "Steps",
        "RedAlive", "BlueAlive",
        "RedMissilesFired", "BlueMissilesFired",
        "RedMissileHits", "BlueMissileHits",
        "RedMissileHitRate", "BlueMissileHitRate",
        "RedDeathsMissile", "RedDeathsCrash",
        "BlueDeathsMissile", "BlueDeathsCrash",
        "KD_Red", "ObsAdapter", "EntityDim", "RewardVersion",
    ]

    rows = []
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        for ep in range(1, args.episodes + 1):
            row = run_one_episode(
                actor=actor,
                rnn_hidden_size=rnn_hidden_size,
                num_red=args.num_red,
                num_blue=args.num_blue,
                max_steps=args.max_steps,
                device=device,
                episode_idx=ep,
                obs_adapter=args.obs_adapter,
                entity_dim=entity_dim,
                enable_blue_gcas=args.enable_blue_gcas,
            )
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"Episode {ep}/{args.episodes}: outcome={row['Outcome']} "
                  f"steps={row['Steps']} red_alive={row['RedAlive']} "
                  f"blue_alive={row['BlueAlive']}", flush=True)

    _print_summary(rows, args.output, args.obs_adapter, entity_dim)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
