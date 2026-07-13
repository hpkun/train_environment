"""Cross-evaluate one red checkpoint against diagnostic blue profiles."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from evaluate_vanilla_mappo import (
    _infer_actor_shapes,
    _set_seed,
    run_one_episode,
)
from my_uav_env.blue_policy_profiles import BLUE_POLICY_PROFILES
from train_vanilla_mappo import (
    ACTION_DISTRIBUTION_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    VanillaActor,
    _compute_obs_dim,
)


MATRIX_FIELDS = (
    "checkpoint", "checkpoint_blue_policy_profile", "profile", "deterministic",
    "seed", "episode_steps", "winner", "red_alive", "blue_alive",
    "red_return", "red_terminal_reward", "red_geometry", "red_lock_mature",
    "red_launches", "red_hits", "blue_geometry", "blue_lock_mature",
    "blue_launches", "blue_hits", "red_missile_deaths", "red_crash_deaths",
    "blue_missile_deaths", "blue_crash_deaths", "blue_target_switches_total",
    "blue_target_dead_switches", "blue_distance_triggered_switches",
    "blue_engaged_triggered_switches", "blue_mws_detected_frames",
    "blue_mws_override_frames", "blue_heading_command_discontinuities",
    "nan_inf_count",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one red checkpoint against multiple blue profiles.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profiles", nargs="+", choices=BLUE_POLICY_PROFILES,
                        default=list(BLUE_POLICY_PROFILES))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-red", type=int, default=3)
    parser.add_argument("--num-blue", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--output-dir", default="results/blue_profile_matrix")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_matrix_actor(path: str, num_red: int, num_blue: int,
                       device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit("checkpoint lacks state_dict")
    if payload.get("model_kind") != "actor":
        raise SystemExit("checkpoint is not an actor checkpoint")
    metadata = payload.get("metadata", {})
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SystemExit("checkpoint schema mismatch")
    if metadata.get("action_distribution") != ACTION_DISTRIBUTION_VERSION:
        raise SystemExit("checkpoint action distribution mismatch")
    if int(metadata.get("num_red", -1)) != num_red or int(
            metadata.get("num_blue", -1)) != num_blue:
        raise SystemExit("checkpoint team size mismatch")
    source_profile = metadata.get("blue_policy_profile")
    if source_profile is None:
        source_profile = "paper_pursuit"
        print("[COMPAT] checkpoint has no blue_policy_profile; source is treated "
              "as paper_pursuit.", flush=True)
    obs_dim = _compute_obs_dim(
        num_red, num_blue, is_red=True,
        obs_mode=metadata.get("obs_mode", "paper_strict"))
    state = payload["state_dict"]
    ckpt_obs_dim, hidden, rnn_hidden = _infer_actor_shapes(state)
    if ckpt_obs_dim != obs_dim:
        raise SystemExit(
            f"checkpoint actor_obs_dim={ckpt_obs_dim}, expected {obs_dim}")
    actor = VanillaActor(obs_dim, 3, hidden=hidden,
                         rnn_hidden=rnn_hidden).to(device)
    actor.load_state_dict(state)
    actor.eval()
    return actor, rnn_hidden, source_profile, metadata


def _matrix_row(row: dict, checkpoint: str, source_profile: str,
                profile: str, deterministic: bool, seed: int) -> dict:
    return {
        "checkpoint": checkpoint,
        "checkpoint_blue_policy_profile": source_profile,
        "profile": profile,
        "deterministic": bool(deterministic),
        "seed": seed,
        "episode_steps": row["Steps"],
        "winner": row["Outcome"],
        "red_alive": row["RedAlive"],
        "blue_alive": row["BlueAlive"],
        "red_return": row["EpisodeRewardRed"],
        "red_terminal_reward": row["RedTerminalReward"],
        "red_geometry": row["RedGeometry"],
        "red_lock_mature": row["RedLockMature"],
        "red_launches": row["RedMissilesFired"],
        "red_hits": row["RedMissileHits"],
        "blue_geometry": row["BlueGeometry"],
        "blue_lock_mature": row["BlueLockMature"],
        "blue_launches": row["BlueMissilesFired"],
        "blue_hits": row["BlueMissileHits"],
        "red_missile_deaths": row["RedDeathsMissile"],
        "red_crash_deaths": row["RedDeathsCrash"],
        "blue_missile_deaths": row["BlueDeathsMissile"],
        "blue_crash_deaths": row["BlueDeathsCrash"],
        "blue_target_switches_total": row["blue_target_switches_total"],
        "blue_target_dead_switches": row["blue_target_dead_switches"],
        "blue_distance_triggered_switches": row[
            "blue_distance_triggered_switches"],
        "blue_engaged_triggered_switches": row[
            "blue_engaged_triggered_switches"],
        "blue_mws_detected_frames": row["blue_mws_detected_frames"],
        "blue_mws_override_frames": row["blue_mws_override_frames"],
        "blue_heading_command_discontinuities": row[
            "blue_heading_command_discontinuities"],
        "nan_inf_count": row["NaNInfCount"],
    }


def _summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    summary = {}
    for profile, items in grouped.items():
        returns = np.asarray([float(r["red_return"]) for r in items])
        summary[profile] = {
            "episodes": len(items),
            "red_return_mean": float(np.mean(returns)),
            "red_return_std": float(np.std(returns)),
            "red_win_rate": float(np.mean([r["winner"] == "red" for r in items])),
            "episode_steps_mean": float(np.mean([r["episode_steps"] for r in items])),
        }
        for key in (
                "red_geometry", "red_lock_mature", "red_launches", "red_hits",
                "blue_geometry", "blue_lock_mature", "blue_launches", "blue_hits",
                "blue_target_switches_total", "blue_target_dead_switches",
                "blue_distance_triggered_switches",
                "blue_engaged_triggered_switches", "blue_mws_detected_frames",
                "blue_mws_override_frames", "red_crash_deaths",
                "blue_crash_deaths", "nan_inf_count"):
            summary[profile][f"{key}_mean"] = float(np.mean(
                [float(r[key]) for r in items]))
    return summary


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    device = _device(args.device)
    actor, rnn_hidden, source_profile, metadata = _load_matrix_actor(
        args.checkpoint, args.num_red, args.num_blue, device)
    seeds = [args.seed_start + i for i in range(args.episodes)]
    rows = []
    for profile in args.profiles:
        for episode_index, seed in enumerate(seeds, start=1):
            _set_seed(seed)
            raw = run_one_episode(
                actor, rnn_hidden, args.num_red, args.num_blue, args.max_steps,
                device, episode_index, False,
                metadata.get("obs_mode", "paper_strict"),
                metadata.get("obs_normalization", "paper_fixed_v1"),
                metadata.get("pid_profile", "paper"),
                float(metadata.get("pid_throttle_base", 0.0)),
                metadata.get("reward_mode", "paper_joint"),
                metadata.get("missile_guidance_mode", "paper_eq9"),
                profile, seed, args.deterministic)
            rows.append(_matrix_row(
                raw, args.checkpoint, source_profile, profile,
                args.deterministic, seed))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    with open(os.path.join(args.output_dir, "profile_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "report.txt"), "w",
              encoding="utf-8") as handle:
        for profile in args.profiles:
            values = summary[profile]
            handle.write(
                f"{profile}: return={values['red_return_mean']:.6f} +/- "
                f"{values['red_return_std']:.6f}, red_win_rate="
                f"{values['red_win_rate']:.6f}, steps="
                f"{values['episode_steps_mean']:.2f}, red_attack="
                f"{values['red_geometry_mean']:.2f}/"
                f"{values['red_lock_mature_mean']:.2f}/"
                f"{values['red_launches_mean']:.2f}/"
                f"{values['red_hits_mean']:.2f}, blue_switches="
                f"{values['blue_target_switches_total_mean']:.2f}, mws="
                f"{values['blue_mws_detected_frames_mean']:.2f}/"
                f"{values['blue_mws_override_frames_mean']:.2f}, crashes="
                f"{values['red_crash_deaths_mean']:.2f}/"
                f"{values['blue_crash_deaths_mean']:.2f}\n")
    print(f"Wrote {len(rows)} episodes to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
