"""Batch evaluation for the vanilla MAPPO baseline without Tacview output."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from my_uav_env import UavCombatEnv
from my_uav_env.alignment.reward_utils import (
    DEFAULT_ALTITUDE_REWARD_CONFIG,
    REWARD_VERSION,
)
from rule_based_agent import blue_coordinated_actions
from train_vanilla_mappo import (
    CHECKPOINT_SCHEMA_VERSION,
    ACTION_DISTRIBUTION_VERSION,
    VanillaActor,
    _classify_death_reason,
    _compute_global_state_dim,
    _compute_obs_dim,
    _episode_outcome,
    _flatten_obs,
    _joint_team_reward_once,
    _ratio_with_denominator_zero,
    _safe_div,
    _unpack_and_validate_checkpoint,
)


EVALUATION_FIELDNAMES = [
    "Episode", "Outcome", "RedWin", "BlueWin", "Draw", "Steps",
    "EpisodeRewardRed", "RedAlive", "BlueAlive",
    "RedMissilesFired", "BlueMissilesFired",
    "RedMissileHits", "BlueMissileHits",
    "RedMissileHitRate", "BlueMissileHitRate",
    "RedDeathsMissile", "RedDeathsCrash",
    "RedDeathsOther", "BlueDeathsMissile", "BlueDeathsCrash",
    "BlueDeathsOther", "CheckpointSchema", "NumRed", "NumBlue", "MaxSteps",
    "EnableBlueGCAS", "RewardVersion", "RewardMode", "ObsNormalization",
    "PIDProfile", "PIDThrottleBase", "MissileGuidanceMode",
    "ActionDistribution", "AltitudeRewardConfigVersion", "AltitudeRewardConfig",
    "BluePolicyProfile", "RedGeometry", "RedLockMature", "BlueGeometry",
    "BlueLockMature", "RedTerminalReward", "NaNInfCount",
    "blue_target_switches_total", "blue_target_dead_switches",
    "blue_distance_triggered_switches", "blue_engaged_triggered_switches",
    "blue_mws_detected_frames", "blue_mws_override_frames",
    "blue_route_phase_changes", "blue_heading_command_discontinuities",
    "blue_altitude_recovery_frames",
]

EVALUATION_SUMMARY_FIELDNAMES = [
    "Episodes", "RedWins", "BlueWins", "Draws",
    "RedWinRate", "BlueWinRate", "RWR", "RWRDenominatorZero",
    "RedDeathsAll", "BlueDeathsAll",
    "RedDeathsMissile", "BlueDeathsMissile",
    "KD_Red_AllDeaths", "KD_Red_MissileOnly",
    "MeanRedAlive", "MeanBlueAlive",
    "RedMissilesFired", "BlueMissilesFired",
    "RedMissileHits", "BlueMissileHits",
    "RedMissileHitRate", "BlueMissileHitRate",
    "CheckpointSchema", "NumRed", "NumBlue", "MaxSteps", "EnableBlueGCAS",
    "RewardVersion", "RewardMode", "ObsNormalization",
    "PIDProfile", "PIDThrottleBase", "MissileGuidanceMode",
    "ActionDistribution", "AltitudeRewardConfigVersion", "AltitudeRewardConfig",
    "BluePolicyProfile",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate vanilla MAPPO baseline over multiple episodes.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--num-red", type=int, default=3)
    parser.add_argument("--num-blue", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--enable-blue-gcas", action="store_true", default=False)
    parser.add_argument("--blue-policy-profile", choices=(
        "paper_pursuit", "fixed_pair_pursuit_v1", "fixed_pair_no_mws_v1",
        "fixed_pair_hold_after_kill_v1", "frozen_route_blue_v1"),
        default="paper_pursuit")
    parser.add_argument("--obs-mode", type=str,
                        choices=("paper_strict", "engineering"),
                        default="paper_strict")
    parser.add_argument("--obs-normalization", type=str,
                        choices=("paper_fixed_v1", "none"),
                        default="paper_fixed_v1")
    parser.add_argument("--pid-profile", choices=("paper", "engineering_safe"),
                        default="paper")
    parser.add_argument("--pid-throttle-base", type=float, default=0.0)
    parser.add_argument("--reward-mode", choices=("paper_joint", "engineering_local"),
                        default="paper_joint")
    parser.add_argument("--missile-guidance-mode",
                        choices=("paper_eq9", "legacy_simplified"),
                        default="paper_eq9")
    parser.add_argument("--output", type=str,
                        default="results/eval_vanilla_mappo.csv")
    parser.add_argument("--summary-output", type=str, default=None)
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


def _infer_actor_shapes(state: dict):
    obs_dim = None
    hidden = 128
    rnn_hidden = 128
    for key, tensor in state.items():
        if key == "fc_in.weight":
            hidden = int(tensor.shape[0])
            obs_dim = int(tensor.shape[1])
        elif key == "rnn.weight_ih":
            rnn_hidden = int(tensor.shape[0] // 3)
    return obs_dim, hidden, rnn_hidden


def _resolve_checkpoint(path: str | None) -> str | None:
    if path:
        return path
    candidates = [
        os.path.join("checkpoints", "vanilla_actor_best.pt"),
        os.path.join("checkpoints", "vanilla_actor_final.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_actor(args, device: torch.device):
    if args.random:
        print("[INFO] --random set; red team uses random actions.", flush=True)
        return None, 128, None

    checkpoint = _resolve_checkpoint(args.checkpoint)
    if checkpoint is None:
        print("[WARN] No checkpoint found; red team uses random actions.",
              flush=True)
        return None, 128, None

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    obs_dim = _compute_obs_dim(
        args.num_red, args.num_blue, is_red=True, obs_mode=args.obs_mode)
    expected_metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "obs_mode": args.obs_mode,
        "obs_normalization": args.obs_normalization,
        "reward_version": REWARD_VERSION,
        "reward_mode": args.reward_mode,
        "pid_profile": args.pid_profile,
        "pid_throttle_base": float(args.pid_throttle_base),
        "missile_guidance_mode": args.missile_guidance_mode,
        "altitude_reward_config": asdict(DEFAULT_ALTITUDE_REWARD_CONFIG),
        "action_distribution": ACTION_DISTRIBUTION_VERSION,
        "blue_policy_profile": args.blue_policy_profile,
        "num_red": args.num_red,
        "num_blue": args.num_blue,
        "global_state_dim": _compute_global_state_dim(args.num_red, args.obs_mode),
        "actor_obs_dim": obs_dim,
    }
    try:
        state = _unpack_and_validate_checkpoint(
            payload, expected_metadata, "actor")
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    ckpt_obs_dim, hidden, rnn_hidden = _infer_actor_shapes(state)
    if ckpt_obs_dim != obs_dim:
        raise SystemExit(
            "ERROR: checkpoint obs_dim does not match current evaluation scale.\n"
            f"  checkpoint obs_dim: {ckpt_obs_dim}\n"
            f"  current obs_dim:    {obs_dim}\n"
            "  vanilla MLP baseline has fixed flattened observation size and "
            "cannot be evaluated zero-shot across a different scale."
        )

    actor = VanillaActor(obs_dim=obs_dim, action_dim=3,
                         hidden=hidden, rnn_hidden=rnn_hidden).to(device)
    actor.load_state_dict(state)
    actor.eval()
    print(f"[INFO] Loaded actor checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Actor shape: obs_dim={obs_dim}, hidden={hidden}, "
          f"rnn_hidden={rnn_hidden}", flush=True)
    return actor, rnn_hidden, checkpoint


def _death_counts(death_reasons: dict[str, str], ids: list[str]) -> Counter:
    counts = Counter()
    for aid in ids:
        reason = death_reasons.get(aid)
        if reason:
            counts[_classify_death_reason(reason)] += 1
    return counts


def run_one_episode(actor, rnn_hidden_size: int, num_red: int, num_blue: int,
                    max_steps: int, device: torch.device, episode_idx: int,
                    enable_blue_gcas: bool, obs_mode: str,
                    obs_normalization: str = "paper_fixed_v1",
                    pid_profile: str = "paper",
                    pid_throttle_base: float = 0.0,
                    reward_mode: str = "paper_joint",
                    missile_guidance_mode: str = "paper_eq9",
                    blue_policy_profile: str = "paper_pursuit",
                    seed: int | None = None,
                    deterministic: bool = True):
    env = UavCombatEnv(
        max_num_blue=num_blue,
        max_num_red=num_red,
        max_steps=max_steps,
        obs_mode=obs_mode,
        pid_profile=pid_profile,
        pid_throttle_base=pid_throttle_base,
        reward_mode=reward_mode,
        missile_guidance_mode=missile_guidance_mode,
        blue_policy_profile=blue_policy_profile,
        enable_gcas_for_blue=enable_blue_gcas,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _ = env.reset() if seed is None else env.reset(seed=seed)
        red_ids = [f"red_{i}" for i in range(num_red)]
        blue_ids = [f"blue_{i}" for i in range(num_blue)]
        rnn_a = np.zeros((num_red, rnn_hidden_size), dtype=np.float32)
        death_reasons: dict[str, str] = {}
        red_missiles_fired = 0.0
        blue_missiles_fired = 0.0
        info = {}
        steps = 0
        red_episode_joint_reward = 0.0
        red_terminal_reward = 0.0
        launch_diag_totals = {"red": Counter(), "blue": Counter()}
        nan_inf_count = 0

        done = False
        while not done:
            actions = {}

            blue_obs_dict = {bid: obs[bid] for bid in blue_ids}
            if hasattr(env, "blue_policy_actions"):
                actions.update(env.blue_policy_actions(blue_obs_dict))
            else:
                engaged = env.refresh_engaged_targets()
                kinematics = env.get_blue_own_kinematics()
                actions.update(blue_coordinated_actions(
                    blue_obs_dict, num_blue, num_red, engaged_targets=engaged,
                    own_positions={
                        aid: value["position"] for aid, value in kinematics.items()
                        if "position" in value},
                    own_headings={
                        aid: value["heading"] for aid, value in kinematics.items()
                        if "heading" in value}))

            if actor is not None:
                alive_indices = []
                obs_batch = []
                for i, rid in enumerate(red_ids):
                    obs_np = obs[rid]
                    alive = not np.allclose(obs_np["ego_state"], 0.0)
                    if alive:
                        obs_batch.append(_flatten_obs(
                            obs_np, obs_mode=obs_mode,
                            obs_normalization=obs_normalization))
                        alive_indices.append(i)
                    else:
                        actions[rid] = np.zeros(3, dtype=np.float32)

                if alive_indices:
                    obs_t = torch.as_tensor(np.stack(obs_batch),
                                            dtype=torch.float32, device=device)
                    rnn_t = torch.as_tensor(rnn_a[alive_indices],
                                            dtype=torch.float32, device=device)
                    with torch.no_grad():
                        action_dist, new_rnn = actor(obs_t, rnn_t)
                        act = action_dist.mode if deterministic else action_dist.sample()
                    for k, i in enumerate(alive_indices):
                        actions[red_ids[i]] = act[k].cpu().numpy().astype(np.float32)
                        rnn_a[i] = new_rnn[k].cpu().numpy()
            else:
                for rid in red_ids:
                    actions[rid] = np.random.uniform(-1, 1, 3).astype(np.float32)

            obs, _rewards, terminated, truncated, info = env.step(actions)
            nan_inf_count += sum(int(not np.all(np.isfinite(value)))
                                 for value in actions.values())
            nan_inf_count += sum(int(not np.isfinite(float(value)))
                                 for value in _rewards.values())
            steps += 1
            red_episode_joint_reward += _joint_team_reward_once(
                _rewards, red_ids)
            red_terminal_reward += float(info.get("__reward_summary__", {}).get(
                "red_team_terminal_reward", 0.0))
            for team in ("red", "blue"):
                launch_diag_totals[team].update(
                    info.get("__launch_diag__", {}).get(team, {}))

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
        red_deaths_other = red_deaths["other"]
        blue_deaths_other = blue_deaths["other"]
        red_missile_hits = blue_deaths_missile
        blue_missile_hits = red_deaths_missile
        blue_diag = info.get("__blue_policy_diag__", {})

        return {
            "Episode": episode_idx,
            "Outcome": outcome,
            "RedWin": 1 if outcome == "red" else 0,
            "BlueWin": 1 if outcome == "blue" else 0,
            "Draw": 1 if outcome == "draw" else 0,
            "Steps": steps,
            "EpisodeRewardRed": red_episode_joint_reward,
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
            "RedDeathsOther": red_deaths_other,
            "BlueDeathsMissile": blue_deaths_missile,
            "BlueDeathsCrash": blue_deaths_crash,
            "BlueDeathsOther": blue_deaths_other,
            "CheckpointSchema": CHECKPOINT_SCHEMA_VERSION,
            "NumRed": num_red,
            "NumBlue": num_blue,
            "MaxSteps": max_steps,
            "EnableBlueGCAS": bool(enable_blue_gcas),
            "RewardVersion": REWARD_VERSION,
            "RewardMode": reward_mode,
            "ObsNormalization": obs_normalization,
            "PIDProfile": pid_profile,
            "PIDThrottleBase": float(pid_throttle_base),
            "MissileGuidanceMode": missile_guidance_mode,
            "ActionDistribution": ACTION_DISTRIBUTION_VERSION,
            "AltitudeRewardConfigVersion": DEFAULT_ALTITUDE_REWARD_CONFIG.version,
            "AltitudeRewardConfig": json.dumps(
                asdict(DEFAULT_ALTITUDE_REWARD_CONFIG), sort_keys=True,
                separators=(",", ":")),
            "BluePolicyProfile": blue_policy_profile,
            "RedGeometry": int(launch_diag_totals["red"]["geometry_ok_pairs"]),
            "RedLockMature": int(launch_diag_totals["red"]["lock_mature_pairs"]),
            "BlueGeometry": int(launch_diag_totals["blue"]["geometry_ok_pairs"]),
            "BlueLockMature": int(launch_diag_totals["blue"]["lock_mature_pairs"]),
            "RedTerminalReward": red_terminal_reward,
            "NaNInfCount": nan_inf_count,
            **{field: int(blue_diag.get(field, 0)) for field in (
                "blue_target_switches_total", "blue_target_dead_switches",
                "blue_distance_triggered_switches", "blue_engaged_triggered_switches",
                "blue_mws_detected_frames", "blue_mws_override_frames",
                "blue_route_phase_changes", "blue_heading_command_discontinuities",
                "blue_altitude_recovery_frames")},
        }
    finally:
        env.close()


def _aggregate_evaluation_summary(rows: list[dict]) -> dict:
    episodes = len(rows)
    red_wins = sum(int(row["RedWin"]) for row in rows)
    blue_wins = sum(int(row["BlueWin"]) for row in rows)
    draws = sum(int(row["Draw"]) for row in rows)
    rwr, rwr_zero = _ratio_with_denominator_zero(red_wins, blue_wins)
    red_deaths_missile = sum(int(row["RedDeathsMissile"]) for row in rows)
    blue_deaths_missile = sum(int(row["BlueDeathsMissile"]) for row in rows)
    red_deaths_all = sum(
        int(row[key]) for row in rows
        for key in ("RedDeathsMissile", "RedDeathsCrash", "RedDeathsOther"))
    blue_deaths_all = sum(
        int(row[key]) for row in rows
        for key in ("BlueDeathsMissile", "BlueDeathsCrash", "BlueDeathsOther"))
    kd_all, _ = _ratio_with_denominator_zero(blue_deaths_all, red_deaths_all)
    kd_missile, _ = _ratio_with_denominator_zero(
        blue_deaths_missile, red_deaths_missile)
    red_fired = sum(float(row["RedMissilesFired"]) for row in rows)
    blue_fired = sum(float(row["BlueMissilesFired"]) for row in rows)
    red_hits = sum(float(row["RedMissileHits"]) for row in rows)
    blue_hits = sum(float(row["BlueMissileHits"]) for row in rows)
    semantics = rows[0] if rows else {}
    return {
        "Episodes": episodes,
        "RedWins": red_wins,
        "BlueWins": blue_wins,
        "Draws": draws,
        "RedWinRate": _safe_div(red_wins, episodes),
        "BlueWinRate": _safe_div(blue_wins, episodes),
        "RWR": rwr,
        "RWRDenominatorZero": rwr_zero,
        "RedDeathsAll": red_deaths_all,
        "BlueDeathsAll": blue_deaths_all,
        "RedDeathsMissile": red_deaths_missile,
        "BlueDeathsMissile": blue_deaths_missile,
        "KD_Red_AllDeaths": kd_all,
        "KD_Red_MissileOnly": kd_missile,
        "MeanRedAlive": float(np.mean([r["RedAlive"] for r in rows])) if rows else 0.0,
        "MeanBlueAlive": float(np.mean([r["BlueAlive"] for r in rows])) if rows else 0.0,
        "RedMissilesFired": red_fired,
        "BlueMissilesFired": blue_fired,
        "RedMissileHits": red_hits,
        "BlueMissileHits": blue_hits,
        "RedMissileHitRate": _safe_div(red_hits, red_fired),
        "BlueMissileHitRate": _safe_div(blue_hits, blue_fired),
        **{
            key: semantics.get(key, "") for key in (
                "CheckpointSchema", "NumRed", "NumBlue", "MaxSteps",
                "EnableBlueGCAS", "RewardVersion", "RewardMode", "ObsNormalization",
                "PIDProfile", "PIDThrottleBase", "MissileGuidanceMode",
                "ActionDistribution", "AltitudeRewardConfigVersion",
                "AltitudeRewardConfig", "BluePolicyProfile",
            )
        },
    }


def _write_and_print_summary(rows: list[dict], output_path: str,
                             summary_path: str) -> dict:
    summary = _aggregate_evaluation_summary(rows)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EVALUATION_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)
    print("=" * 70)
    print("Summary")
    print(f"Episodes: {summary['Episodes']}")
    print(f"Reward version: {REWARD_VERSION}")
    print(f"Red / Blue win rate: {summary['RedWinRate']:.6f} / "
          f"{summary['BlueWinRate']:.6f}")
    print(f"RWR: {summary['RWR']} "
          f"(denominator_zero={summary['RWRDenominatorZero']})")
    print(f"KD all / missile: {summary['KD_Red_AllDeaths']} / "
          f"{summary['KD_Red_MissileOnly']}")
    print(f"Output path: {output_path}")
    print(f"Summary path: {summary_path}")
    return summary


def main():
    args = parse_args()
    _set_seed(args.seed)
    device = _select_device(args.device)
    actor, rnn_hidden_size, _checkpoint = _load_actor(args, device)
    print(f"enable_blue_gcas: {args.enable_blue_gcas}", flush=True)
    print(f"reward_version: {REWARD_VERSION}", flush=True)
    print(f"pid_throttle_base: {args.pid_throttle_base}", flush=True)
    print(f"action_distribution: {ACTION_DISTRIBUTION_VERSION}", flush=True)
    print("altitude_reward_config: "
          f"{json.dumps(asdict(DEFAULT_ALTITUDE_REWARD_CONFIG), sort_keys=True)}",
          flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_FIELDNAMES)
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
                enable_blue_gcas=args.enable_blue_gcas,
                obs_mode=args.obs_mode,
                obs_normalization=args.obs_normalization,
                pid_profile=args.pid_profile,
                pid_throttle_base=args.pid_throttle_base,
                reward_mode=args.reward_mode,
                missile_guidance_mode=args.missile_guidance_mode,
                blue_policy_profile=args.blue_policy_profile,
                seed=None if args.seed is None else args.seed + ep - 1,
            )
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(f"Episode {ep}/{args.episodes}: outcome={row['Outcome']} "
                  f"steps={row['Steps']} red_alive={row['RedAlive']} "
                  f"blue_alive={row['BlueAlive']}", flush=True)

    summary_path = args.summary_output
    if summary_path is None:
        root, extension = os.path.splitext(args.output)
        summary_path = f"{root}_summary{extension or '.csv'}"
    _write_and_print_summary(rows, args.output, summary_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
