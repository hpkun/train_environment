"""Diagnose MAV pitch-up / stall / crash from best checkpoint.

Outputs mav_control_trace.csv with per-step control chain data for red_0.
Also runs safe-action and zero-action MAV comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.hetero_entity_recurrent_policy import (
    HeteroEntityRecurrentPolicy,
    validate_entity_policy_meta,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
from uav_env.JSBSim.pid_controller import F22MavEnergyPIDController


def _load_model(exp_dir: str, checkpoint: str):
    """Load policy from experiment directory."""
    exp = Path(exp_dir)
    # Try best_combined first, then best, then latest
    for ckpt_dir in [f"{checkpoint}_combined", checkpoint, "latest"]:
        model_path = exp / ckpt_dir / "model.pt"
        meta_path = exp / ckpt_dir / "meta.json"
        if model_path.exists() and meta_path.exists():
            break
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {exp_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_entity_policy_meta(meta)
    policy = HeteroEntityRecurrentPolicy(
        entity_dim=int(meta.get("entity_dim", 21)),
        action_dim=3,
        hidden_dim=int(meta.get("hidden_dim", 128)),
        rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        num_attention_heads=int(meta.get("num_attention_heads", 4)),
    ).to(device)
    policy.load(str(model_path), map_location=device)
    policy.eval()
    return policy, device, meta


def _safe_mav_action(speed_mps: float) -> np.ndarray:
    """Return a safety-first MAV action: level flight, moderate speed."""
    target_pitch_deg = 2.0  # slight nose-up
    target_heading = 0.0
    target_speed = max(speed_mps + 5.0, 200.0)  # gentle speed
    # Normalise to [-1,1]
    pitch_norm = target_pitch_deg / 90.0
    heading_norm = target_heading / 180.0
    speed_norm = (target_speed - 102.0) / (408.0 - 102.0) * 2.0 - 1.0
    return np.array([pitch_norm, heading_norm, speed_norm], dtype=np.float32)


def _zero_mav_action() -> np.ndarray:
    """Zero action: level flight at moderate speed."""
    return np.array([0.0, 0.0, 0.3], dtype=np.float32)


def run_diagnostic(policy, device, config_path: str, output_dir: str,
                   episodes: int = 3, max_steps: int = 300,
                   mav_mode: str = "policy",
                   seed: int = 2000):
    """Run diagnostic episodes, trace MAV control chain."""
    from uav_env import make_env

    adapter = HeteroEntitySetAdapter()
    opponent = OpponentPolicy(mode="brma_rule", seed=seed + 33)

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    summaries = []

    for ep in range(episodes):
        ep_seed = seed + ep
        env = make_env(config_path, env_type="jsbsim_hetero")
        obs, info = env.reset(seed=ep_seed)

        ep_rows = []
        step = 0
        mav_dead_step = -1

        while step < max_steps:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_tokens = adapted["actor_entity_tokens"].copy()
            actor_keep = adapted["actor_keep_mask"].copy()
            critic_tokens = adapted["critic_entity_tokens"].copy()
            critic_keep = adapted["critic_keep_mask"].copy()
            critic_counts = adapted.get("critic_counts", np.zeros(4, dtype=np.float32))
            roles = adapted["role_ids"]

            # --- Policy forward pass ---
            with torch.no_grad():
                out = policy.act(
                    torch.as_tensor(actor_tokens, device=device),
                    torch.as_tensor(actor_keep, device=device),
                    torch.as_tensor(roles, device=device),
                    torch.as_tensor(critic_tokens, device=device),
                    torch.as_tensor(critic_keep, device=device),
                    deterministic=True,
                    critic_counts=torch.as_tensor(critic_counts, device=device),
                )
            raw_action = out["action"].cpu().numpy()

            # --- MAV action override for safe/zero modes ---
            effective_action = raw_action.copy()
            if mav_mode in ("safe", "zero"):
                mav_sim = env.red_planes.get("red_0")
                if mav_sim is not None and mav_sim.is_alive:
                    vel = mav_sim.get_velocity()
                    speed = float(np.linalg.norm(vel))
                    if mav_mode == "safe":
                        effective_action[0] = _safe_mav_action(speed)
                    else:
                        effective_action[0] = _zero_mav_action()

            action_dict = {rid: effective_action[i].astype(np.float32)
                          for i, rid in enumerate(env.red_ids)}
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))

            # --- MAV state BEFORE step ---
            mav_sim = env.red_planes.get("red_0")
            if mav_sim is not None and mav_sim.is_alive:
                rpy = mav_sim.get_rpy()
                vel = mav_sim.get_velocity()
                alt = mav_sim.get_geodetic()[2]
                speed = float(np.linalg.norm(vel))

                # PID state
                pid = env.pid_controllers.get("red_0")
                eg_active = False
                eg_level = ""
                pitch_clamped = False
                roll_clamped = False
                throttle_boosted = False
                elev_cmd = 0.0
                aileron_cmd = 0.0
                throttle_cmd_val = 0.0
                if pid is not None and isinstance(pid, F22MavEnergyPIDController):
                    eg_active = pid.last_energy_guard_active
                    eg_level = pid.last_energy_guard_level
                    pitch_clamped = pid.last_pitch_clamped
                    roll_clamped = pid.last_roll_clamped
                    throttle_boosted = pid.last_throttle_boosted
                    # Read last applied control surfaces
                    try:
                        elev_cmd = float(mav_sim.get_property_value("fcs/elevator-cmd-norm"))
                    except Exception:
                        pass
                    try:
                        aileron_cmd = float(mav_sim.get_property_value("fcs/aileron-cmd-norm"))
                    except Exception:
                        pass
                    try:
                        throttle_cmd_val = float(mav_sim.get_property_value("fcs/throttle-cmd-norm"))
                    except Exception:
                        pass

                # Map raw action to physical targets (same as env._parse_actions layer 3)
                PITCH_DEG = 90.0
                VEL_MIN, VEL_MAX = 102.0, 408.0
                ra0 = raw_action[0]
                target_pitch_deg = float(ra0[0]) * PITCH_DEG
                target_heading_deg = float(ra0[1]) * 180.0
                target_vel = VEL_MIN + (float(ra0[2]) + 1.0) / 2.0 * (VEL_MAX - VEL_MIN)
                # Effective action targets
                ea0 = effective_action[0]
                eff_pitch_deg = float(ea0[0]) * PITCH_DEG
                eff_heading_deg = float(ea0[1]) * 180.0
                eff_vel = VEL_MIN + (float(ea0[2]) + 1.0) / 2.0 * (VEL_MAX - VEL_MIN)

                row = {
                    "episode": ep, "step": step,
                    "raw_a0": float(ra0[0]), "raw_a1": float(ra0[1]), "raw_a2": float(ra0[2]),
                    "eff_a0": float(ea0[0]), "eff_a1": float(ea0[1]), "eff_a2": float(ea0[2]),
                    "target_pitch_deg": round(target_pitch_deg, 2),
                    "target_heading_deg": round(target_heading_deg, 2),
                    "target_vel_mps": round(target_vel, 1),
                    "eff_pitch_deg": round(eff_pitch_deg, 2),
                    "eff_heading_deg": round(eff_heading_deg, 2),
                    "eff_vel_mps": round(eff_vel, 1),
                    "cur_pitch_deg": round(float(np.rad2deg(rpy[1])), 2),
                    "cur_roll_deg": round(float(np.rad2deg(rpy[0])), 2),
                    "cur_yaw_deg": round(float(np.rad2deg(rpy[2])), 2),
                    "speed_mps": round(speed, 1),
                    "altitude_m": round(alt, 1),
                    "vertical_speed_mps": round(float(vel[2]), 2),
                    "elevator_cmd": round(elev_cmd, 4),
                    "aileron_cmd": round(aileron_cmd, 4),
                    "throttle_cmd": round(throttle_cmd_val, 4),
                    "energy_guard_active": int(eg_active),
                    "energy_guard_level": eg_level,
                    "pitch_clamped": int(pitch_clamped),
                    "roll_clamped": int(roll_clamped),
                    "throttle_boosted": int(throttle_boosted),
                    "alive": 1,
                    "crash_reason": "",
                }
            else:
                row = {
                    "episode": ep, "step": step,
                    "raw_a0": 0, "raw_a1": 0, "raw_a2": 0,
                    "eff_a0": 0, "eff_a1": 0, "eff_a2": 0,
                    "target_pitch_deg": 0, "target_heading_deg": 0, "target_vel_mps": 0,
                    "eff_pitch_deg": 0, "eff_heading_deg": 0, "eff_vel_mps": 0,
                    "cur_pitch_deg": 0, "cur_roll_deg": 0, "cur_yaw_deg": 0,
                    "speed_mps": 0, "altitude_m": 0, "vertical_speed_mps": 0,
                    "elevator_cmd": 0, "aileron_cmd": 0, "throttle_cmd": 0,
                    "energy_guard_active": 0, "energy_guard_level": "",
                    "pitch_clamped": 0, "roll_clamped": 0, "throttle_boosted": 0,
                    "alive": 0, "crash_reason": "dead",
                }
                if mav_dead_step < 0:
                    mav_dead_step = step
            ep_rows.append(row)

            obs, rewards, terminated, truncated, info = env.step(action_dict)

            # Check MAV status after step
            mav_sim = env.red_planes.get("red_0")
            if mav_sim is not None and not mav_sim.is_alive and mav_dead_step < 0:
                mav_dead_step = step + 1

            step += 1
            if all(terminated.values()) or all(truncated.values()):
                break

        env.close()

        # Episode summary
        alive_end = sum(1 for rid in env.red_ids if env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive) if step > 0 else 0
        summaries.append({
            "episode": ep,
            "steps": step,
            "mav_dead_step": mav_dead_step,
            "mav_mode": mav_mode,
            "red_alive_end": alive_end,
        })
        all_rows.extend(ep_rows)

    # Write CSV
    csv_path = os.path.join(output_dir, f"mav_control_trace_{mav_mode}.csv")
    fieldnames = [
        "episode", "step",
        "raw_a0", "raw_a1", "raw_a2",
        "eff_a0", "eff_a1", "eff_a2",
        "target_pitch_deg", "target_heading_deg", "target_vel_mps",
        "eff_pitch_deg", "eff_heading_deg", "eff_vel_mps",
        "cur_pitch_deg", "cur_roll_deg", "cur_yaw_deg",
        "speed_mps", "altitude_m", "vertical_speed_mps",
        "elevator_cmd", "aileron_cmd", "throttle_cmd",
        "energy_guard_active", "energy_guard_level",
        "pitch_clamped", "roll_clamped", "throttle_boosted",
        "alive", "crash_reason",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved: {csv_path} ({len(all_rows)} rows)")

    # Print first 25 steps summary
    print(f"\n=== {mav_mode.upper()} MAV: first 25 steps ===")
    print(f"{'step':>4s} {'raw_a0':>8s} {'raw_a2':>8s} {'tg_pitch':>8s} {'tg_vel':>7s} {'cur_pitch':>9s} {'cur_roll':>8s} {'speed':>7s} {'alt':>6s} {'elev':>8s} {'thr':>6s} {'EG':>4s}")
    for r in all_rows[:25]:
        if r["episode"] == 0:
            print(f"{r['step']:>4d} {r['raw_a0']:>8.3f} {r['raw_a2']:>8.3f} {r['target_pitch_deg']:>8.1f} {r['target_vel_mps']:>7.0f} {r['cur_pitch_deg']:>9.1f} {r['cur_roll_deg']:>8.1f} {r['speed_mps']:>7.1f} {r['altitude_m']:>6.0f} {r['elevator_cmd']:>8.3f} {r['throttle_cmd']:>6.3f} {r['energy_guard_active']:>4d}")

    print(f"\nSummary: mav_dead_step={summaries[0]['mav_dead_step'] if summaries else '?'}")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir",
                        default="outputs/hetero_entity_recurrent_v2_f22_pid_parallel_env4_500k_main")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--config",
                        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "mav_diagnostics")

    policy, device, meta = _load_model(args.experiment_dir, args.checkpoint)
    print(f"Loaded: {meta.get('policy_arch')} entity_dim={meta.get('entity_dim')}")

    for mode in ["policy", "safe", "zero"]:
        print(f"\n{'='*60}")
        print(f"MAV MODE: {mode}")
        print(f"{'='*60}")
        run_diagnostic(policy, device, args.config, args.output_dir,
                       episodes=args.episodes, max_steps=args.max_steps,
                       mav_mode=mode, seed=args.seed)


if __name__ == "__main__":
    main()
