"""Validate flight behavior of the randomly initialized categorical TAM policy."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import TAMCategoricalRecurrentHAPPOPolicy
from algorithms.happo.rollout_safety import sanitize_policy_inputs, zero_inactive_hidden
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


def _alive_mask(env):
    return np.asarray([
        float(env.red_planes[rid].is_alive) for rid in env.red_ids
    ], dtype=np.float32)


def _fixed_blue_actions(env):
    neutral = np.asarray([env.tam_action_levels - 1, 20, 0, 20], dtype=np.int64)
    return {bid: neutral.copy() for bid in env.blue_ids}


def _disable_blue_missiles(env):
    for sim in env.blue_planes.values():
        sim.num_missiles = 0
        sim.num_left_missiles = 0


def run_validation(
    config: str, *, output_dir: str | Path, episodes: int = 10,
    steps: int = 300, device: str = "cpu",
    modes=("stochastic", "deterministic"), seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = TAMCategoricalRecurrentHAPPOPolicy().to(torch.device(device))
    policy.eval()
    adapter = HeteroObsAdapterV2()
    result = {
        "config": config, "episodes_per_mode": int(episodes),
        "steps_per_episode": int(steps),
        "neutral_action_centers": policy.neutral_action_centers,
        "neutral_action_init_std_bins": policy.neutral_action_init_std_bins,
        "modes": {},
    }
    for mode in modes:
        deterministic = mode == "deterministic"
        episode_rows = []
        all_actions = []
        death_reasons = Counter()
        for episode in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max(steps, 300))
            obs, info = env.reset(seed=seed + episode)
            _disable_blue_missiles(env)
            roles = [0 if rid == "red_0" else 1 for rid in env.red_ids]
            hidden = policy.init_hidden(len(env.red_ids), torch.device(device))
            override_detected = False
            executed_steps = 0
            for step in range(steps):
                adapted = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids
                )
                actor_obs = np.stack([
                    adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
                    for rid in env.red_ids
                ])
                active = _alive_mask(env)
                sanitized = sanitize_policy_inputs(
                    actor_obs, active, critic_state=adapted["critic_state"],
                    rnn_hidden=hidden.detach().cpu().numpy(),
                )
                with torch.no_grad():
                    out = policy.act(
                        torch.as_tensor(sanitized["actor_obs"], device=device),
                        roles=roles,
                        critic_state=torch.as_tensor(sanitized["critic_state"], device=device),
                        deterministic=deterministic,
                        rnn_hidden=torch.as_tensor(sanitized["rnn_hidden"], device=device),
                    )
                red_actions = out["action"].detach().cpu().numpy().astype(np.int64)
                all_actions.append(red_actions.copy())
                hidden_np = zero_inactive_hidden(
                    out["rnn_hidden"].detach().cpu().numpy(), active
                )
                hidden = torch.as_tensor(hidden_np, device=device)
                action_dict = {
                    rid: red_actions[index].copy()
                    for index, rid in enumerate(env.red_ids)
                }
                action_dict.update(_fixed_blue_actions(env))
                obs, _rewards, terminated, truncated, info = env.step(action_dict)
                executed_steps = step + 1
                for index, rid in enumerate(env.red_ids):
                    effective = np.asarray(env._last_effective_actions.get(rid, []))
                    if not np.array_equal(effective, red_actions[index]):
                        override_detected = True
                if all(terminated.values()) or all(truncated.values()):
                    break
            mav_alive = bool(env.red_planes["red_0"].is_alive)
            uav_alive = [bool(env.red_planes[rid].is_alive) for rid in env.red_ids[1:]]
            reason = env._death_reasons.get("red_0") or ("alive" if mav_alive else "unknown")
            death_reasons[reason] += 1
            episode_rows.append({
                "episode": episode, "env_steps": executed_steps,
                "mav_alive": mav_alive,
                "red_uav_survival_rate": float(np.mean(uav_alive)) if uav_alive else 1.0,
                "mav_death_reason": reason,
                "red_action_override_detected": override_detected,
            })
            env.close()
        actions = np.concatenate(all_actions, axis=0)
        result["modes"][mode] = {
            "mav_survival_rate": float(np.mean([row["mav_alive"] for row in episode_rows])),
            "red_uav_survival_rate": float(np.mean([row["red_uav_survival_rate"] for row in episode_rows])),
            "death_reasons": dict(death_reasons),
            "throttle_high_rate": float(np.mean(actions[:, 0] >= policy.action_levels - 4)),
            "surface_middle_rate": float(np.mean((actions[:, [1, 3]] >= 12) & (actions[:, [1, 3]] <= 28))),
            "elevator_bin_mean": float(np.mean(actions[:, 2])),
            "action_bin_usage": [int(np.unique(actions[:, axis]).size) for axis in range(4)],
            "episodes": episode_rows,
        }

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tam_categorical_initial_policy_flight_validation.json"
    md_path = out_dir / "tam_categorical_initial_policy_flight_validation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# TAM Categorical Initial Policy Flight Validation", ""]
    for mode, record in result["modes"].items():
        lines.extend([
            f"## {mode}",
            f"- MAV survival: {record['mav_survival_rate']:.3f}",
            f"- Red UAV survival: {record['red_uav_survival_rate']:.3f}",
            f"- Death reasons: {record['death_reasons']}",
            f"- Throttle high-bin rate: {record['throttle_high_rate']:.3f}",
            f"- Surface middle-bin rate: {record['surface_middle_rate']:.3f}",
            f"- Per-axis bin usage: {record['action_bin_usage']}", "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_validation(
        args.config, output_dir=args.output_dir, episodes=args.episodes,
        steps=args.steps, device=args.device, seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
