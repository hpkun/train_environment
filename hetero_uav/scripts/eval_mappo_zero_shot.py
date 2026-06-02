"""Smoke-style zero-shot evaluation for Stage 1 MAPPO baseline.

This is not a formal win-rate experiment. It only verifies that one saved
shared-policy MAPPO model can run across heterogeneous composition configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter  # noqa: E402
from algorithms.mappo.policy import MAPPOActorCritic  # noqa: E402
from algorithms.mappo.opponent_policy import OpponentPolicy  # noqa: E402


DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def _eval_config(config_path: str, model: MAPPOActorCritic,
                 adapter: HeteroObsAdapter, device: torch.device,
                 episodes: int, seed: int, opponent_policy: str) -> dict:
    env = make_env(config_path, env_type="jsbsim_hetero", max_steps=500)
    opponent = OpponentPolicy(mode=opponent_policy, seed=seed + 17)
    returns = []
    lengths = []
    red_alive_counts = []
    blue_alive_counts = []
    nan_detected = False
    actor_dim_ok = True
    critic_dim_ok = True

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_ret = 0.0
            ep_len = 0
            while True:
                result = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                actor_obs_np = np.stack([
                    result["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                    for rid in env.red_ids
                ])
                critic_state_np = result["critic_state"]
                actor_dim_ok = actor_dim_ok and actor_obs_np.shape[1] == model.actor_obs_dim
                critic_dim_ok = critic_dim_ok and critic_state_np.shape[0] == model.critic_state_dim

                actor_obs_t = torch.as_tensor(actor_obs_np, device=device)
                critic_t = torch.as_tensor(critic_state_np, device=device).unsqueeze(0)
                with torch.no_grad():
                    _dist, _value, action, _log_prob, _entropy = model(
                        actor_obs_t, critic_t, deterministic=True)

                actions_dict = {
                    rid: action[i].cpu().numpy().astype(np.float32)
                    for i, rid in enumerate(env.red_ids)
                }
                actions_dict.update(opponent.act(obs, env.blue_ids))

                obs, rewards, terminated, truncated, info = env.step(actions_dict)
                ep_ret += sum(float(rewards.get(rid, 0.0)) for rid in env.red_ids)
                ep_len += 1

                if np.isnan(ep_ret) or _obs_has_nan(obs):
                    nan_detected = True
                    break
                if all(terminated.values()) or all(truncated.values()):
                    break

            red_alive, blue_alive = _alive_counts(env)
            returns.append(ep_ret)
            lengths.append(ep_len)
            red_alive_counts.append(red_alive)
            blue_alive_counts.append(blue_alive)
    finally:
        env.close()

    return {
        "config": config_path,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_red_alive": float(np.mean(red_alive_counts)) if red_alive_counts else 0.0,
        "avg_blue_alive": float(np.mean(blue_alive_counts)) if blue_alive_counts else 0.0,
        "nan_detected": nan_detected,
        "actor_obs_dim_check": actor_dim_ok,
        "critic_state_dim_check": critic_dim_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/mappo_baseline/latest/model.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy",
                        choices=["zero", "random", "rule_nearest"],
                        default="rule_nearest")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = MAPPOActorCritic().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()
    adapter = HeteroObsAdapter()

    print(f"model: {args.model}")
    print(f"opponent_policy: {args.opponent_policy}")
    print(f"episodes: {args.episodes}")
    for config_path in args.configs:
        if not Path(config_path).exists():
            continue
        result = _eval_config(
            config_path, model, adapter, device,
            args.episodes, args.seed, args.opponent_policy)
        print(f"=== {result['config']} ===")
        print(f"avg_return: {result['avg_return']:.2f}")
        print(f"avg_length: {result['avg_length']:.1f}")
        print(f"avg_red_alive: {result['avg_red_alive']:.2f}")
        print(f"avg_blue_alive: {result['avg_blue_alive']:.2f}")
        print(f"nan_detected: {result['nan_detected']}")
        print(f"actor_obs_dim check: {result['actor_obs_dim_check']}")
        print(f"critic_state_dim check: {result['critic_state_dim_check']}")


if __name__ == "__main__":
    main()
