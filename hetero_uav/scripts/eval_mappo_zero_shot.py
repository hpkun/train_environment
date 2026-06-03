"""Zero-shot evaluation for MAPPO baseline. Auto-infers v1/v2 from meta."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from algorithms.mappo.adapter_utils import (
    load_model_meta,
    make_mappo_model_for_adapter,
    make_obs_adapter,
    resolve_obs_adapter_version,
    validate_model_dims,
)
from algorithms.mappo.opponent_policy import OpponentPolicy

V1_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]
V2_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy",
                        choices=["zero", "random", "rule_nearest"],
                        default="rule_nearest")
    parser.add_argument("--obs-adapter-version", choices=["v1", "v2"],
                        default=None)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    meta = load_model_meta(args.model)
    version = resolve_obs_adapter_version(args.obs_adapter_version, meta)
    adapter = make_obs_adapter(version)
    validate_model_dims(adapter, meta)
    actor_dim = adapter.flat_actor_obs_dim
    critic_dim = adapter.critic_state_dim

    model = make_mappo_model_for_adapter(adapter, device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()

    explicit_configs = args.configs is not None
    configs = args.configs or (V2_CONFIGS if version == "v2" else V1_CONFIGS)

    summary_records: list[dict] = []

    print(f"obs_adapter_version: {version}")
    print(f"actor_obs_dim: {actor_dim}")
    print(f"critic_state_dim: {critic_dim}")
    print(f"episodes: {args.episodes}")
    print(f"configs: {configs}")

    for cfg_path in configs:
        if not Path(cfg_path).exists():
            print(f"SKIP {cfg_path} (not found)")
            continue
        env = None
        try:
            env = make_env(cfg_path, env_type="jsbsim_hetero", max_steps=500)
            obs_mode = getattr(env, "observation_mode", "brma_sensor")
            if version == "v2" and obs_mode != "mav_shared_geo":
                message = (
                    f"v2 requires observation_mode=mav_shared_geo, "
                    f"got {obs_mode} for {cfg_path}"
                )
                if explicit_configs:
                    raise ValueError(message)
                print(f"SKIP {cfg_path}: {message}")
                continue
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=0)
            returns, lengths, red_alive_counts, blue_alive_counts = [], [], [], []
            nan_detected = False
            actor_dim_ok = True
            critic_dim_ok = True
            for ep in range(args.episodes):
                obs, info = env.reset(seed=args.seed + ep)
                ep_ret, ep_len = 0.0, 0
                while True:
                    if _obs_has_nan(obs):
                        nan_detected = True
                        break
                    result = adapter.adapt_all(
                        obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                    actor_obs_np = np.stack([
                        result["actor_obs"].get(
                            rid, np.zeros(actor_dim, dtype=np.float32))
                        for rid in env.red_ids])
                    critic_state_np = result["critic_state"]
                    actor_dim_ok = actor_dim_ok and actor_obs_np.shape[1] == actor_dim
                    critic_dim_ok = critic_dim_ok and critic_state_np.shape[0] == critic_dim
                    if np.isnan(actor_obs_np).any() or np.isnan(critic_state_np).any():
                        nan_detected = True
                        break
                    actor_obs_t = torch.as_tensor(actor_obs_np, device=device)
                    critic_t = torch.as_tensor(critic_state_np, device=device).unsqueeze(0)
                    with torch.no_grad():
                        _, _, action, _, _ = model(
                            actor_obs_t, critic_t, deterministic=True)
                    action_np = action.cpu().numpy()
                    if np.isnan(action_np).any():
                        nan_detected = True
                        break
                    actions_dict = {
                        rid: action_np[i].astype(np.float32)
                        for i, rid in enumerate(env.red_ids)
                    }
                    actions_dict.update(opponent.act(obs, env.blue_ids))
                    obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
                    ep_ret += sum(float(rewards_dict.get(rid, 0.0))
                                  for rid in env.red_ids)
                    ep_len += 1
                    if np.isnan(ep_ret):
                        nan_detected = True
                        break
                    if all(terminated.values()) or all(truncated.values()):
                        break
                r_alive, b_alive = _alive_counts(env)
                returns.append(ep_ret)
                lengths.append(ep_len)
                red_alive_counts.append(r_alive)
                blue_alive_counts.append(b_alive)

            print(f"=== {cfg_path} ===")
            print(f"avg_return: {np.mean(returns):.2f}")
            print(f"avg_length: {np.mean(lengths):.1f}")
            print(f"avg_red_alive: {np.mean(red_alive_counts):.2f}")
            print(f"avg_blue_alive: {np.mean(blue_alive_counts):.2f}")
            print(f"nan_detected: {nan_detected}")
            print(f"actor_dim_ok: {actor_dim_ok}")
            print(f"critic_dim_ok: {critic_dim_ok}")

            summary_records.append({
                "obs_adapter_version": version,
                "config": cfg_path,
                "episodes": args.episodes,
                "avg_return": float(np.mean(returns)),
                "avg_length": float(np.mean(lengths)),
                "avg_red_alive": float(np.mean(red_alive_counts)),
                "avg_blue_alive": float(np.mean(blue_alive_counts)),
                "nan_detected": nan_detected,
                "actor_dim_ok": actor_dim_ok,
                "critic_dim_ok": critic_dim_ok,
            })
        finally:
            if env is not None:
                env.close()

    if args.summary_json and summary_records:
        import os as _os
        _os.makedirs(_os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(summary_records, f, indent=2)


if __name__ == "__main__":
    main()
