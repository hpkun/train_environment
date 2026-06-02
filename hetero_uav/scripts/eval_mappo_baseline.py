"""Smoke-evaluate MAPPO baseline. Auto-infers v1/v2 from model meta.json."""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy",
                        choices=["zero", "random", "rule_nearest"],
                        default="rule_nearest")
    parser.add_argument("--obs-adapter-version", choices=["v1", "v2"],
                        default=None)
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
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17)

    print(f"model: {args.model}")
    print(f"obs_adapter_version: {version}")
    print(f"actor_obs_dim: {actor_dim}")
    print(f"critic_state_dim: {critic_dim}")
    print(f"observation_mode: {meta.get('observation_mode', '?')}")
    print(f"config: {args.config}")
    print(f"opponent_policy: {args.opponent_policy}")

    env = None
    try:
        env = make_env(args.config, env_type="jsbsim_hetero", max_steps=500)
        obs_mode = getattr(env, "observation_mode", "brma_sensor")
        if version == "v2" and obs_mode != "mav_shared_geo":
            raise SystemExit(
                "--obs-adapter-version v2 requires observation_mode=mav_shared_geo")

        returns = []
        lengths = []
        crashes = 0
        nan_count = 0

        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep + 1)
            ep_ret = 0.0
            ep_len = 0
            while True:
                result = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids,
                    blue_ids=env.blue_ids)
                actor_obs_list = [
                    result["actor_obs"].get(rid,
                                            np.zeros(actor_dim, dtype=np.float32))
                    for rid in env.red_ids]
                actor_obs_t = torch.as_tensor(np.stack(actor_obs_list),
                                              device=device)

                with torch.no_grad():
                    _, _, action, _, _ = model(
                        actor_obs_t,
                        torch.zeros(1, critic_dim, device=device),
                        deterministic=True)

                actions_dict = {}
                for i, rid in enumerate(env.red_ids):
                    actions_dict[rid] = action[i].cpu().numpy().astype(np.float32)
                actions_dict.update(opponent.act(obs, env.blue_ids))

                obs, rewards_dict, terminated, truncated, info = env.step(
                    actions_dict)
                ep_ret += sum(float(rewards_dict.get(rid, 0.0))
                              for rid in env.red_ids)
                ep_len += 1

                if np.isnan(ep_ret):
                    nan_count += 1
                    break
                if all(terminated.values()) or all(truncated.values()):
                    for rid in env.red_ids:
                        dr = str(info.get(rid, {}).get("death_reason", ""))
                        if "crash" in dr.lower():
                            crashes += 1
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)

        print(f"episodes: {len(returns)}")
        print(f"avg_return: {np.mean(returns):.2f}")
        print(f"avg_length: {np.mean(lengths):.1f}")
        print(f"crashes: {crashes}")
        print(f"nan_detected: {nan_count > 0}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
