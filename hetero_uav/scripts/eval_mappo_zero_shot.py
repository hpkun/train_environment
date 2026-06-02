"""Zero-shot evaluation for MAPPO baseline. Auto-infers v1/v2 from meta."""
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

    configs = args.configs or (V2_CONFIGS if version == "v2" else V1_CONFIGS)

    print(f"obs_adapter_version: {version}")
    print(f"actor_obs_dim: {actor_dim}")
    print(f"critic_state_dim: {critic_dim}")
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
                print(f"SKIP {cfg_path}: v2 requires mav_shared_geo, got {obs_mode}")
                continue
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=0)
            obs, info = env.reset(seed=args.seed)
            ep_ret, ep_len, nan = 0.0, 0, False
            while True:
                result = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                actor_obs_np = np.stack([
                    result["actor_obs"].get(rid, np.zeros(actor_dim, dtype=np.float32))
                    for rid in env.red_ids])
                actor_obs_t = torch.as_tensor(actor_obs_np, device=device)
                with torch.no_grad():
                    _, _, action, _, _ = model(
                        actor_obs_t,
                        torch.zeros(1, critic_dim, device=device),
                        deterministic=True)
                actions_dict = {}
                for i, rid in enumerate(env.red_ids):
                    actions_dict[rid] = action[i].cpu().numpy().astype(np.float32)
                actions_dict.update(opponent.act(obs, env.blue_ids))
                obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
                ep_ret += sum(float(rewards_dict.get(rid, 0.0))
                              for rid in env.red_ids)
                ep_len += 1
                if np.isnan(ep_ret):
                    nan = True; break
                if all(terminated.values()) or all(truncated.values()):
                    break
            r_alive = sum(1 for s in env.red_planes.values() if s.is_alive)
            b_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
            print(f"{Path(cfg_path).stem:45s} | ret={ep_ret:+7.1f} "
                  f"len={ep_len:4d} r_alive={r_alive} b_alive={b_alive} "
                  f"nan={nan} actor_dim_ok={actor_obs_np.shape[1]==actor_dim} "
                  f"critic_dim_ok={result['critic_state'].shape[0]==critic_dim}")
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
