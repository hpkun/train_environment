"""V2 zero-shot smoke: same checkpoint on 3v2 and 5v4 shared_geo configs."""
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
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from algorithms.mappo.policy import MAPPOActorCritic
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    device = torch.device(args.device)
    adapter = HeteroObsAdapterV2()
    actor_dim = adapter.flat_actor_obs_dim
    critic_dim = adapter.critic_state_dim

    meta_path = Path(args.model).parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("obs_adapter_version") != "v2":
            raise ValueError(f"Expected obs_adapter_version=v2, got {meta.get('obs_adapter_version')}")
        if meta.get("actor_obs_dim") != actor_dim:
            raise ValueError(f"Actor dim mismatch: meta={meta.get('actor_obs_dim')} adapter={actor_dim}")
        if meta.get("critic_state_dim") != critic_dim:
            raise ValueError(f"Critic dim mismatch: meta={meta.get('critic_state_dim')} adapter={critic_dim}")
    else:
        print(f"warning: missing meta.json for {args.model}")

    model = MAPPOActorCritic(actor_obs_dim=actor_dim,
                             critic_state_dim=critic_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()

    for cfg in CONFIGS:
        env = None
        try:
            env = make_env(cfg, env_type="jsbsim_hetero", max_steps=500)
            if getattr(env, "observation_mode", "brma_sensor") != "mav_shared_geo":
                raise ValueError(f"V2 zero-shot smoke requires mav_shared_geo config: {cfg}")
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=0)
            returns, lengths, red_alive_counts, blue_alive_counts = [], [], [], []
            nan_detected = False
            actor_dim_ok = True
            critic_dim_ok = True
            for ep in range(args.episodes):
                obs, info = env.reset(seed=ep)
                ep_ret, ep_len = 0.0, 0
                while True:
                    if _obs_has_nan(obs):
                        nan_detected = True
                        break
                    result = adapter.adapt_all(obs, info=info,
                                               red_ids=env.red_ids,
                                               blue_ids=env.blue_ids)
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
            print(f"=== {cfg} ===")
            print(f"episodes: {args.episodes}")
            print(f"avg_return: {np.mean(returns):.2f}")
            print(f"avg_length: {np.mean(lengths):.1f}")
            print(f"avg_red_alive: {np.mean(red_alive_counts):.2f}")
            print(f"avg_blue_alive: {np.mean(blue_alive_counts):.2f}")
            print(f"nan_detected: {nan_detected}")
            print(f"actor_dim_ok: {actor_dim_ok}")
            print(f"critic_dim_ok: {critic_dim_ok}")
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
