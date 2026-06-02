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
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


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
        with open(meta_path) as f:
            meta = json.load(f)
        saved_dim = meta.get("actor_obs_dim", actor_dim)
        assert saved_dim == actor_dim, \
            f"Actor dim mismatch: meta={saved_dim} adapter={actor_dim}"

    model = MAPPOActorCritic(actor_obs_dim=actor_dim,
                             critic_state_dim=critic_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()

    for cfg in CONFIGS:
        env = None
        try:
            env = make_env(cfg, env_type="jsbsim_hetero", max_steps=500)
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=0)
            obs, info = env.reset(seed=0)
            ep_ret, ep_len, nan = 0.0, 0, False
            while True:
                result = adapter.adapt_all(obs, info=info,
                                           red_ids=env.red_ids,
                                           blue_ids=env.blue_ids)
                actor_obs_list = [
                    result["actor_obs"].get(
                        rid, np.zeros(actor_dim, dtype=np.float32))
                    for rid in env.red_ids]
                actor_obs_t = torch.as_tensor(
                    np.stack(actor_obs_list), device=device)
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
                    nan = True
                    break
                if all(terminated.values()) or all(truncated.values()):
                    break
            r_alive = sum(1 for s in env.red_planes.values() if s.is_alive)
            b_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
            print(f"{Path(cfg).stem:42s} | ret={ep_ret:+7.1f} "
                  f"len={ep_len:4d} r_alive={r_alive} b_alive={b_alive} "
                  f"nan={nan} actor_dim={actor_dim} critic_dim={critic_dim}")
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
