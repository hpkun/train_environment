"""Stage 1 zero-shot diagnostic: same checkpoint across compositions."""
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
from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
from algorithms.mappo.policy import MAPPOActorCritic
from algorithms.mappo.opponent_policy import OpponentPolicy


CONFIGS = [
    "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]


def _alive(env):
    r = sum(1 for s in env.red_planes.values() if s.is_alive)
    b = sum(1 for s in env.blue_planes.values() if s.is_alive)
    return r, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = MAPPOActorCritic().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.eval()
    adapter = HeteroObsAdapter()

    for cfg in CONFIGS:
        env = None
        try:
            env = make_env(cfg, env_type="jsbsim_hetero", max_steps=500)
            opponent = OpponentPolicy(mode=args.opponent_policy, seed=0)
            obs, info = env.reset(seed=0)
            ep_ret, ep_len, crashes, nan = 0.0, 0, 0, False
            while True:
                result = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids,
                    blue_ids=env.blue_ids)
                actor_obs_list = []
                for rid in env.red_ids:
                    actor_obs_list.append(result["actor_obs"].get(
                        rid, np.zeros(140, dtype=np.float32)))
                actor_obs_t = torch.as_tensor(
                    np.stack(actor_obs_list), device=device)

                with torch.no_grad():
                    _, _, action, _, _ = model(
                        actor_obs_t,
                        torch.zeros(1, 700, device=device),
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
                    for rid in env.red_ids:
                        dr = str(info.get(rid, {}).get("death_reason", ""))
                        if "crash" in dr.lower():
                            crashes += 1
                    break
            r_alive, b_alive = _alive(env)
            result0 = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            r0dim = result0["actor_obs"].get(env.red_ids[0], np.zeros(140)).shape[0]
            csdim = result0["critic_state"].shape[0]

            print(f"{Path(cfg).stem:45s} | "
                  f"ret={ep_ret:+7.1f} len={ep_len:4d} "
                  f"r_ali={r_alive} b_ali={b_alive} "
                  f"nan={nan} crashes={crashes} "
                  f"actor_dim={r0dim} critic_dim={csdim}")
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
