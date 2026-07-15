"""Deterministic evaluation for a formal-v1 Pure HAPPO checkpoint."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from algorithms.pure_happo import PureHAPPOPolicy
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ENV_TYPE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/formal_v1_eval.json")
    a = p.parse_args()
    env = make_env(str(ROOT / a.config) if not Path(a.config).is_absolute() else a.config)
    model = Path(a.checkpoint); model = model if model.is_absolute() else ROOT / model
    meta = json.loads((model.parent / "meta.json").read_text(encoding="utf-8"))
    if meta.get("formal_contract") != ENV_TYPE or meta.get("credit_mode") != "shared_alive_team_mean":
        raise ValueError("checkpoint is not a formal-v1 shared-credit checkpoint")
    policy = PureHAPPOPolicy(meta["actor_obs_dim"], meta["critic_state_dim"],
                             meta["action_dim"], meta["num_agents"]).to(a.device)
    policy.load_state_dict(torch.load(model, map_location=a.device, weights_only=True)); policy.eval()
    rows=[]
    for episode in range(a.episodes):
        obs, info = env.reset(seed=a.seed + episode); total=0.0
        while True:
            actor=np.stack([obs[x]["flat"] for x in env.red_ids])
            with torch.no_grad(): out=policy.act(actor, critic_state=info["critic_state"], deterministic=True)
            action=out["action"].cpu().numpy()*info["active_mask"][:,None]
            obs,reward,term,trunc,info=env.step({x:action[i] for i,x in enumerate(env.red_ids)})
            total += float(np.mean(list(reward.values())))
            if info["team_done"]: break
        rows.append({"episode":episode,"return":total,"outcome":info["outcome"],
                     "red_alive":info["red_alive"],"blue_alive":info["blue_alive"],
                     "launches":sum(e["event"]=="launch" for e in env.event_log),
                     "hits":sum(e["event"]=="hit" for e in env.event_log)})
    summary={"episodes":rows,"avg_return":float(np.mean([x["return"] for x in rows])),
             "red_win_rate":float(np.mean([x["outcome"]=="red_win" for x in rows]))}
    out=Path(a.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
    env.close()

if __name__ == "__main__": main()
