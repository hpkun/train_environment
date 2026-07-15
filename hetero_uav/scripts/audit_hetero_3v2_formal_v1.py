"""Short rule-v-rule or random finite-rollout audit; never trains."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    p.add_argument("--episodes",type=int,default=3); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--red-policy",choices=("rule","random"),default="rule")
    p.add_argument("--max-steps",type=int,default=1000)
    p.add_argument("--perturb-initial",action="store_true")
    p.add_argument("--output",default="outputs/hetero_3v2_formal_v1_audit.json")
    a=p.parse_args(); env=make_env(str(ROOT/a.config)); rule=PaperGreedyOpponent(); rng=np.random.default_rng(a.seed)
    rows=[]
    for ep in range(a.episodes):
        perturbation={}
        if a.perturb_initial:
            local=np.random.default_rng(a.seed+ep)
            perturbation={aid:{"lat_deg":float(local.uniform(-0.001,0.001)),
                               "lon_deg":float(local.uniform(-0.001,0.001)),
                               "altitude_m":float(local.uniform(-50,50)),
                               "speed_mps":float(local.uniform(-3,3)),
                               "yaw_deg":float(local.uniform(-2,2))}
                          for aid in (*env.red_ids,*env.blue_ids)}
        _,info=env.reset(seed=a.seed+ep,options={"audit_initial_perturbation":perturbation})
        for step in range(a.max_steps):
            actions=(rule.actions(env,"red") if a.red_policy=="rule" else
                     {x:rng.uniform(-1,1,3).astype(np.float32) for x in env.red_ids})
            obs,reward,term,trunc,info=env.step(actions)
            if not np.isfinite(np.r_[info["critic_state"],list(reward.values())]).all():
                raise ValueError("non-finite formal rollout")
            if info["team_done"]: break
        red_launches=sum(x["event"]=="launch" and x["shooter_id"].startswith("red") for x in env.event_log)
        blue_launches=sum(x["event"]=="launch" and x["shooter_id"].startswith("blue") for x in env.event_log)
        red_hits=sum(x["event"]=="hit" and x["shooter_id"].startswith("red") for x in env.event_log)
        blue_hits=sum(x["event"]=="hit" and x["shooter_id"].startswith("blue") for x in env.event_log)
        rows.append({"episode":ep,"steps":step+1,"outcome":info["outcome"],
                     "launches":red_launches+blue_launches,"hits":red_hits+blue_hits,
                     "red_launches":red_launches,"blue_launches":blue_launches,
                     "red_hits":red_hits,"blue_hits":blue_hits,
                     "red_alive":info["red_alive"],"blue_alive":info["blue_alive"],
                     "perturbation":perturbation})
    result={"red_policy":a.red_policy,"episodes":rows,"total_launches":sum(x["launches"] for x in rows),
            "total_hits":sum(x["hits"] for x in rows),
            "red_launches":sum(x["red_launches"] for x in rows),
            "blue_launches":sum(x["blue_launches"] for x in rows),
            "red_hits":sum(x["red_hits"] for x in rows),
            "blue_hits":sum(x["blue_hits"] for x in rows),
            "red_wins":sum(x["outcome"]=="red_win" for x in rows),
            "blue_wins":sum(x["outcome"]=="blue_win" for x in rows),
            "draws":sum(x["outcome"]=="draw" for x in rows)}
    out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); env.close()
if __name__=="__main__": main()
