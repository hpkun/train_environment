"""Run symmetric GreedyPaperOpponent formal-environment episodes."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_env import TAMPaperCombatEnv

def run(scenario="paper_nominal_2v2",episodes=20,seed=1,max_steps=1000,
        blue_policy="greedy"):
    env=TAMPaperCombatEnv(scenario,"all",max_steps=max_steps);rows=[]
    try:
        for ep in range(episodes):
            env.reset(seed=seed+ep);info={}
            for step in range(max_steps):
                actions=env.build_rule_actions()
                if blue_policy=="level":
                    from aircombat_env_v1.paper_opponent import MANOEUVRES
                    import numpy as np
                    for aid in [x for x in env.controlled_ids if x.startswith("blue_")]:
                        actions[aid]=np.asarray(MANOEUVRES["level"],dtype=np.int64)
                _,_,terminated,truncated,info=env.step(actions)
                if terminated or truncated:break
            row={"episode":ep,"winner":info["winner"],"steps":step+1,"red_alive":info["alive_red"],"blue_alive":info["alive_blue"],
                 "red_missile_kills":info["red_missile_kills"],"blue_missile_kills":info["blue_missile_kills"],
                 "red_crashes":info["red_crashes"],"blue_crashes":info["blue_crashes"],"missiles_fired":info["missiles_fired"],
                 "missile_hits":info["missile_hits"],"timeout":info["termination_reason"]=="episode_limit",
                 "invalid":info["invalid_episode"],"envelope_violation":info["flight_envelope_violation"]}
            rows.append(row);print(json.dumps(row))
    finally:env.close()
    summary={"scenario":scenario,"blue_policy":blue_policy,"episodes":episodes,"valid_terminations":sum(not r["timeout"] and not r["invalid"] for r in rows),
             "missiles_fired":sum(r["missiles_fired"] for r in rows),"missile_hits":sum(r["missile_hits"] for r in rows),
             "red_missile_kills":sum(r["red_missile_kills"] for r in rows),"blue_missile_kills":sum(r["blue_missile_kills"] for r in rows),
             "invalid":sum(r["invalid"] for r in rows),"envelope_violations":sum(r["envelope_violation"] for r in rows)}
    print(json.dumps(summary,indent=2));return summary,rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--scenario",default="paper_nominal_2v2",choices=("paper_nominal_1v1","paper_nominal_2v2"));p.add_argument("--episodes",type=int,default=20);p.add_argument("--seed",type=int,default=1);p.add_argument("--max-steps",type=int,default=1000);p.add_argument("--blue-policy",choices=("greedy","level"),default="greedy");a=p.parse_args();run(a.scenario,a.episodes,a.seed,a.max_steps,a.blue_policy)
if __name__=="__main__":main()
