"""Run deterministic rule-vs-rule checks in the simplified environment."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import SimpleTAMCombatEnv

def run_episode(scenario,seed,weapon_enabled_agent_ids=None):
    env=SimpleTAMCombatEnv(scenario,"all",weapon_enabled_agent_ids=weapon_enabled_agent_ids);env.reset(seed=seed);launches=hits=0;events=[]
    try:
        for step in range(1000):
            obs,rewards,terminated,truncated,info=env.step(env.build_rule_actions());events.extend(info["events"])
            assert all(np.isfinite(x).all() for x in obs.values()) and np.isfinite(list(rewards.values())).all()
            if terminated or truncated:break
        launches=sum(e.get("event_type")=="missile_launch" for e in events);hits=sum(e.get("hit",False) for e in events)
        return {"steps":step+1,"winner":info["winner"],"termination_reason":info["termination_reason"],"missile_launches":launches,
          "missile_hits":hits,"red_missile_kills":info["red_missile_kills"],"blue_missile_kills":info["blue_missile_kills"],
          "red_crashes":info["red_crashes"],"blue_crashes":info["blue_crashes"],"boundary_deaths":info["boundary_deaths"],
          "numerical_invalid":info["numerical_invalid"],"valid_target_reselections":info["valid_target_reselections"],
          "flight_envelope_violation":info["flight_envelope_violation"]}
    finally:env.close()

def run_group(scenario,episodes,seed,weapon_enabled_agent_ids=None):
    rows=[run_episode(scenario,seed+i,weapon_enabled_agent_ids) for i in range(episodes)]
    return {"scenario":scenario,"episodes":episodes,"missile_launches":sum(x["missile_launches"] for x in rows),
      "missile_hits":sum(x["missile_hits"] for x in rows),"red_missile_kills":sum(x["red_missile_kills"] for x in rows),
      "blue_missile_kills":sum(x["blue_missile_kills"] for x in rows),"red_crashes":sum(x["red_crashes"] for x in rows),
      "blue_crashes":sum(x["blue_crashes"] for x in rows),"boundary_deaths":sum(x["boundary_deaths"] for x in rows),
      "numerical_invalid":sum(x["numerical_invalid"] for x in rows),"hit_episodes":sum(x["missile_hits"]>0 for x in rows),
      "launch_episodes":sum(x["missile_launches"]>0 for x in rows),"crash_terminated_episodes":sum(x["termination_reason"] in ("red_eliminated","blue_eliminated") and x["missile_hits"]==0 for x in rows),
      "valid_target_reselections":sum(x["valid_target_reselections"] for x in rows),"envelope_violation_episodes":sum(x["flight_envelope_violation"] for x in rows),"rows":rows}

def main():
    p=argparse.ArgumentParser();p.add_argument("--check",choices=("all","bilateral_1v1","red_only_1v1","rule_2v2"),default="all");p.add_argument("--episodes",type=int,default=10);p.add_argument("--seed",type=int,default=1);p.add_argument("--weapon-enabled-agent-ids",nargs="*");a=p.parse_args()
    result={}
    if a.check in ("all","bilateral_1v1"):result["bilateral_1v1"]=run_group("simple_paper_1v1",a.episodes,a.seed,a.weapon_enabled_agent_ids)
    if a.check in ("all","red_only_1v1"):result["red_only_1v1"]=run_group("simple_paper_1v1",a.episodes,a.seed,{"red_0"})
    if a.check in ("all","rule_2v2"):result["rule_2v2"]=run_group("simple_paper_2v2",a.episodes,a.seed,a.weapon_enabled_agent_ids)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
