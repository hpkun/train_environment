"""Short acceptance checks for simple_paper_3v2_hetero."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import HETERO_SCENARIO,SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

def record(info,result):
    result["numerical_invalid"]+=info["numerical_invalid"];result["crash"]+=info["red_crashes"]+info["blue_crashes"]
    result["boundary"]+=info["boundary_deaths"]
    for event in info.get("events",[]):
        if event.get("event_type")=="missile_launch":result["mav_launch_count" if event["shooter_id"]=="red_mav_0" else "uav_launch_count"]+=1

def random_rollout(steps=64):
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=steps);obs,info=env.reset(seed=1);rng=np.random.default_rng(1)
    result={"completed_steps":0,"reward_finite":True,"numerical_invalid":0,"crash":0,"boundary":0,"mav_launch_count":0,"uav_launch_count":0,"termination_reasons":[]}
    try:
        for _ in range(steps):
            actions={aid:rng.uniform(-1,1,3).astype(np.float32) for aid in env.controlled_ids}
            obs,rewards,terminated,truncated,info=env.step(actions);result["completed_steps"]+=1
            result["reward_finite"]&=bool(np.isfinite(list(rewards.values())).all() and all(np.isfinite(x).all() for x in obs.values()));record(info,result)
            if terminated or truncated:result["termination_reasons"].append(info["termination_reason"]);break
        return result
    finally:env.close()

def rule_checks(episodes=3):
    results=[]
    for episode in range(episodes):
        env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=64);env.reset(seed=100+episode);summary={"episode":episode,"steps":0,"winner":None,"termination_reason":None}
        try:
            for _ in range(64):
                _,_,terminated,truncated,info=env.step(env.build_rule_actions());summary["steps"]+=1
                if terminated or truncated:summary.update({"winner":info["winner"],"termination_reason":info["termination_reason"]});break
            results.append(summary)
        finally:env.close()
    return results

def mappo_smoke(steps=256):
    env=SimpleTAMCombatEnv(HETERO_SCENARIO);adapter=SimpleMAPPOAdapter(env);obs,state,_=adapter.reset(seed=200)
    model=SharedMAPPOActorCritic(81,243);buffer=RolloutBuffer(steps,3,81,243);active=np.ones(3,np.float32);invalid=crash=boundary=0
    try:
        for index in range(steps):
            with torch.no_grad():actions,logp,value,_=model.act(torch.tensor(obs),torch.tensor(state))
            next_obs,next_state,rewards,done,next_active,info=adapter.step(actions.numpy());buffer.store(obs,state,actions.numpy(),logp.numpy(),rewards,done,value.item(),active)
            invalid+=info["numerical_invalid"];crash+=info["red_crashes"]+info["blue_crashes"];boundary+=info["boundary_deaths"]
            if done:obs,state,_=adapter.reset(seed=201+index);active=np.ones(3,np.float32)
            else:obs,state,active=next_obs,next_state,next_active
        next_value=0. if buffer.team_dones[-1] else model.value(torch.tensor(state)).item();stats=MAPPOTrainer(model,ppo_epochs=1).update(buffer,next_value)
        return {"completed_env_steps":steps,"update_finite":bool(np.isfinite(list(stats.values())).all()),"stats":stats,
                "numerical_invalid":invalid,"crash":crash,"boundary":boundary}
    finally:env.close()

def run_checks():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO);obs,info=env.reset(seed=1);adapter=SimpleMAPPOAdapter(env)
    reset={"obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,"agent_roles":info["agent_roles"],
           "missile_capacity":{a.agent_id:a.missile_capacity for a in env.agents},"observation_finite":all(np.isfinite(x).all() for x in obs.values())};env.close()
    return {"reset":reset,"random_rollout_64":random_rollout(),"rule_checks":rule_checks(),"mappo_smoke_256":mappo_smoke()}

def main():print(json.dumps(run_checks(),indent=2))
if __name__=="__main__":main()
