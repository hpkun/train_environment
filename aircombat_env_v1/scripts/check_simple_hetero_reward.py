"""Short formula, event, environment, and MAPPO checks for MAV reward v2."""
from __future__ import annotations
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import HETERO_SCENARIO,SimpleTAMCombatEnv
from aircombat_env_v1.simple_hetero_reward import LegacySimpleMAVReward,PaperTable1MAVReward
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

def aircraft(agent_id,side,role,position,velocity=(1,0,0),alive=True,death_reason=None):
    return SimpleNamespace(agent_id=agent_id,side=side,role=role,position=np.asarray(position,float),velocity=np.asarray(velocity,float),alive=alive,death_reason=death_reason)

def call(reward,mav,agents,missiles=(),events=(),alive_start=None,perception=None,out=()):
    return reward.compute(mav,agents,list(missiles),list(events),alive_start or {mav.agent_id:mav.alive},set(out),perception or {})

def formula_checks():
    expected={0:-1.,7000:-.5,13999:-1/14000,14000:-.5,21000:-.25,28000:.2,30000:.2}
    distance={str(d):PaperTable1MAVReward.distance_reward(d) for d in expected}
    for d,value in expected.items():assert distance[str(d)]==value or np.isclose(distance[str(d)],value)
    mav=aircraft("red_mav_0","red","mav",(0,0,0));b0=aircraft("blue_0","blue","uav",(10000,0,0),(-1,0,0));b1=aircraft("blue_1","blue","uav",(0,10000,0),(0,-1,0));agents=[mav,b0,b1]
    _,clear=call(PaperTable1MAVReward(),mav,agents);missile=SimpleNamespace(alive=True,target_id=mav.agent_id);_,threat=call(PaperTable1MAVReward(),mav,agents,[missile])
    assert clear["r_safety_threat"]==0 and threat["r_safety_threat"]==-1 and clear["r_safety_aspect"]==-2
    return {"distance_segments":distance,"threat_clear":clear["r_safety_threat"],"threat_incoming":threat["r_safety_threat"],"two_enemy_aspect_sum":clear["r_safety_aspect"]}

def support_checks():
    mav=aircraft("red_mav_0","red","mav",(100,0,0));r0=aircraft("red_uav_0","red","uav",(0,0,0));r1=aircraft("red_uav_1","red","uav",(2,0,0));b0=aircraft("blue_0","blue","uav",(4,0,0));b1=aircraft("blue_1","blue","uav",(6,0,0));agents=[mav,r0,r1,b0,b1]
    initial=PaperTable1MAVReward.battlefield_center(agents).tolist();b1.position=np.array([10.,0,0]);moved=PaperTable1MAVReward.battlefield_center(agents).tolist();b1.alive=False;after_kill=PaperTable1MAVReward.battlefield_center(agents).tolist()
    assert initial==[3.,0.,0.] and moved==[4.,0.,0.] and after_kill==[2.,0.,0.]
    mav.position=np.zeros(3);mav.velocity=np.array([1.,0,0]);b0.position=np.array([1000.,0,0]);b0.alive=True;b1.position=np.array([2000.,0,0]);b1.alive=True
    reward=PaperTable1MAVReward();_,none=call(reward,mav,agents,perception={"mav_detected_enemy_ids":[]})
    reward.reset();_,one=call(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0"]})
    reward.reset();_,two=call(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0","blue_1"],"relay_only_track_count":0})
    reward.reset();relay_total,relay=call(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0","blue_1"],"relay_only_track_count":50})
    assert none["r_support_awareness"]==0 and np.isclose(one["r_support_awareness"],.3) and np.isclose(two["r_support_awareness"],.6)
    assert np.isclose(relay_total,two["total"]) and relay["relay_only_track_count_log"]==50
    return {"battlefield_center_initial":initial,"battlefield_center_after_move":moved,"battlefield_center_after_kill":after_kill,
      "awareness_zero_one_two":[none["r_support_awareness"],one["r_support_awareness"],two["r_support_awareness"]],"relay_only_changed_total":False}

def event_checks():
    mav=aircraft("red_mav_0","red","mav",(0,0,0));red=aircraft("red_uav_0","red","uav",(0,0,0));blue=aircraft("blue_0","blue","uav",(0,0,0));agents=[mav,red,blue];event={"reason":"hit","shooter_id":"red_uav_0","target_id":"blue_0"};reward=PaperTable1MAVReward()
    awards=[call(reward,mav,agents,events=[event])[1]["r_event_team_contribution"] for _ in range(3)];assert awards==[100.,100.,0.]
    reward.reset();reset_credit=reward.team_credit_awarded_so_far;mav.alive=False;mav.death_reason="boundary"
    _,death=call(reward,mav,agents,alive_start={mav.agent_id:True},out={mav.agent_id});_,repeat=call(reward,mav,agents,alive_start={mav.agent_id:True},out={mav.agent_id})
    assert death["r_event"]==-200 and repeat["r_event_death"]==0 and reset_credit==0
    return {"team_credit_per_kill":awards,"credit_after_reset":reset_credit,"boundary_death_event":death["r_event"],"repeat_death_event":repeat["r_event_death"]}

def reward_mode_comparison():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO);env.reset(seed=1)
    try:
        mav=env.by_id["red_mav_0"];alive={a.agent_id:a.alive for a in env.agents};result={}
        for mode,reward in (("legacy_v1",LegacySimpleMAVReward()),("paper_table1_v2",PaperTable1MAVReward())):
            total,c=reward.compute(mav,env.agents,[],[],alive,set(),env.perception_result)
            result[mode]={"safety":c["r_safety"],"support":c["r_support"],"event":c["r_event"],"total":total}
        return result
    finally:env.close()

def rule_rollout(perception_mode,reward_mode,episodes=2,max_steps=200):
    rows=[]
    for episode in range(min(episodes,2)):
        env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=min(max_steps,200),hetero_perception_mode=perception_mode,hetero_reward_mode=reward_mode);_,info=env.reset(seed=100+episode)
        returns={aid:0. for aid in env.controlled_ids};safety=support=event=relay=0.;abs_mav_dense=abs_uav_reward=0.
        try:
            for step in range(min(max_steps,200)):
                _,rewards,terminated,truncated,info=env.step(env.build_rule_actions())
                for aid,value in rewards.items():returns[aid]+=float(value)
                c=info["reward_components"]["red_mav_0"];safety+=c["r_safety"];support+=c["r_support"];event+=c["r_event"];relay+=info["relay_only_track_count"]
                abs_mav_dense+=abs(float(c.get("total_dense",c.get("r_safety",0.)+c.get("r_support",0.))))
                abs_uav_reward+=sum(abs(float(rewards.get(aid,0.))) for aid in ("red_uav_0","red_uav_1"))/2.
                if terminated or truncated:break
            steps=step+1;mav_dense=safety+support;role_sum=sum(returns.values())
            mean_abs_mav_dense=abs_mav_dense/max(steps,1);mean_abs_uav=abs_uav_reward/max(steps,1)
            rows.append({"episode":episode,"steps":step+1,"termination_reason":info["termination_reason"],"mav_return":returns["red_mav_0"],
              "mean_uav_return":float(np.mean([returns["red_uav_0"],returns["red_uav_1"]])),"mav_safety":safety,"mav_support":support,"mav_event":event,
              "team_return":float(role_sum),"mav_dense_return":float(mav_dense),"mav_event_return":float(event),
              "mean_abs_mav_dense_per_step":float(mean_abs_mav_dense),"mean_abs_red_uav_reward_per_step":float(mean_abs_uav),
              "mav_dense_to_uav_reward_scale_ratio":float(mean_abs_mav_dense/mean_abs_uav) if mean_abs_uav>0 else 0.0,
              "mav_total_reward_fraction_of_role_sum":float(abs(returns["red_mav_0"])/sum(abs(value) for value in returns.values())) if any(returns.values()) else 0.0,
              "relay_only_accumulated":int(relay),"red_missile_kills":info["red_missile_kills"],"blue_missile_kills":info["blue_missile_kills"],
              "mav_alive":info["mav_alive"],"numerical_invalid":info["numerical_invalid"],"crash":info["red_crashes"]+info["blue_crashes"],"boundary":info["boundary_deaths"]})
        finally:env.close()
    return rows

def mappo_smoke(steps=256):
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_reward_mode="paper_table1_v2",weapon_enabled_agent_ids=set());adapter=SimpleMAPPOAdapter(env);obs,state,_=adapter.reset(seed=200);model=SharedMAPPOActorCritic(81,243);buffer=RolloutBuffer(steps,3,81,243);active=np.ones(3,np.float32);reward_finite=True
    try:
        for index in range(steps):
            with torch.no_grad():actions,logp,value,_=model.act(torch.tensor(obs),torch.tensor(state))
            next_obs,next_state,rewards,done,next_active,info=adapter.step(actions.numpy());reward_finite&=bool(np.isfinite(rewards).all());buffer.store(obs,state,actions.numpy(),logp.numpy(),rewards,done,value.item(),active)
            if done:obs,state,_=adapter.reset(seed=201+index);active=np.ones(3,np.float32)
            else:obs,state,active=next_obs,next_state,next_active
        trainer=MAPPOTrainer(model,ppo_epochs=1);data=buffer.to_tensors();advantages,_=trainer.compute_gae(data,0.);stats=trainer.update(buffer,0.)
        assert obs.shape==(3,81) and state.shape==(243,) and actions.shape==(3,3) and reward_finite and torch.isfinite(advantages).all() and np.isfinite(list(stats.values())).all()
        return {"completed_env_steps":steps,"obs_shape":[3,81],"state_shape":[243],"action_shape":[3,3],"reward_finite":reward_finite,"advantage_finite":True,"update_finite":True,"stats":stats,
          "numerical_invalid":info["numerical_invalid"],"crash":info["red_crashes"]+info["blue_crashes"],"boundary":info["boundary_deaths"]}
    finally:env.close()

def run_checks():
    return {"formula":formula_checks(),"support":support_checks(),"event":event_checks(),"reward_mode_comparison":reward_mode_comparison(),
      "rule_combinations":{"paper_fused+legacy_v1":rule_rollout("paper_fused","legacy_v1"),
        "paper_fused+paper_table1_v2":rule_rollout("paper_fused","paper_table1_v2"),
        "uav_only_ablation+paper_table1_v2":rule_rollout("uav_only_ablation","paper_table1_v2")},"mappo_smoke_256":mappo_smoke()}

def main():print(json.dumps(run_checks(),indent=2))
if __name__=="__main__":main()
