"""Short acceptance checks for paper-aligned simple 3v2 perception/support."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_situation import paper_situation_score
from aircombat_env_v1.simple_env import HETERO_SCENARIO,SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

UAV_IDS=("red_uav_0","red_uav_1")

def initial_perception_checks():
    result={}
    for mode in ("paper_fused","uav_only_ablation"):
        env=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode=mode);obs,info=env.reset(seed=1)
        try:
            distances={aid:{bid:float(np.linalg.norm(env.by_id[bid].position-env.by_id[aid].position)) for bid in ("blue_0","blue_1")} for aid in UAV_IDS}
            assert min(min(value.values()) for value in distances.values())>14000.
            assert all(info["direct_enemy_ids_by_agent"][aid]==[] for aid in UAV_IDS)
            assert info["mav_detected_enemy_ids"]==["blue_0","blue_1"]
            expected=["blue_0","blue_1"] if mode=="paper_fused" else []
            assert all(info["visible_enemy_ids_by_agent"][aid]==expected for aid in UAV_IDS)
            assert all(np.array_equal(obs[aid][69:71],[1,1] if expected else [0,0]) for aid in UAV_IDS)
            assert all(obs[aid].shape==(81,) and np.array_equal(obs[aid][-2:],[0,1]) for aid in UAV_IDS)
            result[mode]={"minimum_uav_enemy_distance_m":min(min(value.values()) for value in distances.values()),
              "mav_detected_enemy_ids":info["mav_detected_enemy_ids"],"direct_enemy_ids_by_uav":{aid:info["direct_enemy_ids_by_agent"][aid] for aid in UAV_IDS},
              "visible_enemy_ids_by_uav":{aid:info["visible_enemy_ids_by_agent"][aid] for aid in UAV_IDS},
              "initial_relay_only_track_count":info["relay_only_track_count"],"enemy_masks":{aid:obs[aid][69:71].astype(int).tolist() for aid in UAV_IDS}}
        finally:env.close()
    return result

def target_selection_check():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,weapon_enabled_agent_ids=set());env.reset(seed=1)
    try:
        ego=env.by_id["red_uav_0"];near=env.by_id["blue_0"];far=env.by_id["blue_1"]
        near.position=ego.position+np.array([1000.,0.,3000.]);near.velocity=ego.velocity.copy()
        far.position=ego.position+np.array([5000.,0.,-3000.]);far.velocity=ego.velocity.copy()
        env._update_perception();env._update_targets(False)
        scores={target.agent_id:paper_situation_score(ego.position,ego.velocity,target.position,target.velocity) for target in (near,far)}
        assert np.linalg.norm(near.position-ego.position)<np.linalg.norm(far.position-ego.position)
        assert env.current_targets[ego.agent_id]==max(scores,key=scores.get)=="blue_1"
        return {"nearest_target":"blue_0","highest_score_target":env.current_targets[ego.agent_id],"scores":scores}
    finally:env.close()

def direct_detection_check():
    result={}
    for mode in ("paper_fused","uav_only_ablation"):
        env=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode=mode,weapon_enabled_agent_ids=set());env.reset(seed=1)
        try:
            uav=env.by_id["red_uav_0"];env.by_id["blue_0"].position=uav.position+np.array([1000.,0.,0.])
            env._update_perception();env._update_targets(False);info=env._info(None,None)
            assert "blue_0" in info["direct_enemy_ids_by_agent"][uav.agent_id]
            assert "blue_0" in info["visible_enemy_ids_by_agent"][uav.agent_id]
            result[mode]={"direct":info["direct_enemy_ids_by_agent"][uav.agent_id],"relay_only":info["relay_only_tracks_by_uav"][uav.agent_id],"target":uav.current_target}
        finally:env.close()
    return result

def fire_control_check():
    hidden=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode="uav_only_ablation",max_steps=1);hidden.reset(seed=1)
    try:
        _,_,_,_,hidden_info=hidden.step(hidden.build_rule_actions())
        hidden_red_launches=sum(e["event_type"]=="missile_launch" and e["shooter_id"].startswith("red_") for e in hidden_info["events"])
    finally:hidden.close()
    direct=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=1);direct.reset(seed=1)
    try:
        uav=direct.by_id["red_uav_0"];direct.by_id["blue_0"].position=uav.position+np.array([1000.,0.,0.])
        direct._update_perception();direct._update_targets(False)
        _,_,_,_,direct_info=direct.step(direct.build_rule_actions())
        launchers=[e["shooter_id"] for e in direct_info["events"] if e["event_type"]=="missile_launch"]
        assert hidden_red_launches==0 and "red_uav_0" in launchers and "red_mav_0" not in launchers
        return {"hidden_target_red_launches":hidden_red_launches,"direct_target_launchers":launchers,
          "shared_target_selected_initially":True,"mav_erroneous_launches":launchers.count("red_mav_0")}
    finally:direct.close()

def rule_comparison(mode,episodes=2,max_steps=200):
    episodes=min(int(episodes),2);max_steps=min(int(max_steps),200);rows=[]
    for episode in range(episodes):
        env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=max_steps,hetero_perception_mode=mode);_,info=env.reset(seed=100+episode)
        row={"episode":episode,"first_target_visible_step":None,"first_target_selection_step":None,"first_direct_detection_step":None,
          "first_launch_step":None,"relay_only_tracks_accumulated":0,"termination_reason":None,"mav_erroneous_launches":0,
          "numerical_invalid":0,"crash":0,"boundary":0}
        try:
            for step in range(max_steps):
                if row["first_target_visible_step"] is None and any(info["visible_enemy_ids_by_agent"][aid] for aid in UAV_IDS):row["first_target_visible_step"]=step
                if row["first_target_selection_step"] is None and any(info["current_targets"][aid] is not None for aid in UAV_IDS):row["first_target_selection_step"]=step
                if row["first_direct_detection_step"] is None and any(info["direct_enemy_ids_by_agent"][aid] for aid in UAV_IDS):row["first_direct_detection_step"]=step
                row["relay_only_tracks_accumulated"]+=int(info["relay_only_track_count"])
                _,_,terminated,truncated,info=env.step(env.build_rule_actions())
                launches=[e for e in info["events"] if e["event_type"]=="missile_launch"]
                if launches and row["first_launch_step"] is None:row["first_launch_step"]=step
                row["mav_erroneous_launches"]+=sum(e["shooter_id"]=="red_mav_0" for e in launches)
                row["numerical_invalid"]+=int(info["numerical_invalid"]);row["crash"]+=int(info["red_crashes"]+info["blue_crashes"]);row["boundary"]+=int(info["boundary_deaths"])
                if terminated or truncated:break
            row.update({"completed_steps":step+1,"termination_reason":info["termination_reason"],
              "shared_track_launches":int(info["red_uav_launches_using_shared_track"]),"direct_track_launches":int(info["red_uav_launches_using_direct_track"])})
            rows.append(row)
        finally:env.close()
    return rows

def mappo_smoke(mode,steps=256):
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode=mode,weapon_enabled_agent_ids=set());adapter=SimpleMAPPOAdapter(env);obs,state,_=adapter.reset(seed=200)
    model=SharedMAPPOActorCritic(81,243);buffer=RolloutBuffer(steps,3,81,243);active=np.ones(3,np.float32);invalid=crash=boundary=0
    try:
        assert obs.shape==(3,81) and state.shape==(243,)
        for index in range(steps):
            with torch.no_grad():actions,logp,value,_=model.act(torch.tensor(obs),torch.tensor(state))
            assert actions.shape==(3,3)
            next_obs,next_state,rewards,done,next_active,info=adapter.step(actions.numpy());buffer.store(obs,state,actions.numpy(),logp.numpy(),rewards,done,value.item(),active)
            invalid+=int(info["numerical_invalid"]);crash+=int(info["red_crashes"]+info["blue_crashes"]);boundary+=int(info["boundary_deaths"])
            if done:obs,state,_=adapter.reset(seed=201+index);active=np.ones(3,np.float32)
            else:obs,state,active=next_obs,next_state,next_active
        next_value=0. if buffer.team_dones[-1] else model.value(torch.tensor(state)).item();stats=MAPPOTrainer(model,ppo_epochs=1).update(buffer,next_value)
        return {"completed_env_steps":steps,"obs_shape":[3,81],"state_shape":[243],"action_shape":[3,3],
          "update_finite":bool(np.isfinite(list(stats.values())).all()),"stats":stats,"numerical_invalid":invalid,"crash":crash,"boundary":boundary}
    finally:env.close()

def run_checks():
    return {"initial_perception":initial_perception_checks(),"visibility_and_target_selection":target_selection_check(),
      "direct_detection":direct_detection_check(),"fire_control":fire_control_check(),
      "rule_comparison":{mode:rule_comparison(mode) for mode in ("paper_fused","uav_only_ablation")},
      "mappo_smoke_256":{mode:mappo_smoke(mode) for mode in ("paper_fused","uav_only_ablation")}}

def main():print(json.dumps(run_checks(),indent=2))
if __name__=="__main__":main()
