"""Evaluate a SimpleTAMCombatEnv MAPPO checkpoint."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

def resolve_device(name):return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else "cpu" if name=="auto" else name)

def evaluate_model(model,scenario,episodes,device,deterministic=True,seed=1,return_rows=False,hetero_perception_mode="paper_fused",hetero_reward_mode="paper_table1_v2"):
    numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state();cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        rows=[]
        for episode in range(episodes):
            episode_seed=seed+episode
            np.random.seed(episode_seed);torch.manual_seed(episode_seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(episode_seed)
            env=SimpleTAMCombatEnv(scenario,"red",hetero_perception_mode=hetero_perception_mode,hetero_reward_mode=hetero_reward_mode) if scenario=="simple_paper_3v2_hetero" else SimpleTAMCombatEnv(scenario,"red")
            adapter=SimpleMAPPOAdapter(env);obs,state,info=adapter.reset(seed=episode_seed)
            active=np.ones(adapter.num_agents,np.float32);episode_return=0.;action_trace=[]
            relay_tracks=support_steps=shared_selections=target_selections=0
            visible_enemies=direct_enemies=shared_enemies=0
            role_returns={aid:0. for aid in adapter.agent_ids};mav_safety=mav_support=mav_event=mav_death=mav_team=mav_awareness=mav_position=mav_dense=0.
            try:
                for step in range(1000):
                    with torch.no_grad():actions,_,_,_=model.act(torch.as_tensor(obs,device=device),torch.as_tensor(state,device=device),deterministic)
                    action_np=actions.cpu().numpy();action_trace.append(action_np.tolist())
                    obs,state,rewards,done,next_active,info=adapter.step(action_np)
                    episode_return+=float((rewards*active).sum()/max(active.sum(),1.))
                    for index,aid in enumerate(adapter.agent_ids):role_returns[aid]+=float(rewards[index]*active[index])
                    active=next_active
                    if scenario=="simple_paper_3v2_hetero":
                        relay_tracks+=int(info["relay_only_track_count"]);support_steps+=int(info["mav_support_active"])
                        for aid in ("red_uav_0","red_uav_1"):
                            source=info["target_selection_source"][aid]
                            target_selections+=int(source in ("direct","mav_shared"));shared_selections+=int(source=="mav_shared")
                            visible_enemies+=len(info["visible_enemy_ids_by_agent"][aid])
                            direct_enemies+=len(info["direct_enemy_ids_by_agent"][aid]);shared_enemies+=len(info["shared_enemy_ids_by_agent"][aid])
                        components=info["reward_components"]["red_mav_0"]
                        mav_safety+=float(components.get("r_safety",0.));mav_support+=float(components.get("r_support",0.));mav_event+=float(components.get("r_event",0.))
                        mav_death+=float(components.get("r_event_death",0.));mav_team+=float(components.get("r_event_team_contribution",0.));mav_awareness+=float(components.get("r_support_awareness",0.));mav_position+=float(components.get("r_support_position",0.));mav_dense+=float(components.get("total_dense",components.get("r_safety",0.)+components.get("r_support",0.)))
                    if done:break
                row={"return":float(episode_return),"length":int(step+1),"winner":str(info["winner"]),
                  "timeout":bool(info["termination_reason"]=="timeout"),"red_alive":int(info["alive_red"]),
                  "blue_alive":int(info["alive_blue"]),"missile_launches":int(info["missiles_fired"]),
                  "missile_hits":int(info["missile_hits"]),"crashes":int(info["red_crashes"]+info["blue_crashes"]),
                  "boundary_deaths":int(info["boundary_deaths"]),"numerical_invalid":int(info["numerical_invalid"]),
                  "envelope":bool(info["flight_envelope_violation"]),"action_trace":action_trace}
                if scenario=="simple_paper_3v2_hetero":
                    row.update({"mav_alive":bool(info["mav_alive"]),"red_uav_alive":int(info["red_uav_alive"]),
                      "mav_lost":bool(info["red_team_failed_by_mav_loss"]),
                      "red_uav_team_lost":bool(info["red_team_failed_by_uav_loss"]),
                      "red_missile_kills":int(info["red_missile_kills"]),"blue_missile_kills":int(info["blue_missile_kills"]),
                      "relay_only_track_steps":int(relay_tracks),"mav_support_steps":int(support_steps),
                      "shared_target_selections":int(shared_selections),"uav_target_selections":int(target_selections),
                      "visible_enemy_observations":int(visible_enemies),"direct_enemy_observations":int(direct_enemies),
                      "shared_enemy_observations":int(shared_enemies),
                      "red_uav_launches_using_shared_track":int(info["red_uav_launches_using_shared_track"]),
                      "red_uav_launches_using_direct_track":int(info["red_uav_launches_using_direct_track"]),
                      "mav_return":float(role_returns.get("red_mav_0",0.)),
                      "mean_red_uav_return":float(np.mean([role_returns.get("red_uav_0",0.),role_returns.get("red_uav_1",0.)])),
                      "mav_safety_return":float(mav_safety),"mav_support_return":float(mav_support),"mav_event_return":float(mav_event),
                      "mav_death_penalty":float(mav_death),"mav_team_contribution":float(mav_team),"mav_awareness_return":float(mav_awareness),
                      "mav_position_return":float(mav_position),"mav_dense_return":float(mav_dense)})
                rows.append(row)
            finally:env.close()
        n=max(len(rows),1)
        result={"episodes":int(len(rows)),"mean_return":float(np.mean([r["return"] for r in rows])),"return_std":float(np.std([r["return"] for r in rows])),"mean_episode_length":float(np.mean([r["length"] for r in rows])),
          "red_win_rate":float(sum(r["winner"]=="red" for r in rows)/n),"blue_win_rate":float(sum(r["winner"]=="blue" for r in rows)/n),
          "draw_rate":float(sum(r["winner"]=="draw" for r in rows)/n),"timeout_rate":float(sum(r["timeout"] for r in rows)/n),
          "mean_red_alive":float(np.mean([r["red_alive"] for r in rows])),"mean_blue_alive":float(np.mean([r["blue_alive"] for r in rows])),
          "missile_launches":int(sum(r["missile_launches"] for r in rows)),"missile_hits":int(sum(r["missile_hits"] for r in rows)),
          "crashes":int(sum(r["crashes"] for r in rows)),"boundary_deaths":int(sum(r["boundary_deaths"] for r in rows)),
          "numerical_invalid_episodes":int(sum(r["numerical_invalid"]>0 for r in rows)),
          "flight_envelope_violation_episodes":int(sum(r["envelope"] for r in rows))}
        if scenario=="simple_paper_3v2_hetero":
            total_steps=max(sum(r["length"] for r in rows),1);uav_observation_slots=max(2*total_steps,1)
            target_selection_count=max(sum(r["uav_target_selections"] for r in rows),1)
            result.update({"mav_survival_rate":float(sum(r["mav_alive"] for r in rows)/n),
              "mean_red_uav_alive":float(np.mean([r["red_uav_alive"] for r in rows])),
              "mav_loss_rate":float(sum(r["mav_lost"] for r in rows)/n),
              "red_uav_team_loss_rate":float(sum(r["red_uav_team_lost"] for r in rows)/n),
              "red_missile_kills":int(sum(r["red_missile_kills"] for r in rows)),
              "blue_missile_kills":int(sum(r["blue_missile_kills"] for r in rows)),
              "mean_relay_only_tracks_per_step":float(sum(r["relay_only_track_steps"] for r in rows)/total_steps),
              "fraction_steps_with_mav_support":float(sum(r["mav_support_steps"] for r in rows)/total_steps),
              "fraction_uav_target_selections_from_shared_tracks":float(sum(r["shared_target_selections"] for r in rows)/target_selection_count),
              "red_uav_launches_using_shared_track":int(sum(r["red_uav_launches_using_shared_track"] for r in rows)),
              "red_uav_launches_using_direct_track":int(sum(r["red_uav_launches_using_direct_track"] for r in rows)),
              "mean_visible_enemies_per_red_uav":float(sum(r["visible_enemy_observations"] for r in rows)/uav_observation_slots),
              "mean_direct_enemies_per_red_uav":float(sum(r["direct_enemy_observations"] for r in rows)/uav_observation_slots),
              "mean_shared_enemies_per_red_uav":float(sum(r["shared_enemy_observations"] for r in rows)/uav_observation_slots),
              "hetero_reward_mode":hetero_reward_mode,"reward_contract_version":"heterogeneous_reward_v2",
              "checkpoint_reward_contract_known":bool(getattr(model,"checkpoint_reward_contract_known",False)),
              "mean_mav_return":float(np.mean([r["mav_return"] for r in rows])),"mean_red_uav_return":float(np.mean([r["mean_red_uav_return"] for r in rows])),
              "mean_mav_safety_return":float(np.mean([r["mav_safety_return"] for r in rows])),"mean_mav_support_return":float(np.mean([r["mav_support_return"] for r in rows])),
              "mean_mav_event_return":float(np.mean([r["mav_event_return"] for r in rows])),"mean_mav_death_penalty":float(np.mean([r["mav_death_penalty"] for r in rows])),
              "mean_mav_team_contribution":float(np.mean([r["mav_team_contribution"] for r in rows])),"mean_mav_awareness_return":float(np.mean([r["mav_awareness_return"] for r in rows])),
              "mean_mav_position_return":float(np.mean([r["mav_position_return"] for r in rows])),"mean_mav_dense_return":float(np.mean([r["mav_dense_return"] for r in rows]))})
        if return_rows:result["rows"]=rows
        return result
    finally:
        np.random.set_state(numpy_state);torch.random.set_rng_state(torch_state)
        if cuda_states is not None:torch.cuda.set_rng_state_all(cuda_states)

def load_checkpoint(path,scenario,device,hetero_perception_mode="paper_fused",hetero_reward_mode="paper_table1_v2",allow_reward_mode_override=False):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    contract=checkpoint.get("environment_contract");known=contract is not None
    if known and contract.get("hetero_reward_mode")!=hetero_reward_mode and not allow_reward_mode_override:
        raise ValueError(f"requested hetero_reward_mode={hetero_reward_mode!r} does not match checkpoint reward mode {contract.get('hetero_reward_mode')!r}")
    env=SimpleTAMCombatEnv(scenario,hetero_perception_mode=hetero_perception_mode,hetero_reward_mode=hetero_reward_mode) if scenario=="simple_paper_3v2_hetero" else SimpleTAMCombatEnv(scenario)
    adapter=SimpleMAPPOAdapter(env);env.close()
    expected={"scenario":scenario,"obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,"action_dim":adapter.action_dim}
    for key,value in expected.items():
        if checkpoint.get(key)!=value:raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r} does not match expected {value!r}")
    if list(checkpoint.get("agent_ids",[]))!=adapter.agent_ids:raise ValueError("checkpoint agent_ids/num_agents mismatch")
    if known:
        for key,value in expected.items():
            if contract.get(key)!=value:raise ValueError(f"checkpoint environment_contract {key}={contract.get(key)!r} does not match expected {value!r}")
        if list(contract.get("agent_ids",[]))!=adapter.agent_ids:raise ValueError("checkpoint environment_contract agent_ids/num_agents mismatch")
    model=SharedMAPPOActorCritic(adapter.obs_dim,adapter.state_dim,adapter.action_dim).to(device);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    model.checkpoint_reward_contract_known=bool(known);model.checkpoint_environment_contract=contract
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--scenario",choices=("simple_paper_1v1","simple_paper_2v2","simple_paper_3v2_hetero"),required=True);p.add_argument("--episodes",type=int,default=5);p.add_argument("--device",default="auto");p.add_argument("--deterministic",action="store_true");p.add_argument("--seed",type=int,default=1);p.add_argument("--hetero-perception-mode",choices=("paper_fused","uav_only_ablation"),default="paper_fused");p.add_argument("--hetero-reward-mode",choices=("legacy_v1","paper_table1_v2"),default="paper_table1_v2");p.add_argument("--allow-reward-mode-override",action="store_true");a=p.parse_args()
    device=resolve_device(a.device);model=load_checkpoint(a.model,a.scenario,device,a.hetero_perception_mode,a.hetero_reward_mode,a.allow_reward_mode_override);print(json.dumps(evaluate_model(model,a.scenario,a.episodes,device,a.deterministic,a.seed,False,a.hetero_perception_mode,a.hetero_reward_mode),indent=2))
if __name__=="__main__":main()
