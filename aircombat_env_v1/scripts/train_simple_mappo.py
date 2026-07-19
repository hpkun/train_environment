"""Train the minimal feed-forward MAPPO baseline on SimpleTAMCombatEnv."""
from __future__ import annotations
import argparse,csv,sys,time
from collections import deque
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import SimpleTAMCombatEnv
from aircombat_env_v1.simple_hetero_reward import reward_contract_metadata
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model,resolve_device
import torch

TRAIN_FIELDS=("update","env_steps","episodes_completed","mean_episode_return","mean_episode_length","red_win_rate","blue_win_rate","draw_rate","timeout_rate","mean_red_alive","mean_blue_alive","missile_hits_per_episode","actor_loss","critic_loss","entropy","approx_kl","clip_fraction","action_mean_abs","action_saturation_rate","numerical_invalid_episodes","mean_mav_episode_return","mean_red_uav_episode_return","mean_mav_safety_sum","mean_mav_support_sum","mean_mav_event_sum","mean_relay_only_track_steps")
EVAL_FIELDS=("env_steps","hetero_perception_mode","hetero_reward_mode","reward_contract_version","mean_return","red_win_rate","blue_win_rate","draw_rate","timeout_rate","mean_episode_length","mav_survival_rate","mean_red_uav_alive","mean_mav_return","mean_red_uav_return","mean_mav_safety_return","mean_mav_support_return","mean_mav_event_return","mean_mav_death_penalty","mean_mav_team_contribution","mean_mav_awareness_return","mean_mav_position_return","mean_mav_dense_return","mean_relay_only_tracks_per_step","fraction_steps_with_mav_support")
EPISODE_COMPONENT_FIELDS=("episode","env_steps","length","winner","termination_reason","team_return","mav_return","red_uav_0_return","red_uav_1_return","mean_red_uav_return","mav_safety_sum","mav_support_sum","mav_event_sum","mav_event_death_sum","mav_event_team_contribution_sum","mav_awareness_sum","mav_position_sum","mav_dense_sum","red_uav_height_sum","red_uav_speed_sum","red_uav_angle_sum","red_uav_distance_sum","red_uav_dodge_sum","red_uav_event_sum","relay_only_track_steps","mav_support_steps","red_missile_kills","blue_missile_kills","mav_alive","red_uav_alive","numerical_invalid","crashes","boundary_deaths")
COMPONENT_SUM_FIELDS=EPISODE_COMPONENT_FIELDS[10:26]
ENVIRONMENT_CONTRACT_SCHEMA_VERSION="1"

def build_environment_contract(args,adapter):
    heterogeneous=args.scenario=="simple_paper_3v2_hetero"
    reward_meta=reward_contract_metadata(args.hetero_reward_mode) if heterogeneous else None
    return {"environment_contract_schema_version":ENVIRONMENT_CONTRACT_SCHEMA_VERSION,"scenario":args.scenario,
      "hetero_perception_mode":args.hetero_perception_mode if heterogeneous else None,
      "hetero_reward_mode":args.hetero_reward_mode if heterogeneous else None,
      "reward_contract_schema_version":reward_meta["reward_contract_schema_version"] if reward_meta else None,
      "reward_contract_version":reward_meta["reward_contract_version"] if reward_meta else None,
      "reward_config":reward_meta["reward_config"] if reward_meta else None,
      "obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,"action_dim":adapter.action_dim,
      "agent_ids":list(adapter.agent_ids)}

def build_episode_component_row(episode,env_steps,length,info,team_return,role_returns,component_sums,scenario):
    sums={key:float(component_sums.get(key,0.)) for key in COMPONENT_SUM_FIELDS}
    heterogeneous=scenario=="simple_paper_3v2_hetero"
    uav_returns=[float(role_returns.get(aid,0.)) for aid in ("red_uav_0","red_uav_1")] if heterogeneous else []
    return {"episode":int(episode),"env_steps":int(env_steps),"length":int(length),
      "winner":str(info.get("winner") or ""),"termination_reason":str(info.get("termination_reason") or ""),
      "team_return":float(team_return),"mav_return":float(role_returns.get("red_mav_0",0.)),
      "red_uav_0_return":float(role_returns.get("red_uav_0",0.)),"red_uav_1_return":float(role_returns.get("red_uav_1",0.)),
      "mean_red_uav_return":float(np.mean(uav_returns)) if uav_returns else 0.0,**sums,
      "red_missile_kills":int(info.get("red_missile_kills",0)),"blue_missile_kills":int(info.get("blue_missile_kills",0)),
      "mav_alive":int(bool(info.get("mav_alive",False))) if heterogeneous else 0,
      "red_uav_alive":int(info.get("red_uav_alive",0)) if heterogeneous else 0,
      "numerical_invalid":int(info.get("numerical_invalid",0)),
      "crashes":int(info.get("red_crashes",0))+int(info.get("blue_crashes",0)),
      "boundary_deaths":int(info.get("boundary_deaths",0))}

def save_checkpoint(path,model,adapter,args,env_steps):
    contract=build_environment_contract(args,adapter)
    torch.save({"model_state_dict":model.state_dict(),"scenario":args.scenario,"obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,
      "action_dim":adapter.action_dim,"agent_ids":adapter.agent_ids,"environment_contract":contract,"training_args":vars(args),"env_steps":env_steps},path)

def recent_stats(records):
    if not records:return {"mean_episode_return":0.,"mean_episode_length":0.,"red_win_rate":0.,"blue_win_rate":0.,"draw_rate":0.,"timeout_rate":0.,"mean_red_alive":0.,"mean_blue_alive":0.,"missile_hits_per_episode":0.,"numerical_invalid_episodes":0,"mean_mav_episode_return":0.,"mean_red_uav_episode_return":0.,"mean_mav_safety_sum":0.,"mean_mav_support_sum":0.,"mean_mav_event_sum":0.,"mean_relay_only_track_steps":0.}
    rows=list(records);n=len(rows)
    return {"mean_episode_return":np.mean([r["return"] for r in rows]),"mean_episode_length":np.mean([r["length"] for r in rows]),
      "red_win_rate":sum(r["winner"]=="red" for r in rows)/n,"blue_win_rate":sum(r["winner"]=="blue" for r in rows)/n,
      "draw_rate":sum(r["winner"]=="draw" for r in rows)/n,"timeout_rate":sum(r["timeout"] for r in rows)/n,
      "mean_red_alive":np.mean([r["red_alive"] for r in rows]),"mean_blue_alive":np.mean([r["blue_alive"] for r in rows]),
      "missile_hits_per_episode":np.mean([r["missile_hits"] for r in rows]),"numerical_invalid_episodes":sum(r["invalid"] for r in rows),
      "mean_mav_episode_return":np.mean([r.get("mav_return",0.) for r in rows]),"mean_red_uav_episode_return":np.mean([r.get("mean_red_uav_return",0.) for r in rows]),
      "mean_mav_safety_sum":np.mean([r.get("mav_safety_sum",0.) for r in rows]),"mean_mav_support_sum":np.mean([r.get("mav_support_sum",0.) for r in rows]),
      "mean_mav_event_sum":np.mean([r.get("mav_event_sum",0.) for r in rows]),"mean_relay_only_track_steps":np.mean([r.get("relay_only_track_steps",0.) for r in rows])}

def build_parser():
    p=argparse.ArgumentParser();p.add_argument("--scenario",choices=("simple_paper_1v1","simple_paper_2v2","simple_paper_3v2_hetero"),required=True);p.add_argument("--total-env-steps",type=int,default=10000);p.add_argument("--rollout-length",type=int,default=256);p.add_argument("--seed",type=int,default=1);p.add_argument("--device",default="auto");p.add_argument("--output-dir",required=True);p.add_argument("--eval-interval",type=int,default=2500);p.add_argument("--eval-episodes",type=int,default=5);p.add_argument("--deterministic-eval",action=argparse.BooleanOptionalAction,default=True);p.add_argument("--actor-lr",type=float,default=3e-4);p.add_argument("--critic-lr",type=float,default=3e-4);p.add_argument("--entropy-coef",type=float,default=.01);p.add_argument("--ppo-epochs",type=int,default=4)
    p.add_argument("--hetero-perception-mode",choices=("paper_fused","uav_only_ablation"),default="paper_fused")
    p.add_argument("--hetero-reward-mode",choices=("legacy_v1","paper_table1_v2"),default="paper_table1_v2")
    return p

def main():
    args=build_parser().parse_args()
    np.random.seed(args.seed);torch.manual_seed(args.seed);device=resolve_device(args.device);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    env=SimpleTAMCombatEnv(args.scenario,"red",hetero_perception_mode=args.hetero_perception_mode,hetero_reward_mode=args.hetero_reward_mode);adapter=SimpleMAPPOAdapter(env);model=SharedMAPPOActorCritic(adapter.obs_dim,adapter.state_dim,adapter.action_dim).to(device)
    trainer=MAPPOTrainer(model,args.actor_lr,args.critic_lr,entropy_coef=args.entropy_coef,ppo_epochs=args.ppo_epochs)
    train_file=(output/"train_log.csv").open("w",newline="",encoding="utf-8");train_writer=csv.DictWriter(train_file,fieldnames=TRAIN_FIELDS);train_writer.writeheader()
    eval_file=(output/"eval_log.csv").open("w",newline="",encoding="utf-8");eval_writer=csv.DictWriter(eval_file,fieldnames=EVAL_FIELDS);eval_writer.writeheader()
    episode_file=(output/"episode_reward_components.csv").open("w",newline="",encoding="utf-8");episode_writer=csv.DictWriter(episode_file,fieldnames=EPISODE_COMPONENT_FIELDS);episode_writer.writeheader()
    def run_eval(step):
        model.eval();summary=evaluate_model(model,args.scenario,args.eval_episodes,device,args.deterministic_eval,args.seed+100000+step,False,args.hetero_perception_mode,args.hetero_reward_mode);model.train()
        eval_writer.writerow({key:step if key=="env_steps" else summary.get(key,"") for key in EVAL_FIELDS});eval_file.flush();return summary
    best_score=(-float("inf"),-float("inf"));initial_eval=run_eval(0);save_checkpoint(output/"initial.pt",model,adapter,args,0);save_checkpoint(output/"best.pt",model,adapter,args,0);best_score=(initial_eval["red_win_rate"],initial_eval["mean_return"])
    obs,state,info=adapter.reset(seed=args.seed);active=np.ones(adapter.num_agents,np.float32);env_steps=updates=episodes_completed=0
    episode_return=0.;episode_length=0;recent=deque(maxlen=20);next_eval=args.eval_interval;start=time.perf_counter()
    role_returns={aid:0. for aid in adapter.agent_ids};component_sums={key:0. for key in COMPONENT_SUM_FIELDS}
    try:
        while env_steps<args.total_env_steps:
            rollout=min(args.rollout_length,args.total_env_steps-env_steps);buffer=RolloutBuffer(rollout,adapter.num_agents,adapter.obs_dim,adapter.state_dim,adapter.action_dim);rollout_actions=[]
            for _ in range(rollout):
                with torch.no_grad():actions,log_probs,value,_=model.act(torch.as_tensor(obs,device=device),torch.as_tensor(state,device=device),False)
                action_np=actions.cpu().numpy();next_obs,next_state,rewards,team_done,next_active,info=adapter.step(action_np)
                buffer.store(obs,state,action_np,log_probs.cpu().numpy(),rewards,team_done,value.item(),active);rollout_actions.append(action_np)
                episode_return+=float((rewards*active).sum()/max(active.sum(),1.));episode_length+=1;env_steps+=1
                for index,aid in enumerate(adapter.agent_ids):role_returns[aid]+=float(rewards[index]*active[index])
                if args.scenario=="simple_paper_3v2_hetero":
                    mav_components=info["reward_components"]["red_mav_0"]
                    component_sums["mav_safety_sum"]+=float(mav_components.get("r_safety",0.));component_sums["mav_support_sum"]+=float(mav_components.get("r_support",0.));component_sums["mav_event_sum"]+=float(mav_components.get("r_event",0.))
                    component_sums["mav_event_death_sum"]+=float(mav_components.get("r_event_death",0.));component_sums["mav_event_team_contribution_sum"]+=float(mav_components.get("r_event_team_contribution",0.));component_sums["mav_awareness_sum"]+=float(mav_components.get("r_support_awareness",0.));component_sums["mav_position_sum"]+=float(mav_components.get("r_support_position",0.));component_sums["mav_dense_sum"]+=float(mav_components.get("total_dense",mav_components.get("r_safety",0.)+mav_components.get("r_support",0.)))
                    for aid in ("red_uav_0","red_uav_1"):
                        u=info["reward_components"][aid]
                        for target,key in (("red_uav_height_sum","r_height"),("red_uav_speed_sum","r_speed"),("red_uav_angle_sum","r_angle"),("red_uav_distance_sum","r_distance"),("red_uav_dodge_sum","r_dodge"),("red_uav_event_sum","r_event")):component_sums[target]+=float(u.get(key,0.))
                    component_sums["relay_only_track_steps"]+=int(info["relay_only_track_count"]);component_sums["mav_support_steps"]+=int(info["mav_support_active"])
                obs,state,active=next_obs,next_state,next_active
                if team_done:
                    episodes_completed+=1;episode_row=build_episode_component_row(episodes_completed,env_steps,episode_length,info,episode_return,role_returns,component_sums,args.scenario);mean_uav=episode_row["mean_red_uav_return"]
                    episode_writer.writerow(episode_row);episode_file.flush();recent.append({"return":episode_return,"length":episode_length,"winner":info["winner"],
                      "timeout":info["termination_reason"]=="timeout","red_alive":info["alive_red"],"blue_alive":info["alive_blue"],
                      "missile_hits":info["missile_hits"],"invalid":info["numerical_invalid"]>0,"mav_return":episode_row["mav_return"],"mean_red_uav_return":mean_uav,
                      "mav_safety_sum":component_sums["mav_safety_sum"],"mav_support_sum":component_sums["mav_support_sum"],"mav_event_sum":component_sums["mav_event_sum"],"relay_only_track_steps":component_sums["relay_only_track_steps"]})
                    obs,state,info=adapter.reset(seed=args.seed+env_steps);active=np.ones(adapter.num_agents,np.float32);episode_return=0.;episode_length=0
                    role_returns={aid:0. for aid in adapter.agent_ids};component_sums={key:0. for key in COMPONENT_SUM_FIELDS}
            with torch.no_grad():next_value=0. if buffer.team_dones[-1] else model.value(torch.as_tensor(state,device=device)).item()
            stats=trainer.update(buffer,next_value);updates+=1;actions_all=np.concatenate(rollout_actions,axis=0);window=recent_stats(recent)
            row={"update":updates,"env_steps":env_steps,"episodes_completed":episodes_completed,**window,**{k:stats[k] for k in ("actor_loss","critic_loss","entropy","approx_kl","clip_fraction")},
              "action_mean_abs":float(np.mean(np.abs(actions_all))),"action_saturation_rate":float(np.mean(np.abs(actions_all)>=.999))}
            train_writer.writerow(row);train_file.flush();save_checkpoint(output/"latest.pt",model,adapter,args,env_steps)
            while env_steps>=next_eval and next_eval<=args.total_env_steps:
                summary=run_eval(next_eval);score=(summary["red_win_rate"],summary["mean_return"])
                if score>best_score:best_score=score;save_checkpoint(output/"best.pt",model,adapter,args,env_steps)
                next_eval+=args.eval_interval
            print(f"update={updates} env_steps={env_steps} episodes={episodes_completed} return={window['mean_episode_return']:.3f} actor_loss={stats['actor_loss']:.4f} critic_loss={stats['critic_loss']:.4f}",flush=True)
        if (args.total_env_steps==0 or (args.total_env_steps%args.eval_interval)!=0):
            summary=run_eval(env_steps);score=(summary["red_win_rate"],summary["mean_return"])
            if score>best_score:save_checkpoint(output/"best.pt",model,adapter,args,env_steps)
        save_checkpoint(output/"latest.pt",model,adapter,args,env_steps)
        print(f"training_seconds={time.perf_counter()-start:.3f}",flush=True)
    finally:train_file.close();eval_file.close();episode_file.close();env.close()
if __name__=="__main__":main()
