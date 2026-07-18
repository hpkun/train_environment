"""Train the minimal feed-forward MAPPO baseline on SimpleTAMCombatEnv."""
from __future__ import annotations
import argparse,csv,sys,time
from collections import deque
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.simple_env import SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
from aircombat_env_v1.scripts.eval_simple_mappo import evaluate_model,resolve_device
import torch

TRAIN_FIELDS=("update","env_steps","episodes_completed","mean_episode_return","mean_episode_length","red_win_rate","blue_win_rate","draw_rate","timeout_rate","mean_red_alive","mean_blue_alive","missile_hits_per_episode","actor_loss","critic_loss","entropy","approx_kl","clip_fraction","action_mean_abs","action_saturation_rate","numerical_invalid_episodes")
EVAL_FIELDS=("env_steps","mean_return","red_win_rate","blue_win_rate","draw_rate","timeout_rate","mean_episode_length")

def save_checkpoint(path,model,adapter,args,env_steps):
    torch.save({"model_state_dict":model.state_dict(),"scenario":args.scenario,"obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,
      "action_dim":adapter.action_dim,"agent_ids":adapter.agent_ids,"training_args":vars(args),"env_steps":env_steps},path)

def recent_stats(records):
    if not records:return {"mean_episode_return":0.,"mean_episode_length":0.,"red_win_rate":0.,"blue_win_rate":0.,"draw_rate":0.,"timeout_rate":0.,"mean_red_alive":0.,"mean_blue_alive":0.,"missile_hits_per_episode":0.,"numerical_invalid_episodes":0}
    rows=list(records);n=len(rows)
    return {"mean_episode_return":np.mean([r["return"] for r in rows]),"mean_episode_length":np.mean([r["length"] for r in rows]),
      "red_win_rate":sum(r["winner"]=="red" for r in rows)/n,"blue_win_rate":sum(r["winner"]=="blue" for r in rows)/n,
      "draw_rate":sum(r["winner"]=="draw" for r in rows)/n,"timeout_rate":sum(r["timeout"] for r in rows)/n,
      "mean_red_alive":np.mean([r["red_alive"] for r in rows]),"mean_blue_alive":np.mean([r["blue_alive"] for r in rows]),
      "missile_hits_per_episode":np.mean([r["missile_hits"] for r in rows]),"numerical_invalid_episodes":sum(r["invalid"] for r in rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--scenario",choices=("simple_paper_1v1","simple_paper_2v2","simple_paper_3v2_hetero"),required=True);p.add_argument("--total-env-steps",type=int,default=10000);p.add_argument("--rollout-length",type=int,default=256);p.add_argument("--seed",type=int,default=1);p.add_argument("--device",default="auto");p.add_argument("--output-dir",required=True);p.add_argument("--eval-interval",type=int,default=2500);p.add_argument("--eval-episodes",type=int,default=5);p.add_argument("--deterministic-eval",action=argparse.BooleanOptionalAction,default=True);p.add_argument("--actor-lr",type=float,default=3e-4);p.add_argument("--critic-lr",type=float,default=3e-4);p.add_argument("--entropy-coef",type=float,default=.01);p.add_argument("--ppo-epochs",type=int,default=4);args=p.parse_args()
    np.random.seed(args.seed);torch.manual_seed(args.seed);device=resolve_device(args.device);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    env=SimpleTAMCombatEnv(args.scenario,"red");adapter=SimpleMAPPOAdapter(env);model=SharedMAPPOActorCritic(adapter.obs_dim,adapter.state_dim,adapter.action_dim).to(device)
    trainer=MAPPOTrainer(model,args.actor_lr,args.critic_lr,entropy_coef=args.entropy_coef,ppo_epochs=args.ppo_epochs)
    train_file=(output/"train_log.csv").open("w",newline="",encoding="utf-8");train_writer=csv.DictWriter(train_file,fieldnames=TRAIN_FIELDS);train_writer.writeheader()
    eval_file=(output/"eval_log.csv").open("w",newline="",encoding="utf-8");eval_writer=csv.DictWriter(eval_file,fieldnames=EVAL_FIELDS);eval_writer.writeheader()
    def run_eval(step):
        model.eval();summary=evaluate_model(model,args.scenario,args.eval_episodes,device,args.deterministic_eval,args.seed+100000+step);model.train()
        eval_writer.writerow({key:step if key=="env_steps" else summary[key] for key in EVAL_FIELDS});eval_file.flush();return summary
    best_score=(-float("inf"),-float("inf"));initial_eval=run_eval(0);save_checkpoint(output/"initial.pt",model,adapter,args,0);save_checkpoint(output/"best.pt",model,adapter,args,0);best_score=(initial_eval["red_win_rate"],initial_eval["mean_return"])
    obs,state,info=adapter.reset(seed=args.seed);active=np.ones(adapter.num_agents,np.float32);env_steps=updates=episodes_completed=0
    episode_return=0.;episode_length=0;recent=deque(maxlen=20);next_eval=args.eval_interval;start=time.perf_counter()
    try:
        while env_steps<args.total_env_steps:
            rollout=min(args.rollout_length,args.total_env_steps-env_steps);buffer=RolloutBuffer(rollout,adapter.num_agents,adapter.obs_dim,adapter.state_dim,adapter.action_dim);rollout_actions=[]
            for _ in range(rollout):
                with torch.no_grad():actions,log_probs,value,_=model.act(torch.as_tensor(obs,device=device),torch.as_tensor(state,device=device),False)
                action_np=actions.cpu().numpy();next_obs,next_state,rewards,team_done,next_active,info=adapter.step(action_np)
                buffer.store(obs,state,action_np,log_probs.cpu().numpy(),rewards,team_done,value.item(),active);rollout_actions.append(action_np)
                episode_return+=float((rewards*active).sum()/max(active.sum(),1.));episode_length+=1;env_steps+=1
                obs,state,active=next_obs,next_state,next_active
                if team_done:
                    episodes_completed+=1;recent.append({"return":episode_return,"length":episode_length,"winner":info["winner"],
                      "timeout":info["termination_reason"]=="timeout","red_alive":info["alive_red"],"blue_alive":info["alive_blue"],
                      "missile_hits":info["missile_hits"],"invalid":info["numerical_invalid"]>0})
                    obs,state,info=adapter.reset(seed=args.seed+env_steps);active=np.ones(adapter.num_agents,np.float32);episode_return=0.;episode_length=0
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
    finally:train_file.close();eval_file.close();env.close()
if __name__=="__main__":main()
