"""Train recurrent mixed-action PPO on the nominal paper 1v1 scenario."""
from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
for n,v in (("OMP_NUM_THREADS","1"),("MKL_NUM_THREADS","1"),("KMP_DUPLICATE_LIB_OK","TRUE")): os.environ.setdefault(n,v)
import numpy as np
import torch
if __package__ in (None,""): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.evaluation import evaluate_policy
from aircombat_env_v1.recurrent_ppo import RecurrentActor,RecurrentCritic,RecurrentRolloutBuffer,recurrent_ppo_update
from aircombat_env_v1.seeds import NOMINAL_VALIDATION_SEEDS
from aircombat_env_v1.training import TrainingConfig,append_csv,resolve_device,save_model,set_random_seeds,write_json
from aircombat_env_v1.vec_env import SubprocVecEnv


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--total-steps",type=int,default=200000)
    p.add_argument("--num-envs",type=int,default=8); p.add_argument("--rollout-steps",type=int,default=256)
    p.add_argument("--sequence-length",type=int,default=32); p.add_argument("--eval-interval",type=int,default=10000)
    p.add_argument("--eval-episodes",type=int,default=50); p.add_argument("--seed",type=int,default=1)
    p.add_argument("--device",default="auto"); p.add_argument("--output-dir")
    p.add_argument("--resume-checkpoint"); p.add_argument("--initial-step",type=int,default=0)
    return p.parse_args()


def main():
    a=parse_args(); cfg=TrainingConfig(total_steps=a.total_steps,num_envs=a.num_envs,rollout_steps=a.rollout_steps,
        eval_interval=a.eval_interval,eval_episodes=a.eval_episodes,seed=a.seed,device=a.device)
    out=Path(a.output_dir) if a.output_dir else Path("aircombat_env_v1/outputs")/f"recurrent_ppo_1v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True,exist_ok=True); write_json(out/"config.json",{**asdict(cfg),"sequence_length":a.sequence_length,"scenario":"paper_nominal_1v1"})
    set_random_seeds(a.seed); device=resolve_device(a.device); actor=RecurrentActor().to(device); critic=RecurrentCritic().to(device)
    if a.resume_checkpoint:
        payload=torch.load(a.resume_checkpoint,map_location=device,weights_only=False)
        actor.load_state_dict(payload["actor"]); critic.load_state_dict(payload["critic"])
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=cfg.learning_rate)
    step=int(a.initial_step); updates=0
    next_eval=((step//a.eval_interval)+1)*a.eval_interval
    best=None; consecutive=0; start_time=time.time()
    initial=evaluate_policy("recurrent_ppo",scenario="paper_nominal_1v1",
        seeds=NOMINAL_VALIDATION_SEEDS[:a.eval_episodes],actor=actor,device=device)
    append_csv(out/"evaluation_log.csv",{"global_step":step,**initial})
    initial_key=(initial["red_missile_kill_rate"],-initial["blue_missile_kill_rate"],
                 -(initial["mean_hit_time_s"] or 1e9),-initial["numerical_invalid"])
    best={"key":initial_key,"global_step":step,**initial}
    save_model(out/"best_nominal.pt",actor,critic,cfg,step)
    consecutive=1 if initial["red_missile_kill_rate"]>=.8 and initial["numerical_invalid"]==0 else 0
    with SubprocVecEnv(a.num_envs,a.seed,{"scenario_mode":"paper_nominal_1v1","opponent_policy":"paper_greedy","max_steps":1000}) as envs:
        obs,_=envs.reset(); actor_h=actor.initial_hidden(a.num_envs,device); critic_h=critic.initial_hidden(a.num_envs,device)
        episode_starts=np.ones(a.num_envs,np.float32)
        while step<a.total_steps:
            rs=min(a.rollout_steps,max(1,int(np.ceil((a.total_steps-step)/a.num_envs))))
            buf=RecurrentRolloutBuffer(rs,a.num_envs); rewards_all=[]
            rollout_red_hits=rollout_blue_hits=rollout_launches=0
            for _ in range(rs):
                ot=torch.as_tensor(obs,dtype=torch.float32,device=device); st=torch.as_tensor(episode_starts,dtype=torch.float32,device=device)[:,None]
                with torch.no_grad():
                    m,f,lp,_,new_ah=actor.act(ot,actor_h,st)
                    values,new_ch=critic(ot,critic_h,st); values=values[:,0]
                actions=[{"maneuver":x,"fire":int(y)} for x,y in zip(m.cpu().numpy(),f.cpu().numpy())]
                next_obs,reward,terminated,truncated,infos=envs.step(actions); done=np.logical_or(terminated,truncated)
                for info in infos:
                    terminal=info.get("terminal_info")
                    if terminal:
                        rollout_red_hits += int(terminal.get("event")=="red_hit")
                        rollout_blue_hits += int(terminal.get("event")=="blue_hit")
                        rollout_launches += int(terminal.get("red_launch_count",0))
                bootstrap=next_obs.copy()
                for i,info in enumerate(infos):
                    if truncated[i]: bootstrap[i]=info["terminal_observation"]
                with torch.no_grad(): next_values,_=critic(torch.as_tensor(bootstrap,dtype=torch.float32,device=device),new_ch,
                    torch.as_tensor(done.astype(np.float32),device=device)[:,None])
                buf.add(observations=obs,maneuvers=m.cpu().numpy(),fire=f.cpu().numpy(),log_probs=lp.cpu().numpy(),
                    values=values.cpu().numpy(),next_values=next_values[:,0].cpu().numpy(),rewards=reward,
                    terminated=terminated,truncated=truncated,episode_starts=episode_starts,
                    actor_hidden=actor_h[0].cpu().numpy(),critic_hidden=critic_h[0].cpu().numpy())
                obs=next_obs; actor_h=new_ah; critic_h=new_ch; episode_starts=done.astype(np.float32)
                actor_h[:,done]=0; critic_h[:,done]=0; rewards_all.extend(reward.tolist())
            step+=rs*a.num_envs; updates+=1
            chunks=buf.sequences(a.sequence_length,cfg.gamma,cfg.gae_lambda)
            metrics=recurrent_ppo_update(actor,critic,optimizer,chunks,device,clip_epsilon=cfg.clip_epsilon,
                update_epochs=cfg.update_epochs,entropy_coef=cfg.entropy_coef,value_coef=cfg.value_coef,
                max_grad_norm=cfg.max_grad_norm,target_kl=cfg.target_kl)
            append_csv(out/"training_log.csv",{"global_step":step,"update":updates,
                "mean_rollout_reward":float(np.mean(rewards_all)),"red_missile_hits":rollout_red_hits,
                "blue_missile_hits":rollout_blue_hits,"red_launches":rollout_launches,**metrics})
            save_model(out/"latest.pt",actor,critic,cfg,step)
            if step>=next_eval or step>=a.total_steps:
                seeds=NOMINAL_VALIDATION_SEEDS[:a.eval_episodes]
                result=evaluate_policy("recurrent_ppo",scenario="paper_nominal_1v1",seeds=seeds,actor=actor,device=device)
                append_csv(out/"evaluation_log.csv",{"global_step":step,**result})
                key=(result["red_missile_kill_rate"],-result["blue_missile_kill_rate"],
                     -(result["mean_hit_time_s"] or 1e9),-result["numerical_invalid"])
                if best is None or key>best["key"]:
                    best={"key":key,"global_step":step,**result}; save_model(out/"best_nominal.pt",actor,critic,cfg,step)
                gate=result["red_missile_kill_rate"]>=.8 and result["numerical_invalid"]==0
                consecutive=consecutive+1 if gate else 0
                while next_eval<=step: next_eval+=a.eval_interval
                print(f"eval step={step} kill={result['red_missile_kill_rate']:.3f} invalid={result['numerical_invalid']}",flush=True)
                if consecutive>=2: break
            print(f"step={step} reward={np.mean(rewards_all):.3f} kl={metrics['approximate_kl']:.5f}",flush=True)
    summary={"output_dir":str(out.resolve()),"global_step":step,"updates":updates,"best":best,
             "early_stop":consecutive>=2,"elapsed_seconds":time.time()-start_time}
    write_json(out/"summary.json",summary); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
