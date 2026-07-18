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

def evaluate_model(model,scenario,episodes,device,deterministic=True,seed=1,return_rows=False):
    numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state();cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        rows=[]
        for episode in range(episodes):
            episode_seed=seed+episode
            np.random.seed(episode_seed);torch.manual_seed(episode_seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(episode_seed)
            env=SimpleTAMCombatEnv(scenario,"red");adapter=SimpleMAPPOAdapter(env);obs,state,info=adapter.reset(seed=episode_seed)
            active=np.ones(adapter.num_agents,np.float32);episode_return=0.;action_trace=[]
            try:
                for step in range(1000):
                    with torch.no_grad():actions,_,_,_=model.act(torch.as_tensor(obs,device=device),torch.as_tensor(state,device=device),deterministic)
                    action_np=actions.cpu().numpy();action_trace.append(action_np.tolist())
                    obs,state,rewards,done,next_active,info=adapter.step(action_np)
                    episode_return+=float((rewards*active).sum()/max(active.sum(),1.));active=next_active
                    if done:break
                rows.append({"return":episode_return,"length":step+1,"winner":info["winner"],"timeout":info["termination_reason"]=="timeout",
                  "red_alive":info["alive_red"],"blue_alive":info["alive_blue"],"missile_launches":info["missiles_fired"],"missile_hits":info["missile_hits"],
                  "crashes":info["red_crashes"]+info["blue_crashes"],"boundary_deaths":info["boundary_deaths"],
                  "numerical_invalid":info["numerical_invalid"],"envelope":info["flight_envelope_violation"],"action_trace":action_trace})
            finally:env.close()
        n=max(len(rows),1)
        result={"episodes":len(rows),"mean_return":float(np.mean([r["return"] for r in rows])),"return_std":float(np.std([r["return"] for r in rows])),"mean_episode_length":float(np.mean([r["length"] for r in rows])),
          "red_win_rate":sum(r["winner"]=="red" for r in rows)/n,"blue_win_rate":sum(r["winner"]=="blue" for r in rows)/n,
          "draw_rate":sum(r["winner"]=="draw" for r in rows)/n,"timeout_rate":sum(r["timeout"] for r in rows)/n,
          "mean_red_alive":float(np.mean([r["red_alive"] for r in rows])),"mean_blue_alive":float(np.mean([r["blue_alive"] for r in rows])),
          "missile_launches":sum(r["missile_launches"] for r in rows),"missile_hits":sum(r["missile_hits"] for r in rows),
          "crashes":sum(r["crashes"] for r in rows),"boundary_deaths":sum(r["boundary_deaths"] for r in rows),
          "numerical_invalid_episodes":sum(r["numerical_invalid"]>0 for r in rows),"flight_envelope_violation_episodes":sum(r["envelope"] for r in rows)}
        if return_rows:result["rows"]=rows
        return result
    finally:
        np.random.set_state(numpy_state);torch.random.set_rng_state(torch_state)
        if cuda_states is not None:torch.cuda.set_rng_state_all(cuda_states)

def load_checkpoint(path,scenario,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False);env=SimpleTAMCombatEnv(scenario);adapter=SimpleMAPPOAdapter(env);env.close()
    expected={"scenario":scenario,"obs_dim":adapter.obs_dim,"state_dim":adapter.state_dim,"action_dim":adapter.action_dim}
    for key,value in expected.items():
        if checkpoint.get(key)!=value:raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r} does not match expected {value!r}")
    if list(checkpoint.get("agent_ids",[]))!=adapter.agent_ids:raise ValueError("checkpoint agent_ids/num_agents mismatch")
    model=SharedMAPPOActorCritic(adapter.obs_dim,adapter.state_dim,adapter.action_dim).to(device);model.load_state_dict(checkpoint["model_state_dict"]);model.eval();return model

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--scenario",choices=("simple_paper_1v1","simple_paper_2v2","simple_paper_3v2_hetero"),required=True);p.add_argument("--episodes",type=int,default=5);p.add_argument("--device",default="auto");p.add_argument("--deterministic",action="store_true");p.add_argument("--seed",type=int,default=1);a=p.parse_args()
    device=resolve_device(a.device);model=load_checkpoint(a.model,a.scenario,device);print(json.dumps(evaluate_model(model,a.scenario,a.episodes,device,a.deterministic,a.seed),indent=2))
if __name__=="__main__":main()
