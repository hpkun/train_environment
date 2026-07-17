"""Deterministic baseline and recurrent-policy evaluation."""
from __future__ import annotations
from collections import Counter
import numpy as np
import torch
from .combat import pursuit_action
from .env import AirCombat1v1Env


def record_event(counts,event):
    mapping={"red_hit":"red_hits","blue_hit":"blue_hits","red_crash":"red_crashes",
             "blue_crash":"blue_crashes","timeout":"timeouts"}
    if event in mapping: counts[mapping[event]]+=1
    elif "numerical_invalid" in str(event) or event=="physics_exception": counts["numerical_invalid"]+=1
    else: counts["draws"]+=1


def evaluate_policy(policy,episodes=20,scenario="paper_nominal_1v1",opponent="paper_greedy",
                    seed=1,seeds=None,stochastic=False,actor=None,device="cpu"):
    del stochastic
    seeds=list(seeds) if seeds is not None else [seed+i for i in range(int(episodes))]
    env=AirCombat1v1Env(scenario_mode=scenario,opponent_policy=opponent,max_steps=1000)
    rng=np.random.default_rng(seed); counts=Counter(); returns=[]; lengths=[]; launches=[]; hit_times=[]; envelope=0
    try:
        for episode_seed in seeds:
            obs,_=env.reset(seed=int(episode_seed)); total=0.; final={}
            hidden=actor.initial_hidden(1,device) if actor is not None and hasattr(actor,"initial_hidden") else None
            start=torch.ones(1,1,device=device)
            for step in range(env.max_steps):
                if policy in ("zero","zero_no_fire"):
                    action={"maneuver":np.zeros(3,np.float32),"fire":0}
                elif policy=="random":
                    action={"maneuver":rng.uniform(-1,1,3).astype(np.float32),"fire":int(rng.integers(2))}
                elif policy in ("pursuit_rule","pursuit_fire_rule"):
                    action={"maneuver":pursuit_action(env.red_state,env.blue_state),"fire":1}
                elif policy in ("ppo","recurrent_ppo"):
                    if actor is None: raise ValueError("actor is required")
                    with torch.no_grad():
                        m,f,_,_,hidden=actor.act(torch.as_tensor(obs,dtype=torch.float32,device=device)[None],hidden,start,True)
                    start.zero_(); action={"maneuver":m[0].cpu().numpy(),"fire":int(f[0].item())}
                else: raise ValueError(f"unknown policy: {policy}")
                obs,reward,terminated,truncated,final=env.step(action); total+=reward
                if terminated or truncated: break
            record_event(counts,final.get("event","timeout")); returns.append(total); lengths.append(step+1)
            launches.append(final.get("red_launch_count",0)); envelope+=int(final.get("flight_envelope_violation",False))
            if final.get("hit_time_s") is not None: hit_times.append(final["hit_time_s"])
    finally: env.close()
    n=max(len(seeds),1)
    result={"policy":policy,"scenario":scenario,"opponent":opponent,"episodes":len(seeds),
        **{k:counts[k] for k in ("red_hits","blue_hits","red_crashes","blue_crashes","numerical_invalid","draws","timeouts")},
        "opponent_failures":counts["blue_crashes"],"red_missile_kill_rate":counts["red_hits"]/n,
        "blue_missile_kill_rate":counts["blue_hits"]/n,"red_hit_rate":counts["red_hits"]/n,
        "blue_hit_rate":counts["blue_hits"]/n,"red_crash_rate":counts["red_crashes"]/n,
        "timeout_rate":counts["timeouts"]/n,"invalid_rate":counts["numerical_invalid"]/n,
        "mean_return":float(np.mean(returns)),"mean_steps":float(np.mean(lengths)),
        "mean_launch_count":float(np.mean(launches)),"mean_hit_time_s":float(np.mean(hit_times)) if hit_times else None,
        "flight_envelope_violation_rate":envelope/n}
    result["best_eligible"]=counts["numerical_invalid"]==0 and counts["blue_crashes"]==0
    return result
