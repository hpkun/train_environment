"""Formal Gate A/B/C diagnostics with no learning or parameter tuning."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_env import TAMPaperCombatEnv
from aircombat_env_v1.paper_opponent import MANOEUVRES

LEVEL=np.asarray(MANOEUVRES["level"],np.int64)

def _run_episode(gate,seed,swap=False):
    mode="paper_nominal_2v2" if gate=="C" else "paper_nominal_1v1"
    enabled={"red_0"} if gate in ("A","C") else None
    env=TAMPaperCombatEnv(mode,"all",weapon_enabled_agent_ids=enabled)
    env.reset(seed=seed)
    if swap:
        for a in env.agents:a.side="blue" if a.side=="red" else "red"
        env._update_targets(count_changes=False)
    red0_history=[];all_events=[];first_hit_step=None
    for step in range(1000):
        actions=env.build_rule_actions()
        if gate=="A":actions["blue_0"]=LEVEL.copy()
        elif gate=="C":
            for aid in ("red_1","blue_0","blue_1"):actions[aid]=LEVEL.copy()
        target_before=env.current_targets.get("red_0");_,_,terminated,truncated,info=env.step(actions)
        events=info.get("events",[]);all_events.extend(events)
        if first_hit_step is None and any(e.get("hit") and e.get("shooter_id")=="red_0" for e in events):first_hit_step=step+1
        red0_history.append({"decision_step":step,"current_target":target_before,"missiles_left":env.by_id["red_0"].missile_left,
            "events":[e for e in events if e.get("shooter_id")=="red_0"]})
        if terminated or truncated:break
    launches=[e for e in all_events if e.get("event_type")=="missile_launch"]
    hits=[e for e in all_events if e.get("hit")]
    red_launches=[e for e in launches if e["shooter_id"]=="red_0"]
    red_hits=[e for e in hits if e["shooter_id"]=="red_0"]
    second=next((e for e in red_launches if e.get("launch_number")==2),None)
    row={"gate":gate,"steps":step+1,"winner":info["winner"],"red_launches":len(red_launches),"red_missile_hits":len(red_hits),
         "blue_missile_hits":sum(env.by_id[e["shooter_id"]].side=="blue" for e in hits),"red_crashes":info["red_crashes"],"blue_crashes":info["blue_crashes"],
         "timeout":info["termination_reason"]=="episode_limit","invalid":info["invalid_episode"],"envelope_violation":info["flight_envelope_violation"],
         "first_hit_step":first_hit_step,"first_hit_time_s":None if first_hit_step is None else first_hit_step*.2,
         "target_changes":sum(e["agent_id"]=="red_0" and e["current_target"] is not None for e in info["target_change_events"]),"second_launch":second is not None,
         "second_launch_target":None if second is None else second["target_id"],"red_hit_targets":[e["target_id"] for e in red_hits],
         "missiles_fired":info["missiles_fired"],"missile_hits":info["missile_hits"],"simultaneous_kills":info["simultaneous_kills"],
         "history":red0_history if gate=="C" else None}
    env.close();return row

def run_gate(gate,episodes=20,seed=1):
    rows=[_run_episode(gate,seed+i) for i in range(episodes)]
    summary={"gate":gate,"episodes":episodes,"red_launches":sum(r["red_launches"] for r in rows),"red_missile_hit_episodes":sum(r["red_missile_hits"]>0 for r in rows),
      "red_missile_hits":sum(r["red_missile_hits"] for r in rows),"blue_missile_hits":sum(r["blue_missile_hits"] for r in rows),
      "red_crashes":sum(r["red_crashes"] for r in rows),"blue_crashes":sum(r["blue_crashes"] for r in rows),"timeouts":sum(r["timeout"] for r in rows),
      "invalid":sum(r["invalid"] for r in rows),"envelope_violations":sum(r["envelope_violation"] for r in rows),
      "target_reselection_episodes":sum(r["target_changes"]>0 for r in rows),"second_launch_episodes":sum(r["second_launch"] for r in rows),
      "two_target_kill_episodes":sum(len(set(r["red_hit_targets"]))>=2 for r in rows),
      "mean_hit_step":float(np.mean([r["first_hit_step"] for r in rows if r["first_hit_step"] is not None])) if any(r["first_hit_step"] for r in rows) else None,
      "mean_hit_time_s":float(np.mean([r["first_hit_time_s"] for r in rows if r["first_hit_time_s"] is not None])) if any(r["first_hit_time_s"] for r in rows) else None}
    if gate=="B":
        swapped=[_run_episode("B",seed+i,True) for i in range(episodes)]
        summary["symmetric_draws"]=sum(r["winner"]=="draw" and r["simultaneous_kills"]>0 for r in rows)
        summary["swapped_symmetric_draws"]=sum(r["winner"]=="draw" and r["simultaneous_kills"]>0 for r in swapped)
    return summary,rows

def main():
    p=argparse.ArgumentParser();p.add_argument("--gate",choices=("A","B","C","all"),default="all");p.add_argument("--episodes",type=int,default=20);p.add_argument("--seed",type=int,default=1);a=p.parse_args()
    result={g:run_gate(g,a.episodes,a.seed)[0] for g in ("A","B","C") if a.gate in (g,"all")};print(json.dumps(result,indent=2,default=lambda value:value.item()))
if __name__=="__main__":main()
