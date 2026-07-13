"""Paper-grounded heterogeneous reward, not exact paper reproduction."""
from __future__ import annotations

import math
import numpy as np

V4_COMPONENT_FIELDS = (
    "v4_uav_height_raw", "v4_uav_speed_raw", "v4_uav_angle_offense_raw",
    "v4_uav_reverse_angle_threat_raw", "v4_uav_angle_advantage_raw",
    "v4_uav_distance_offense_raw", "v4_uav_reverse_distance_threat_raw",
    "v4_uav_distance_advantage_raw", "v4_uav_dodge_raw", "v4_uav_dense_raw",
    "v4_uav_event_kill_raw", "v4_uav_event_death_raw", "v4_uav_event_oob_raw",
    "v4_uav_event_raw", "v4_mav_dist_safe_raw", "v4_mav_missile_safe_raw",
    "v4_mav_aspect_safe_raw", "v4_mav_safety_raw", "v4_mav_support_position_raw",
    "v4_mav_awareness_raw", "v4_mav_support_raw", "v4_mav_death_event_raw",
    "v4_mav_team_kill_raw", "v4_mav_shared_kill_raw", "v4_mav_event_raw",
    "v4_terminal_raw", "v4_flight_raw", "v4_dense_scaled", "v4_event_scaled",
    "v4_mav_safety_scaled", "v4_mav_support_scaled", "v4_mav_event_scaled",
    "v4_terminal_scaled", "v4_flight_scaled", "v4_total", "v4_component_sum",
    "v4_identity_error", "v4_j_combat", "v4_terminal_applied",
)


def reset_v4_state(env):
    env._v4_uav_death_seen = set(); env._v4_uav_oob_seen = set()
    env._v4_mav_death_seen = False; env._v4_mav_team_credit_used = 0.0
    env._v4_terminal_applied = False


def _mean(values): return float(np.mean(values)) if values else 0.0


def _pair(env, attacker, target):
    g = env._brma_tam_3d_geometry(attacker, target)
    ata=float(g["tam_ata_rad"]); aa=float(g["tam_aa_rad"]); d=float(g["target_distance_m"])
    angle=float(np.clip(1.0-(ata+aa)/math.pi,-1.0,1.0))
    distance=float(env._tam_table1_distance_raw(d))
    return angle, distance


def _terminal(env, alive_red, alive_blue, cfg):
    if env._v4_terminal_applied: return 0.0, 0.0
    done = not alive_red or not alive_blue or env.current_step >= env.max_steps
    if not done: return 0.0, 0.0
    env._v4_terminal_applied=True
    if not alive_blue and alive_red: return 1.0, 1.0
    if not alive_red and alive_blue: return -1.0, 1.0
    if not alive_red and not alive_blue: return 0.0, 1.0
    nr=len(alive_red)/max(len(env.red_ids),1); nb=len(alive_blue)/max(len(env.blue_ids),1)
    mav_alive=any(env.agent_roles.get(r)=="mav" for r in alive_red)
    score=nr-nb
    if not mav_alive and len(alive_blue)==len(env.blue_ids): score=min(score,-0.5)
    return float(np.clip(score,-1.0,1.0)),1.0


def compute_v4_reward(env, base_rewards, components):
    cfg=env.brma_tam_paper_calibrated_v4_config; scales=cfg["scales"]
    alive_blue={b:s for b,s in env.blue_planes.items() if s.is_alive}
    alive_red={r:s for r,s in env.red_planes.items() if s.is_alive}
    terminal,terminal_applied=_terminal(env,alive_red,alive_blue,cfg)
    mav_id=next((r for r in env.red_ids if env.agent_roles.get(r)=="mav"),None)
    mav=env.red_planes.get(mav_id)
    for rid in env.red_ids:
        role=env.agent_roles.get(rid); sim=env.red_planes.get(rid); comp=components.setdefault(rid,{})
        vals={k:0.0 for k in V4_COMPONENT_FIELDS}
        if role=="attack_uav" and sim is not None:
            h,_=env._tam_table1_uav_height_raw(sim,cfg); speed=np.linalg.norm(sim.get_velocity())
            ref=float(cfg["uav"]["reference_speed_mps"]); v=env._tam_table1_speed_raw(speed,ref)
            ao=[];at=[];do=[];dt=[]
            for blue in alive_blue.values():
                a,d=_pair(env,sim,blue); ra,rd=_pair(env,blue,sim)
                ao.append(a);do.append(d);at.append(ra);dt.append(rd)
            dodge,*_=env._tam_table1_dodge_raw(sim,cfg)
            A=_mean(ao)-0.8*_mean(at); D=_mean(do)-0.8*_mean(dt)
            dense=(10*h+10*v+15*A+10*D+30*dodge)/75.0
            kills=int(env._step_kill_count.get(rid,0)); death=0.0; oob=0.0
            if not sim.is_alive and rid not in env._v4_uav_death_seen: death=-1.0; env._v4_uav_death_seen.add(rid)
            if env._tam_v7_out_of_zone(sim) and rid not in env._v4_uav_oob_seen: oob=-0.5; env._v4_uav_oob_seen.add(rid)
            event=float(kills)+death+oob
            vals.update(v4_uav_height_raw=h,v4_uav_speed_raw=v,v4_uav_angle_offense_raw=_mean(ao),
                v4_uav_reverse_angle_threat_raw=_mean(at),v4_uav_angle_advantage_raw=A,
                v4_uav_distance_offense_raw=_mean(do),v4_uav_reverse_distance_threat_raw=_mean(dt),
                v4_uav_distance_advantage_raw=D,v4_uav_dodge_raw=dodge,v4_uav_dense_raw=dense,
                v4_uav_event_kill_raw=float(kills),v4_uav_event_death_raw=death,v4_uav_event_oob_raw=oob,v4_uav_event_raw=event)
            vals["v4_dense_scaled"]=scales["uav_dense_scale"]*dense; vals["v4_event_scaled"]=scales["uav_event_scale"]*event
        elif role=="mav":
            dist,_=env._tam_table1_mav_dist(mav,cfg); missile=env._tam_table1_mav_threat(mav); aspect=env._tam_table1_mav_aspect(mav)
            safety=.5*dist+.3*missile+.2*aspect
            pos,_=env._tam_table1_mav_position(mav,cfg); pairs=shared=0
            for uid in env.red_ids:
                if env.agent_roles.get(uid)!="attack_uav": continue
                for bid in alive_blue:
                    pairs+=1; ts=env._mav_shared_track_state(uid,bid)
                    shared+=int(ts.get("mav_shared_visible",False) and not ts.get("direct_visible",False))
            aware=shared/max(pairs,1); support=.6*pos+.4*aware
            death=0.0
            if mav is not None and not mav.is_alive and not env._v4_mav_death_seen: death=-1.0; env._v4_mav_death_seen=True
            team_kills=sum(int(env._step_kill_count.get(r,0)) for r in env.red_ids if env.agent_roles.get(r)=="attack_uav")
            available=max(0.0,float(scales["mav_team_kill_credit_cap"])-env._v4_mav_team_credit_used)
            credit=min(team_kills*float(scales["mav_team_kill_credit_scale"]),available); env._v4_mav_team_credit_used+=credit
            shared_kills=sum(1 for x in getattr(env,"_launch_quality_done_step_records",[]) or []
                             if str(x.get("shooter_id","")).startswith("red_")
                             and x.get("raw_termination_reason")=="hit"
                             and x.get("launch_track_source") in {"mav_shared","direct_and_mav_shared"})
            event=scales["mav_death_event_scale"]*death+credit
            vals.update(v4_mav_dist_safe_raw=dist,v4_mav_missile_safe_raw=missile,v4_mav_aspect_safe_raw=aspect,
                v4_mav_safety_raw=safety,v4_mav_support_position_raw=pos,v4_mav_awareness_raw=aware,
                v4_mav_support_raw=support,v4_mav_death_event_raw=death,v4_mav_team_kill_raw=float(team_kills),
                v4_mav_shared_kill_raw=float(shared_kills),v4_mav_event_raw=event)
            vals["v4_mav_safety_scaled"]=scales["mav_safety_scale"]*safety; vals["v4_mav_support_scaled"]=scales["mav_support_scale"]*support
            vals["v4_mav_event_scaled"]=scales["mav_event_scale"]*event
        flight=float(comp.get("r_pitch",0))+float(comp.get("r_roll",0))+float(comp.get("r_vel",0))
        vals["v4_flight_raw"]=flight; vals["v4_flight_scaled"]=scales["flight_scale"]*flight
        vals["v4_terminal_raw"]=terminal; vals["v4_terminal_applied"]=terminal_applied; vals["v4_terminal_scaled"]=scales["terminal_scale"]*terminal
        vals["v4_j_combat"]=len(alive_red)/max(len(env.red_ids),1)-len(alive_blue)/max(len(env.blue_ids),1)
        total=sum(vals[k] for k in ("v4_dense_scaled","v4_event_scaled","v4_mav_safety_scaled","v4_mav_support_scaled","v4_mav_event_scaled","v4_terminal_scaled","v4_flight_scaled"))
        vals["v4_total"]=total; vals["v4_component_sum"]=total; vals["v4_identity_error"]=0.0
        if not all(np.isfinite(float(vals[k])) for k in V4_COMPONENT_FIELDS): raise ValueError(f"non-finite v4 reward agent={rid}")
        comp.update(vals); comp["total"]=total; base_rewards[rid]=total
    return base_rewards,components
