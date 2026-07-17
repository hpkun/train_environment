"""Formal TAM attack-UAV reward without legacy overlays."""
from __future__ import annotations
import math
import numpy as np

def paper_height_reward(altitude,minimum=750.):
    if altitude<=0:return -1.
    if altitude<minimum:return float(-1+altitude/minimum)
    return float(np.clip((altitude-minimum)/(6000-minimum),0,1))
def uav_speed_reward(ego,target):
    ego=max(float(ego),1e-6)
    return 1. if target<.5*ego else float(2-2*target/ego) if target<=1.5*ego else -1.
def uav_angle_reward(ata,aa): return float(1-(ata+aa)/np.pi)
def uav_distance_reward(distance):
    km=distance/1000.; return 1. if km<=5 else float(math.exp(-.921*(km-5))) if km<10 else -1.

class PaperReward:
    def compute(self,agents,targets,pairs,missiles,events,alive_start,out_of_zone):
        by_id={a.agent_id:a for a in agents}; killed={}
        for e in events:
            if e.get("reason")=="hit": killed[e["shooter_id"]]=killed.get(e["shooter_id"],0)+1
        rewards={}; components={}
        for a in agents:
            if not alive_start.get(a.agent_id,False):
                components[a.agent_id]={k:0. for k in ("r_height","r_speed","r_angle","r_distance","r_dodge_angle","r_dodge_speed","r_dodge","r_event","total")}; rewards[a.agent_id]=0.; continue
            target=by_id.get(targets.get(a.agent_id)); pair=pairs.get(a.agent_id)
            rh=paper_height_reward(a.position[2]); rs=uav_speed_reward(a.speed,target.speed) if target and pair else 0.
            ra=uav_angle_reward(pair.ata_rad,pair.aa_rad) if pair else 0.; rd=uav_distance_reward(pair.distance_m) if pair else 0.
            incoming=[m for m in missiles if m.alive and m.target_id==a.agent_id]; rda=rds=0.
            if incoming:
                threat=min(incoming,key=lambda m:np.linalg.norm(m.position-a.position)/max(m.speed_mps,1))
                los=a.position-threat.position; lam=float(np.arccos(np.clip(np.dot(los,threat.velocity)/max(np.linalg.norm(los)*np.linalg.norm(threat.velocity),1e-8),-1,1)))
                rda=-float(np.cos(lam)); rds=(threat.decision_start_speed_mps-threat.speed_mps)/1000.
            event=200.*killed.get(a.agent_id,0)
            if not a.alive and not a.out_of_boundary:event-=200.
            if a.agent_id in out_of_zone:event-=100.
            total=10*rh+10*rs+15*ra+10*rd+30*(rda+rds)+event
            components[a.agent_id]={"r_height":rh,"r_speed":rs,"r_angle":ra,"r_distance":rd,"r_dodge_angle":rda,
                "r_dodge_speed":rds,"r_dodge":rda+rds,"r_event":event,"total":float(total)}; rewards[a.agent_id]=float(total)
        return rewards,components
