"""Small engineering MAV reward for the simplified heterogeneous scenario."""
from __future__ import annotations
import numpy as np
from .paper_situation import assess_pair

D_DANGER_M=5000.
D_SAFE_M=10000.
D_OPT_M=4000.
D_SUPPORT_MAX_M=12000.

def _angle(a,b):
    return float(np.arccos(np.clip(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-8),-1.,1.)))

class SimpleMAVReward:
    component_names=("r_safety_distance","r_safety_threat","r_safety_aspect","r_safety",
                     "r_support_position","r_support_awareness","r_support","r_event","total")

    def compute(self,mav,agents,missiles,events,alive_start,out_of_zone):
        if not alive_start.get(mav.agent_id,False):return 0.,{name:0. for name in self.component_names}
        enemies=[a for a in agents if a.side=="blue" and a.alive]
        attack_uavs=[a for a in agents if a.side=="red" and a.role=="uav" and a.alive]
        if enemies:
            distance=min(float(np.linalg.norm(enemy.position-mav.position)) for enemy in enemies)
            if distance<D_DANGER_M:r_distance=-(1-distance/D_DANGER_M)
            elif distance<D_SAFE_M:r_distance=-.5*(1-(distance-D_DANGER_M)/(D_SAFE_M-D_DANGER_M))
            else:r_distance=.2
            aspects=[]
            for enemy in enemies:
                threat_angle=_angle(enemy.velocity,mav.position-enemy.position)
                aspects.append(-(1-threat_angle/(np.pi/4))) if threat_angle<np.pi/4 else aspects.append(0.)
            r_aspect=float(min(aspects))
        else:r_distance=.2;r_aspect=0.
        r_threat=-1. if any(m.alive and m.target_id==mav.agent_id for m in missiles) else 0.
        r_safety=.5*r_distance+.3*r_threat+.2*r_aspect
        if attack_uavs:
            centre=np.mean([a.position for a in attack_uavs],axis=0);d_support=float(np.linalg.norm(mav.position-centre))
            if d_support<D_OPT_M:r_position=d_support/D_OPT_M-1
            elif d_support<D_SUPPORT_MAX_M:r_position=1-(d_support-D_OPT_M)/(D_SUPPORT_MAX_M-D_OPT_M)
            else:r_position=-.5
        else:r_position=-.5
        awareness=[.3*(1-pair.ata_rad/(np.pi/2)) for enemy in enemies
                   if (pair:=assess_pair(mav.position,mav.velocity,enemy.position,enemy.velocity)).ata_rad<np.pi/2]
        r_awareness=float(max(awareness,default=0.));r_support=.6*r_position+.4*r_awareness
        kills=sum(1 for event in events if event.get("reason")=="hit" and
                  (shooter:=next((a for a in agents if a.agent_id==event.get("shooter_id")),None)) is not None and
                  shooter.side=="red" and shooter.role=="uav" and
                  next((a for a in agents if a.agent_id==event.get("target_id")),shooter).side=="blue")
        r_event=min(100.*kills,200.)
        if not mav.alive and mav.death_reason!="boundary":r_event-=200.
        if mav.agent_id in out_of_zone:r_event-=100.
        total=10*r_safety+10*r_support+r_event
        components={"r_safety_distance":float(r_distance),"r_safety_threat":float(r_threat),"r_safety_aspect":float(r_aspect),
          "r_safety":float(r_safety),"r_support_position":float(r_position),"r_support_awareness":float(r_awareness),
          "r_support":float(r_support),"r_event":float(r_event),"total":float(total)}
        return float(total),components
