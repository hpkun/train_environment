"""Selectable MAV reward contracts for the simplified heterogeneous scenario."""
from __future__ import annotations
import numpy as np
from .paper_situation import assess_pair

# Legacy engineering constants.  Kept unchanged for old experiment replay.
D_DANGER_M=5000.
D_SAFE_M=10000.
D_OPT_M=4000.
D_SUPPORT_MAX_M=12000.

# PaperTable1 structure with paper_unspecified_engineering project mappings.
MAV_DANGER_DISTANCE_M=14000.0
MAV_SAFE_DISTANCE_M=28000.0
MAV_SUPPORT_OPTIMAL_DISTANCE_M=14000.0
MAV_SUPPORT_MAX_DISTANCE_M=28000.0
MAV_DEATH_COST=200.0
MAV_TEAM_KILL_CREDIT=100.0
MAV_TEAM_CREDIT_CAP=200.0
HETERO_REWARD_MODES=("legacy_v1","paper_table1_v2")
HETERO_REWARD_CONTRACT_VERSION="heterogeneous_reward_v2"

def _angle(a,b):
    return float(np.arccos(np.clip(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-8),-1.,1.)))

class LegacySimpleMAVReward:
    """The original learnability-oriented reward, preserved numerically."""
    component_names=("r_safety_distance","r_safety_threat","r_safety_aspect","r_safety",
                     "r_support_position","r_support_awareness","r_support","r_event","total")

    def reset(self):
        pass

    def compute(self,mav,agents,missiles,events,alive_start,out_of_zone,perception_result=None):
        del perception_result
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

class PaperTable1MAVReward:
    """MAV Safety + Support + Event reward defined by Table 1 structure."""
    component_names=("r_safety_distance","r_safety_threat","r_safety_aspect","r_safety",
      "battlefield_center_distance_m","r_support_position","r_support_awareness","r_support",
      "r_event_death","r_event_team_contribution","r_event","team_credit_awarded_so_far",
      "mav_detected_enemy_count_log","relay_only_track_count_log","total_dense","total")

    def __init__(self):self.reset()

    def reset(self):
        self.team_credit_awarded_so_far=0.0;self.mav_death_penalty_awarded=False

    @staticmethod
    def distance_reward(distance_m):
        d=float(distance_m)
        if d<MAV_DANGER_DISTANCE_M:return float(-(1-d/MAV_DANGER_DISTANCE_M))
        if d<MAV_SAFE_DISTANCE_M:return float(-.5*(1-(d-MAV_DANGER_DISTANCE_M)/(MAV_SAFE_DISTANCE_M-MAV_DANGER_DISTANCE_M)))
        return .2

    @staticmethod
    def position_reward(distance_m):
        d=float(distance_m)
        if d<MAV_SUPPORT_OPTIMAL_DISTANCE_M:return float(d/MAV_SUPPORT_OPTIMAL_DISTANCE_M-1.)
        if d<MAV_SUPPORT_MAX_DISTANCE_M:return float(1-(d-MAV_SUPPORT_OPTIMAL_DISTANCE_M)/(MAV_SUPPORT_MAX_DISTANCE_M-MAV_SUPPORT_OPTIMAL_DISTANCE_M))
        return -.5

    @staticmethod
    def battlefield_center(agents):
        entities=[a for a in agents if a.alive and ((a.side=="red" and a.role=="uav") or a.side=="blue")]
        return None if not entities else np.mean([a.position for a in entities],axis=0)

    def compute(self,mav,agents,missiles,events,alive_start,out_of_zone,perception_result=None):
        del out_of_zone
        perception_result=perception_result or {}
        enemies=[a for a in agents if a.side=="blue" and a.alive]
        by_id={a.agent_id:a for a in agents}
        nearest=min((float(np.linalg.norm(enemy.position-mav.position)) for enemy in enemies),default=MAV_SAFE_DISTANCE_M)
        r_distance=self.distance_reward(nearest)
        r_threat=-1. if any(m.alive and m.target_id==mav.agent_id for m in missiles) else 0.
        r_aspect=0.0
        for enemy in enemies:
            ta=_angle(enemy.velocity,mav.position-enemy.position)
            if ta<np.pi/4:r_aspect-=1-ta/(np.pi/4)
        r_safety=.5*r_distance+.3*r_threat+.2*r_aspect
        center=self.battlefield_center(agents)
        center_distance=0.0 if center is None else float(np.linalg.norm(mav.position-center))
        r_position=0.0 if center is None else self.position_reward(center_distance)
        detected=[by_id[aid] for aid in perception_result.get("mav_detected_enemy_ids",[]) if aid in by_id and by_id[aid].alive and by_id[aid].side=="blue"]
        r_awareness=0.0
        for enemy in detected:
            ao=assess_pair(mav.position,mav.velocity,enemy.position,enemy.velocity).ata_rad
            if ao<np.pi/2:r_awareness+=.3*(1-ao/(np.pi/2))
        r_support=.6*r_position+.4*r_awareness
        r_death=0.0
        if alive_start.get(mav.agent_id,False) and not mav.alive and not self.mav_death_penalty_awarded:
            r_death=-MAV_DEATH_COST;self.mav_death_penalty_awarded=True
        valid_kills=0
        for event in events:
            shooter=by_id.get(event.get("shooter_id"));target=by_id.get(event.get("target_id"))
            if event.get("reason")=="hit" and shooter is not None and target is not None and shooter.side=="red" and shooter.role=="uav" and target.side=="blue":valid_kills+=1
        available=max(MAV_TEAM_CREDIT_CAP-self.team_credit_awarded_so_far,0.0)
        r_team=float(min(MAV_TEAM_KILL_CREDIT*valid_kills,available));self.team_credit_awarded_so_far+=r_team
        r_event=r_death+r_team;total_dense=r_safety+r_support;total=total_dense+r_event
        values={"r_safety_distance":r_distance,"r_safety_threat":r_threat,"r_safety_aspect":r_aspect,"r_safety":r_safety,
          "battlefield_center_distance_m":center_distance,"r_support_position":r_position,"r_support_awareness":r_awareness,
          "r_support":r_support,"r_event_death":r_death,"r_event_team_contribution":r_team,"r_event":r_event,
          "team_credit_awarded_so_far":self.team_credit_awarded_so_far,
          "mav_detected_enemy_count_log":len(detected),"relay_only_track_count_log":int(perception_result.get("relay_only_track_count",0)),
          "total_dense":total_dense,"total":total}
        components={key:(int(value) if key.endswith("_count_log") else float(value)) for key,value in values.items()}
        if not np.isfinite([float(value) for value in components.values()]).all():raise FloatingPointError("non-finite MAV reward")
        self.last_battlefield_center=None if center is None else [float(x) for x in center]
        self.last_battlefield_center_distance_m=float(center_distance)
        return float(total),components

SimpleMAVReward=LegacySimpleMAVReward

def build_mav_reward(mode):
    if mode=="legacy_v1":return LegacySimpleMAVReward()
    if mode=="paper_table1_v2":return PaperTable1MAVReward()
    raise ValueError(f"hetero_reward_mode must be one of {HETERO_REWARD_MODES}")

def mav_reward_config():
    return {"mav_danger_distance_m":MAV_DANGER_DISTANCE_M,"mav_safe_distance_m":MAV_SAFE_DISTANCE_M,
      "mav_support_optimal_distance_m":MAV_SUPPORT_OPTIMAL_DISTANCE_M,"mav_support_max_distance_m":MAV_SUPPORT_MAX_DISTANCE_M,
      "mav_death_cost":MAV_DEATH_COST,"mav_team_kill_credit":MAV_TEAM_KILL_CREDIT,"mav_team_credit_cap":MAV_TEAM_CREDIT_CAP}
