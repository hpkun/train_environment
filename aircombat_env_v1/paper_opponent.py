"""Direct adaptation of TAM's observation-consistent greedy opponent."""
from __future__ import annotations
import numpy as np
from .paper_action_semantics import map_action_indices
from .paper_reward import paper_height_reward,uav_speed_reward,uav_angle_reward,uav_distance_reward
from .paper_situation import assess_pair

MANOEUVRES={"level":(24,20,20,20),"accelerate":(34,20,20,20),"decelerate":(14,20,20,20),
 "left_turn":(27,10,20,14),"right_turn":(27,29,20,25),"climb":(27,20,14,20),"dive":(24,20,25,20)}

class GreedyPaperOpponent:
    def act(self,agent,target,incoming):
        if not agent.alive or target is None or not target.alive:return np.asarray(MANOEUVRES["level"],np.int64),{"manoeuvre":"level"}
        candidates={}; dt=.2; target_pos=target.position+target.velocity*dt
        for name,indices in MANOEUVRES.items():
            pos,vel,pitch,heading,speed=self._predict(agent,indices); pair=assess_pair(pos,vel,target_pos,target.velocity)
            dodge=self._dodge(pos,incoming,dt); total=10*paper_height_reward(pos[2])+10*uav_speed_reward(speed,target.speed)+15*uav_angle_reward(pair.ata_rad,pair.aa_rad)+10*uav_distance_reward(pair.distance_m)+30*dodge
            candidates[name]={"total_dense_reward":float(total),"r_dodge":float(dodge),"action_indices":list(indices)}
        name=max(MANOEUVRES,key=lambda n:(candidates[n]["total_dense_reward"],-list(MANOEUVRES).index(n)))
        return np.asarray(MANOEUVRES[name],np.int64),{"manoeuvre":name,"candidates":candidates}
    def _predict(self,a,indices):
        throttle,aileron,elevator,rudder=map_action_indices(indices); dt=.2
        speed=max(1,a.speed+(55*(throttle-.4)-10)*dt); roll=float(np.clip(a.roll+aileron*np.deg2rad(70)*dt,-np.deg2rad(75),np.deg2rad(75)))
        pitch=float(np.clip(a.pitch-elevator*np.deg2rad(25)*dt,-np.deg2rad(40),np.deg2rad(40)))
        heading=float((a.heading+(np.sin(roll)*np.deg2rad(100)+rudder*np.deg2rad(40))*dt+np.pi)%(2*np.pi)-np.pi)
        cp=np.cos(pitch); vel=np.array([cp*np.cos(heading),cp*np.sin(heading),np.sin(pitch)])*speed
        return a.position+vel*dt,vel,pitch,heading,float(speed)
    def _dodge(self,pos,incoming,dt):
        active=[m for m in incoming if m.alive]
        if not active:return 0.
        m=min(active,key=lambda x:np.linalg.norm(x.position+x.velocity*dt-pos)/max(x.speed_mps,1)); los=pos-(m.position+m.velocity*dt)
        lam=float(np.arccos(np.clip(np.dot(los,m.velocity)/max(np.linalg.norm(los)*np.linalg.norm(m.velocity),1e-8),-1,1)))
        return -float(np.cos(lam))+(m.decision_start_speed_mps-m.speed_mps)/1000.
