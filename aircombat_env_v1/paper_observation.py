"""Direct adaptation of TAM's fixed-slot 7D/5D paper observation."""
from __future__ import annotations
import numpy as np
from .paper_situation import assess_pair

OBS_KEYS=("ego_state","ally_states","enemy_states","incoming_missile_states","ally_mask","enemy_mask","incoming_missile_mask")

class PaperObservation:
    def __init__(self,max_red,max_blue):
        self.max_red=max_red; self.max_blue=max_blue; self.max_incoming=8
        self.position_norm_m=28000.; self.altitude_norm_m=6000.
    def shapes_for(self,side): return ((self.max_red if side=="red" else self.max_blue)-1,self.max_blue if side=="red" else self.max_red)
    def build(self,agents,missiles,visible_enemy_ids_by_agent=None):
        sides={s:sorted([a for a in agents if a.side==s],key=lambda a:a.agent_id) for s in ("red","blue")}; result={}
        for ego in agents:
            allies=[a for a in sides[ego.side] if a.agent_id!=ego.agent_id]; enemies=sides["blue" if ego.side=="red" else "red"]
            na,ne=self.shapes_for(ego.side); ally=np.zeros((na,5),np.float32); enemy=np.zeros((ne,5),np.float32)
            am=np.zeros(na,np.float32); em=np.zeros(ne,np.float32)
            if ego.alive:
                for i,a in enumerate(allies[:na]):
                    if a.alive: ally[i]=self._relative(ego,a.position,a.velocity); am[i]=1
                visible=None if visible_enemy_ids_by_agent is None else set(visible_enemy_ids_by_agent.get(ego.agent_id,()))
                for i,a in enumerate(enemies[:ne]):
                    if a.alive and (visible is None or a.agent_id in visible):
                        enemy[i]=self._relative(ego,a.position,a.velocity); em[i]=1
            incoming=sorted([m for m in missiles if m.alive and m.target_id==ego.agent_id],key=lambda m:(np.linalg.norm(m.position-ego.position)/max(m.speed_mps,1),m.missile_id))
            ms=np.zeros((self.max_incoming,5),np.float32); mm=np.zeros(self.max_incoming,np.float32)
            if ego.alive:
                for i,m in enumerate(incoming[:self.max_incoming]): ms[i]=self._relative(ego,m.position,m.velocity); mm[i]=1
            item={"ego_state":self._ego(ego),"ally_states":ally,"enemy_states":enemy,"incoming_missile_states":ms,
                  "ally_mask":am,"enemy_mask":em,"incoming_missile_mask":mm}
            if not all(v.dtype==np.float32 and np.isfinite(v).all() for v in item.values()): raise FloatingPointError(f"non-finite observation {ego.agent_id}")
            result[ego.agent_id]=item
        return result
    def flatten(self,item): return np.concatenate([item[k].reshape(-1) for k in OBS_KEYS]).astype(np.float32)
    def _ego(self,a):
        if not a.alive:return np.zeros(7,np.float32)
        return np.array([a.position[0]/28000,a.position[1]/28000,a.position[2]/6000,a.speed/400,a.pitch/np.pi,a.heading/np.pi,a.roll/np.pi],np.float32)
    def _relative(self,ego,pos,vel):
        p=assess_pair(ego.position,ego.velocity,pos,vel)
        return np.array([p.relative_speed_mps/400,p.relative_altitude_m/6000,p.distance_m/28000,p.ata_rad/np.pi,p.aa_rad/np.pi],np.float32)
