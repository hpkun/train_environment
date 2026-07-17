"""Direct adaptation of TAM automatic paper weapon management."""
from __future__ import annotations
import numpy as np
from .paper_missile import PaperMissile

class PaperWeaponManager:
    def __init__(self,published,inferred): self.published=published; self.inferred=inferred; self.reset()
    def reset(self):
        self.missiles=[]; self.last_launch_time_s={}; self.counter=0; self.total_fired=0; self.total_hits=0; self._reported=set()
    def begin_decision_step(self):
        for missile in self.missiles: missile.mark_decision_start()
    def try_launch(self,shooter,target,simulation_time_s):
        if not shooter.alive or shooter.missile_left<=0 or target is None or not target.alive:return None
        distance=float(np.linalg.norm(target.position-shooter.position))
        if distance<=1e-6 or distance>float(self.published["maximum_attack_range_m"]):return None
        if simulation_time_s-self.last_launch_time_s.get(shooter.agent_id,-1e30)+1e-9<float(self.published["launch_interval_s"]):return None
        self.counter+=1; mid=f"{shooter.agent_id}_M{self.counter:04d}"; config={**self.published,**self.inferred}
        missile=PaperMissile(mid,shooter.agent_id,target.agent_id,shooter.position.copy(),target.position-shooter.position,config)
        self.missiles.append(missile); shooter.missile_left-=1; self.last_launch_time_s[shooter.agent_id]=simulation_time_s; self.total_fired+=1
        return {"event_type":"missile_launch","missile_id":mid,"shooter_id":shooter.agent_id,"target_id":target.agent_id,"distance_m":distance}
    def step_physics_once(self,by_id,dt):
        events=[]
        for missile in self.missiles:
            if not missile.alive:continue
            target=by_id.get(missile.target_id)
            if target is None or not target.alive: missile.status,missile.termination_reason="miss","target_dead"
            else: missile.step(target.position,target.velocity,dt)
            if missile.termination_reason and missile.missile_id not in self._reported:
                hit=missile.termination_reason=="hit" and target is not None and target.alive
                if hit: target.kill("shotdown"); self.total_hits+=1
                self._reported.add(missile.missile_id)
                events.append({"event_type":"missile_termination","missile_id":missile.missile_id,"shooter_id":missile.shooter_id,
                    "target_id":missile.target_id,"reason":missile.termination_reason,"hit":hit})
        return events
