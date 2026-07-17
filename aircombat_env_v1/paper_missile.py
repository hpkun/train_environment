"""Direct adaptation of TAM's formal paper point-mass missile."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

G=9.80665

def analytic_los_rates(relative_position,relative_velocity):
    x,y,z=np.asarray(relative_position,dtype=np.float64); xd,yd,zd=np.asarray(relative_velocity,dtype=np.float64)
    rho2=x*x+y*y; rho=float(np.sqrt(max(rho2,1e-12))); range2=max(rho2+z*z,1e-12)
    yaw=float(np.arctan2(y,x)); pitch=float(np.arctan2(z,rho))
    yaw_rate=float((x*yd-y*xd)/max(rho2,1e-12)); rho_rate=float((x*xd+y*yd)/rho)
    pitch_rate=float((rho*zd-z*rho_rate)/range2)
    return yaw,pitch,yaw_rate,pitch_rate

def _wrap_pi(v): return float((v+np.pi)%(2*np.pi)-np.pi)

@dataclass
class PaperMissile:
    missile_id:str; shooter_id:str; target_id:str; position:np.ndarray; velocity:np.ndarray; config:dict
    def __post_init__(self):
        self.position=np.asarray(self.position,dtype=np.float64).copy(); initial=np.asarray(self.velocity,dtype=np.float64)
        direction=initial/max(float(np.linalg.norm(initial)),1e-8)
        self.speed_mps=float(self.config["missile_initial_speed_mps"])
        self.yaw_rad=float(np.arctan2(direction[1],direction[0])); self.pitch_rad=float(np.arctan2(direction[2],np.linalg.norm(direction[:2])))
        self.velocity=self._direction()*self.speed_mps; self.previous_speed_mps=self.speed_mps
        self.decision_start_speed_mps=self.speed_mps; self.flight_time_s=0.; self.los_yaw_rad=self.los_pitch_rad=0.
        self.los_yaw_rate_rps=self.los_pitch_rate_rps=0.; self.longitudinal_overload_g=0.
        self.lateral_overload_y_g=self.lateral_overload_z_g=self.commanded_overload_g=0.
        self.distance_m=float("inf"); self.status="flying"; self.termination_reason=None
        self.navigation_gain_y=float(self.config["navigation_gain_y"]); self.navigation_gain_z=float(self.config["navigation_gain_z"])
        self.timeout_s=2.*float(self.config["maximum_attack_range_m"])/float(self.config["missile_initial_speed_mps"])
    @property
    def alive(self): return self.status=="flying"
    def mark_decision_start(self):
        if self.alive: self.decision_start_speed_mps=self.speed_mps
    def _direction(self):
        cp=np.cos(self.pitch_rad); return np.array([cp*np.cos(self.yaw_rad),cp*np.sin(self.yaw_rad),np.sin(self.pitch_rad)])
    def step(self,target_position,target_velocity,dt):
        if not self.alive:return self.termination_reason
        relative=np.asarray(target_position)-self.position; relative_velocity=np.asarray(target_velocity)-self.velocity
        self.distance_m=float(np.linalg.norm(relative))
        if self.distance_m<=float(self.config["hit_radius_m"]):
            self.status,self.termination_reason="hit","hit"; return "hit"
        self.los_yaw_rad,self.los_pitch_rad,self.los_yaw_rate_rps,self.los_pitch_rate_rps=analytic_los_rates(relative,relative_velocity)
        cp=float(np.cos(self.pitch_rad)); self.lateral_overload_y_g=self.navigation_gain_y*self.speed_mps/G*cp*self.los_yaw_rate_rps
        self.lateral_overload_z_g=self.navigation_gain_z*self.speed_mps/G*self.los_pitch_rate_rps+cp
        norm=float(np.hypot(self.lateral_overload_y_g,self.lateral_overload_z_g)); limit=float(self.config["maximum_overload_g"])
        if norm>limit:
            scale=limit/norm; self.lateral_overload_y_g*=scale; self.lateral_overload_z_g*=scale; norm=limit
        self.commanded_overload_g=norm; safe_cp=cp if abs(cp)>=1e-4 else np.copysign(1e-4,cp or 1.)
        self.yaw_rad=_wrap_pi(self.yaw_rad+G*self.lateral_overload_y_g/(max(self.speed_mps,1.)*safe_cp)*dt)
        self.pitch_rad=float(np.clip(self.pitch_rad+G/max(self.speed_mps,1.)*(self.lateral_overload_z_g-cp)*dt,-np.pi/2+1e-4,np.pi/2-1e-4))
        drag=float(self.config["effective_quadratic_drag_per_m"])*self.speed_mps**2
        thrust=float(self.config["powered_acceleration_mps2"]) if self.flight_time_s<float(self.config["powered_duration_s"]) else 0.
        self.longitudinal_overload_g=(thrust-drag)/G; self.previous_speed_mps=self.speed_mps
        self.speed_mps=float(max(self.speed_mps+G*(self.longitudinal_overload_g-np.sin(self.pitch_rad))*dt,1.))
        self.velocity=self._direction()*self.speed_mps; self.position+=self.velocity*dt; self.flight_time_s+=dt
        if not np.isfinite(np.r_[self.position,self.velocity,self.speed_mps]).all(): self.status,self.termination_reason="miss","nonfinite"
        elif self.flight_time_s>=self.timeout_s: self.status,self.termination_reason="miss","timeout"
        return self.termination_reason
    def telemetry(self):
        return {"missile_id":self.missile_id,"target_id":self.target_id,"n_y_g":self.lateral_overload_y_g,
            "n_z_g":self.lateral_overload_z_g,"total_overload_g":self.commanded_overload_g,"speed_mps":self.speed_mps,
            "decision_start_speed_mps":self.decision_start_speed_mps,"distance_m":self.distance_m,
            "derived_timeout_s":self.timeout_s,"termination_reason":self.termination_reason}
