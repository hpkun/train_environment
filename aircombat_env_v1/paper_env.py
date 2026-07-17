"""Formal paper-aligned TAM direct-flight-control 1v1/2v2 environment."""
from __future__ import annotations
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .aircraft import AircraftSimulator
from .geometry import LLA2NEU
from .scenario import ORIGIN
from .paper_action_semantics import ACTION_LEVELS,INACTIVE_ACTION_PLACEHOLDER,map_action_indices
from .paper_observation import PaperObservation
from .paper_opponent import GreedyPaperOpponent
from .paper_reward import PaperReward
from .paper_situation import assess_pair
from .paper_weapon import PaperWeaponManager

PUBLISHED={"simulation_frequency_hz":60,"decision_frequency_hz":5,"physics_frames_per_action":12,
 "episode_limit_steps":1000,"maximum_speed_mps":400.,"minimum_safe_altitude_m":750.,
 "maximum_aircraft_overload_g":9.,"maximum_attack_range_m":14000.,"launch_interval_s":25.,
 "maximum_overload_g":30.,"navigation_gain_y":3.,"navigation_gain_z":3.,
 "missile_mass_kg":84.,"missile_length_m":2.87,"missile_diameter_m":.127}
INFERRED={"missile_initial_speed_mps":500.,"powered_duration_s":3.,"powered_acceleration_mps2":110.,
 "hit_radius_m":60.,"effective_quadratic_drag_per_m":.00012,"missile_speed_reward_norm_mps":1000.,"reward_global_scale":1.}

SCENARIOS={
 "paper_nominal_1v1":[("red_0","red",120.,60.,0.),("blue_0","blue",120.,60.2,180.)],
 "paper_nominal_2v2":[("red_0","red",120.,60.,0.),("red_1","red",120.02,60.,0.),
                         ("blue_0","blue",120.,60.2,180.),("blue_1","blue",120.02,60.2,180.)]}

@dataclass
class PaperAircraft:
    agent_id:str; side:str; longitude:float; latitude:float; initial_heading_deg:float
    def __post_init__(self):
        self.simulator=AircraftSimulator(60); self.missile_left=2; self.alive=True; self.death_reason=None
        self.out_of_boundary=False; self.current_target=None; self.incoming_missiles=[]; self._state=None
        self.reset()
    def reset(self):
        self.missile_left=2; self.alive=True; self.death_reason=None; self.out_of_boundary=False; self.current_target=None; self.incoming_missiles=[]
        # Formal direct-FCS initialization: engine running, gear/flaps retracted by
        # the bundled model, minimum published throttle command, no PID trim.
        self._state=self.simulator.reset(6000.,250.,self.initial_heading_deg,0.,0.,self.latitude,self.longitude,0.,.4)
        self._update(); return self
    def apply_direct_fcs_command(self,command):
        throttle,aileron,elevator,rudder=np.asarray(command,float).reshape(4)
        self.simulator.set_controls(aileron,elevator,rudder,throttle)
    def step_physics_once(self):
        if not self.alive:return
        try:self._state=self.simulator.run()
        except (RuntimeError,ValueError,FloatingPointError):self.kill("numerical_invalid");return
        self._update()
        if not np.isfinite(np.r_[self.position,self.velocity,self.roll,self.pitch,self.heading,self.speed,self.load_factor_g]).all():self.kill("numerical_invalid")
        elif self.position[2]<=0:self.kill("crash")
    def _update(self):
        s=self._state; self.position=LLA2NEU(s["longitude"],s["latitude"],s["altitude"],*ORIGIN)
        self.velocity=np.array([s["v_north"],s["v_east"],-s["v_down"]],float)
        self.roll=float(s["roll"]);self.pitch=float(s["pitch"]);self.heading=float(s["heading"])
        self.speed=float(s["true_airspeed"]);self.load_factor_g=float(s["load_factor"])
    def kill(self,reason):self.alive=False;self.death_reason=reason;self.out_of_boundary=reason=="boundary"
    def close(self):self.simulator=None

class TAMPaperCombatEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,scenario_mode="paper_nominal_2v2",controlled_side="red",max_steps=1000,
                 weapon_enabled_agent_ids=None):
        if scenario_mode not in SCENARIOS:raise ValueError("scenario_mode must be paper_nominal_1v1 or paper_nominal_2v2")
        if controlled_side not in ("red","all"):raise ValueError("controlled_side must be red or all")
        self.scenario_mode=scenario_mode;self.controlled_side=controlled_side;self.max_steps=int(max_steps)
        self.weapon_enabled_agent_ids=(None if weapon_enabled_agent_ids is None
                                       else set(weapon_enabled_agent_ids))
        self.agent_specs=SCENARIOS[scenario_mode];self.max_red=sum(x[1]=="red" for x in self.agent_specs);self.max_blue=sum(x[1]=="blue" for x in self.agent_specs)
        self.controlled_ids=[x[0] for x in self.agent_specs if controlled_side=="all" or x[1]=="red"]
        self.action_space=spaces.Dict({aid:spaces.MultiDiscrete([40]*4) for aid in self.controlled_ids})
        self.paper_observation=PaperObservation(self.max_red,self.max_blue); self.observation_space=self._observation_space()
        self.weapon=PaperWeaponManager(PUBLISHED,INFERRED);self.reward_model=PaperReward();self.opponent=GreedyPaperOpponent();self.agents=[]
    def _observation_space(self):
        result={}
        side_by_id={x[0]:x[1] for x in self.agent_specs}
        for aid in self.controlled_ids:
            na,ne=self.paper_observation.shapes_for(side_by_id[aid]);result[aid]=spaces.Dict({
             "ego_state":spaces.Box(-np.inf,np.inf,(7,),dtype=np.float32),"ally_states":spaces.Box(-np.inf,np.inf,(na,5),dtype=np.float32),
             "enemy_states":spaces.Box(-np.inf,np.inf,(ne,5),dtype=np.float32),"incoming_missile_states":spaces.Box(-np.inf,np.inf,(8,5),dtype=np.float32),
             "ally_mask":spaces.Box(0,1,(na,),dtype=np.float32),"enemy_mask":spaces.Box(0,1,(ne,),dtype=np.float32),
             "incoming_missile_mask":spaces.Box(0,1,(8,),dtype=np.float32)})
        return spaces.Dict(result)
    @staticmethod
    def map_action(indices):return map_action_indices(indices)
    def flatten_observation(self,item):return self.paper_observation.flatten(item)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);del options
        if not self.agents:self.agents=[PaperAircraft(*spec) for spec in self.agent_specs]
        else:
            for a in self.agents:a.reset()
        self.by_id={a.agent_id:a for a in self.agents};self.weapon.reset();self.step_count=0;self.simulation_time_s=0.
        self.invalid_episode=False;self.red_missile_kills=self.blue_missile_kills=0;self.red_crashes=self.blue_crashes=0
        self.flight_envelope_violation=False;self.minimum_altitude_m=min(a.position[2] for a in self.agents)
        self.maximum_speed_mps=max(a.speed for a in self.agents);self.maximum_load_factor_g=max(abs(a.load_factor_g) for a in self.agents)
        self.current_targets={};self.target_change_events=[];self.target_changes=0;self.second_launches=0
        self._launch_counts={a.agent_id:0 for a in self.agents};self.simultaneous_kills=0
        self._update_targets(count_changes=False);obs=self.paper_observation.build(self.agents,self.weapon.missiles)
        return {aid:obs[aid] for aid in self.controlled_ids},self._info(None,None)
    def _update_targets(self,count_changes=True):
        for a in self.agents:
            previous=self.current_targets.get(a.agent_id)
            enemies=sorted([e for e in self.agents if e.side!=a.side and e.alive],key=lambda e:e.agent_id)
            target=min(enemies,key=lambda e:(float(np.linalg.norm(e.position-a.position)),e.agent_id)) if a.alive and enemies else None
            a.current_target=target.agent_id if target else None;self.current_targets[a.agent_id]=a.current_target
            if count_changes and previous is not None and previous!=a.current_target:
                self.target_changes+=1;self.target_change_events.append({"agent_id":a.agent_id,"previous_target":previous,
                    "current_target":a.current_target,"decision_step":self.step_count,"simulation_time_s":self.simulation_time_s})
            a.incoming_missiles=[m for m in self.weapon.missiles if m.alive and m.target_id==a.agent_id]
    def build_rule_actions(self,agent_ids=None):
        selected=set(agent_ids or self.controlled_ids);result={}
        for aid in selected:
            a=self.by_id[aid];target=self.by_id.get(a.current_target);result[aid]=self.opponent.act(a,target,a.incoming_missiles)[0]
        return result
    def step(self,actions):
        self._update_targets();self.weapon.begin_decision_step();alive_start={a.agent_id:a.alive for a in self.agents};events=[]
        action_map={str(k):np.asarray(v,dtype=np.int64) for k,v in actions.items()}
        for aid in self.controlled_ids:
            if self.by_id[aid].alive and aid not in action_map:raise KeyError(f"missing action for {aid}")
        if self.controlled_side=="red":
            for a in self.agents:
                if a.side=="blue" and a.alive:action_map[a.agent_id]=self.opponent.act(a,self.by_id.get(a.current_target),a.incoming_missiles)[0]
        for a in self.agents:
            if self.weapon_enabled_agent_ids is None or a.agent_id in self.weapon_enabled_agent_ids:
                launch=self.weapon.try_launch(a,self.by_id.get(a.current_target),self.simulation_time_s)
                if launch:
                    self._launch_counts[a.agent_id]+=1
                    if self._launch_counts[a.agent_id]==2:self.second_launches+=1
                    launch.update({"simulation_time_s":self.simulation_time_s,"decision_step":self.step_count,
                                   "missiles_left":a.missile_left,"launch_number":self._launch_counts[a.agent_id]})
                    events.append(launch)
        commands={a.agent_id:map_action_indices(action_map.get(a.agent_id,INACTIVE_ACTION_PLACEHOLDER)) for a in self.agents}
        out_step=set()
        for frame_index in range(12):
            for a in self.agents:
                if a.alive:a.apply_direct_fcs_command(commands[a.agent_id])
            for a in self.agents:
                if a.alive:a.step_physics_once()
            self._check_frame(out_step);frame_events=self.weapon.step_physics_once(self.by_id,1/60)
            for event in frame_events:event.update({"physics_frame_index":frame_index,"simulation_time_s":self.simulation_time_s+1/60})
            hit_sides={self.by_id[e["shooter_id"]].side for e in frame_events if e.get("hit")}
            if hit_sides=={"red","blue"}:self.simultaneous_kills+=1
            events.extend(frame_events)
            self.simulation_time_s+=1/60
        for e in events:
            if e.get("reason")=="hit":
                side=self.by_id[e["shooter_id"]].side
                if side=="red":self.red_missile_kills+=1
                else:self.blue_missile_kills+=1
        self.step_count+=1;pairs={}
        for a in self.agents:
            t=self.by_id.get(a.current_target)
            if t is not None:pairs[a.agent_id]=assess_pair(a.position,a.velocity,t.position,t.velocity)
        rewards,components=self.reward_model.compute(self.agents,self.current_targets,pairs,self.weapon.missiles,events,alive_start,out_step)
        red_alive=sum(a.alive for a in self.agents if a.side=="red");blue_alive=sum(a.alive for a in self.agents if a.side=="blue")
        terminated=red_alive==0 or blue_alive==0;truncated=not terminated and self.step_count>=self.max_steps
        winner=("draw" if red_alive==blue_alive==0 else "red" if blue_alive==0 else "blue" if red_alive==0 else "draw" if truncated else None)
        reason=("mutual_elimination" if red_alive==blue_alive==0 else "blue_eliminated" if blue_alive==0 else "red_eliminated" if red_alive==0 else "episode_limit" if truncated else None)
        self._update_targets();obs=self.paper_observation.build(self.agents,self.weapon.missiles);info=self._info(winner,reason);info["reward_components"]=components;info["events"]=events
        return {aid:obs[aid] for aid in self.controlled_ids},{aid:rewards[aid] for aid in self.controlled_ids},terminated,truncated,info
    def _check_frame(self,out_step):
        for a in self.agents:
            self.minimum_altitude_m=min(self.minimum_altitude_m,float(a.position[2]));self.maximum_speed_mps=max(self.maximum_speed_mps,a.speed)
            self.maximum_load_factor_g=max(self.maximum_load_factor_g,abs(a.load_factor_g))
            self.flight_envelope_violation|=a.position[2]<750 or a.speed>400 or abs(a.load_factor_g)>9
            if a.death_reason=="numerical_invalid":self.invalid_episode=True
            if a.alive and np.linalg.norm(a.position[:2])>28000:a.kill("boundary");out_step.add(a.agent_id)
        self.red_crashes=sum(a.death_reason=="crash" for a in self.agents if a.side=="red")
        self.blue_crashes=sum(a.death_reason=="crash" for a in self.agents if a.side=="blue")
    def _info(self,winner,reason):
        red=[a for a in self.agents if a.side=="red"];blue=[a for a in self.agents if a.side=="blue"]
        return {"scenario_mode":self.scenario_mode,"simulation_time_s":self.simulation_time_s,"decision_step":self.step_count,
         "alive_red":sum(a.alive for a in red),"alive_blue":sum(a.alive for a in blue),"red_missiles_left":sum(a.missile_left for a in red),
         "blue_missiles_left":sum(a.missile_left for a in blue),"missiles_fired":self.weapon.total_fired,"missile_hits":self.weapon.total_hits,
         "red_missile_kills":self.red_missile_kills,"blue_missile_kills":self.blue_missile_kills,"red_crashes":self.red_crashes,"blue_crashes":self.blue_crashes,
         "active_missiles":sum(m.alive for m in self.weapon.missiles),"current_targets":dict(self.current_targets),
         "flight_envelope_violation":self.flight_envelope_violation,"minimum_altitude_m":self.minimum_altitude_m,
         "maximum_speed_mps":self.maximum_speed_mps,"maximum_load_factor_g":self.maximum_load_factor_g,"invalid_episode":self.invalid_episode,
         "red_numerical_invalid":sum(a.death_reason=="numerical_invalid" for a in red),
         "blue_numerical_invalid":sum(a.death_reason=="numerical_invalid" for a in blue),
         "boundary_deaths":sum(a.death_reason=="boundary" for a in self.agents),
         "simultaneous_kills":self.simultaneous_kills,"target_changes":self.target_changes,
         "target_change_events":list(self.target_change_events),"second_launches":self.second_launches,
         "weapon_enabled_agent_ids":(None if self.weapon_enabled_agent_ids is None else sorted(self.weapon_enabled_agent_ids)),
         "termination_reason":reason,"winner":winner,"death_reason":{a.agent_id:a.death_reason for a in self.agents}}
    def close(self):
        for a in self.agents:a.close()
