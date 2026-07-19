"""Simplified learnable JSBSim 1v1/2v2 environment with fixed PID control."""
from __future__ import annotations
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .aircraft import AircraftSimulator
from .combat import action_to_targets
from .config import DEFAULT_CONFIG,load_config
from .geometry import LLA2NEU
from .hetero_perception import HeterogeneousPerceptionSystem,PERCEPTION_MODES
from .opponent import paper_greedy_action
from .paper_observation import PaperObservation
from .paper_reward import PaperReward
from .simple_hetero_reward import SimpleMAVReward
from .paper_situation import assess_pair,paper_situation_score
from .paper_weapon import PaperWeaponManager
from .pid import PaperAutopilot
from .scenario import ORIGIN

PUBLISHED={"simulation_frequency_hz":60,"decision_frequency_hz":5,"physics_frames_per_action":12,
 "episode_limit_steps":1000,"maximum_speed_mps":400.,"minimum_safe_altitude_m":750.,
 "maximum_aircraft_overload_g":9.,"maximum_attack_range_m":14000.,"launch_interval_s":25.,
 "maximum_overload_g":30.,"navigation_gain_y":3.,"navigation_gain_z":3.,
 "missile_mass_kg":84.,"missile_length_m":2.87,"missile_diameter_m":.127}
INFERRED={"missile_initial_speed_mps":500.,"powered_duration_s":3.,"powered_acceleration_mps2":110.,
 "hit_radius_m":60.,"effective_quadratic_drag_per_m":.00012,"missile_speed_reward_norm_mps":1000.,"reward_global_scale":1.}
SIMPLE_SCENARIOS={
 "simple_paper_1v1":[("red_0","red",120.,60.,0.),("blue_0","blue",120.,60.2,180.)],
 "simple_paper_2v2":[("red_0","red",120.,60.,0.),("red_1","red",120.02,60.,0.),
                        ("blue_0","blue",120.,60.2,180.),("blue_1","blue",120.02,60.2,180.)],
 "simple_paper_3v2_hetero":[("red_uav_0","red",120.,60.,0.),("red_uav_1","red",120.04,60.,0.),
                        ("red_mav_0","red",120.02,59.98,0.),("blue_0","blue",120.,60.2,180.),
                        ("blue_1","blue",120.04,60.2,180.)]}
HETERO_SCENARIO="simple_paper_3v2_hetero"
HETERO_ROLES={"red_uav_0":"uav","red_uav_1":"uav","red_mav_0":"mav","blue_0":"uav","blue_1":"uav"}
HETERO_MISSILES={"red_uav_0":2,"red_uav_1":2,"red_mav_0":0,"blue_0":2,"blue_1":2}

@dataclass
class SimplePIDAircraft:
    agent_id:str;side:str;longitude:float;latitude:float;initial_heading_deg:float;config:dict
    role:str="uav";missile_capacity:int=2
    def __post_init__(self):
        self.simulator=AircraftSimulator(self.config["timing"]["sim_frequency_hz"])
        self.autopilot=PaperAutopilot.from_config(self.config);self._state=None;self.reset()
    def reset(self):
        trim=self.config["trim"];self.alive=True;self.death_reason=None;self.out_of_boundary=False
        self.missile_left=self.missile_capacity;self.current_target=None;self.incoming_missiles=[];self.autopilot.reset()
        self._state=self.simulator.reset(6000.,250.,self.initial_heading_deg,0.,0.,self.latitude,self.longitude,
                                         trim["elevator_trim"],trim["throttle_base"])
        for name in ("gear/gear-cmd-norm","gear/gear-pos-norm","fcs/flap-cmd-norm","fcs/flap-pos-norm"):
            self.simulator.set_property(name,0.)
        for index in range(3):self.simulator.set_property(f"gear/unit[{index}]/pos-norm",0.)
        self.target_pitch=0.;self.target_heading=np.deg2rad(self.initial_heading_deg);self.target_speed=250.
        self._update();return self
    def set_high_level_action(self,action):
        self.target_pitch,self.target_heading,self.target_speed=action_to_targets(action,self.heading)
    def step_physics_once(self):
        if not self.alive:return
        try:
            controls=self.autopilot.step(self.roll,self.pitch,self.heading,self.speed,self.target_pitch,
                self.target_heading,self.target_speed,self.simulator.dt)
            self.simulator.set_controls(*controls);self._state=self.simulator.run();self._update()
        except (RuntimeError,ValueError,FloatingPointError):self.kill("numerical_invalid");return
        if not np.isfinite(np.r_[self.position,self.velocity,self.roll,self.pitch,self.heading,self.speed,
                                 self.vertical_speed,self.load_factor_g,self.alpha,self.beta]).all():self.kill("numerical_invalid")
        elif self.position[2]<=0:self.kill("crash")
    def _update(self):
        s=self._state;self.position=LLA2NEU(s["longitude"],s["latitude"],s["altitude"],*ORIGIN)
        self.velocity=np.array([s["v_north"],s["v_east"],-s["v_down"]],float)
        self.roll=float(s["roll"]);self.pitch=float(s["pitch"]);self.heading=float(s["heading"])
        self.speed=float(s["true_airspeed"]);self.vertical_speed=float(-s["v_down"])
        self.load_factor_g=float(self.simulator.get_property("accelerations/Nz"));self.alpha=float(s["alpha"]);self.beta=float(s["beta"])
        self.state=dict(s,load_factor=self.load_factor_g)
    def kill(self,reason):self.alive=False;self.death_reason=reason;self.out_of_boundary=reason=="boundary"
    def close(self):self.simulator=None

class SimpleTAMCombatEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,scenario_mode="simple_paper_1v1",controlled_side="red",max_steps=1000,
                 weapon_enabled_agent_ids=None,config_path=DEFAULT_CONFIG,hetero_perception_mode="paper_fused"):
        if scenario_mode not in SIMPLE_SCENARIOS:raise ValueError("unsupported simple scenario")
        if controlled_side not in ("red","all"):raise ValueError("controlled_side must be red or all")
        if scenario_mode!=HETERO_SCENARIO and hetero_perception_mode!="paper_fused":
            raise ValueError("hetero_perception_mode is only configurable for simple_paper_3v2_hetero")
        if hetero_perception_mode not in PERCEPTION_MODES:
            raise ValueError(f"hetero_perception_mode must be one of {PERCEPTION_MODES}")
        self.scenario_mode=scenario_mode;self.controlled_side=controlled_side;self.max_steps=int(max_steps)
        self.hetero_perception_mode=hetero_perception_mode
        self.weapon_enabled_agent_ids=None if weapon_enabled_agent_ids is None else set(weapon_enabled_agent_ids)
        self.config=load_config(config_path);self.agent_specs=SIMPLE_SCENARIOS[scenario_mode]
        self.max_red=sum(x[1]=="red" for x in self.agent_specs);self.max_blue=sum(x[1]=="blue" for x in self.agent_specs)
        self.controlled_ids=[x[0] for x in self.agent_specs if controlled_side=="all" or x[1]=="red"]
        self.action_space=spaces.Dict({aid:spaces.Box(-1.,1.,(3,),np.float32) for aid in self.controlled_ids})
        self.paper_observation=PaperObservation(self.max_red,self.max_blue)
        base_dim=7+6*((self.max_red-1)+self.max_blue+8);dim=base_dim+(2 if scenario_mode==HETERO_SCENARIO else 0)
        self.observation_space=spaces.Dict({aid:spaces.Box(-np.inf,np.inf,(dim,),np.float32) for aid in self.controlled_ids})
        self.weapon=PaperWeaponManager(PUBLISHED,INFERRED);self.reward_model=PaperReward();self.mav_reward_model=SimpleMAVReward();self.agents=[]
        self.hetero_perception=HeterogeneousPerceptionSystem(hetero_perception_mode) if scenario_mode==HETERO_SCENARIO else None
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);del options
        if not self.agents:
            self.agents=[SimplePIDAircraft(*spec,self.config,role=HETERO_ROLES.get(spec[0],"uav"),
                         missile_capacity=HETERO_MISSILES.get(spec[0],2)) for spec in self.agent_specs]
        else:
            for agent in self.agents:agent.reset()
        self.by_id={a.agent_id:a for a in self.agents};self.weapon.reset();self.step_count=0;self.simulation_time_s=0.
        self.current_targets={};self.target_change_events=[];self.valid_target_reselections=0
        self.perception_result={};self.target_selection_scores={};self.target_selection_source={}
        self.red_uav_launches_using_shared_track=0;self.red_uav_launches_using_direct_track=0
        self.invalid_episode=False;self.red_missile_kills=self.blue_missile_kills=0;self.simultaneous_kills=0
        self.minimum_altitude_by_agent={a.agent_id:a.position[2] for a in self.agents};self.maximum_speed_by_agent={a.agent_id:a.speed for a in self.agents}
        self.minimum_altitude_m=min(self.minimum_altitude_by_agent.values());self.maximum_altitude_m=max(a.position[2] for a in self.agents)
        self.minimum_speed_mps=min(a.speed for a in self.agents);self.maximum_speed_mps=max(self.maximum_speed_by_agent.values())
        self.maximum_load_factor_g=max(abs(a.load_factor_g) for a in self.agents);self.flight_envelope_violation=False
        self._update_perception();self._update_targets(False);return self._observations(),self._info(None,None)
    def _update_perception(self):
        if self.hetero_perception is not None:self.perception_result=self.hetero_perception.build(self.agents)
    def _update_targets(self,count_changes=True):
        for a in self.agents:
            previous=self.current_targets.get(a.agent_id);enemies=[e for e in self.agents if e.side!=a.side and e.alive]
            scores={};source="none"
            if self.scenario_mode==HETERO_SCENARIO and a.alive and a.side=="red":
                key="mav_detected_enemy_ids" if a.role=="mav" else "visible_enemy_ids_by_agent"
                ids=self.perception_result[key] if a.role=="mav" else self.perception_result[key].get(a.agent_id,[])
                candidates=[self.by_id[aid] for aid in ids if self.by_id[aid].alive]
                scores={e.agent_id:paper_situation_score(a.position,a.velocity,e.position,e.velocity) for e in candidates}
                target=min(candidates,key=lambda e:(-scores[e.agent_id],e.agent_id)) if candidates else None
                if target is not None:
                    direct=self.perception_result["direct_enemy_ids_by_agent"].get(a.agent_id,[])
                    source="direct" if target.agent_id in direct else "mav_shared"
            else:
                target=min(enemies,key=lambda e:(float(np.linalg.norm(e.position-a.position)),e.agent_id)) if a.alive and enemies else None
                if self.scenario_mode==HETERO_SCENARIO and target is not None:source="global_rule_opponent"
            a.current_target=target.agent_id if target else None;self.current_targets[a.agent_id]=a.current_target
            if self.scenario_mode==HETERO_SCENARIO:
                self.target_selection_scores[a.agent_id]={key:float(value) for key,value in sorted(scores.items())}
                self.target_selection_source[a.agent_id]=source
            if count_changes and previous is not None and a.current_target is not None and previous!=a.current_target:
                self.valid_target_reselections+=1;self.target_change_events.append({"agent_id":a.agent_id,"previous_target":previous,
                  "current_target":a.current_target,"decision_step":self.step_count,"simulation_time_s":self.simulation_time_s})
            a.incoming_missiles=[m for m in self.weapon.missiles if m.alive and m.target_id==a.agent_id]
    def build_rule_actions(self,agent_ids=None):
        selected=set(self.controlled_ids if agent_ids is None else agent_ids);result={}
        for aid in selected:
            a=self.by_id[aid];target=self.by_id.get(a.current_target)
            result[aid]=np.array([0.,-.5,0.],np.float32) if a.role=="mav" else np.zeros(3,np.float32) if target is None else paper_greedy_action(a.state,target.state)
        return result
    def step(self,actions):
        self._update_perception();self._update_targets();self.weapon.begin_decision_step();alive_start={a.agent_id:a.alive for a in self.agents};events=[]
        action_map={str(k):np.asarray(v,dtype=np.float32) for k,v in actions.items()}
        for aid in self.controlled_ids:
            if self.by_id[aid].alive and aid not in action_map:raise KeyError(f"missing action for {aid}")
        if self.controlled_side=="red":
            for a in self.agents:
                if a.side=="blue" and a.alive:
                    target=self.by_id.get(a.current_target);action_map[a.agent_id]=np.zeros(3,np.float32) if target is None else paper_greedy_action(a.state,target.state)
        for a in self.agents:
            if self.weapon_enabled_agent_ids is None or a.agent_id in self.weapon_enabled_agent_ids:
                target=self.by_id.get(a.current_target)
                allowed=not (self.scenario_mode==HETERO_SCENARIO and a.side=="red" and
                    (a.role=="mav" or target is None or target.agent_id not in self.perception_result["visible_enemy_ids_by_agent"].get(a.agent_id,[])))
                launch=self.weapon.try_launch(a,target,self.simulation_time_s) if allowed else None
                if launch:
                    launch.update({"simulation_time_s":self.simulation_time_s,"decision_step":self.step_count});events.append(launch)
                    if self.scenario_mode==HETERO_SCENARIO and a.side=="red" and a.role=="uav":
                        if target.agent_id in self.perception_result["direct_enemy_ids_by_agent"].get(a.agent_id,[]):
                            self.red_uav_launches_using_direct_track+=1
                        else:self.red_uav_launches_using_shared_track+=1
        for a in self.agents:
            if a.alive:a.set_high_level_action(action_map.get(a.agent_id,np.zeros(3,np.float32)))
        out_step=set()
        for frame in range(12):
            for a in self.agents:a.step_physics_once()
            self._check_frame(out_step);frame_events=self.weapon.step_physics_once(self.by_id,1/60)
            for event in frame_events:event.update({"physics_frame_index":frame,"simulation_time_s":self.simulation_time_s+1/60})
            hit_sides={self.by_id[e["shooter_id"]].side for e in frame_events if e.get("hit")}
            if hit_sides=={"red","blue"}:self.simultaneous_kills+=1
            events.extend(frame_events);self.simulation_time_s+=1/60
        for e in events:
            if e.get("reason")=="hit":
                if self.by_id[e["shooter_id"]].side=="red":self.red_missile_kills+=1
                else:self.blue_missile_kills+=1
        self.step_count+=1;pairs={}
        for a in self.agents:
            target=self.by_id.get(a.current_target)
            if target is not None:pairs[a.agent_id]=assess_pair(a.position,a.velocity,target.position,target.velocity)
        rewards,components=self.reward_model.compute(self.agents,self.current_targets,pairs,self.weapon.missiles,events,alive_start,out_step)
        if self.scenario_mode==HETERO_SCENARIO:
            mav=self.by_id["red_mav_0"];rewards[mav.agent_id],components[mav.agent_id]=self.mav_reward_model.compute(
                mav,self.agents,self.weapon.missiles,events,alive_start,out_step)
        red_alive=sum(a.alive for a in self.agents if a.side=="red");blue_alive=sum(a.alive for a in self.agents if a.side=="blue")
        if self.scenario_mode==HETERO_SCENARIO:
            mav_lost=not self.by_id["red_mav_0"].alive;uavs_lost=not any(self.by_id[aid].alive for aid in ("red_uav_0","red_uav_1"))
            red_failed=mav_lost or uavs_lost;blue_eliminated=blue_alive==0;terminated=red_failed or blue_eliminated
            winner="draw" if red_failed and blue_eliminated else "blue" if red_failed else "red" if blue_eliminated else None
            reason="mutual_mission_failure" if red_failed and blue_eliminated else "mav_lost" if mav_lost else "red_uavs_eliminated" if uavs_lost else "blue_eliminated" if blue_eliminated else None
        else:
            terminated=red_alive==0 or blue_alive==0
            winner="draw" if red_alive==blue_alive==0 else "red" if blue_alive==0 else "blue" if red_alive==0 else None
            reason="mutual_elimination" if red_alive==blue_alive==0 else "blue_eliminated" if blue_alive==0 else "red_eliminated" if red_alive==0 else None
        truncated=not terminated and self.step_count>=self.max_steps
        if truncated:winner="draw";reason="timeout"
        self._update_perception();self._update_targets();info=self._info(winner,reason);info["events"]=events;info["reward_components"]=components
        return self._observations(),{aid:float(rewards[aid]) for aid in self.controlled_ids},bool(terminated),bool(truncated),info
    def _check_frame(self,out_step):
        for a in self.agents:
            self.minimum_altitude_by_agent[a.agent_id]=min(self.minimum_altitude_by_agent[a.agent_id],float(a.position[2]))
            self.maximum_speed_by_agent[a.agent_id]=max(self.maximum_speed_by_agent[a.agent_id],a.speed)
            self.minimum_altitude_m=min(self.minimum_altitude_m,float(a.position[2]));self.maximum_altitude_m=max(self.maximum_altitude_m,float(a.position[2]))
            self.minimum_speed_mps=min(self.minimum_speed_mps,a.speed);self.maximum_speed_mps=max(self.maximum_speed_mps,a.speed)
            self.maximum_load_factor_g=max(self.maximum_load_factor_g,abs(a.load_factor_g))
            self.flight_envelope_violation|=a.position[2]<750 or a.speed>400 or abs(a.load_factor_g)>9
            if a.death_reason=="numerical_invalid":self.invalid_episode=True
            if a.alive and np.linalg.norm(a.position[:2])>28000:a.kill("boundary");out_step.add(a.agent_id)
    def _observations(self):
        visible=self.perception_result.get("visible_enemy_ids_by_agent") if self.scenario_mode==HETERO_SCENARIO else None
        structured=self.paper_observation.build(self.agents,self.weapon.missiles,visible)
        result={}
        for aid in self.controlled_ids:
            flat=self.paper_observation.flatten(structured[aid])
            if self.scenario_mode==HETERO_SCENARIO:
                flat=np.concatenate([flat,np.array([1.,0.] if self.by_id[aid].role=="mav" else [0.,1.],np.float32)])
            result[aid]=flat.astype(np.float32)
        return result
    def _info(self,winner,reason):
        red=[a for a in self.agents if a.side=="red"];blue=[a for a in self.agents if a.side=="blue"]
        info={"scenario_mode":self.scenario_mode,"decision_step":self.step_count,"simulation_time_s":self.simulation_time_s,
          "winner":winner,"termination_reason":reason,"alive_red":sum(a.alive for a in red),"alive_blue":sum(a.alive for a in blue),
          "missiles_fired":self.weapon.total_fired,"missile_hits":self.weapon.total_hits,"red_missile_kills":self.red_missile_kills,
          "blue_missile_kills":self.blue_missile_kills,"red_crashes":sum(a.death_reason=="crash" for a in red),
          "blue_crashes":sum(a.death_reason=="crash" for a in blue),"boundary_deaths":sum(a.death_reason=="boundary" for a in self.agents),
          "numerical_invalid":sum(a.death_reason=="numerical_invalid" for a in self.agents),"invalid_episode":self.invalid_episode,
          "simultaneous_kills":self.simultaneous_kills,"current_targets":dict(self.current_targets),
          "valid_target_reselections":self.valid_target_reselections,"target_change_events":list(self.target_change_events),
          "minimum_altitude_m":self.minimum_altitude_m,"maximum_altitude_m":self.maximum_altitude_m,
          "minimum_speed_mps":self.minimum_speed_mps,"maximum_speed_mps":self.maximum_speed_mps,
          "maximum_load_factor_g":self.maximum_load_factor_g,"flight_envelope_violation":self.flight_envelope_violation,
          "minimum_altitude_by_agent":dict(self.minimum_altitude_by_agent),"maximum_speed_by_agent":dict(self.maximum_speed_by_agent),
          "death_reason":{a.agent_id:a.death_reason for a in self.agents},
          "weapon_enabled_agent_ids":None if self.weapon_enabled_agent_ids is None else sorted(self.weapon_enabled_agent_ids)}
        if self.scenario_mode==HETERO_SCENARIO:
            mav=self.by_id["red_mav_0"];uavs=[self.by_id[aid] for aid in ("red_uav_0","red_uav_1")]
            info.update({"agent_roles":{a.agent_id:a.role for a in self.agents},"mav_alive":mav.alive,
              "red_uav_alive":sum(a.alive for a in uavs),"red_uav_missiles_left":{a.agent_id:a.missile_left for a in uavs},
              "mav_missiles_left":mav.missile_left,"red_team_failed_by_mav_loss":not mav.alive,
              "red_team_failed_by_uav_loss":not any(a.alive for a in uavs),"hetero_perception_mode":self.hetero_perception_mode,
              **self.perception_result,"target_selection_scores":self.target_selection_scores,
              "target_selection_source":self.target_selection_source,
              "red_uav_targets_visible":{a.agent_id:bool(a.current_target is not None and a.current_target in
                self.perception_result["visible_enemy_ids_by_agent"].get(a.agent_id,[])) for a in uavs},
              "red_uav_launches_using_shared_track":int(self.red_uav_launches_using_shared_track),
              "red_uav_launches_using_direct_track":int(self.red_uav_launches_using_direct_track)})
        return info
    def close(self):
        for a in self.agents:a.close()
