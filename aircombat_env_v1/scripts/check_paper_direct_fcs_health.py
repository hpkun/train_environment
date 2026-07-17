"""Long-horizon formal direct-FCS diagnostics without PID or weapons."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_action_semantics import map_action_indices
from aircombat_env_v1.paper_env import PaperAircraft,TAMPaperCombatEnv
from aircombat_env_v1.paper_opponent import MANOEUVRES

SEQUENCE=[("level",50),("accelerate",20),("level",30),("left_turn",20),("level",30),
          ("right_turn",20),("level",30),("climb",15),("level",40),("dive",15),
          ("level",40),("decelerate",20),("level",50)]

def run_commands(names,steps=1000,heading_deg=0.):
    aircraft=PaperAircraft("health_0","red",120.,60.,heading_deg);completed=0
    metrics={"minimum_altitude_m":aircraft.position[2],"maximum_altitude_m":aircraft.position[2],
             "minimum_speed_mps":aircraft.speed,"maximum_speed_mps":aircraft.speed,
             "maximum_abs_load_factor_g":abs(aircraft.load_factor_g),
             "maximum_abs_alpha_deg":abs(float(np.rad2deg(aircraft.alpha))),
             "maximum_abs_beta_deg":abs(float(np.rad2deg(aircraft.beta)))}
    try:
        for name in names:
            aircraft.apply_direct_fcs_command(map_action_indices(MANOEUVRES[name]))
            for _ in range(12):
                aircraft.step_physics_once()
                metrics["minimum_altitude_m"]=min(metrics["minimum_altitude_m"],aircraft.position[2]);metrics["maximum_altitude_m"]=max(metrics["maximum_altitude_m"],aircraft.position[2])
                metrics["minimum_speed_mps"]=min(metrics["minimum_speed_mps"],aircraft.speed);metrics["maximum_speed_mps"]=max(metrics["maximum_speed_mps"],aircraft.speed)
                metrics["maximum_abs_load_factor_g"]=max(metrics["maximum_abs_load_factor_g"],abs(aircraft.load_factor_g))
                metrics["maximum_abs_alpha_deg"]=max(metrics["maximum_abs_alpha_deg"],abs(float(np.rad2deg(aircraft.alpha))))
                metrics["maximum_abs_beta_deg"]=max(metrics["maximum_abs_beta_deg"],abs(float(np.rad2deg(aircraft.beta))))
                if not aircraft.alive:break
            completed+=1
            if not aircraft.alive or completed>=steps:break
        metrics.update({"requested_steps":steps,"completed_steps":completed,"survived":aircraft.alive and completed==steps,
            "numerical_invalid":int(aircraft.death_reason=="numerical_invalid"),"crash":int(aircraft.death_reason=="crash"),"boundary":0,
            "final_vertical_speed_mps":aircraft.vertical_speed,"envelope_violation":bool(metrics["minimum_altitude_m"]<750 or metrics["maximum_speed_mps"]>400 or metrics["maximum_abs_load_factor_g"]>9),
            "death_reason":aircraft.death_reason})
        return metrics
    finally:aircraft.close()

def run_greedy(steps=1000):
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=steps,weapon_enabled_agent_ids=set());env.reset(seed=1);info={}
    try:
        for completed in range(1,steps+1):
            _,_,terminated,truncated,info=env.step(env.build_rule_actions())
            if terminated or truncated:break
        deaths=info["death_reason"];boundary=sum(v=="boundary" for v in deaths.values());crash=sum(v=="crash" for v in deaths.values())
        return {"requested_steps":steps,"completed_steps":completed,"survived":crash==0 and not info["invalid_episode"],
                "crash":crash,"numerical_invalid":int(info["invalid_episode"]),"boundary":boundary,
                "minimum_altitude_m":info["minimum_altitude_m"],"maximum_altitude_m":info["maximum_altitude_m"],
                "minimum_speed_mps":info["minimum_speed_mps"],"maximum_speed_mps":info["maximum_speed_mps"],
                "maximum_abs_load_factor_g":info["maximum_load_factor_g"],"maximum_abs_alpha_deg":info["maximum_alpha_deg"],
                "maximum_abs_beta_deg":info["maximum_beta_deg"],"final_vertical_speed_mps":info["final_vertical_speed_mps"],
                "envelope_violation":info["flight_envelope_violation"],"death_reason":deaths}
    finally:env.close()

def main():
    cycle=[name for name,count in SEQUENCE for _ in range(count)]
    result={"level_heading_0":run_commands(("level" for _ in range(1000)),heading_deg=0.),
            "level_heading_180":run_commands(("level" for _ in range(1000)),heading_deg=180.),
            "basic_manoeuvre_cycle":run_commands((cycle[i%len(cycle)] for i in range(1000))),
            "greedy_no_weapons":run_greedy()}
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
