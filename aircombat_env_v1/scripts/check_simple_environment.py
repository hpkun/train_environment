"""Acceptance checks for the simplified fixed-PID JSBSim environment."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.config import load_config
from aircombat_env_v1.simple_env import SimplePIDAircraft

def run_action(action,steps):
    a=SimplePIDAircraft("check_0","red",120.,60.,0.,load_config());start_alt=a.position[2];start_speed=a.speed;start_heading=a.heading
    metrics={"minimum_altitude_m":a.position[2],"maximum_altitude_m":a.position[2],"minimum_speed_mps":a.speed,
      "maximum_speed_mps":a.speed,"maximum_abs_load_factor_g":abs(a.load_factor_g)}
    try:
        for step in range(steps):
            a.set_high_level_action(np.asarray(action,np.float32))
            for _ in range(12):
                a.step_physics_once();metrics["minimum_altitude_m"]=min(metrics["minimum_altitude_m"],a.position[2]);metrics["maximum_altitude_m"]=max(metrics["maximum_altitude_m"],a.position[2])
                metrics["minimum_speed_mps"]=min(metrics["minimum_speed_mps"],a.speed);metrics["maximum_speed_mps"]=max(metrics["maximum_speed_mps"],a.speed)
                metrics["maximum_abs_load_factor_g"]=max(metrics["maximum_abs_load_factor_g"],abs(a.load_factor_g))
                if not a.alive:break
            if not a.alive:break
        metrics.update({"completed_steps":step+1,"crash":int(a.death_reason=="crash"),"numerical_invalid":int(a.death_reason=="numerical_invalid"),
          "altitude_change_m":a.position[2]-start_alt,"speed_change_mps":a.speed-start_speed,
          "heading_change_deg":float(np.rad2deg((a.heading-start_heading+np.pi)%(2*np.pi)-np.pi)),
          "flight_envelope_violation":metrics["minimum_altitude_m"]<750 or metrics["maximum_speed_mps"]>400 or metrics["maximum_abs_load_factor_g"]>9})
        return metrics
    finally:a.close()

def run_checks():
    actions={"level":[0,0,0],"left_turn":[0,-.35,0],"right_turn":[0,.35,0],"climb":[.25,0,0],
             "descend":[-.25,0,0],"accelerate":[0,0,1],"decelerate":[0,0,-.4]}
    short={name:run_action(action,50) for name,action in actions.items()}
    return {"pid_level_1000":run_action(actions["level"],1000),"basic_actions_50":short,
      "direction_checks":{"turns_opposite":np.sign(short["left_turn"]["heading_change_deg"])==-np.sign(short["right_turn"]["heading_change_deg"]),
       "vertical_opposite":np.sign(short["climb"]["altitude_change_m"])==-np.sign(short["descend"]["altitude_change_m"]),
       "speed_opposite":np.sign(short["accelerate"]["speed_change_mps"])==-np.sign(short["decelerate"]["speed_change_mps"])}}

def main():print(json.dumps(run_checks(),indent=2,default=lambda x:x.item()))
if __name__=="__main__":main()
