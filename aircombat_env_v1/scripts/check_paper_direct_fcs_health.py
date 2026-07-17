"""Long-horizon direct-FCS diagnostics without PID or weapons."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_action_semantics import map_action_indices
from aircombat_env_v1.paper_env import PaperAircraft
from aircombat_env_v1.paper_opponent import MANOEUVRES

SEQUENCE=[("level",50),("accelerate",20),("level",30),("left_turn",20),("level",30),
          ("right_turn",20),("level",30),("climb",15),("level",40),("dive",15),
          ("level",40),("decelerate",20),("level",50)]

def run_commands(names,steps=1000):
    aircraft=PaperAircraft("health_0","red",120.,60.,0.);metrics={"minimum_altitude_m":aircraft.position[2],
        "maximum_altitude_m":aircraft.position[2],"minimum_speed_mps":aircraft.speed,"maximum_speed_mps":aircraft.speed,
        "maximum_abs_load_factor_g":abs(aircraft.load_factor_g)};completed=0
    try:
        for name in names:
            aircraft.apply_direct_fcs_command(map_action_indices(MANOEUVRES[name]))
            for _ in range(12):
                aircraft.step_physics_once();metrics["minimum_altitude_m"]=min(metrics["minimum_altitude_m"],aircraft.position[2])
                metrics["maximum_altitude_m"]=max(metrics["maximum_altitude_m"],aircraft.position[2]);metrics["minimum_speed_mps"]=min(metrics["minimum_speed_mps"],aircraft.speed)
                metrics["maximum_speed_mps"]=max(metrics["maximum_speed_mps"],aircraft.speed);metrics["maximum_abs_load_factor_g"]=max(metrics["maximum_abs_load_factor_g"],abs(aircraft.load_factor_g))
                if not aircraft.alive:break
            completed+=1
            if not aircraft.alive or completed>=steps:break
        metrics.update({"requested_steps":steps,"completed_steps":completed,"survived":aircraft.alive and completed==steps,
            "numerical_invalid":int(aircraft.death_reason=="numerical_invalid"),"crash":int(aircraft.death_reason=="crash"),
            "flight_envelope_violation":bool(metrics["minimum_altitude_m"]<750 or metrics["maximum_speed_mps"]>400 or metrics["maximum_abs_load_factor_g"]>9),
            "final_roll_rad":aircraft.roll,"final_pitch_rad":aircraft.pitch,"final_heading_rad":aircraft.heading,"death_reason":aircraft.death_reason})
        return metrics
    finally:aircraft.close()

def main():
    level=run_commands(("level" for _ in range(1000)))
    cycle=[name for name,count in SEQUENCE for _ in range(count)];maneuvers=run_commands((cycle[i%len(cycle)] for i in range(1000)))
    print(json.dumps({"level":level,"basic_manoeuvre_cycle":maneuvers},indent=2))
if __name__=="__main__":main()
