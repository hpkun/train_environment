"""Offline local-JSBSim calibration of the seven paper basic manoeuvres."""
from __future__ import annotations
import argparse,json,sys
from itertools import product
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.paper_action_semantics import map_action_indices
from aircombat_env_v1.paper_env import PaperAircraft

TRIM={"throttle":.430859375,"aileron":0.,"elevator":-.05078125,"rudder":0.}

def nearest_indices():
    return (int(round((TRIM["throttle"]-.4)/.5*39)),int(round((TRIM["aileron"]+1)/2*39)),
            int(round((TRIM["elevator"]+1)/2*39)),int(round((TRIM["rudder"]+1)/2*39)))

def _wrap_deg(value):return float((value+180.)%360.-180.)

def run_fixed(aircraft,indices,steps):
    aircraft.reset();aircraft._initialize_paper_direct_fcs(indices)
    start_alt=float(aircraft.position[2]);start_heading=float(np.rad2deg(aircraft.heading));vertical=[]
    result={"completed_steps":0,"crash":False,"numerical_invalid":False,"minimum_altitude_m":start_alt,
            "maximum_altitude_m":start_alt,"minimum_speed_mps":aircraft.speed,"maximum_speed_mps":aircraft.speed,
            "maximum_abs_load_factor_g":abs(aircraft.load_factor_g),"maximum_abs_alpha_deg":abs(float(np.rad2deg(aircraft.alpha))),
            "maximum_abs_beta_deg":abs(float(np.rad2deg(aircraft.beta)))}
    command=map_action_indices(indices)
    for step in range(steps):
        aircraft.apply_direct_fcs_command(command)
        for _ in range(12):
            aircraft.step_physics_once();vertical.append(aircraft.vertical_speed)
            result["minimum_altitude_m"]=min(result["minimum_altitude_m"],float(aircraft.position[2]))
            result["maximum_altitude_m"]=max(result["maximum_altitude_m"],float(aircraft.position[2]))
            result["minimum_speed_mps"]=min(result["minimum_speed_mps"],aircraft.speed)
            result["maximum_speed_mps"]=max(result["maximum_speed_mps"],aircraft.speed)
            result["maximum_abs_load_factor_g"]=max(result["maximum_abs_load_factor_g"],abs(aircraft.load_factor_g))
            result["maximum_abs_alpha_deg"]=max(result["maximum_abs_alpha_deg"],abs(float(np.rad2deg(aircraft.alpha))))
            result["maximum_abs_beta_deg"]=max(result["maximum_abs_beta_deg"],abs(float(np.rad2deg(aircraft.beta))))
            if not aircraft.alive:break
        result["completed_steps"]=step+1
        envelope=(result["minimum_altitude_m"]<750 or result["maximum_speed_mps"]>400 or result["maximum_abs_load_factor_g"]>9)
        if not aircraft.alive or envelope:break
    result.update({"survived":aircraft.alive and result["completed_steps"]==steps,
                   "crash":aircraft.death_reason=="crash","numerical_invalid":aircraft.death_reason=="numerical_invalid",
                   "envelope_violation":result["minimum_altitude_m"]<750 or result["maximum_speed_mps"]>400 or result["maximum_abs_load_factor_g"]>9,
                   "final_altitude_m":float(aircraft.position[2]),"final_speed_mps":aircraft.speed,
                   "final_vertical_speed_mps":aircraft.vertical_speed,"mean_vertical_speed_mps":float(np.mean(vertical)),
                   "final_roll_deg":float(np.rad2deg(aircraft.roll)),
                   "heading_change_deg":_wrap_deg(float(np.rad2deg(aircraft.heading))-start_heading)})
    return result

def level_rank(row):
    cases=row["cases"]
    return (-int(all(c["survived"] for c in cases)),-int(not any(c["numerical_invalid"] for c in cases)),
            -int(not any(c["envelope_violation"] for c in cases)),-min(c["minimum_altitude_m"] for c in cases),
            max(c["maximum_altitude_m"]-c["minimum_altitude_m"] for c in cases),
            max(abs(c["final_vertical_speed_mps"]) for c in cases),max(abs(c["final_speed_mps"]-250) for c in cases),
            max(abs(c["final_roll_deg"]) for c in cases),max(abs(c["heading_change_deg"]) for c in cases))

def evaluate_level(indices,steps,aircraft_by_heading):
    return {"indices":list(indices),"controls":map_action_indices(indices).tolist(),
            "cases":[run_fixed(aircraft_by_heading[h],indices,steps) for h in (0,180)]}

def calibrate_level():
    center=nearest_indices();ranges=[range(max(0,center[0]-4),min(39,center[0]+4)+1),(19,20),
                                     range(max(0,center[2]-4),min(39,center[2]+4)+1),(19,20)]
    aircraft={h:PaperAircraft(f"cal_{h}","red",120.,60.,float(h)) for h in (0,180)}
    try:
        stage1=[evaluate_level(x,300,aircraft) for x in product(*ranges)]
        stage1.sort(key=level_rank);stage2=[evaluate_level(x["indices"],1000,aircraft) for x in stage1[:16]]
        stage2.sort(key=level_rank)
    finally:
        for a in aircraft.values():a.close()
    best=stage2[0];best["qualified"]=all(c["survived"] and not c["numerical_invalid"] and not c["envelope_violation"] for c in best["cases"])
    return {"trim_center_indices":list(center),"search_ranges":[list(x) for x in ranges],"stage1_candidate_count":len(stage1),
            "stage1_top16":[{"indices":x["indices"],"rank":level_rank(x)} for x in stage1[:16]],"stage2_top10":stage2[:10],"best":best}

def calibrate_other(level):
    a=PaperAircraft("cal_other","red",120.,60.,0.)
    try:
        base50=run_fixed(a,level,50);base40=run_fixed(a,level,40);t,ai,e,r=level
        result={"level":tuple(level)}
        for name,direction in (("accelerate",1),("decelerate",-1)):
            candidates=[]
            for offset in range(1,9):
                idx=int(np.clip(t+direction*offset,0,39));case=run_fixed(a,(idx,ai,e,r),50)
                delta=case["final_speed_mps"]-base50["final_speed_mps"]
                valid=case["survived"] and not case["envelope_violation"] and direction*delta>1.
                candidates.append((offset,idx,delta,valid,case))
            chosen=next((x for x in candidates if x[3]),None);result[name]=tuple((chosen[1],ai,e,r) if chosen else level)
        pitch=[]
        for offset in range(1,7):
            for sign in (-1,1):
                idx=int(np.clip(e+sign*offset,0,39));case=run_fixed(a,(t,ai,idx,r),40)
                pitch.append((offset,idx,case["final_altitude_m"]-base40["final_altitude_m"],case["mean_vertical_speed_mps"]-base40["mean_vertical_speed_mps"],case))
        up=next((x for x in sorted(pitch) if x[2]>10 and x[3]>.5 and x[4]["survived"] and not x[4]["envelope_violation"]),None)
        down=next((x for x in sorted(pitch) if x[2]<-10 and x[3]<-.5 and x[4]["survived"] and not x[4]["envelope_violation"]),None)
        result["climb"]=(t,ai,up[1],r) if up else tuple(level);result["dive"]=(t,ai,down[1],r) if down else tuple(level)
        turns=[]
        offsets=[x for x in range(-8,9) if x]
        for da,dr,dt in product(offsets,offsets,range(3)):
            indices=(min(t+dt,39),int(np.clip(ai+da,0,39)),e,int(np.clip(r+dr,0,39)))
            case=run_fixed(a,indices,40);loss=base40["final_altitude_m"]-case["final_altitude_m"]
            if case["survived"] and not case["envelope_violation"] and abs(case["heading_change_deg"])>3 and loss<500:
                turns.append((indices,case))
        left=sorted((x for x in turns if x[1]["heading_change_deg"]<0),key=lambda x:(sum(abs(x[0][i]-level[i]) for i in (0,1,3)),abs(x[1]["heading_change_deg"]),max(0,base40["final_altitude_m"]-x[1]["final_altitude_m"])))[0]
        right=sorted((x for x in turns if x[1]["heading_change_deg"]>0),key=lambda x:(sum(abs(x[0][i]-level[i]) for i in (0,1,3)),abs(x[1]["heading_change_deg"]),max(0,base40["final_altitude_m"]-x[1]["final_altitude_m"])))[0]
        result["left_turn"]=tuple(left[0]);result["right_turn"]=tuple(right[0])
        return result
    finally:a.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("aircombat_env_v1/outputs/paper_manoeuvre_calibration.json"));args=p.parse_args()
    level=calibrate_level();result={"fidelity":"paper_unspecified_local_jsbsim_calibration","level_calibration":level}
    if level["best"]["qualified"]:result["manoeuvres"]={k:list(v) for k,v in calibrate_other(level["best"]["indices"]).items()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
