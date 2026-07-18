"""Solve and audit the local 6000 m / 250 m/s formal F-16 trim."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from itertools import product
from pathlib import Path
import numpy as np
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from aircombat_env_v1.aircraft import AircraftSimulator
from aircombat_env_v1.paper_trim import PaperTrimState,save_trim

def _prepare(sim,heading,pitch,alpha,elevator,throttle):
    sim.reset(6000.,250.,heading,0.,pitch,60.,120.,elevator,throttle,
              alpha_deg=alpha,beta_deg=0.,flight_path_angle_deg=0.)
    for name in ("gear/gear-cmd-norm","gear/gear-pos-norm","fcs/flap-cmd-norm","fcs/flap-pos-norm"):
        sim.set_property(name,0.)
    for index in range(3):sim.set_property(f"gear/unit[{index}]/pos-norm",0.)

def evaluate(sim,candidate,heading,decision_steps):
    _prepare(sim,heading,candidate["pitch_deg"],candidate["alpha_deg"],candidate["elevator"],candidate["throttle"])
    command=(candidate.get("aileron",0.),candidate["elevator"],candidate.get("rudder",0.),candidate["throttle"])
    initial_pitch=sim.state()["pitch"];metrics={"completed_steps":0,"minimum_altitude_m":6000.,"maximum_altitude_m":6000.,
      "minimum_speed_mps":250.,"maximum_speed_mps":250.,"maximum_abs_load_factor_g":0.,"maximum_abs_alpha_deg":0.,"maximum_abs_beta_deg":0.}
    reason=None;state=sim.state()
    for step in range(decision_steps):
        for _ in range(12):
            sim.set_controls(*command);state=sim.run();values=np.asarray(tuple(state.values()),float)
            metrics["minimum_altitude_m"]=min(metrics["minimum_altitude_m"],state["altitude"]);metrics["maximum_altitude_m"]=max(metrics["maximum_altitude_m"],state["altitude"])
            metrics["minimum_speed_mps"]=min(metrics["minimum_speed_mps"],state["true_airspeed"]);metrics["maximum_speed_mps"]=max(metrics["maximum_speed_mps"],state["true_airspeed"])
            metrics["maximum_abs_load_factor_g"]=max(metrics["maximum_abs_load_factor_g"],abs(sim.get_property("accelerations/Nz")))
            metrics["maximum_abs_alpha_deg"]=max(metrics["maximum_abs_alpha_deg"],abs(float(np.rad2deg(state["alpha"]))))
            metrics["maximum_abs_beta_deg"]=max(metrics["maximum_abs_beta_deg"],abs(float(np.rad2deg(state["beta"]))))
            if not np.isfinite(values).all():reason="numerical_invalid";break
            if state["altitude"]<=0:reason="crash";break
            if state["altitude"]<750:reason="altitude_below_750_m";break
            if state["true_airspeed"]>400:reason="speed_above_400_mps";break
            if metrics["maximum_abs_load_factor_g"]>9:reason="load_above_9_g";break
        metrics["completed_steps"]=step+1
        if reason:break
    metrics.update({"heading_deg":heading,"survived":reason is None and metrics["completed_steps"]==decision_steps,
      "failure_reason":reason,"envelope_violation":reason in {"altitude_below_750_m","speed_above_400_mps","load_above_9_g"},
      "final_vertical_speed_mps":-state["v_down"],"altitude_drift_m":state["altitude"]-6000.,
      "speed_drift_mps":state["true_airspeed"]-250.,"pitch_drift_deg":float(np.rad2deg(state["pitch"]-initial_pitch)),
      "alpha_drift_deg":float(np.rad2deg(state["alpha"])-candidate["alpha_deg"]),"final_roll_deg":float(np.rad2deg(state["roll"]))})
    return metrics

def rank(row):
    cases=row["cases"]
    return (-int(all(c["survived"] for c in cases)),-int(not any(c["envelope_violation"] for c in cases)),
      max(abs(c["final_vertical_speed_mps"]) for c in cases),max(abs(c["altitude_drift_m"]) for c in cases),
      max(abs(c["speed_drift_mps"]) for c in cases),max(abs(c["pitch_drift_deg"]) for c in cases),
      max(abs(c["alpha_drift_deg"]) for c in cases))

def built_in_full_trim():
    states=[]
    for heading in (0.,180.):
        sim=AircraftSimulator(60);_prepare(sim,heading,0.,0.,0.,.43);sim.fdm.do_trim(1)
        pitch=float(np.rad2deg(sim.get_property("attitude/pitch-rad")));alpha=float(np.rad2deg(sim.get_property("aero/alpha-rad")))
        candidate={"pitch_deg":pitch,"alpha_deg":alpha,"throttle":sim.get_property("fcs/throttle-cmd-norm"),
          "aileron":sim.get_property("fcs/aileron-cmd-norm"),
          "elevator":sim.get_property("fcs/elevator-cmd-norm")+sim.get_property("fcs/pitch-trim-cmd-norm"),
          "rudder":sim.get_property("fcs/rudder-cmd-norm")+sim.get_property("fcs/yaw-trim-cmd-norm")}
        states.append({"heading_deg":heading,"candidate":candidate,"gamma_deg":float(np.rad2deg(sim.get_property("flight-path/gamma-rad"))),
                       "beta_deg":float(np.rad2deg(sim.get_property("aero/beta-rad"))),"roll_deg":float(np.rad2deg(sim.get_property("attitude/roll-rad")))})
    validation=[]
    for state in states:
        sim=AircraftSimulator(60);validation.append(evaluate(sim,state["candidate"],state["heading_deg"],1000))
    return {"available":True,"mode":"tFull","states":states,"validation":validation,"qualified":all(x["survived"] for x in validation)}

def offline_search():
    sims={h:AircraftSimulator(60) for h in (0.,180.)}
    def rows_for(candidates,steps,headings=(0.,)):
        rows=[]
        for c in candidates:rows.append({**c,"cases":[evaluate(sims[h],c,h,steps) for h in headings]})
        rows.sort(key=rank);return rows
    coarse=[{"pitch_deg":a,"alpha_deg":a,"elevator":e,"throttle":t,"aileron":0.,"rudder":0.}
            for a,e,t in product(np.linspace(-2,10,7),np.linspace(-.15,.05,5),np.linspace(.40,.55,7))]
    coarse_rows=rows_for(coarse,150)
    fine_set=set()
    for c in coarse_rows[:20]:
        for da,de,dt in product((-1.,0.,1.),(-.0125,0.,.0125),(-.0125,0.,.0125)):
            fine_set.add((round(float(np.clip(c["alpha_deg"]+da,-2,10)),8),round(float(np.clip(c["elevator"]+de,-.15,.05)),8),round(float(np.clip(c["throttle"]+dt,.4,.55)),8)))
    fine=[{"pitch_deg":a,"alpha_deg":a,"elevator":e,"throttle":t,"aileron":0.,"rudder":0.} for a,e,t in sorted(fine_set)]
    fine_rows=rows_for(fine,150);long_rows=rows_for([{k:c[k] for k in ("pitch_deg","alpha_deg","elevator","throttle","aileron","rudder")} for c in fine_rows[:10]],1000,(0.,180.))
    best=long_rows[0];qualified=all(c["survived"] for c in best["cases"])
    reasons=Counter(c["failure_reason"] or "passed" for row in long_rows for c in row["cases"])
    return {"coarse_candidate_count":len(coarse),"fine_candidate_count":len(fine),"coarse_top20":coarse_rows[:20],
            "fine_top20":fine_rows[:20],"long_top10":long_rows,"failure_reason_distribution":dict(reasons),"best":best,"qualified":qualified}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("aircombat_env_v1/outputs/paper_trim"));args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    builtin=built_in_full_trim();result={"builtin":builtin}
    if not builtin["qualified"]:result["offline"]=offline_search()
    selected=None
    if builtin["qualified"]:selected=builtin["states"][0]["candidate"]
    elif result["offline"]["qualified"]:selected=result["offline"]["best"]
    if selected:
        cases=(builtin["validation"] if builtin["qualified"] else result["offline"]["best"]["cases"])
        trim=PaperTrimState(pitch_deg=selected["pitch_deg"],alpha_deg=selected["alpha_deg"],throttle=selected["throttle"],
          aileron=selected["aileron"],elevator=selected["elevator"],rudder=selected["rudder"],
          source="paper_unspecified_local_jsbsim_trim",validation_metrics={"heading_cases":cases})
        save_trim(trim,args.output_dir/"paper_trim_state.json")
    (args.output_dir/"trim_search.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"builtin_qualified":builtin["qualified"],"offline_qualified":result.get("offline",{}).get("qualified"),
      "trim_state_written":selected is not None,"output_dir":str(args.output_dir)},indent=2))
if __name__=="__main__":main()
