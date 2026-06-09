"""Diagnose how blue detects and responds to red. No training."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from uav_env import make_env
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
REF = Path("c:/Users/HPK/Desktop/train_environment/rule_based_agent.py")
ORIGINAL_BLUE_FOUND = REF.exists()

def rd(r): return [math.degrees(float(x)) for x in r]

def bearing_to(pos_from, pos_to):
    d = pos_to - pos_from
    return math.atan2(d[1], d[0])

def diagnose(blue_mode, steps, red_fn_name="level_cruise"):
    env = make_env(CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    blue_data = {bid: dict(records=[], first_detection_step=None, target_switch_count=0,
                           prev_target_slot=None, target_lost_steps=0) for bid in env.blue_ids}

    for s in range(steps):
        acts = {}
        for rid in env.red_ids: acts[rid] = np.array([0.0, 0.0, 0.3], np.float32)
        opp = OpponentPolicy(mode=blue_mode, seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)

        for bid in env.blue_ids:
            bo = obs.get(bid, {})
            es = bo.get("enemy_states")
            has_es = es is not None and np.asarray(es).size > 0 and not np.allclose(np.asarray(es), 0.0)
            egs = bo.get("enemy_geo_states")
            has_egs = egs is not None and np.asarray(egs).size > 0
            eom = bo.get("enemy_observed_mask")
            eam = bo.get("enemy_alive_mask")

            bsim = env.blue_planes.get(bid)
            tgt_slot = -1; tgt_dist = -1.0; brg = 0.0; hdg_err = 0.0
            if bsim and bsim.is_alive:
                nr_dists = []
                for rid in env.red_ids:
                    rs = env.red_planes.get(rid)
                    if rs and rs.is_alive:
                        d = float(np.linalg.norm(bsim.get_position() - rs.get_position()))
                        nr_dists.append((d, rid))
                if nr_dists:
                    tgt_dist, tgt_rid = min(nr_dists, key=lambda x: x[0])
                    tgt_slot = int(tgt_rid.split("_")[1])
                    rs_pos = env.red_planes[tgt_rid].get_position()
                    brg = bearing_to(bsim.get_position(), rs_pos)
                    hdg_err = math.atan2(math.sin(brg - bsim.get_rpy()[2]), math.cos(brg - bsim.get_rpy()[2]))

            # Detection check
            just_detected = False
            if blue_data[bid]["first_detection_step"] is None and has_es and tgt_slot >= 0:
                blue_data[bid]["first_detection_step"] = s
                just_detected = True

            # Target switch
            prev_slot = blue_data[bid]["prev_target_slot"]
            if prev_slot is not None and tgt_slot >= 0 and tgt_slot != prev_slot:
                blue_data[bid]["target_switch_count"] += 1
            if tgt_slot < 0 and prev_slot is not None and prev_slot >= 0:
                blue_data[bid]["target_lost_steps"] += 1
            blue_data[bid]["prev_target_slot"] = tgt_slot if tgt_slot >= 0 else prev_slot

            act = acts.get(bid, [0,0,0])
            if hasattr(act, "tolist"): act = act.tolist()
            is_default = abs(act[0])<1e-6 and abs(act[1])<1e-6 and abs(act[2]-0.3)<1e-3

            blue_data[bid]["records"].append(dict(
                step=s, has_enemy_states=has_es, has_enemy_geo_states=has_egs,
                enemy_observed=int(np.sum(np.asarray(eom))) if eom is not None else 0,
                enemy_alive=int(np.sum(np.asarray(eam))) if eam is not None else 0,
                selected_target_slot=tgt_slot, selected_target_distance=round(tgt_dist,1),
                heading_error_to_target=round(hdg_err,4) if tgt_dist>0 else None,
                action=list(act), default_action_used=is_default,
                just_detected=just_detected))

        if all(terminated.values()) or all(truncated.values()): break

    env.close()

    result = {}
    for bid in env.blue_ids:
        recs = blue_data[bid]["records"]
        fd_step = blue_data[bid]["first_detection_step"]
        # Action before/after detection
        act_before = recs[0]["action"] if recs else []
        act_after = recs[fd_step]["action"] if fd_step is not None and fd_step < len(recs) else []
        act_changed = act_before != act_after

        # Does heading command match target bearing?
        hdg_matches = False
        if fd_step is not None and fd_step < len(recs):
            r = recs[fd_step]
            hdg_matches = r.get("heading_error_to_target") is not None and abs(r["heading_error_to_target"]) < 0.5

        dists = [r["selected_target_distance"] for r in recs if r["selected_target_distance"] > 0]
        result[bid] = dict(
            first_detection_step=fd_step,
            target_switch_count=blue_data[bid]["target_switch_count"],
            target_lost_steps=blue_data[bid]["target_lost_steps"],
            action_before_detection=act_before,
            action_after_detection=act_after,
            action_change_after_detection=act_changed,
            heading_command_matches_target_bearing=hdg_matches,
            distance_start=round(dists[0],1) if dists else -1,
            distance_end=round(dists[-1],1) if dists else -1,
        )
    return result

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-json", default="outputs/flight_audit/blue_detection_and_response.json")
    p.add_argument("--output-md", default="outputs/flight_audit/blue_detection_and_response.md")
    args = p.parse_args()
    st = args.steps

    rn = diagnose("rule_nearest", st)
    gf = diagnose("greedy_fsm", st)
    data = dict(rule_nearest=rn, greedy_fsm=gf,
                original_brma_blue_found=ORIGINAL_BLUE_FOUND,
                original_brma_blue_path=str(REF) if ORIGINAL_BLUE_FOUND else "NOT FOUND")

    md = ["# Blue Detection and Response Audit", "",
          f"Original BRMA blue policy found: {ORIGINAL_BLUE_FOUND}"]
    if ORIGINAL_BLUE_FOUND:
        md.append(f"Path: {REF}")
        md.append("The original uses a 4-layer state machine with: search_acquire, combat, cruise, climb states.")
        md.append("The original reads enemy_states (11-dim body-frame entity vectors).")
        md.append("")
    else:
        md.append("Original BRMA-MAPPO blue policy code NOT found in parent project.")
        md.append("")
    for mode, res in [("rule_nearest", rn), ("greedy_fsm", gf)]:
        md.append(f"## {mode}")
        for bid, d in res.items():
            md.append(f"### {bid}")
            md.append(f"- first_detection_step: {d['first_detection_step']}")
            md.append(f"- action_before_detection: {d['action_before_detection']}")
            md.append(f"- action_after_detection: {d['action_after_detection']}")
            md.append(f"- action_change_after_detection: {d['action_change_after_detection']}")
            md.append(f"- heading_matches_target_bearing: {d['heading_command_matches_target_bearing']}")
            md.append(f"- target_switch_count: {d['target_switch_count']}")
            md.append(f"- target_lost_steps: {d['target_lost_steps']}")
            md.append(f"- distance_start: {d['distance_start']} m, end: {d['distance_end']} m")
            md.append("")
    md.append("## Conclusion")
    md.append("- Distance closing alone is insufficient — initial head-on geometry causes automatic closing.")
    md.append("- Detection check: first_detection_step tells when blue first saw a valid enemy.")
    md.append("- Response check: action_change_after_detection and heading_match verify response quality.")
    md.append("- Original BRMA state machine is much more sophisticated than current opponent_policy.py.")

    for path, content in [(args.output_json, json.dumps(data, indent=2)), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)

    print(f"output_json: {args.output_json}"); print(f"output_md: {args.output_md}")
    print(f"original_brma_blue_found: {ORIGINAL_BLUE_FOUND}")

if __name__ == "__main__": main()
