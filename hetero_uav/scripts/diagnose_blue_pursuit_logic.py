"""Diagnose blue pursuit effectiveness. No training."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from uav_env import make_env
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"

def rd(r): return [math.degrees(float(x)) for x in r]

def diagnose_blue(blue_mode, steps, red_mode="zero"):
    env = make_env(CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    blue_data = {bid: dict(distances=[], actions=[], headings=[], targets=[]) for bid in env.blue_ids}
    for s in range(steps):
        if red_mode == "zero":
            acts = {rid: np.zeros(3, np.float32) for rid in env.red_ids}
        else:
            acts = {rid: np.array([0.0, 0.0, 0.3], np.float32) for rid in env.red_ids}
        opp = OpponentPolicy(mode=blue_mode, seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)

        for bid in env.blue_ids:
            bo = obs.get(bid, {})
            es = bo.get("enemy_states")
            has_es = es is not None and np.asarray(es).size > 0 and not np.allclose(np.asarray(es), 0.0)
            bsim = env.blue_planes.get(bid)
            nr = -1.0; tgt_slot = -1; tgt_dist = -1.0
            if bsim and bsim.is_alive:
                nr_dists = []
                for rid in env.red_ids:
                    rs = env.red_planes.get(rid)
                    if rs and rs.is_alive:
                        d = float(np.linalg.norm(bsim.get_position() - rs.get_position()))
                        nr_dists.append(d)
                if nr_dists:
                    nr = min(nr_dists)
                    tgt_slot = int(np.argmin(nr_dists))
                    tgt_dist = nr_dists[tgt_slot]
            act = acts.get(bid, [0,0,0])
            if hasattr(act, "tolist"): act = act.tolist()
            hdg = float(sim.get_rpy()[2]) if (sim := (env.blue_planes.get(bid))) and sim.is_alive else 0.0
            blue_data[bid]["distances"].append(nr if nr > 0 else None)
            blue_data[bid]["actions"].append(list(act))
            blue_data[bid]["headings"].append(round(hdg, 4))
            blue_data[bid]["targets"].append(dict(slot=tgt_slot, dist=round(tgt_dist, 1) if tgt_dist>0 else -1))
            blue_data[bid]["has_enemy_states"] = has_es

        if all(terminated.values()) or all(truncated.values()): break
    env.close()

    result = {}
    for bid in env.blue_ids:
        bd = blue_data[bid]
        dists = [d for d in bd["distances"] if d is not None and d > 0]
        result[bid] = dict(
            enemy_states_nonzero=bd.get("has_enemy_states", False),
            distance_start=round(dists[0], 1) if dists else -1,
            distance_end=round(dists[-1], 1) if dists else -1,
            distance_decreased=bool(dists and dists[-1] < dists[0]),
            distance_decreased_ratio=round((dists[0]-dists[-1])/dists[0], 3) if dists and dists[0]>0 else 0,
            first_action=bd["actions"][0] if bd["actions"] else [],
            default_action_used=bool(bd["actions"] and all(abs(a[0])<1e-6 and abs(a[1])<1e-6 and abs(a[2]-0.3)<1e-3 for a in bd["actions"][:5])),
            first_target=bd["targets"][0] if bd["targets"] else {},
        )
    return result

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-json", default="outputs/flight_audit/blue_pursuit_logic.json")
    p.add_argument("--output-md", default="outputs/flight_audit/blue_pursuit_logic.md")
    args = p.parse_args()
    st = args.steps

    rn = diagnose_blue("rule_nearest", st)
    gf = diagnose_blue("greedy_fsm", st)

    data = dict(rule_nearest=rn, greedy_fsm=gf)

    md = ["# Blue Pursuit Logic Diagnostic", ""]
    for mode, res in [("rule_nearest", rn), ("greedy_fsm", gf)]:
        md.append(f"## {mode}")
        for bid, d in res.items():
            md.append(f"### {bid}")
            md.append(f"- enemy_states_nonzero: {d['enemy_states_nonzero']}")
            md.append(f"- distance start: {d['distance_start']} m, end: {d['distance_end']} m")
            md.append(f"- distance decreased: {d['distance_decreased']}")
            md.append(f"- distance decreased ratio: {d['distance_decreased_ratio']}")
            md.append(f"- first action: {d['first_action']}")
            md.append(f"- default action used: {d['default_action_used']}")
            md.append(f"- first target: slot={d['first_target'].get('slot')} dist={d['first_target'].get('dist')}")
            md.append("")
    md.append("## Conclusions")
    md.append("- If distance_decreased_ratio > 0, blue is closing in.")
    md.append("- If first_action is near [0,0,0.3], blue may be using default cruise.")
    md.append("- Compare rule_nearest vs greedy_fsm pursuit behavior.")

    for path, d in [(args.output_json, data), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2) if isinstance(d, dict) else d)
    print(f"output_json: {args.output_json}"); print(f"output_md: {args.output_md}")

if __name__ == "__main__": main()
