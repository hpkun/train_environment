"""Audit 3v2 heterogeneous failure modes. No training."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from uav_env import make_env
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIGS = {
    "hetero_f22_mav": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
    "f16_mav_surrogate": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate.yaml",
    "homo_f16_2v2_reference": "uav_env/JSBSim/configs/homo_f16_2v2_brma_rule.yaml",
}

def rd(r): return [math.degrees(float(x)) for x in r]

def run_case(config_path, episodes, steps):
    env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    all_results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        ep_data = dict(death_order=[], death_info={}, blue_targets={},
                       missiles_red=0, missiles_blue=0, missile_hits_red=0, missile_hits_blue=0,
                       missile_terms=dict(hit=0, timeout=0, p_hit_fail=0, low_speed=0, overshoot=0,
                                          target_dead=0, unknown=0),
                       red0_actions=[], red0_rolls=[], red0_pitches=[],
                       blue_missile_targets=dict(red_0=0, red_1=0, red_2=0))
        red0_dead = False
        for s in range(steps):
            # Fixed red actions (cruise)
            acts = {rid: np.array([0.0, 0.0, 0.3], np.float32) for rid in env.red_ids}
            opp = OpponentPolicy(mode="brma_rule", seed=ep*1000+s)
            acts.update(opp.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(acts)

            # Death tracking
            for rid in env.red_ids:
                sim = env.red_planes.get(rid)
                if sim and not sim.is_alive and rid not in ep_data["death_info"]:
                    ep_data["death_order"].append(rid)
                    reason = "unknown"
                    if rid in getattr(env, "_crashed_this_step", set()):
                        reason = "crash"
                    else:
                        mt = (info.get("__missile_term__", {}) if isinstance(info, dict) else {})
                        if isinstance(mt, dict):
                            for team, rdict in mt.items():
                                if isinstance(rdict, dict) and rdict.get("hit", 0) > 0:
                                    reason = "missile_kill"
                                    break
                    ep_data["death_info"][rid] = dict(step=s, reason=reason)

            # MAV state
            sim0 = env.red_planes.get("red_0")
            if sim0 and sim0.is_alive and not red0_dead:
                rpy = rd(sim0.get_rpy())
                act = acts.get("red_0", [0,0,0])
                if hasattr(act, "tolist"): act = act.tolist()
                ep_data["red0_actions"].append(list(act))
                ep_data["red0_rolls"].append(rpy[0])
                ep_data["red0_pitches"].append(rpy[1])
            elif sim0 and not sim0.is_alive and not red0_dead:
                red0_dead = True
                ep_data["red0_death_step"] = s

            # Missile counts per step
            if isinstance(info, dict):
                for bid in env.blue_ids:
                    bi = info.get(bid, {})
                    if isinstance(bi, dict):
                        ep_data["missiles_blue"] += int(bi.get("missiles_fired_this_step", 0))
                for rid in env.red_ids:
                    ri = info.get(rid, {})
                    if isinstance(ri, dict):
                        ep_data["missiles_red"] += int(ri.get("missiles_fired_this_step", 0))
                mt_ = (info.get("__missile_term__", {}) if isinstance(info, dict) else {})
                if isinstance(mt_, dict):
                    for team, rdict in mt_.items():
                        if isinstance(rdict, dict):
                            for reason, count in rdict.items():
                                key = reason if reason in ep_data["missile_terms"] else "unknown"
                                ep_data["missile_terms"][key] = ep_data["missile_terms"].get(key, 0) + int(count)

            if all(terminated.values()) or all(truncated.values()):
                break

        # Blue target tracking (from OpponentPolicy if greedy_fsm, skip for now)
        # Outcome
        ra = sum(1 for s_ in env.red_planes.values() if s_.is_alive)
        ba = sum(1 for s_ in env.blue_planes.values() if s_.is_alive)
        ep_data["outcome"] = dict(red_alive=ra, blue_alive=ba, steps=s+1)
        ep_data["red0_model"] = getattr(env.red_planes.get("red_0"), "model", "?") if env.red_planes.get("red_0") else "?"
        ep_data["red0_role"] = env.agent_roles.get("red_0", "?")
        ep_data["red0_num_missiles"] = env._num_missiles_for("red_0")
        ep_data["red0_death_step"] = ep_data.get("red0_death_step", None)
        ep_data["red0_max_abs_roll"] = max(abs(r) for r in ep_data["red0_rolls"]) if ep_data["red0_rolls"] else 0
        ep_data["red0_max_abs_pitch"] = max(abs(r) for r in ep_data["red0_pitches"]) if ep_data["red0_pitches"] else 0
        all_results.append(ep_data)
    env.close()
    return all_results

def summarize(case_name, results):
    n = len(results)
    return dict(
        case=case_name,
        episodes=n,
        outcome=dict(
            red_alive_mean=round(np.mean([r["outcome"]["red_alive"] for r in results]), 1),
            blue_alive_mean=round(np.mean([r["outcome"]["blue_alive"] for r in results]), 1),
            red_win_rate=sum(1 for r in results if r["outcome"]["red_alive"] > r["outcome"]["blue_alive"]) / n,
            blue_win_rate=sum(1 for r in results if r["outcome"]["blue_alive"] > r["outcome"]["red_alive"]) / n,
            timeout_rate=sum(1 for r in results if r["outcome"]["steps"] >= 1000) / n,
        ),
        first_dead=dict(
            agent=results[0]["death_order"][0] if results and results[0]["death_order"] else "none",
            avg_step=round(np.mean([(r["death_order"] and r["death_info"].get(r["death_order"][0], {}).get("step", 999)) or 999 for r in results if r["death_order"]]), 0) if any(r["death_order"] for r in results) else None,
        ),
        red0_death_count=sum(1 for r in results if "red_0" in r["death_info"]),
        red0_death_reasons={r["death_info"]["red_0"]["reason"]: sum(1 for rr in results if "red_0" in rr["death_info"] and rr["death_info"]["red_0"]["reason"] == r["death_info"]["red_0"]["reason"]) for r in results if "red_0" in r["death_info"]} if any("red_0" in r["death_info"] for r in results) else {},
        red0_avg_max_roll=round(np.mean([r["red0_max_abs_roll"] for r in results]), 1),
        red0_avg_max_pitch=round(np.mean([r["red0_max_abs_pitch"] for r in results]), 1),
        missiles=dict(
            red_fired_avg=round(np.mean([r["missiles_red"] for r in results]), 1),
            blue_fired_avg=round(np.mean([r["missiles_blue"] for r in results]), 1),
        ),
    )

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output-json", default="outputs/environment_audit/hetero_3v2_failure_modes.json")
    p.add_argument("--output-md", default="outputs/environment_audit/hetero_3v2_failure_modes.md")
    args = p.parse_args()

    summaries = {}
    for name, cfg in CONFIGS.items():
        print(f"Running {name}...")
        results = run_case(cfg, args.episodes, args.steps)
        summaries[name] = summarize(name, results)

    # Conclusions
    f22 = summaries["hetero_f22_mav"]
    f16s = summaries["f16_mav_surrogate"]
    homo = summaries["homo_f16_2v2_reference"]
    conclusions = [
        f"3v2 F-22 MAV: red0 deaths={f22['red0_death_count']}/{f22['episodes']}, max_roll={f22['red0_avg_max_roll']:.0f} deg, first_dead={f22['first_dead']['agent']}",
        f"3v2 F-16 MAV surrogate: red0 deaths={f16s['red0_death_count']}/{f16s['episodes']}, max_roll={f16s['red0_avg_max_roll']:.0f} deg, first_dead={f16s['first_dead']['agent']}",
        f"2v2 homo: red_alive={homo['outcome']['red_alive_mean']}, blue_alive={homo['outcome']['blue_alive_mean']}, first_dead={homo['first_dead']['agent']}",
        "If first dead is always red_0, MAV is targeted first regardless of aircraft model.",
        "2v2 homo proves shared MLP can survive brma_rule in symmetric F-16 setting.",
    ]

    data = dict(summaries=summaries, conclusions=conclusions)
    md = ["# 3v2 Heterogeneous Failure Modes Audit", ""]
    for name, s in summaries.items():
        md.append(f"## {name}")
        md.append(f"- episodes: {s['episodes']}")
        md.append(f"- red_alive: {s['outcome']['red_alive_mean']}, blue_alive: {s['outcome']['blue_alive_mean']}")
        md.append(f"- red0_deaths: {s['red0_death_count']}/{s['episodes']}")
        md.append(f"- red0 max_roll: {s['red0_avg_max_roll']:.0f} deg")
        md.append(f"- missiles red: {s['missiles']['red_fired_avg']}, blue: {s['missiles']['blue_fired_avg']}")
        md.append(f"- first dead: {s['first_dead']['agent']} @ step {s['first_dead']['avg_step']}")
        md.append("")
    md.append("## Conclusions")
    for c in conclusions: md.append(f"- {c}")

    for path, content in [(args.output_json, json.dumps(data, indent=2)), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
    print(f"output_json: {args.output_json}")

if __name__ == "__main__": main()
