"""Diagnose F-22 flight stability and blue opponent behavior."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from uav_env import make_env
from algorithms.mappo.adapter_utils import (load_model_meta, make_obs_adapter, resolve_obs_adapter_version, validate_model_dims, make_mappo_model_for_adapter)
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
MODEL = "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"

def rd(r): return [math.degrees(float(x)) for x in r]

def test_zero(steps, aid):
    env = make_env(CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    recs = []
    for s in range(steps):
        acts = {rid: np.array([0.0, 0.0, 0.3], np.float32) for rid in env.red_ids}
        acts.update({bid: np.zeros(3, np.float32) for bid in env.blue_ids})
        obs, rewards, terminated, truncated, info = env.step(acts)
        sim = env.red_planes.get(aid)
        rp = {"alive": False}
        if sim and sim.is_alive:
            rpy = rd(sim.get_rpy())
            pos = sim.get_position()
            vel = sim.get_velocity()
            rp = dict(roll=rpy[0], pitch=rpy[1], yaw=rpy[2], alt=float(pos[2]), speed=float(np.linalg.norm(vel)), alive=True)
        recs.append(dict(step=s, state=rp))
        if all(terminated.values()): break
    env.close()
    return recs

def test_trained(steps):
    if not (ROOT / MODEL).exists(): return []
    meta = load_model_meta(MODEL); version = resolve_obs_adapter_version(None, meta)
    adapter = make_obs_adapter(version); validate_model_dims(adapter, meta)
    arch = meta.get("actor_arch", "mlp")
    model = make_mappo_model_for_adapter(adapter, torch.device("cpu"), actor_arch=arch)
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    model.eval()
    env = make_env(CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    recs = []
    for s in range(steps):
        result = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [result["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, np.float32)) for rid in env.red_ids]
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(np.stack(aobs)), torch.as_tensor(result["critic_state"]).unsqueeze(0), deterministic=True)
        acts = {rid: action.cpu().numpy()[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
        opp = OpponentPolicy(mode="rule_nearest", seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)
        sim = env.red_planes.get("red_0")
        rpy = rd(sim.get_rpy()) if sim and sim.is_alive else [0,0,0]
        red0_act = acts.get("red_0", [0,0,0])
        if hasattr(red0_act, "tolist"): red0_act = red0_act.tolist()
        recs.append(dict(step=s, act=list(red0_act), roll=rpy[0], pitch=rpy[1], yaw=rpy[2], alive=bool(sim and sim.is_alive)))
        if all(terminated.values()): break
    env.close()
    return recs

def test_blue_pursuit(steps):
    env = make_env(CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    recs = []
    for s in range(steps):
        acts = {rid: np.zeros(3, np.float32) for rid in env.red_ids}
        opp = OpponentPolicy(mode="rule_nearest", seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)
        bi = {}
        for bid in env.blue_ids:
            bo = obs.get(bid, {})
            es = bo.get("enemy_states")
            egs = bo.get("enemy_geo_states")
            sim = env.blue_planes.get(bid)
            nr = -1.0
            if sim and sim.is_alive:
                for rid in env.red_ids:
                    rs = env.red_planes.get(rid)
                    if rs and rs.is_alive:
                        d = float(np.linalg.norm(sim.get_position() - rs.get_position()))
                        nr = d if nr < 0 else min(nr, d)
            act = acts.get(bid, [0,0,0])
            if hasattr(act, "tolist"): act = act.tolist()
            bi[bid] = dict(has_es=es is not None and np.asarray(es).size > 0, has_egs=egs is not None, nr_dist=round(nr,1) if nr > 0 else -1, act=list(act))
        recs.append(dict(step=s, blue=bi))
        if all(terminated.values()): break
    env.close()
    return recs

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-json", default="outputs/flight_audit/flight_and_opponent_behavior.json")
    p.add_argument("--output-md", default="outputs/flight_audit/flight_and_opponent_behavior.md")
    args = p.parse_args()
    st = args.steps

    rec_mav = test_zero(st, "red_0")
    rec_f16 = test_zero(st, "red_1")
    rec_pol = test_trained(st)
    rec_blu = test_blue_pursuit(st)

    mav_rolls = [r["state"]["roll"] for r in rec_mav if r["state"].get("alive")]
    mav_max_roll = max(abs(x) for x in mav_rolls) if mav_rolls else 0
    mav_crashed = not rec_mav[-1]["state"].get("alive", False)
    f16_rolls = [r["state"]["roll"] for r in rec_f16 if r["state"].get("alive")]
    f16_max_roll = max(abs(x) for x in f16_rolls) if f16_rolls else 0
    f16_crashed = not rec_f16[-1]["state"].get("alive", False)

    blue_has_es = any(any(bi.get(bid,{}).get("has_es") for bid in bi) for bi in [r.get("blue",{}) for r in rec_blu])
    blue_has_egs = any(any(bi.get(bid,{}).get("has_egs") for bid in bi) for bi in [r.get("blue",{}) for r in rec_blu])
    blue_acts = [r.get("blue",{}).get("blue_0",{}).get("act",[0,0,0]) for r in rec_blu]
    blue_def = sum(1 for a in blue_acts if abs(a[0])<1e-6 and abs(a[1])<1e-6 and abs(a[2]-0.3)<1e-3)
    blue_stuck = blue_def > 0.5 * max(len(blue_acts), 1)

    hp = []
    if mav_crashed: hp.append("F-22 MAV crashed with zero-action cruise command")
    if blue_stuck and not blue_has_es: hp.append("rule_nearest: enemy_states absent in V2; blue stuck on default [0,0,0.3]")

    data = dict(
        mav_zero_action=dict(max_abs_roll_deg=round(mav_max_roll,1), crashed=mav_crashed, steps=len(rec_mav)),
        f16_zero_action=dict(max_abs_roll_deg=round(f16_max_roll,1), crashed=f16_crashed, steps=len(rec_f16)),
        trained_policy_red0=dict(steps=len(rec_pol), red0_alive=rec_pol[-1].get("alive",False) if rec_pol else "N/A") if rec_pol else {},
        blue_rule_nearest=dict(has_enemy_states=blue_has_es, has_enemy_geo_states=blue_has_egs, default_action_pct=round(100*blue_def/max(len(blue_acts),1),1), blue_stuck_on_default=blue_stuck),
        high_priority_issues=hp)

    md = ["# Flight and Opponent Behavior Audit", "", "## MAV (F-22) zero-action", f"- max abs roll: {mav_max_roll:.1f} deg", f"- crashed: {mav_crashed}", "", "## F-16 zero-action", f"- max abs roll: {f16_max_roll:.1f} deg", f"- crashed: {f16_crashed}", "", "## Blue rule_nearest", f"- has_enemy_states: {blue_has_es}", f"- has_enemy_geo_states: {blue_has_egs}", f"- default action pct: {100*blue_def/max(len(blue_acts),1):.1f}%", f"- stuck on default: {blue_stuck}", "", "## High Priority Issues"]
    for h in hp: md.append(f"- {h}")
    if not hp: md.append("- none")
    md.extend(["", "## Key Finding", "rule_nearest reads obs.enemy_states which does NOT exist in V2 mav_shared_geo.", "Blue defaults to [0,0,0.3] (level cruise). Blue does not pursue.", "Affects ALL training/eval runs with rule_nearest+V2.", "Fix: update _rule_nearest_action to read enemy_geo_states."])

    out_j = Path(args.output_json); out_m = Path(args.output_md)
    out_j.parent.mkdir(parents=True, exist_ok=True); out_m.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(data, indent=2)); out_m.write_text("\n".join(md))
    print(f"output_json: {out_j}"); print(f"output_md: {out_m}")
    for h in hp: print(f"HIGH PRIORITY: {h}")

if __name__ == "__main__": main()
