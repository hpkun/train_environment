"""Counterfactual MAV action diagnostics. No training."""
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

MODEL = "outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix/latest/model.pt"
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"

def rd(r): return [math.degrees(float(x)) for x in r]

def rollout(env, model, adapter, steps, red_fn, disable_trim=False):
    obs, info = env.reset(seed=0)
    if disable_trim:
        env.set_action_trim_enabled(False)
    red0_alive = True
    red0_death_step = None
    red0_max_abs_roll = 0.0
    red0_max_abs_pitch = 0.0
    red0_min_alt = float("inf")
    red0_init_raw = None
    red0_init_eff = None
    red0_roll_exceeds_90_step = None
    last_alt = None
    alt_drops = []
    recs = []
    for s in range(steps):
        result = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [result["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, np.float32)) for rid in env.red_ids]
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(np.stack(aobs)), torch.as_tensor(result["critic_state"]).unsqueeze(0), deterministic=True)
        acts = {}
        for i, rid in enumerate(env.red_ids):
            raw_a = action.cpu().numpy()[i].astype(np.float32)
            acts[rid] = red_fn(rid, raw_a, s)
        opp = OpponentPolicy(mode="rule_nearest", seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)

        sim = env.red_planes.get("red_0")
        if sim and sim.is_alive:
            rpy = rd(sim.get_rpy())
            alt = float(sim.get_position()[2])
            red0_max_abs_roll = max(red0_max_abs_roll, abs(rpy[0]))
            red0_max_abs_pitch = max(red0_max_abs_pitch, abs(rpy[1]))
            red0_min_alt = min(red0_min_alt, alt)
            if red0_roll_exceeds_90_step is None and abs(rpy[0]) > 90:
                red0_roll_exceeds_90_step = s
            if last_alt is not None:
                alt_drops.append(last_alt - alt)
            last_alt = alt
        elif red0_alive:
            red0_alive = False
            red0_death_step = s

        if s == 0:
            red0_init_raw = [round(float(v),4) for v in acts.get("red_0",[0,0,0]).tolist()] if hasattr(acts.get("red_0"),"tolist") else None
            eff = env._last_effective_actions.get("red_0", None)
            red0_init_eff = [round(float(v),4) for v in eff] if eff else None

        if all(terminated.values()) or all(truncated.values()): break

    ra = sum(1 for s_ in env.red_planes.values() if s_.is_alive)
    ba = sum(1 for s_ in env.blue_planes.values() if s_.is_alive)
    return dict(
        red0_alive_final=red0_alive, red0_death_step=red0_death_step,
        red0_max_abs_roll_deg=round(red0_max_abs_roll,1),
        red0_max_abs_pitch_deg=round(red0_max_abs_pitch,1),
        red0_min_altitude=round(red0_min_alt,1) if red0_min_alt < float("inf") else None,
        red0_initial_raw_action=red0_init_raw, red0_initial_effective_action=red0_init_eff,
        red0_roll_exceeds_90_step=red0_roll_exceeds_90_step,
        red0_altitude_drop_rate=round(np.mean(alt_drops),2) if alt_drops else 0,
        red_alive_final=ra, blue_alive_final=ba,
        steps=len(recs))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-json", default="outputs/flight_audit/mav_policy_action_counterfactuals.json")
    p.add_argument("--output-md", default="outputs/flight_audit/mav_policy_action_counterfactuals.md")
    args = p.parse_args()
    st = args.steps

    meta = load_model_meta(args.model)
    adapter = make_obs_adapter(resolve_obs_adapter_version(None, meta))
    model = make_mappo_model_for_adapter(adapter, torch.device("cpu"), actor_arch=meta.get("actor_arch","mlp"))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    def policy_fn(rid, raw_a, s):
        return raw_a  # pass through

    # A: current policy
    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rA = rollout(env, model, adapter, st, policy_fn)
    env.close()

    # B: red0_zero_override
    def red0_zero_fn(rid, raw_a, s):
        if rid == "red_0": return np.array([0.0, 0.0, 0.3], np.float32)
        return raw_a
    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rB = rollout(env, model, adapter, st, red0_zero_fn)
    env.close()

    # C: red0_level_override
    def red0_level_fn(rid, raw_a, s):
        if rid == "red_0": return np.array([0.0, 0.0, 0.5], np.float32)
        return raw_a
    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rC = rollout(env, model, adapter, st, red0_level_fn)
    env.close()

    # D: no trim
    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rD = rollout(env, model, adapter, st, policy_fn, disable_trim=True)
    env.close()

    # E: clamped
    def clamped_fn(rid, raw_a, s):
        if rid == "red_0":
            a = raw_a.copy()
            a[0] = np.clip(a[0], -0.3, 0.3)  # pitch
            a[1] = np.clip(a[1], -0.3, 0.3)  # heading
            return a
        return raw_a
    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    rE = rollout(env, model, adapter, st, clamped_fn)
    env.close()

    data = dict(current_policy=rA, red0_zero_override=rB, red0_level_override=rC,
                current_policy_no_trim=rD, red0_clamped_policy=rE)

    md = ["# MAV Policy Action Counterfactuals", ""]
    for name, r in data.items():
        md.append(f"## {name}")
        md.append(f"- red0_alive: {r['red0_alive_final']} death_step: {r['red0_death_step']}")
        md.append(f"- max abs roll: {r['red0_max_abs_roll_deg']} deg, max abs pitch: {r['red0_max_abs_pitch_deg']} deg")
        md.append(f"- min altitude: {r['red0_min_altitude']}")
        md.append(f"- init raw action: {r['red0_initial_raw_action']}")
        md.append(f"- init effective action: {r['red0_initial_effective_action']}")
        md.append(f"- roll>90 step: {r['red0_roll_exceeds_90_step']}")
        md.append(f"- altitude drop rate: {r['red0_altitude_drop_rate']}")
        md.append(f"- red_alive final: {r['red_alive_final']} blue_alive: {r['blue_alive_final']}")
        md.append("")

    for path, d in [(args.output_json, data), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2) if isinstance(d, dict) else d)
    print(f"output_json: {args.output_json}"); print(f"output_md: {args.output_md}")

if __name__ == "__main__": main()
