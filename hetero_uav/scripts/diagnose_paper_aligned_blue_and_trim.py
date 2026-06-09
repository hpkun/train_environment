"""Paper-aligned blue opponent and trim comparison. No training."""
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
CONFIG_DEFAULT = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
CONFIG_NO_TRIM = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml"

def rd(r): return [math.degrees(float(x)) for x in r]

def rollout(steps, config, opponent_mode, red0_override_zero=False):
    meta = load_model_meta(MODEL)
    adapter = make_obs_adapter(resolve_obs_adapter_version(None, meta))
    model = make_mappo_model_for_adapter(adapter, torch.device("cpu"), actor_arch=meta.get("actor_arch","mlp"))
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    model.eval()

    env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    result = dict(red0=dict(alive=True, death_step=None, max_abs_roll=0, max_abs_pitch=0, min_alt=float("inf")),
                  blue=dict(detection_step=None, heading_errors=[], missiles_fired=0))
    for s in range(steps):
        rad = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [rad["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, np.float32)) for rid in env.red_ids]
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(np.stack(aobs)), torch.as_tensor(rad["critic_state"]).unsqueeze(0), deterministic=True)
        acts = {rid: action.cpu().numpy()[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
        if red0_override_zero:
            acts["red_0"] = np.array([0.0, 0.0, 0.3], np.float32)

        if s == 0:
            raw = acts.get("red_0", [0,0,0])
            result["red0"]["initial_raw_action"] = list(raw.tolist()) if hasattr(raw,"tolist") else list(raw)
            result["red0"]["initial_trim"] = env._last_action_trim_applied.get("red_0", None)
            result["red0"]["initial_effective_action"] = env._last_effective_actions.get("red_0", None)

        opp = OpponentPolicy(mode=opponent_mode, seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)

        # MAV state
        sim = env.red_planes.get("red_0")
        if sim and sim.is_alive:
            rpy = rd(sim.get_rpy()); alt = float(sim.get_position()[2])
            result["red0"]["max_abs_roll"] = max(result["red0"]["max_abs_roll"], abs(rpy[0]))
            result["red0"]["max_abs_pitch"] = max(result["red0"]["max_abs_pitch"], abs(rpy[1]))
            result["red0"]["min_alt"] = min(result["red0"]["min_alt"], alt)
        elif result["red0"]["alive"]:
            result["red0"]["alive"] = False; result["red0"]["death_step"] = s

        # Blue detection + response
        for bid in env.blue_ids:
            bo = obs.get(bid, {}); es = bo.get("enemy_states")
            has_es = es is not None and np.asarray(es).size > 0 and not np.allclose(np.asarray(es), 0.0)
            if result["blue"]["detection_step"] is None and has_es:
                result["blue"]["detection_step"] = s
            bsim = env.blue_planes.get(bid)
            if bsim and bsim.is_alive and has_es:
                nr_dists = []
                for rid in env.red_ids:
                    rs = env.red_planes.get(rid)
                    if rs and rs.is_alive:
                        nr_dists.append((float(np.linalg.norm(bsim.get_position()-rs.get_position())), rid))
                if nr_dists:
                    _, tgt_rid = min(nr_dists, key=lambda x: x[0])
                    rs_pos = env.red_planes[tgt_rid].get_position()
                    brg = math.atan2(rs_pos[1]-bsim.get_position()[1], rs_pos[0]-bsim.get_position()[0])
                    he = math.atan2(math.sin(brg-bsim.get_rpy()[2]), math.cos(brg-bsim.get_rpy()[2]))
                    result["blue"]["heading_errors"].append(he)
            mf = info.get(bid, {}).get("missiles_fired_this_step", 0) if isinstance(info, dict) else 0
            result["blue"]["missiles_fired"] += int(mf)

        if all(terminated.values()) or all(truncated.values()): break

    env.close()
    ra = sum(1 for s_ in env.red_planes.values() if s_.is_alive) if hasattr(env,"red_planes") else 0
    ba = sum(1 for s_ in env.blue_planes.values() if s_.is_alive) if hasattr(env,"blue_planes") else 0
    result["outcome"] = dict(red_alive=ra, blue_alive=ba, mav_alive=result["red0"]["alive"])
    result["red0"]["min_alt"] = round(result["red0"]["min_alt"],1) if result["red0"]["min_alt"] < float("inf") else None
    he = result["blue"]["heading_errors"]
    result["blue"]["heading_error_start"] = round(he[0],4) if he else None
    result["blue"]["heading_error_after_det"] = round(he[min(10,len(he)-1)],4) if len(he)>10 else (round(he[-1],4) if he else None)
    return result

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output-json", default="outputs/flight_audit/paper_aligned_blue_and_trim.json")
    p.add_argument("--output-md", default="outputs/flight_audit/paper_aligned_blue_and_trim.md")
    args = p.parse_args()
    st = args.steps

    cases = {
        "current_rule_nearest_current_trim": lambda: rollout(st, CONFIG_DEFAULT, "rule_nearest"),
        "current_greedy_fsm_current_trim": lambda: rollout(st, CONFIG_DEFAULT, "greedy_fsm"),
        "brma_rule_current_trim": lambda: rollout(st, CONFIG_DEFAULT, "brma_rule"),
        "brma_rule_no_mav_trim": lambda: rollout(st, CONFIG_NO_TRIM, "brma_rule"),
    }
    results = {}
    for name, fn in cases.items():
        print(f"Running {name}...")
        results[name] = fn()

    md = ["# Paper-Aligned Blue and Trim Comparison", ""]
    for name, r in results.items():
        r0 = r["red0"]; b = r["blue"]; o = r["outcome"]
        md.append(f"## {name}")
        md.append(f"- MAV alive: {r0['alive']} death_step: {r0['death_step']}")
        md.append(f"- max_abs_roll: {r0['max_abs_roll']:.1f} max_abs_pitch: {r0['max_abs_pitch']:.1f}")
        md.append(f"- min_alt: {r0['min_alt']}")
        md.append(f"- init_raw: {r0.get('initial_raw_action')}")
        md.append(f"- init_trim: {r0.get('initial_trim')}")
        md.append(f"- init_eff: {r0.get('initial_effective_action')}")
        md.append(f"- blue det_step: {b['detection_step']} he_start: {b['heading_error_start']} he_after: {b['heading_error_after_det']}")
        md.append(f"- missiles_fired: {b['missiles_fired']}")
        md.append(f"- red_alive: {o['red_alive']} blue_alive: {o['blue_alive']} mav_alive: {o['mav_alive']}")
        md.append("")

    for path, content in [(args.output_json, json.dumps(results,indent=2)), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
    print(f"output_json: {args.output_json}")

if __name__ == "__main__": main()
