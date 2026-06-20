"""Diagnose MAV survival failure: is it missile kill or flight control?"""
from __future__ import annotations
import argparse, json, sys, os, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

def _load_policy(args):
    from algorithms.happo import BRMARecurrentMaskedHAPPOReferencePolicy
    import torch, json as jj
    meta = jj.loads((Path(args.checkpoint).parent / "meta.json").read_text())
    policy = BRMARecurrentMaskedHAPPOReferencePolicy(
        entity_dim=meta.get("entity_dim", 19),
        critic_state_dim=meta.get("critic_state_dim", 480),
        action_dim=meta.get("action_dim", 4),
        rnn_hidden_size=meta.get("rnn_hidden_size", 128),
        random_scale_mask=meta.get("random_scale_mask", False),
        biased_mask=meta.get("biased_mask", False))
    policy.load(args.checkpoint, map_location=args.device)
    policy.eval()
    return policy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="outputs/mav_survival_failure_diagnosis")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.mappo.opponent_policy import OpponentPolicy
    import torch

    policy = _load_policy(args)
    adapter = HeteroObsAdapterV2()
    opp = OpponentPolicy(mode="tam_direct_fsm", seed=42)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in ["deterministic", "stochastic"]:
        det = mode == "deterministic"
        mav_deaths = []
        for ep in range(args.episodes):
            env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=True)
            obs, info = env.reset(seed=ep)
            roles = [0 if "mav" in str(env.agent_roles.get(rid,"")).lower() or rid=="red_0" else 1 for rid in env.red_ids]
            rnn_h = None
            if hasattr(policy, 'rnn_hidden_size') and policy.rnn_hidden_size:
                rnn_h = np.zeros((len(env.red_ids), policy.rnn_hidden_size), dtype=np.float32)
            death_step = -1; death_reason = "unknown"
            for s in range(1000):
                adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                actor_obs = np.stack([adapted["actor_obs"].get(rid, np.zeros(96,dtype=np.float32)) for rid in env.red_ids])
                critic = adapted["critic_state"]
                act_kw = {}
                if rnn_h is not None:
                    act_kw["rnn_hidden"] = torch.as_tensor(rnn_h, device=args.device)
                with torch.no_grad():
                    out = policy.act(torch.as_tensor(actor_obs, device=args.device), roles=roles,
                                     critic_state=torch.as_tensor(critic, device=args.device),
                                     deterministic=det, **act_kw)
                actions = out["action"].cpu().numpy()
                if rnn_h is not None and "rnn_hidden" in out:
                    rnn_h = out["rnn_hidden"].cpu().numpy()
                act_dict = {rid: actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
                for bid, bact in opp.act({bid: obs[bid] for bid in env.blue_ids}, env.blue_ids, env=env).items():
                    act_dict[bid] = bact.astype(np.float32)
                obs, rew, term, trunc, info = env.step(act_dict)
                mav = env.red_planes.get("red_0")
                if mav is None or not mav.is_alive:
                    if death_step < 0:
                        death_step = s
                        if s >= env.max_steps - 1: death_reason = "timeout"
                        else:
                            mt = info.get("__missile_term__", {})
                            red_hits = mt.get("red",{}).get("hit",0)
                            death_reason = "missile_kill" if red_hits > 0 else "crash_or_unknown"
                            if mav:
                                alt = mav.get_geodetic()[2]
                                spd = float(np.linalg.norm(mav.get_velocity()))
                                if alt < 2500: death_reason = "crash_low_alt"
                                elif spd < 50: death_reason = "crash_low_speed"
                    break
            mav_deaths.append({"episode": ep, "mode": mode, "death_step": death_step, "death_reason": death_reason})
            env.close()
        results.append({"mode": mode, "mav_deaths": mav_deaths,
                        "mean_death_step": np.mean([d["death_step"] for d in mav_deaths if d["death_step"]>0])})

    # Write outputs
    out_json = out_dir / "mav_survival_failure.json"
    out_md = out_dir / "mav_survival_failure.md"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = ["# MAV Survival Failure Diagnosis", ""]
    for r in results:
        deaths = r["mav_deaths"]
        md.append(f"## {r['mode']}")
        reasons = {}
        for d in deaths:
            reasons[d["death_reason"]] = reasons.get(d["death_reason"], 0) + 1
        md.append(f"- Mean death step: {r['mean_death_step']:.0f}")
        md.append(f"- Death reasons: {reasons}")
        md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_json}")
    for r in results:
        print(f"  {r['mode']}: mean_death={r['mean_death_step']:.0f}")

if __name__ == "__main__":
    main()
