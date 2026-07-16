"""Deterministic evaluation for a formal-v1 Pure HAPPO checkpoint."""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from algorithms.pure_happo import ALGORITHM_CONTRACT, PureHAPPOPolicy
from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ENV_TYPE
from uav_env.JSBSim.formal_v1.reward import REWARD_CONTRACT_VERSION


def _validate_checkpoint_meta(meta: dict) -> None:
    if meta.get("formal_contract") != ENV_TYPE or meta.get("credit_mode") != "shared_alive_team_mean":
        raise ValueError("checkpoint is not a formal-v1 shared-credit checkpoint")
    if meta.get("reward_contract", {}).get("version") != REWARD_CONTRACT_VERSION:
        raise ValueError("checkpoint does not use the current formal role-reward contract")
    expected = {
        "algorithm_contract": ALGORITHM_CONTRACT,
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
        "actor_obs_dim": 68, "critic_state_dim": 204, "action_dim": 3,
        "num_agents": 3,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"incompatible checkpoint {key}: {meta.get(key)!r} != {value!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/formal_v1_eval.json")
    a = p.parse_args()
    env = make_env(str(ROOT / a.config) if not Path(a.config).is_absolute() else a.config)
    model = Path(a.checkpoint); model = model if model.is_absolute() else ROOT / model
    meta = json.loads((model.parent / "meta.json").read_text(encoding="utf-8"))
    _validate_checkpoint_meta(meta)
    policy = PureHAPPOPolicy(meta["actor_obs_dim"], meta["critic_state_dim"],
                             meta["action_dim"], meta["num_agents"]).to(a.device)
    policy.load_state_dict(torch.load(model, map_location=a.device, weights_only=True)); policy.eval()
    rows=[]
    for episode in range(a.episodes):
        obs, info = env.reset(seed=a.seed + episode); totals={x:0.0 for x in env.red_ids}
        stats=defaultdict(float); action_rows=[]
        while True:
            actor=np.stack([obs[x]["flat"] for x in env.red_ids])
            with torch.no_grad(): out=policy.act(actor, critic_state=info["critic_state"], deterministic=True)
            action=out["action"].cpu().numpy()*info["active_mask"][:,None]
            action_rows.append(action.copy())
            active_before=np.asarray(info["active_mask"],np.float32)
            obs,reward,term,trunc,info=env.step({x:action[i] for i,x in enumerate(env.red_ids)})
            for aid in env.red_ids: totals[aid] += float(reward[aid])
            components=info["reward_components"]["per_agent"]
            for key in ("dense","safety","support_position","shared_information"):
                stats[f"mav_{key}"] += float(components["red_0"].get(key,0.0))
            for key in ("dense","flight","speed","angle","distance","dodge"):
                stats[f"uav_{key}"] += float(np.mean([
                    components[aid].get(key,0.0) for aid in ("red_1","red_2")]))
            stats["steps"] += 1
            for i,aid in enumerate(("red_1","red_2"),start=1):
                gate=info.get("fire_gates",{}).get(aid,{})
                if active_before[i] > 0.5 and gate.get("observable",False):
                    stats["geometry_samples"] += 1
                    for key in ("range_ok","ata_ok","ta_ok","geometry_ok"):
                        stats[key] += int(gate.get(key,False))
            if info["team_done"]: break
        red_launches=sum(e["event"]=="launch" and e["shooter_id"].startswith("red") for e in env.event_log)
        blue_launches=sum(e["event"]=="launch" and e["shooter_id"].startswith("blue") for e in env.event_log)
        red_hits=sum(e["event"]=="hit" and e["shooter_id"].startswith("red") for e in env.event_log)
        blue_hits=sum(e["event"]=="hit" and e["shooter_id"].startswith("blue") for e in env.event_log)
        denominator=max(stats["geometry_samples"],1.0)
        step_denominator=max(stats["steps"],1.0)
        actions=np.asarray(action_rows,np.float32)
        row={"episode":episode,"mav_return":totals["red_0"],
                     "uav_return_mean":float(np.mean([totals["red_1"],totals["red_2"]])),
                     "outcome":info["outcome"],
                     "red_alive":info["red_alive"],"blue_alive":info["blue_alive"],
                     "red_launches":red_launches,"blue_launches":blue_launches,
                     "red_hits":red_hits,"blue_hits":blue_hits,
                     "red_kills":red_hits,"blue_kills":blue_hits,
                     "mav_survived":int(info["mav_alive"]),
                     "range_rate":stats["range_ok"]/denominator,
                     "ata_rate":stats["ata_ok"]/denominator,
                     "ta_rate":stats["ta_ok"]/denominator,
                     "geometry_rate":stats["geometry_ok"]/denominator}
        for key in ("dense","safety","support_position","shared_information"):
            row[f"mav_{key}_mean"]=stats[f"mav_{key}"]/step_denominator
        for key in ("dense","flight","speed","angle","distance","dodge"):
            row[f"uav_{key}_mean"]=stats[f"uav_{key}"]/step_denominator
        for role,values in (("mav",actions[:,0,:]),("uav",actions[:,1:,:].reshape(-1,3))):
            for dim,name in enumerate(("pitch","heading","speed")):
                row[f"{role}_action_mean_{name}"]=float(values[:,dim].mean())
                row[f"{role}_action_std_{name}"]=float(values[:,dim].std())
                row[f"{role}_action_saturation_{name}"]=float(
                    np.mean(np.abs(values[:,dim])>=0.999))
        if not all(np.isfinite(value) for value in row.values() if not isinstance(value,str)):
            raise ValueError(f"non-finite formal eval row episode={episode}")
        rows.append(row)
    summary={"episodes":rows,
             "avg_mav_return":float(np.mean([x["mav_return"] for x in rows])),
             "avg_uav_return":float(np.mean([x["uav_return_mean"] for x in rows])),
             "red_win_rate":float(np.mean([x["outcome"]=="red_win" for x in rows])),
             "blue_win_rate":float(np.mean([x["outcome"]=="blue_win" for x in rows])),
             "mutual_elimination_rate":float(np.mean([
                 x["outcome"]=="mutual_elimination" for x in rows])),
             "timeout_rate":float(np.mean([x["outcome"]=="draw" for x in rows])),
             "mav_survival_rate":float(np.mean([x["mav_survived"] for x in rows])),
             "red_alive_final_mean":float(np.mean([x["red_alive"] for x in rows])),
             "blue_alive_final_mean":float(np.mean([x["blue_alive"] for x in rows]))}
    for key in ("red_launches","blue_launches","red_hits","blue_hits","red_kills","blue_kills",
                "range_rate","ata_rate","ta_rate","geometry_rate","mav_shared_information_mean"):
        summary[f"{key}_mean"]=float(np.mean([x[key] for x in rows]))
    summary["finite"]=bool(all(
        np.isfinite(value) for row in rows for value in row.values() if not isinstance(value,str)))
    numeric_episode_keys = [
        key for key, value in rows[0].items()
        if key != "episode" and isinstance(value, (int, float, np.number))]
    summary["episode_metric_std"] = {
        key: float(np.std([float(row[key]) for row in rows]))
        for key in numeric_episode_keys
    }
    out=Path(a.output); out=out if out.is_absolute() else ROOT/out; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
    env.close()

if __name__ == "__main__": main()
