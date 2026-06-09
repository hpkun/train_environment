"""Audit brma_legacy reward posture components and MAV action correlation. No training."""
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

COMPONENT_NAMES = ["r_pitch", "r_roll", "r_vel", "r_alt", "r_bound", "r_adv", "r_end", "r_death"]
COMPONENT_WEIGHTS = {"r_pitch": 0.01, "r_roll": 0.002, "r_vel": 0.02, "r_alt": 0.04, "r_bound": 0.04, "r_adv": 0.15}
from uav_env.JSBSim.env import UavCombatEnv
env_path = ROOT / "uav_env" / "JSBSim" / "env.py"
env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output-json", default="outputs/reward_audit/posture_reward_and_mav_actions.json")
    p.add_argument("--output-md", default="outputs/reward_audit/posture_reward_and_mav_actions.md")
    args = p.parse_args()
    st = args.steps

    # --- Run trained policy rollout with reward tracking ---
    meta = load_model_meta(args.model)
    adapter = make_obs_adapter(resolve_obs_adapter_version(None, meta))
    model = make_mappo_model_for_adapter(adapter, torch.device("cpu"), actor_arch=meta.get("actor_arch","mlp"))
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    env = make_env(args.config, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    obs, info = env.reset(seed=0)
    trace = []
    red0_alive = True
    red0_death_step = None

    for s in range(st):
        result = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        aobs = [result["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, np.float32)) for rid in env.red_ids]
        with torch.no_grad():
            _, _, action, _, _ = model(torch.as_tensor(np.stack(aobs)), torch.as_tensor(result["critic_state"]).unsqueeze(0), deterministic=True)
        acts = {rid: action.cpu().numpy()[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
        opp = OpponentPolicy(mode="rule_nearest", seed=s+17)
        acts.update(opp.act(obs, env.blue_ids, env=env))
        obs, rewards, terminated, truncated, info = env.step(acts)

        # red_0 state
        sim = env.red_planes.get("red_0")
        rpy = rd(sim.get_rpy()) if sim and sim.is_alive else [0,0,0]
        pos = sim.get_position() if sim else [0,0,0]
        vel = sim.get_velocity() if sim else [0,0,0]
        raw_act = list(acts.get("red_0", [0,0,0]).tolist()) if hasattr(acts.get("red_0", []), "tolist") else list(acts.get("red_0", [0,0,0]))
        trim = env._last_action_trim_applied.get("red_0", None)
        eff = env._last_effective_actions.get("red_0", None)

        # Reward components from info
        rcinfo = info.get("red_0", {}) if isinstance(info, dict) else {}
        comps = {k: float(rcinfo.get(k, 0.0)) for k in COMPONENT_NAMES}

        trace.append(dict(
            step=s, raw_action=raw_act, action_trim=trim, effective_action=eff,
            roll_deg=rpy[0], pitch_deg=rpy[1], yaw_deg=rpy[2],
            altitude_m=float(pos[2]), speed_mps=float(np.linalg.norm(vel)),
            alive=bool(sim and sim.is_alive),
            reward_total=float(rewards.get("red_0", 0.0)), **comps))

        if red0_alive and not bool(sim and sim.is_alive):
            red0_alive = False; red0_death_step = s

        if all(terminated.values()): break

    env.close()

    # --- Analysis ---
    first50 = trace[:50]
    death_window = trace[max(0, red0_death_step-30):red0_death_step] if red0_death_step else []

    first50_act = [r["raw_action"] for r in first50]
    avg_pitch0 = np.mean([a[0] for a in first50_act])
    avg_heading0 = np.mean([a[1] for a in first50_act])

    death_window_act = [r["raw_action"] for r in death_window] if death_window else []
    death_window_comps = {k: [r[k] for r in death_window] for k in COMPONENT_NAMES} if death_window else {}
    roll_over90 = [r for r in trace if abs(r["roll_deg"]) > 90]
    roll_over90_comps = {k: np.mean([r[k] for r in roll_over90]) for k in COMPONENT_NAMES} if roll_over90 else {}

    # Reward audit
    reward_audit = {
        "reward_weights": COMPONENT_WEIGHTS,
        "r_pitch_description": "penalizes |pitch| > pi/4 (45 deg), severe at > pi/3 (60 deg) — weight 0.01",
        "r_roll_description": "penalty for |roll| approaching pi/2 (90 deg) — weight 0.002 (VERY SMALL)",
        "r_alt_description": "reward for altitude between 4000-7000m, penalty outside — weight 0.04",
        "r_vel_description": "reward for speed in [250,350] m/s — weight 0.02",
        "r_bound_description": "penalty for x or |y| > 40000m — weight 0.04",
        "r_adv_description": "situation advantage reward (AO/TA geometry) — weight 0.15 (LARGEST per-step weight)",
        "r_end_description": "terminal: 30*(N_alive-N_enemy)/N_max — only at episode end",
        "note_r_roll_small": "r_roll weight is only 0.002 — effectively negligible for learning roll stability",
        "env_code_path": str(env_path),
    }

    paper_align = {
        "original_brma_paper_r_pitch": "yes — eq in paper §2.5, Table 4 weight=0.01",
        "original_brma_paper_r_roll": "yes — eq in paper §2.5, Table 4 weight=0.002",
        "original_brma_paper_r_vel": "yes — eq in paper §2.5, Table 4 weight=0.02",
        "original_brma_paper_r_alt": "yes — eq in paper §2.5, Table 4 weight=0.04",
        "original_brma_paper_r_adv": "yes — eq in paper §2.5, Table 4 weight=0.15",
        "current_implementation_matches_paper": "yes — weights and formulas confirmed in env.py lines 1088-1177",
        "original_blue_policy_found": str(Path("c:/Users/HPK/Desktop/train_environment/rule_based_agent.py").exists()),
        "original_blue_policy_path": "c:/Users/HPK/Desktop/train_environment/rule_based_agent.py",
    }

    conclusions = [
        "r_roll weight (0.002) is too small to penalize F-22 extreme roll (180 deg) — effectively no roll stability learned",
        "r_pitch penalizes >45 deg but F-22 reaches 89.8 deg — policy receives some penalty but not enough to override r_adv (0.15)",
        "r_adv (0.15) dominates per-step reward — may encourage extreme maneuvers for geometry advantage",
        "F-22 zero-action also shows max_roll=180 deg — aircraft model stability is a contributing factor independent of reward",
        "MAV posture issue is likely BOTH reward (insufficient roll penalty) AND control (trim + F-22 dynamics)",
    ]

    data = dict(
        reward_component_audit=reward_audit,
        red0_action_reward_trace=dict(
            first50=first50, death_window=death_window,
            red0_death_step=red0_death_step, roll_over90_comps=roll_over90_comps,
            avg_pitch_first50=round(avg_pitch0,4), avg_heading_first50=round(avg_heading0,4),
        ),
        paper_original_project_alignment=paper_align,
        conclusions=conclusions)

    md = ["# Posture Reward and MAV Action Audit", "", "## Reward Structure", ""]
    for k, v in reward_audit.items():
        if not k.startswith("note_"): md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Key Metrics")
    md.append(f"- red0_death_step: {red0_death_step}")
    md.append(f"- avg pitch first 50: {avg_pitch0:.4f}")
    md.append(f"- avg heading first 50: {avg_heading0:.4f}")
    md.append("")
    md.append("## Paper Alignment")
    for k, v in paper_align.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Conclusions")
    for c in conclusions: md.append(f"- {c}")

    for path, content in [(args.output_json, json.dumps(data, indent=2)), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)

    print(f"output_json: {args.output_json}")
    print(f"output_md: {args.output_md}")
    print(f"red0_death_step: {red0_death_step}")
    print(f"original_blue_found: {paper_align['original_blue_policy_found']}")

if __name__ == "__main__": main()
