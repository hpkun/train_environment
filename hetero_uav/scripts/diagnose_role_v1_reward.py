"""Diagnose role_v1 reward overlay vs brma_legacy. No training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env

LEGACY_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
ROLE_CFG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_role_v1.yaml"

ROLE_KEYS = [
    "r_role_mav_survival", "r_role_mav_death",
    "r_role_mav_support", "r_role_mav_team_contribution",
    "r_role_uav_attack_window", "r_role_uav_kill_bonus",
    "r_role_uav_death_penalty", "r_role_uav_missile_warning",
]


def _run(cfg, steps=30):
    env = make_env(cfg, env_type="jsbsim_hetero", max_steps=60)
    try:
        obs, info = env.reset(seed=0)
        comp_sums = {aid: {k: 0.0 for k in ROLE_KEYS} for aid in env.agent_ids}
        total_rew = {aid: 0.0 for aid in env.agent_ids}
        nan = False
        for _ in range(steps):
            acts = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(acts)
            for aid in env.agent_ids:
                total_rew[aid] += float(rewards.get(aid, 0.0))
                rcinfo = info.get(aid, {})
                for k in ROLE_KEYS:
                    comp_sums[aid][k] += float(rcinfo.get(k, 0.0))
            if np.isnan(sum(total_rew.values())):
                nan = True
                break
            if all(terminated.values()):
                break
        return total_rew, comp_sums, nan, env.hetero_reward_mode
    finally:
        env.close()


def _markdown(data):
    lines = [
        "# role_v1 Reward Diagnostic",
        "",
        "## Summary",
        f"- legacy_role_keys_present: {data['legacy_has_role_keys']}",
        f"- role_v1_role_keys_present: {data['role_has_role_keys']}",
        f"- nan: {data['nan']}",
        "",
        "## Legacy (brma_legacy)",
    ]
    for k in ROLE_KEYS:
        v = data["legacy_comps"].get("red_0", {}).get(k, 0.0)
        lines.append(f"- {k}: {'YES' if abs(v) > 1e-9 else '0'}")
    lines.append("")
    lines.append("## role_v1")
    lines.append("")
    for aid in sorted(data["role_comps"].keys()):
        lines.append(f"### {aid}")
        for k in ROLE_KEYS:
            v = data["role_comps"][aid].get(k, 0.0)
            lines.append(f"- {k}: {v:.6f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/environment_audit/role_v1_reward_diagnostic.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/role_v1_reward_diagnostic.md")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    rew_l, comps_l, nan_l, _ = _run(LEGACY_CFG, args.steps)
    rew_r, comps_r, nan_r, _ = _run(ROLE_CFG, args.steps)

    legacy_has = any(abs(comps_l.get(aid, {}).get(k, 0.0)) > 1e-9 for aid in comps_l for k in ROLE_KEYS)
    role_has = any(abs(comps_r.get(aid, {}).get(k, 0.0)) > 1e-9 for aid in comps_r for k in ROLE_KEYS)

    data = {
        "legacy_has_role_keys": legacy_has,
        "role_has_role_keys": role_has,
        "nan": nan_l or nan_r,
        "legacy_comps": {aid: dict(comps_l.get(aid, {})) for aid in comps_l},
        "role_comps": {aid: dict(comps_r.get(aid, {})) for aid in comps_r},
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")

    print(f"legacy_role_keys_present: {legacy_has}", flush=True)
    print(f"role_v1_role_keys_present: {role_has}", flush=True)
    print(f"nan: {data['nan']}", flush=True)
    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)


if __name__ == "__main__":
    main()
