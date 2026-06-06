"""Diagnose minimal_v1 vs brma_legacy reward overlay. No training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from algorithms.mappo.opponent_policy import OpponentPolicy

CONFIGS = {
    "legacy": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "minimal": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_minimal.yaml",
}
OVERLAY_KEYS = ["r_mav_survival", "r_mav_death", "r_mav_support",
                "r_shared_track_used", "r_attack_kill_bonus"]


def _run(env, policy_mode: str, steps: int = 20):
    obs, info = env.reset(seed=0)
    opponent = OpponentPolicy(mode="greedy_fsm", seed=1)
    comp_sums = {aid: {k: 0.0 for k in OVERLAY_KEYS} for aid in env.agent_ids}
    total_rew = {aid: 0.0 for aid in env.agent_ids}
    nan = False
    for _ in range(steps):
        actions = {}
        if policy_mode == "zero":
            red_acts = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
        else:
            red_acts = {rid: np.random.uniform(-0.5, 0.5, (3,)).astype(np.float32)
                        for rid in env.red_ids}
        actions.update(red_acts)
        actions.update(opponent.act(obs, env.blue_ids))
        obs, rewards, terminated, truncated, info = env.step(actions)
        for aid in env.agent_ids:
            total_rew[aid] += float(rewards.get(aid, 0.0))
            rcinfo = info.get(aid, {})
            for k in OVERLAY_KEYS:
                comp_sums[aid][k] += float(rcinfo.get(k, 0.0))
        if np.isnan(sum(total_rew.values())):
            nan = True
            break
        if all(terminated.values()):
            obs, info = env.reset(seed=0)
    return total_rew, comp_sums, nan


def main():
    for label, cfg in CONFIGS.items():
        print(f"=== {label} ({cfg}) ===")
        for pmode in ["zero", "bounded_random"]:
            env = make_env(cfg, env_type="jsbsim_hetero", max_steps=30)
            try:
                total, comps, nan = _run(env, pmode)
                mav_id = "red_0"
                print(f"  {pmode}: total_red_rew={total[mav_id]:+.3f} "
                      f"nan={nan}")
                for k in OVERLAY_KEYS:
                    v = comps.get(mav_id, {}).get(k, 0.0)
                    present = "YES" if abs(v) > 1e-9 else "0"
                    print(f"    {k}: {present}")
            finally:
                env.close()
        print()


if __name__ == "__main__":
    main()
