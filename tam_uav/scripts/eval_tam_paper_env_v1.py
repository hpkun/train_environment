"""Evaluate fixed or perturbed TAM paper environments (50 episodes per level by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.make_env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/tam_paper_env_v1_3v2.yaml")
    parser.add_argument("--perturbation-levels", nargs="+",
                        choices=("none", "low", "medium", "large"),
                        default=["none", "low", "medium", "large"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--backend", choices=("simple", "jsbsim"), default="jsbsim")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="outputs/tam_paper_env_v1_evaluation.json")
    args = parser.parse_args()
    summaries = {}
    for level_index, level in enumerate(args.perturbation_levels):
        records = []
        env = make_env(str(ROOT / args.config), dynamics_backend=args.backend,
                       initial_perturbation=level)
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + level_index * 10000 + episode)
            while True:
                actions = {aid: env.task.opponent.act(aid, obs[aid])[0]
                           for aid in env.agent_ids}
                obs, rewards, terminated, truncated, info = env.step(actions)
                if all(terminated.values()) or all(truncated.values()):
                    break
            records.append({"winner": info["winner"], "steps": info["episode_step"],
                            "missiles_fired": info["missiles_fired"],
                            "missile_hits": info["missile_hits"],
                            "finite": bool(np.isfinite(list(rewards.values())).all())})
        env.close()
        summaries[level] = {
            "episodes": len(records),
            "red_win_rate": sum(r["winner"] == "red" for r in records) / len(records),
            "mean_steps": float(np.mean([r["steps"] for r in records])),
            "mean_missiles_fired": float(np.mean([r["missiles_fired"] for r in records])),
            "mean_missile_hits": float(np.mean([r["missile_hits"] for r in records])),
            "all_finite": all(r["finite"] for r in records),
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
