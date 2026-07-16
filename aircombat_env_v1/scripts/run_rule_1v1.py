"""Run pure-pursuit red against a selected rule blue opponent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aircombat_env_v1.combat import pursuit_action
from aircombat_env_v1.env import AirCombat1v1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--opponent", choices=("straight", "pursuit"),
                        default="straight")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    counts = {key: 0 for key in (
        "red_hits", "blue_hits", "red_crashes", "blue_crashes",
        "draws", "timeouts", "invalid_episodes")}
    steps = []
    for episode in range(args.episodes):
        seed = episode
        env = AirCombat1v1Env(args.opponent, max_steps=args.max_steps)
        observation, info = env.reset(seed=seed)
        try:
            while info["step"] < args.max_steps:
                action = pursuit_action(env.red_state, env.blue_state)
                observation, reward, terminated, truncated, info = env.step(action)
                if not env.observation_space.contains(observation) or not np.isfinite(reward):
                    raise RuntimeError("invalid output")
                if terminated or truncated:
                    break
            event = info["event"]
            if event == "red_hit":
                counts["red_hits"] += 1
            elif event == "blue_hit":
                counts["blue_hits"] += 1
            elif event == "red_crash":
                counts["red_crashes"] += 1
            elif event == "blue_crash":
                counts["blue_crashes"] += 1
            elif event in ("draw_simultaneous_hit", "draw_both_crash"):
                counts["draws"] += 1
            elif event == "timeout":
                counts["timeouts"] += 1
            else:
                counts["invalid_episodes"] += 1
        except Exception:
            event = "invalid"
            counts["invalid_episodes"] += 1
        steps.append(info["step"])
        print(f"episode={episode} seed={seed} event={event} "
              f"winner={info.get('winner')} steps={info['step']} "
              f"final_distance_m={info['distance_m']:.1f} "
              f"red_dwell={info['red_attack_dwell']:.1f} "
              f"blue_dwell={info['blue_attack_dwell']:.1f}")
        env.close()
    summary = {"episodes": args.episodes, **counts,
               "mean_steps": float(np.mean(steps))}
    print(summary)


if __name__ == "__main__":
    main()
