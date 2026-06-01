from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


def run_episode(env, rng: np.random.Generator) -> dict:
    obs, info = env.reset()
    done = False
    episode_return = 0.0
    missile_used = 0
    length = 0
    reason_counts = Counter()
    while not done:
        before = dict(info.get("missile_left", {}))
        actions = {
            aid: rng.uniform(-1.0, 1.0, env.action_shape).astype(np.float32)
            for aid in env.agent_ids
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        after = info.get("missile_left", {})
        missile_used += sum(max(0, before.get(aid, 0) - after.get(aid, 0)) for aid in after)
        reason_counts.update(info.get("missile_summary", {}).get("reason_counts", {}))
        episode_return += float(np.mean(list(rewards.values()))) if rewards else 0.0
        length += 1
        done = all(terminated.get(aid, False) or truncated.get(aid, False)
                   for aid in env.agent_ids)
    return {
        "red_win": 1.0 if info.get("winner") == "red" else 0.0,
        "mav_survival": float(info.get("mav_survival", 0.0)),
        "red_alive": float(info.get("red_alive", 0.0)),
        "blue_alive": float(info.get("blue_alive", 0.0)),
        "kills": float(info.get("red_kills", 0) + info.get("blue_kills", 0)),
        "missile_used": float(missile_used),
        "episode_return": episode_return,
        "episode_length": float(length),
        "termination_reason": info.get("termination_reason") or "unknown",
        "missile_reason_counts": reason_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.config)
    results = [run_episode(env, rng) for _ in range(args.episodes)]
    term_counts = Counter(r["termination_reason"] for r in results)
    missile_reasons = Counter()
    for result in results:
        missile_reasons.update(result["missile_reason_counts"])

    print(f"episodes: {args.episodes}")
    print(f"win_rate: {np.mean([r['red_win'] for r in results]):.3f}")
    print(f"mav_survival_rate: {np.mean([r['mav_survival'] for r in results]):.3f}")
    print(f"avg_red_alive: {np.mean([r['red_alive'] for r in results]):.3f}")
    print(f"avg_blue_alive: {np.mean([r['blue_alive'] for r in results]):.3f}")
    print(f"avg_kills: {np.mean([r['kills'] for r in results]):.3f}")
    print(f"avg_missile_used: {np.mean([r['missile_used'] for r in results]):.3f}")
    print(f"avg_episode_return: {np.mean([r['episode_return'] for r in results]):.3f}")
    print(f"avg_episode_length: {np.mean([r['episode_length'] for r in results]):.3f}")
    print(f"termination_reason_counts: {dict(term_counts)}")
    print(f"missile_reason_counts: {dict(missile_reasons)}")
    env.close()


if __name__ == "__main__":
    main()
