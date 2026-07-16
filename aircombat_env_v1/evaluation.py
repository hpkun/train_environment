"""Evaluation policies and episode aggregation for the 1v1 environment."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from .combat import pursuit_action
from .env import AirCombat1v1Env
from .ppo import deterministic_action, stochastic_action


EVALUATION_MODES = (
    ("fixed_tail_chase", "straight"),
    ("randomized_tail_chase", "straight"),
    ("offset_tail_chase", "straight"),
    ("randomized_tail_chase", "pursuit"),
)


def evaluate_policy(policy, episodes=20, scenario="fixed_tail_chase",
                    opponent="straight", seed=1, stochastic=False,
                    actor=None, device="cpu"):
    env = AirCombat1v1Env(
        scenario_mode=scenario, opponent_policy=opponent, max_steps=1000)
    rng = np.random.default_rng(seed)
    counts = Counter()
    returns, steps, distances, boresights = [], [], [], []
    try:
        for episode in range(int(episodes)):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            final_info = {}
            for step_index in range(env.max_steps):
                if policy == "zero":
                    action = np.zeros(3, dtype=np.float32)
                elif policy == "random":
                    action = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
                elif policy == "pursuit_rule":
                    action = pursuit_action(env.red_state, env.blue_state)
                elif policy == "ppo":
                    if actor is None:
                        raise ValueError("actor is required for PPO evaluation")
                    tensor = torch.as_tensor(
                        observation, dtype=torch.float32,
                        device=device).unsqueeze(0)
                    if stochastic:
                        action_tensor, _ = stochastic_action(actor, tensor)
                    else:
                        action_tensor = deterministic_action(actor, tensor)
                    action = action_tensor.squeeze(0).cpu().numpy()
                else:
                    raise ValueError(f"unknown policy: {policy}")
                observation, reward, terminated, truncated, final_info = env.step(
                    action)
                episode_return += reward
                if terminated or truncated:
                    break
            event = final_info.get("event", "timeout")
            if event == "red_hit":
                counts["red_hits"] += 1
            elif event == "blue_hit":
                counts["blue_hits"] += 1
            elif event == "red_crash":
                counts["red_crashes"] += 1
            elif event == "blue_crash":
                counts["blue_crashes"] += 1
            elif event == "timeout":
                counts["timeouts"] += 1
            elif "numerical_invalid" in str(event):
                counts["numerical_invalid"] += 1
            else:
                counts["draws"] += 1
            returns.append(episode_return)
            steps.append(step_index + 1)
            distances.append(float(final_info.get("distance_m", np.nan)))
            boresights.append(float(
                final_info.get("red_boresight_deg", np.nan)))
    finally:
        env.close()
    result = {
        "policy": policy,
        "scenario": scenario,
        "opponent": opponent,
        "episodes": int(episodes),
        "red_hits": counts["red_hits"],
        "blue_hits": counts["blue_hits"],
        "red_crashes": counts["red_crashes"],
        "blue_crashes": counts["blue_crashes"],
        "numerical_invalid": counts["numerical_invalid"],
        "draws": counts["draws"],
        "timeouts": counts["timeouts"],
        "red_hit_rate": counts["red_hits"] / max(int(episodes), 1),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps)),
        "mean_final_distance": float(np.nanmean(distances)),
        "mean_red_boresight_deg": float(np.nanmean(boresights)),
    }
    return result
