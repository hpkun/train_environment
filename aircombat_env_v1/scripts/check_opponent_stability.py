"""Run the required paper-greedy blue stability gate."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aircombat_env_v1.combat import pursuit_action
from aircombat_env_v1.env import AirCombat1v1Env


def run_stability(episodes=20, seed=1):
    rows = []
    for scenario in (
            "fixed_tail_chase", "randomized_tail_chase", "offset_tail_chase"):
        for red_policy in ("zero", "pursuit_rule"):
            counts = Counter()
            minimum_altitude = float("inf")
            minimum_speed = float("inf")
            maximum_load = 0.0
            full_timeouts = 0
            for episode in range(episodes):
                env = AirCombat1v1Env(
                    opponent_policy="paper_greedy", scenario_mode=scenario,
                    max_steps=1000)
                try:
                    _, _ = env.reset(seed=seed + episode)
                    for _step in range(1000):
                        action = (np.zeros(3, np.float32)
                                  if red_policy == "zero" else
                                  pursuit_action(env.red_state, env.blue_state))
                        _, _, terminated, truncated, info = env.step(action)
                        blue = env.blue_state
                        minimum_altitude = min(
                            minimum_altitude, blue["altitude"])
                        minimum_speed = min(
                            minimum_speed, blue["true_airspeed"])
                        maximum_load = max(
                            maximum_load, abs(blue["load_factor"]))
                        if terminated or truncated:
                            break
                    event = info.get("event", "timeout")
                    counts[event] += 1
                    if event == "timeout" and info["step"] == 1000:
                        full_timeouts += 1
                finally:
                    env.close()
            rows.append({
                "scenario": scenario, "red_policy": red_policy,
                "episodes": episodes,
                "blue_crash": counts["blue_crash"],
                "blue_numerical_invalid": (
                    counts["blue_numerical_invalid"]
                    + counts["draw_both_numerical_invalid"]
                    + counts["physics_exception"]),
                "red_hit": counts["red_hit"],
                "blue_hit": counts["blue_hit"],
                "timeout": counts["timeout"],
                "full_1000_step_timeouts": full_timeouts,
                "minimum_blue_altitude_m": minimum_altitude,
                "minimum_blue_speed_mps": minimum_speed,
                "maximum_blue_load_factor": maximum_load,
            })
    passed = bool(
        all(row["blue_numerical_invalid"] == 0 for row in rows)
        and all(row["blue_crash"] == 0 for row in rows
                if row["red_policy"] == "zero")
        and all(row["blue_crash"] <= row["episodes"] / 2 for row in rows
                if row["red_policy"] == "pursuit_rule"))
    return {"passed": passed, "rows": rows}


def main():
    result = run_stability()
    output = Path("aircombat_env_v1/outputs") / (
        "opponent_stability_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "stability.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with (output / "stability.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({**result, "output_dir": str(output.resolve())}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
