"""Evaluate zero, random, and pursuit baselines before PPO training."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aircombat_env_v1.evaluation import evaluate_policy


def run_baselines(episodes=20, seed=1):
    rows = []
    for scenario in (
            "fixed_tail_chase", "randomized_tail_chase", "offset_tail_chase"):
        for policy in ("zero", "random", "pursuit_rule"):
            rows.append(evaluate_policy(
                policy, episodes, scenario, "paper_greedy", seed))
    lookup = {(row["scenario"], row["policy"]): row for row in rows}
    fixed = lookup[("fixed_tail_chase", "pursuit_rule")]
    randomized = lookup[("randomized_tail_chase", "pursuit_rule")]
    passed = bool(
        fixed["red_hit_rate"] >= 0.9
        and randomized["red_hit_rate"] >= 0.7
        and all(row["invalid_rate"] == 0.0 for row in rows)
        and all(row["blue_crash_rate"] < 0.5 for row in rows))
    return {"passed": passed, "rows": rows}


def main():
    result = run_baselines()
    output = Path("aircombat_env_v1/outputs") / (
        "rule_baselines_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "baselines.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    with (output / "baselines.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({**result, "output_dir": str(output.resolve())}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
