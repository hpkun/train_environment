"""Compare complete-trajectory reward ordering under common formal seeds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from hetero_3v2_v2_audit_common import (
    AUDIT_DIR, V1_CONFIG, mean_ci, paired_effect, run_episode, write_csv, write_json,
)


CONTROLLERS = ("rule", "random", "hold", "safe", "escape", "ata", "ta", "range", "geometry")


def _run_task(task):
    config, seed, controller, max_steps = task
    return run_episode(config, seed, controller, max_steps, perturb=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V1_CONFIG)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    tasks = [
        (args.config, seed, controller, args.max_steps)
        for seed in range(args.seeds)
        for controller in CONTROLLERS
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_task, tasks))
    else:
        rows = [_run_task(task) for task in tasks]
    output = AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "reward_policy_comparison.csv", rows)
    by_controller = {
        controller: [row for row in rows if row["controller"] == controller]
        for controller in CONTROLLERS
    }
    summary = {
        controller: {
            field: mean_ci([row[field] for row in subset])
            for field in (
                "team_return", "dense_return", "event_return", "mav_return",
                "uav_return", "geometry_rate", "red_launches", "red_hits",
                "blue_alive_final", "shared_positive_step_rate")
        }
        for controller, subset in by_controller.items()
    }
    summary["paired_effects"] = {}
    for left, right in (
            ("rule", "random"), ("rule", "safe"), ("geometry", "hold"),
            ("ta", "hold"), ("ata", "hold"), ("range", "hold")):
        summary["paired_effects"][f"{left}_minus_{right}"] = paired_effect(
            [row["team_return"] for row in by_controller[left]],
            [row["team_return"] for row in by_controller[right]])
    write_json(output / "reward_learnability_summary.json", summary)
    print(output / "reward_learnability_summary.json")


if __name__ == "__main__":
    main()
