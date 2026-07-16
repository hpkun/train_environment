"""Audit whether formal reset states admit timely red attack geometry."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from hetero_3v2_v2_audit_common import (
    AUDIT_DIR, V1_CONFIG, mean_ci, run_episode, write_csv, write_json,
)


def _run_task(task):
    config, seed, controller, max_steps = task
    return run_episode(config, seed, controller, max_steps, perturb=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V1_CONFIG)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    tasks = [
        (args.config, seed, controller, args.max_steps)
        for controller in ("rule", "geometry", "hold")
        for seed in range(args.seeds)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_task, tasks))
    else:
        rows = [_run_task(task) for task in tasks]
    output = AUDIT_DIR if args.output_dir == str(AUDIT_DIR) else AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "reachability.csv", rows)
    summary = {}
    for controller in ("rule", "geometry", "hold"):
        subset = [row for row in rows if row["controller"] == controller]
        geometry_steps = [row["first_red_geometry_step"] for row in subset
                          if row["first_red_geometry_step"] is not None]
        before_death = []
        for row in subset:
            deaths = [
                value for value in (
                    row["first_mav_death_step"], row["first_red_1_death_step"],
                    row["first_red_2_death_step"])
                if value is not None
            ]
            if (row["first_red_geometry_step"] is not None
                    and (not deaths or row["first_red_geometry_step"] < min(deaths))):
                before_death.append(row)
        summary[controller] = {
            "episodes": len(subset),
            "outcomes": dict(Counter(row["outcome"] for row in subset)),
            "red_geometry_reached_rate": sum(
                row["first_red_geometry_step"] is not None for row in subset) / len(subset),
            "red_geometry_before_first_red_death_rate": len(before_death) / len(subset),
            "red_launch_rate": sum(row["red_launches"] > 0 for row in subset) / len(subset),
            "red_hit_rate": sum(row["red_hits"] > 0 for row in subset) / len(subset),
            "blue_launches_first_rate": sum(
                row["first_blue_launch_step"] is not None and (
                    row["first_red_launch_step"] is None
                    or row["first_blue_launch_step"] < row["first_red_launch_step"])
                for row in subset) / len(subset),
            "first_red_geometry_step": mean_ci(geometry_steps),
            "mav_survival_rate": sum(row["mav_alive_final"] for row in subset) / len(subset),
            "geometry_rate": mean_ci([row["geometry_rate"] for row in subset]),
        }
    write_json(output / "timing_summary.json", summary)
    print(output / "timing_summary.json")


if __name__ == "__main__":
    main()
