"""Audit MAV survival, marginal shared information, and timing."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from hetero_3v2_v2_audit_common import (
    AUDIT_DIR, V1_CONFIG, mean_ci, run_episode, write_csv, write_json,
)


def _run_task(task):
    config, seed, max_steps = task
    return run_episode(config, seed, "rule", max_steps, True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V1_CONFIG)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    tasks = [(args.config, seed, args.max_steps) for seed in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_task, tasks))
    else:
        rows = [_run_task(task) for task in tasks]
    output = AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "mav_role_episodes.csv", rows)
    death_steps = [row["first_mav_death_step"] for row in rows
                   if row["first_mav_death_step"] is not None]
    summary = {
        "episodes": len(rows),
        "mav_survival_rate": sum(row["mav_alive_final"] for row in rows) / len(rows),
        "mav_death_step": mean_ci(death_steps),
        "shared_positive_step_rate": mean_ci(
            [row["shared_positive_step_rate"] for row in rows]),
        "mav_return": mean_ci([row["mav_return"] for row in rows]),
        "geometry_rate": mean_ci([row["geometry_rate"] for row in rows]),
        "red_launch_rate": sum(row["red_launches"] > 0 for row in rows) / len(rows),
        "red_hit_rate": sum(row["red_hits"] > 0 for row in rows) / len(rows),
        "note": (
            "shared_positive_step_rate measures absolute shared-only coverage, "
            "not its change; a high near-constant rate is evidence that v1 "
            "rewards state occupancy rather than marginal information gain."
        ),
    }
    write_json(output / "mav_role_summary.json", summary)
    print(output / "mav_role_summary.json")


if __name__ == "__main__":
    main()
