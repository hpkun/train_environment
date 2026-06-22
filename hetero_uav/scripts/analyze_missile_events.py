"""Analyse missile_events.csv from rich logs.

Outputs summary.json and summary.csv with:
  - owner_id x raw_termination_reason pivot
  - shooter_speed_mps by termination reason
  - owner_id low_speed counts
  - hit vs low_speed shooter_speed_mps means
  - red/blue hit rates
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def analyse(missile_csv: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(missile_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # -- owner_id x termination reason --
    reason_by_owner = defaultdict(lambda: defaultdict(int))
    speed_by_reason = defaultdict(list)
    # -- red/blue aggregates --
    red_fired = red_hits = red_low_speed = 0
    blue_fired = blue_hits = blue_low_speed = 0
    hit_speeds = []
    low_speed_speeds = []
    owner_low_speed = defaultdict(int)
    owner_hits = defaultdict(int)

    for r in rows:
        team = r.get("owner_team", "")
        reason = r.get("raw_termination_reason", "") or "in_flight"
        owner = r.get("owner_id", "")
        shooter_speed = float(r.get("shooter_speed_mps", 0) or 0)
        reason_by_owner[owner][reason] += 1
        speed_by_reason[reason].append(shooter_speed)

        if team == "red":
            red_fired += 1
            if reason == "hit":
                red_hits += 1
                hit_speeds.append(shooter_speed)
                owner_hits[owner] += 1
            if reason == "low_speed":
                red_low_speed += 1
                low_speed_speeds.append(shooter_speed)
                owner_low_speed[owner] += 1
        elif team == "blue":
            blue_fired += 1
            if reason == "hit":
                blue_hits += 1
                hit_speeds.append(shooter_speed)

    # -- summary dict --
    summary = {
        "source": str(missile_csv),
        "total_missiles": len(rows),
        "red": {
            "fired": red_fired,
            "hits": red_hits,
            "hit_rate": red_hits / max(red_fired, 1),
            "low_speed": red_low_speed,
            "low_speed_rate": red_low_speed / max(red_fired, 1),
        },
        "blue": {
            "fired": blue_fired,
            "hits": blue_hits,
            "hit_rate": blue_hits / max(blue_fired, 1),
            "low_speed": blue_low_speed,
            "low_speed_rate": blue_low_speed / max(blue_fired, 1),
        },
        "hit_shooter_speed_mean": float(np.mean(hit_speeds)) if hit_speeds else 0.0,
        "low_speed_shooter_speed_mean": float(np.mean(low_speed_speeds)) if low_speed_speeds else 0.0,
        "owner_low_speed_counts": dict(owner_low_speed),
        "owner_hit_counts": dict(owner_hits),
        "reason_by_owner": {
            owner: dict(reasons) for owner, reasons in sorted(reason_by_owner.items())
        },
        "speed_by_reason": {
            reason: float(np.mean(speeds)) for reason, speeds in sorted(speed_by_reason.items())
        },
    }

    # -- JSON --
    json_path = os.path.join(output_dir, "missile_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}", flush=True)

    # -- CSV --
    csv_path = os.path.join(output_dir, "missile_analysis_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for key, val in [
            ("total_missiles", summary["total_missiles"]),
            ("red_fired", red_fired),
            ("red_hits", red_hits),
            ("red_hit_rate", summary["red"]["hit_rate"]),
            ("red_low_speed", red_low_speed),
            ("red_low_speed_rate", summary["red"]["low_speed_rate"]),
            ("blue_fired", blue_fired),
            ("blue_hits", blue_hits),
            ("blue_hit_rate", summary["blue"]["hit_rate"]),
            ("blue_low_speed", blue_low_speed),
            ("blue_low_speed_rate", summary["blue"]["low_speed_rate"]),
            ("hit_shooter_speed_mean", summary["hit_shooter_speed_mean"]),
            ("low_speed_shooter_speed_mean", summary["low_speed_shooter_speed_mean"]),
        ]:
            w.writerow([key, str(val)])
        # per-owner low_speed
        w.writerow([])
        w.writerow(["owner_id", "low_speed_count", "hit_count"])
        for owner in sorted(set(list(owner_low_speed) + list(owner_hits))):
            w.writerow([owner, owner_low_speed.get(owner, 0), owner_hits.get(owner, 0)])
    print(f"Saved: {csv_path}", flush=True)

    # -- Print summary --
    print()
    print(f"Total missiles: {summary['total_missiles']}")
    print(f"Red:  fired={red_fired}  hits={red_hits} (rate={summary['red']['hit_rate']:.3f})  "
          f"low_speed={red_low_speed} (rate={summary['red']['low_speed_rate']:.3f})")
    print(f"Blue: fired={blue_fired}  hits={blue_hits} (rate={summary['blue']['hit_rate']:.3f})  "
          f"low_speed={blue_low_speed}")
    print(f"Hit shooter speed mean:  {summary['hit_shooter_speed_mean']:.1f} m/s")
    print(f"Low_speed shooter speed: {summary['low_speed_shooter_speed_mean']:.1f} m/s")
    print()
    print("Owner low_speed:")
    for owner, count in sorted(owner_low_speed.items()):
        print(f"  {owner}: {count}")
    print()
    print("Speed by termination reason:")
    for reason, spd in summary["speed_by_reason"].items():
        print(f"  {reason}: {spd:.1f} m/s")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missile-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyse(args.missile_csv, args.output_dir)


if __name__ == "__main__":
    main()
