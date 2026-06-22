"""Analyse missile_events.csv from rich logs.

Separates launch rows from termination rows so that fired/hit/speed
statistics are computed from the correct row types.

Outputs summary.json and summary.csv.
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

    all_rows = []
    with open(missile_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)

    # ---- split by event_type ----
    launch_rows = [r for r in all_rows if r.get("event_type", "") == "launch"]
    term_rows = [r for r in all_rows if r.get("event_type", "") != "launch"]

    def _team_rows(rows, team):
        return [r for r in rows if r.get("owner_team", "") == team]

    red_launch = _team_rows(launch_rows, "red")
    blue_launch = _team_rows(launch_rows, "blue")
    red_term = _team_rows(term_rows, "red")
    blue_term = _team_rows(term_rows, "blue")

    red_launch_count = len(red_launch)
    blue_launch_count = len(blue_launch)
    red_term_count = len(red_term)
    blue_term_count = len(blue_term)

    # ---- termination reason counts (from term_rows only) ----
    def _reason_counts(rows):
        counts = defaultdict(int)
        for r in rows:
            reason = r.get("raw_termination_reason", "") or "unresolved"
            counts[reason] += 1
        return dict(counts)

    red_term_reasons = _reason_counts(red_term)
    blue_term_reasons = _reason_counts(blue_term)

    red_hit_count = red_term_reasons.get("hit", 0)
    blue_hit_count = blue_term_reasons.get("hit", 0)
    red_low_speed_count = red_term_reasons.get("low_speed", 0)
    blue_low_speed_count = blue_term_reasons.get("low_speed", 0)

    # ---- rates ----
    red_hit_rate_by_launch = red_hit_count / max(red_launch_count, 1)
    red_hit_rate_by_terminated = red_hit_count / max(red_term_count, 1)
    blue_hit_rate_by_launch = blue_hit_count / max(blue_launch_count, 1)
    blue_hit_rate_by_terminated = blue_hit_count / max(blue_term_count, 1)

    # ---- shooter speed by termination reason (term_rows only) ----
    def _shooter_speed(rows):
        speeds = []
        for r in rows:
            s = r.get("shooter_speed_mps", "")
            try:
                speeds.append(float(s))
            except (ValueError, TypeError):
                pass
        return speeds

    # red-only speeds
    red_hit_speeds = _shooter_speed([r for r in red_term if r.get("raw_termination_reason", "") == "hit"])
    red_low_speed_speeds = _shooter_speed([r for r in red_term if r.get("raw_termination_reason", "") == "low_speed"])

    # per-reason speeds (red term only)
    red_speed_by_reason = {}
    for reason in sorted(set(r.get("raw_termination_reason", "") or "unresolved" for r in red_term)):
        speeds_r = _shooter_speed([r for r in red_term if r.get("raw_termination_reason", "") == reason])
        if speeds_r:
            red_speed_by_reason[reason] = float(np.mean(speeds_r))

    # all-team hit speeds (for reference, explicitly labelled)
    all_hit_speeds = _shooter_speed([r for r in term_rows if r.get("raw_termination_reason", "") == "hit"])

    # ---- owner_id x raw_termination_reason (term_rows only) ----
    owner_reason = defaultdict(lambda: defaultdict(int))
    for r in term_rows:
        owner = r.get("owner_id", "")
        reason = r.get("raw_termination_reason", "") or "unresolved"
        owner_reason[owner][reason] += 1

    red_owner_reason = defaultdict(lambda: defaultdict(int))
    for r in red_term:
        owner = r.get("owner_id", "")
        reason = r.get("raw_termination_reason", "") or "unresolved"
        red_owner_reason[owner][reason] += 1

    # ---- unresolved / in_flight (launches without matching termination) ----
    launch_ids = set(r.get("missile_id", "") for r in launch_rows)
    term_ids = set(r.get("missile_id", "") for r in term_rows)
    unresolved_count = len(launch_ids - term_ids)

    # ---- build summary ----
    summary = {
        "source": str(missile_csv),
        "launch_rows": len(launch_rows),
        "termination_rows": len(term_rows),
        "unresolved_missiles": unresolved_count,
        "red": {
            "launch_count": red_launch_count,
            "term_count": red_term_count,
            "hit_count": red_hit_count,
            "low_speed_count": red_low_speed_count,
            "hit_rate_by_launch": red_hit_rate_by_launch,
            "hit_rate_by_terminated": red_hit_rate_by_terminated,
            "low_speed_rate_by_launch": red_low_speed_count / max(red_launch_count, 1),
            "term_reasons": red_term_reasons,
            "hit_shooter_speed_mean": float(np.mean(red_hit_speeds)) if red_hit_speeds else 0.0,
            "low_speed_shooter_speed_mean": float(np.mean(red_low_speed_speeds)) if red_low_speed_speeds else 0.0,
            "speed_by_reason": red_speed_by_reason,
        },
        "blue": {
            "launch_count": blue_launch_count,
            "term_count": blue_term_count,
            "hit_count": blue_hit_count,
            "low_speed_count": blue_low_speed_count,
            "hit_rate_by_launch": blue_hit_rate_by_launch,
            "hit_rate_by_terminated": blue_hit_rate_by_terminated,
            "term_reasons": blue_term_reasons,
        },
        "all_team_hit_shooter_speed_mean": float(np.mean(all_hit_speeds)) if all_hit_speeds else 0.0,
        "owner_reason_from_term_rows": {
            owner: dict(reasons) for owner, reasons in sorted(owner_reason.items())
        },
        "red_owner_reason_from_term_rows": {
            owner: dict(reasons) for owner, reasons in sorted(red_owner_reason.items())
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
            ("launch_rows", len(launch_rows)),
            ("termination_rows", len(term_rows)),
            ("unresolved_missiles", unresolved_count),
            ("red_launch_count", red_launch_count),
            ("red_term_count", red_term_count),
            ("red_hit_count", red_hit_count),
            ("red_low_speed_count", red_low_speed_count),
            ("red_hit_rate_by_launch", red_hit_rate_by_launch),
            ("red_hit_rate_by_terminated", red_hit_rate_by_terminated),
            ("red_low_speed_rate_by_launch", red_low_speed_count / max(red_launch_count, 1)),
            ("blue_launch_count", blue_launch_count),
            ("blue_term_count", blue_term_count),
            ("blue_hit_count", blue_hit_count),
            ("blue_low_speed_count", blue_low_speed_count),
            ("blue_hit_rate_by_launch", blue_hit_rate_by_launch),
            ("red_hit_shooter_speed_mean", summary["red"]["hit_shooter_speed_mean"]),
            ("red_low_speed_shooter_speed_mean", summary["red"]["low_speed_shooter_speed_mean"]),
            ("all_team_hit_shooter_speed_mean", summary["all_team_hit_shooter_speed_mean"]),
        ]:
            w.writerow([key, str(val)])
        # per-owner red reasons
        w.writerow([])
        w.writerow(["owner_id", "low_speed", "hit", "other"])
        for owner in sorted(red_owner_reason):
            reasons = red_owner_reason[owner]
            ls = reasons.get("low_speed", 0)
            hit = reasons.get("hit", 0)
            other = sum(v for k, v in reasons.items() if k not in ("low_speed", "hit"))
            w.writerow([owner, ls, hit, other])
    print(f"Saved: {csv_path}", flush=True)

    # -- Print --
    print()
    print(f"Launch rows:   {len(launch_rows)}  (unresolved: {unresolved_count})")
    print(f"Termination rows: {len(term_rows)}")
    print(f"Red:  launched={red_launch_count}  terminated={red_term_count}  "
          f"hits={red_hit_count} (by_launch={red_hit_rate_by_launch:.3f}, by_term={red_hit_rate_by_terminated:.3f})  "
          f"low_speed={red_low_speed_count}")
    print(f"Blue: launched={blue_launch_count}  terminated={blue_term_count}  "
          f"hits={blue_hit_count} (by_launch={blue_hit_rate_by_launch:.3f})")
    print(f"Red hit shooter speed mean:       {summary['red']['hit_shooter_speed_mean']:.1f} m/s")
    print(f"Red low_speed shooter speed mean: {summary['red']['low_speed_shooter_speed_mean']:.1f} m/s")
    print(f"All-team hit shooter speed mean:  {summary['all_team_hit_shooter_speed_mean']:.1f} m/s")
    print()
    print("Red owner low_speed:")
    for owner in sorted(red_owner_reason):
        ls = red_owner_reason[owner].get("low_speed", 0)
        if ls > 0:
            print(f"  {owner}: {ls}")
    print()
    print("Red speed by termination reason:")
    for reason, spd in sorted(red_speed_by_reason.items()):
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
