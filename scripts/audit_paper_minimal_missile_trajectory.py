"""Per-physics-frame causal missile audit for paper_minimal_3v3_v1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_minimal_3v3_spec import PAPER_MINIMAL_ENVIRONMENT_PROFILE
from my_uav_env.env import UavCombatEnv
from scripts.audit_paper_minimal_3v3 import _red_rule_actions


VECTOR_FIELDS = (
    "missile_position", "target_position", "missile_velocity", "target_velocity")


def _percentiles(values) -> dict:
    finite = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
    }


def _termination_reason(rows: list[dict]) -> str:
    return next((str(row["termination_reason"]) for row in reversed(rows)
                 if row.get("termination_reason")), "in_flight")


def _causal_class(rows: list[dict]) -> str:
    reason = _termination_reason(rows)
    ever_positive = any(float(row["closing_speed_mps"]) > 1.0 for row in rows)
    minimum = min(float(row["range_m"]) for row in rows)
    final_range = float(rows[-1]["range_m"])
    if reason == "hit":
        return "hit"
    if reason == "p_hit_fail":
        return "p_hit_fail"
    if reason == "timeout":
        return "timeout"
    if reason == "target_dead":
        return "target_dead"
    if not ever_positive:
        return "never_established_positive_closing"
    if reason == "overshoot" and final_range <= minimum + 50.0:
        return "temporary_range_increase_terminated"
    if reason == "overshoot":
        return "genuine_passed_closest_approach"
    return reason


def summarize_trajectories(rows: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["missile_id"]))].append(row)
    teams = {}
    for team in ("red", "blue"):
        missiles = [records for records in grouped.values()
                    if records and records[0]["team"] == team]
        reasons = Counter(_termination_reason(records) for records in missiles)
        classes = Counter(_causal_class(records) for records in missiles)
        overshoots = [records for records in missiles
                      if _termination_reason(records) == "overshoot"]
        ever_positive = [any(float(row["closing_speed_mps"]) > 1.0
                             for row in records) for records in missiles]
        overshoot_ever_positive = [
            any(float(row["closing_speed_mps"]) > 1.0 for row in records)
            for records in overshoots]
        teams[team] = {
            "missiles": len(missiles),
            "termination_reasons": dict(reasons),
            "causal_classes": dict(classes),
            "first_frame_closing_speed_mps": _percentiles(
                records[0]["closing_speed_mps"] for records in missiles),
            "first_frame_velocity_to_los_angle_deg": _percentiles(
                records[0]["missile_velocity_to_LOS_angle_deg"]
                for records in missiles),
            "positive_closing_established_fraction": (
                float(np.mean(ever_positive)) if ever_positive else 0.0),
            "termination_minimum_range_m": _percentiles(
                min(float(row["range_m"]) for row in records)
                for records in missiles),
            "overshoot_ever_positive_closing_fraction": (
                float(np.mean(overshoot_ever_positive))
                if overshoot_ever_positive else 0.0),
            "overshoot_trigger_time_s": _percentiles(
                records[-1]["flight_time_sec"] for records in overshoots),
            "overshoot_departed_historical_minimum_50m_fraction": (
                float(np.mean([
                    float(records[-1]["range_m"])
                    > min(float(row["range_m"]) for row in records) + 50.0
                    for records in overshoots])) if overshoots else 0.0),
        }
    return {"teams": teams, "trajectory_rows": len(rows)}


def collect_trajectory_audit(seeds, max_steps: int) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for seed in (int(value) for value in seeds):
        env = UavCombatEnv(
            max_num_red=3, max_num_blue=3, max_steps=max_steps,
            environment_profile=PAPER_MINIMAL_ENVIRONMENT_PROFILE,
            blue_policy_profile="paper_minimal_fixed_pair_v1",
            suppress_jsbsim_output=True)
        try:
            obs, _ = env.reset(seed=seed)
            env.set_team_mws_enabled("red", False)
            env.set_team_mws_enabled("blue", False)

            def sink(row, *, current_seed=seed, current_env=env):
                row["seed"] = current_seed
                row["physics_frame"] = int(current_env._physics_frame)
                rows.append(row)

            env.set_missile_trajectory_sink(sink)
            for _ in range(max_steps):
                actions = _red_rule_actions(env, obs, "fixed")
                actions.update(env.blue_policy_actions(
                    {blue_id: obs[blue_id] for blue_id in env.blue_ids}))
                obs, _rewards, terminated, truncated, _info = env.step(actions)
                if all(terminated[aid] or truncated[aid] for aid in env.agent_ids):
                    break
        finally:
            env.close()
    summary = summarize_trajectories(rows)
    summary.update({"seeds": [int(seed) for seed in seeds], "max_steps": max_steps})
    return summary, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["seed", "missile_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in VECTOR_FIELDS:
                if field in output:
                    output[field] = json.dumps(output[field], separators=(",", ":"))
            writer.writerow(output)


def _markdown(summary: dict) -> str:
    lines = ["# paper_minimal_3v3_v1 missile trajectory audit"]
    for team, values in summary["teams"].items():
        lines.extend([
            "", f"## {team}", "",
            f"- missiles: {values['missiles']}",
            f"- termination reasons: `{values['termination_reasons']}`",
            f"- causal classes: `{values['causal_classes']}`",
            f"- positive closing established: "
            f"{values['positive_closing_established_fraction']:.6f}",
            f"- overshoot with prior positive closing: "
            f"{values['overshoot_ever_positive_closing_fraction']:.6f}",
            f"- overshoot departed historical minimum by 50 m: "
            f"{values['overshoot_departed_historical_minimum_50m_fraction']:.6f}",
            f"- first-frame closing speed: "
            f"`{values['first_frame_closing_speed_mps']}`",
            f"- first-frame velocity-to-LOS angle: "
            f"`{values['first_frame_velocity_to_los_angle_deg']}`",
            f"- termination minimum range: "
            f"`{values['termination_minimum_range_m']}`",
            f"- overshoot trigger time: "
            f"`{values['overshoot_trigger_time_s']}`",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", default="tmp/minimal_missile_trajectory.json")
    parser.add_argument("--csv", default="tmp/minimal_missile_trajectory.csv")
    parser.add_argument("--markdown", default="tmp/minimal_missile_trajectory.md")
    args = parser.parse_args()
    summary, rows = collect_trajectory_audit(args.seeds, args.steps)
    json_path, csv_path, markdown_path = map(
        Path, (args.json, args.csv, args.markdown))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["teams"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
