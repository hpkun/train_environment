from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _event_value(row, death_penalty: float, per_kill: float, cap: float) -> float:
    death = 1.0 - float(row["mav_alive_final"])
    kills = max(float(row.get("team_kill_alive_raw", 0.0)), 0.0)
    return -death_penalty * death + min(per_kill * kills, cap)


def solve(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    complete = frame[(frame["censored"] == 0) & (frame["terminal_observed"] == 1)].copy()
    notes = []
    if complete.empty:
        return pd.DataFrame(), ["no complete environment episodes"]
    shared_kills = float(complete.get("shared_kill_raw", pd.Series([0.0])).sum())
    productive = complete[complete["blue_alive_final"] < 2]
    mav_alive = complete[complete["mav_alive_final"] > 0.5]
    mav_dead = complete[complete["mav_alive_final"] < 0.5]
    if shared_kills <= 0:
        notes.append("no MAV-shared-only kill was observed; positive shared contribution cannot be identified")
    if productive.empty:
        notes.append("no episode caused blue aircraft loss; productive-event ordering cannot be evaluated")
    if mav_alive.empty or mav_dead.empty:
        notes.append("both MAV-alive and MAV-dead terminal episodes are required for death-event ordering")
    if notes:
        return pd.DataFrame(), notes
    feasible = []
    for death_penalty in np.arange(200.0, 1000.1, 25.0):
        for per_kill in np.arange(5.0, 200.1, 5.0):
            for cap in np.arange(per_kill, 400.1, 10.0):
                events = complete.apply(lambda row: _event_value(row, death_penalty, per_kill, cap), axis=1)
                dead = events[complete["mav_alive_final"] < 0.5]
                alive = events[complete["mav_alive_final"] > 0.5]
                if not dead.empty and not alive.empty and dead.mean() >= alive.mean():
                    continue
                bad = events[(complete["blue_alive_final"] >= 2) & (complete["mav_alive_final"] < 0.5)]
                productive_events = events[complete["blue_alive_final"] < 2]
                if not bad.empty and bad.max() >= productive_events.max():
                    continue
                feasible.append({"mav_death_penalty": death_penalty,
                                 "mav_team_credit_per_kill": per_kill,
                                 "mav_team_credit_cap": cap})
    return pd.DataFrame(feasible), notes


def _select_candidates(feasible: pd.DataFrame) -> pd.DataFrame:
    if feasible.empty:
        return feasible
    mins = feasible.min(); maxs = feasible.max(); spans = (maxs - mins).replace(0, 1.0)
    normalized = (feasible - mins) / spans
    targets = {"lower_feasible_bound": 0.0, "interval_midpoint": 0.5, "upper_feasible_bound": 1.0}
    selected = []
    used = set()
    for name, target in targets.items():
        distances = ((normalized - target) ** 2).sum(axis=1)
        for idx in distances.sort_values().index:
            signature = tuple(feasible.loc[idx].tolist())
            if signature not in used:
                used.add(signature)
                row = feasible.loc[idx].to_dict(); row["candidate"] = name; selected.append(row)
                break
    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.episodes_csv)
    feasible, notes = solve(frame)
    feasible.to_csv(output / "v5_unknown_constants_feasible_points.csv", index=False)
    if feasible.empty:
        status = ("TAM_HAPPO_PAPER_FORMULA_V5_UNPUBLISHED_CONSTANTS_REQUIRE_EMPIRICAL_SELECTION"
                  if notes else "TAM_V5_UNKNOWN_CONSTANTS_INFEASIBLE")
        candidates = pd.DataFrame()
        interval = {}
    else:
        interval = {column: {"min": float(feasible[column].min()), "max": float(feasible[column].max())}
                    for column in feasible.columns}
        candidates = _select_candidates(feasible)
        candidates.to_csv(output / "v5_unknown_constants_candidates.csv", index=False)
        base = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
        for _, candidate in candidates.iterrows():
            config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
            block = config["tam_happo_paper_formula_v5"]
            block["unknown_constants"] = {
                "status": "constraint_generated_candidate",
                "mav_death_penalty": float(candidate["mav_death_penalty"]),
                "mav_team_credit_per_kill": float(candidate["mav_team_credit_per_kill"]),
                "mav_team_credit_cap": float(candidate["mav_team_credit_cap"]),
            }
            path = output / f"tam_v5_{candidate['candidate']}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        status = ("TAM_HAPPO_PAPER_FORMULA_V5_UNPUBLISHED_CONSTANTS_REQUIRE_EMPIRICAL_SELECTION"
                  if notes else "TAM_V5_UNKNOWN_CONSTANTS_FEASIBLE")
    payload = {"status": status, "complete_episode_count": int(((frame.censored == 0) & (frame.terminal_observed == 1)).sum()),
               "feasible_point_count": int(len(feasible)), "feasible_interval": interval,
               "notes": notes,
               "search_domain_is_project_constraint_not_paper_constant": True}
    (output / "v5_unknown_constants_solution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "v5_unknown_constants_solution.md").write_text(
        "# TAM v5 unknown constants\n\n" + f"Status: `{status}`\n\n"
        + "The grid bounds are project constraint-search bounds, not published TAM-HAPPO constants.\n\n"
        + (candidates.to_markdown(index=False) if not candidates.empty else "No feasible candidate."),
        encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
