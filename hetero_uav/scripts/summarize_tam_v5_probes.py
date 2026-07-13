from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.paper_formula_v5 import V5_TRAIN_FIELDS


READY = "TAM_HAPPO_PAPER_FORMULA_V5_READY_FOR_200K_PROBE"
NOT_READY = "TAM_HAPPO_PAPER_FORMULA_V5_NOT_READY"
INSUFFICIENT = "TAM_HAPPO_PAPER_FORMULA_V5_INSUFFICIENT_EVIDENCE"
EXPECTED_SEEDS = {0, 1, 2}


def _finite_probe_metrics(train: pd.DataFrame, episodes: pd.DataFrame) -> bool:
    train_columns = ["total_env_steps_actual", "nan_detected", *V5_TRAIN_FIELDS]
    episode_columns = [
        "identity_error_max_abs", "dead_before_sum", "true_final_j_last",
        "unique_red_launch_last", "unique_red_hit_last",
        "shared_launch_raw_sum", "shared_hit_raw_sum", "shared_kill_raw_sum",
        "direct_launch_raw_sum", "direct_hit_raw_sum", "direct_kill_raw_sum",
        "direct_and_shared_launch_raw_sum", "direct_and_shared_hit_raw_sum",
        "direct_and_shared_kill_raw_sum", "team_kill_after_mav_death_raw_sum",
    ]
    if not set(train_columns) <= set(train.columns) or not set(episode_columns) <= set(episodes.columns):
        return False
    train_values = train[train_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    episode_values = episodes[episode_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(train_values).all() and np.isfinite(episode_values).all())


def _event_counts(run: Path) -> tuple[int, int, int]:
    path = run / "rich_logs/missile_events.csv"
    if not path.exists():
        episodes_path = run / "rich_logs/episode_reward_components.csv"
        if not episodes_path.exists():
            return 0, 0, 0
        episodes = pd.read_csv(episodes_path)
        roles = episodes["role"] if "role" in episodes else pd.Series("", index=episodes.index)
        mav = episodes[roles.astype(str).eq("mav")]
        if mav.empty:
            return 0, 0, 0
        def _sum(columns: tuple[str, ...]) -> int:
            return int(sum(
                pd.to_numeric(
                    mav[column] if column in mav else pd.Series(0.0, index=mav.index),
                    errors="coerce",
                ).fillna(0).sum()
                for column in columns
            ))
        return (
            _sum(("shared_launch_raw_sum", "direct_launch_raw_sum", "direct_and_shared_launch_raw_sum")),
            _sum(("shared_hit_raw_sum", "direct_hit_raw_sum", "direct_and_shared_hit_raw_sum")),
            _sum(("shared_kill_raw_sum", "direct_kill_raw_sum", "direct_and_shared_kill_raw_sum")),
        )
    frame = pd.read_csv(path)
    if frame.empty:
        return 0, 0, 0
    shooter = frame["shooter_id"] if "shooter_id" in frame else pd.Series("", index=frame.index)
    red = frame[shooter.astype(str).str.startswith("red_")].copy()
    key = [column for column in ("episode_id", "missile_id") if column in red.columns]
    launches = red.drop_duplicates(key).shape[0] if key else 0
    reasons = red["raw_termination_reason"] if "raw_termination_reason" in red else pd.Series("", index=red.index)
    hits = red[reasons.astype(str).eq("hit")]
    hit_count = hits.drop_duplicates(key).shape[0] if key else 0
    kill_count = int(pd.to_numeric(hits.get("kill_attributed", 0), errors="coerce").fillna(0).sum()) if "kill_attributed" in hits else 0
    return int(launches), int(hit_count), kill_count


def inspect_run(run: Path, candidate: str, seed: int) -> tuple[dict, list[str]]:
    required = {
        "runner_status": run / "runner_status.json",
        "train_metrics": run / "rich_logs/train_metrics.csv",
        "episodes": run / "rich_logs/episode_reward_components.csv",
        "terminal": run / "terminal_episode_audit.csv",
        "manifest": run / "probe_manifest.json",
        "checkpoint": run / "latest/model.pt",
    }
    reasons = [f"missing_{name}" for name, path in required.items() if not path.exists()]
    row = {"candidate": candidate, "seed": seed, "run_dir": str(run)}
    if reasons:
        return row, reasons
    status = json.loads(required["runner_status"].read_text(encoding="utf-8"))
    train = pd.read_csv(required["train_metrics"])
    episodes = pd.read_csv(required["episodes"])
    terminal = pd.read_csv(required["terminal"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if train.empty or episodes.empty or terminal.empty:
        reasons.append("empty_required_csv")
        return row, reasons
    config = yaml.safe_load(Path(manifest["config"]).read_text(encoding="utf-8"))
    block = config.get("tam_happo_paper_formula_v5", {})
    invariant = (
        abs(float(block.get("global_reward_scale", -1)) - 0.005) <= 1e-12
        and block.get("target_assessment", {}).get("weights")
        == {"angle": 0.35, "distance": 0.25, "height": 0.20, "speed": 0.20}
    )
    last = train.iloc[-1]
    identity = float(pd.to_numeric(train.get("v5_identity_max_abs", 0), errors="coerce").fillna(0).max())
    launches, hits, kills = _event_counts(run)
    dead_after_credit = float(pd.to_numeric(episodes.get("team_kill_after_mav_death_raw_sum", 0), errors="coerce").fillna(0).sum())
    dead_before_total = float(pd.to_numeric(episodes.get("dead_before_sum", 0), errors="coerce").fillna(0).sum())
    launch_gate_path = run / "rich_logs/launch_gate_diagnostics.csv"
    launch_gate_available = launch_gate_path.exists() and launch_gate_path.stat().st_size > 0
    geometry_rate = np.nan
    if launch_gate_available:
        gates = pd.read_csv(launch_gate_path)
        if not gates.empty and "any_geometry_pass" in gates:
            geometry_rate = float(pd.to_numeric(gates["any_geometry_pass"], errors="coerce").fillna(0).mean())
    else:
        reasons.append("evidence_missing_launch_gate_diagnostics")
    if str(status.get("rich_log_mode", "summary")) != "full":
        reasons.append("evidence_dead_before_growth_unverifiable_summary_logging")
    row.update({
        "total_steps": int(status.get("total_env_steps_actual", last.get("total_env_steps_actual", 0))),
        "runner_normal": bool(status.get("runner_completed_normally", status.get("status") == "normal")),
        "finite": _finite_probe_metrics(train, episodes), "identity_max": identity,
        "paper_contract": invariant, "episodes": len(terminal),
        "avg_return": float(pd.to_numeric(
            episodes["episode_return"] if "episode_return" in episodes
            else pd.Series(np.nan, index=episodes.index), errors="coerce").mean()),
        "timeout_rate": float((terminal.get("end_reason", "").astype(str) == "timeout").mean()),
        "mav_survival": float(pd.to_numeric(terminal.get("mav_alive_env", 0), errors="coerce").mean()),
        "blue_loss": float((2.0 - pd.to_numeric(terminal.get("blue_alive_env", 2), errors="coerce")).sum()),
        "red_launch": launches, "red_hit": hits, "red_kill": kills,
        "uav_angle": float(pd.to_numeric(train.get("v5_uav_angle_mean", 0), errors="coerce").mean()),
        "uav_distance": float(pd.to_numeric(train.get("v5_uav_distance_mean", 0), errors="coerce").mean()),
        "mav_safety": float(pd.to_numeric(train.get("v5_mav_safety_mean", 0), errors="coerce").mean()),
        "geometry_rate": geometry_rate, "launch_gate_available": launch_gate_available,
        "dead_after_credit": dead_after_credit, "dead_before_records": dead_before_total,
    })
    if not row["runner_normal"]: reasons.append("runner_not_normal")
    if row["total_steps"] < 20480: reasons.append("insufficient_steps")
    if not row["finite"]: reasons.append("nonfinite_metrics")
    if identity > 1e-8: reasons.append("identity_violation")
    if not invariant: reasons.append("paper_contract_changed")
    if not (0 <= kills <= hits <= launches): reasons.append("event_count_inconsistent")
    if dead_after_credit != 0: reasons.append("team_credit_after_mav_death")
    return row, reasons


def readiness(per_seed: pd.DataFrame, expected_candidates: list[str], baseline: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    if not expected_candidates:
        return INSUFFICIENT, ["no_generated_candidates"]
    for candidate in expected_candidates:
        seeds = set(per_seed.loc[per_seed.candidate == candidate, "seed"].astype(int).tolist())
        if seeds != EXPECTED_SEEDS:
            reasons.append(f"{candidate}:missing_expected_seeds")
    if reasons or baseline.empty:
        if baseline.empty: reasons.append("matched_v3_baseline_missing")
        return INSUFFICIENT, reasons
    baseline_required = {"seed", "red_launch", "blue_loss", "uav_angle", "uav_distance", "geometry_rate"}
    if not baseline_required <= set(baseline.columns):
        missing = sorted(baseline_required - set(baseline.columns))
        return INSUFFICIENT, [f"matched_v3_baseline_missing_columns:{','.join(missing)}"]
    baseline_numeric = baseline[["red_launch", "blue_loss", "uav_angle", "uav_distance", "geometry_rate"]]
    if not np.isfinite(baseline_numeric.to_numpy(dtype=float)).all():
        return INSUFFICIENT, ["matched_v3_baseline_has_unavailable_required_evidence"]
    if "evidence_pass" in per_seed and not bool(per_seed["evidence_pass"].all()):
        return INSUFFICIENT, ["probe_launch_gate_or_dead_before_evidence_incomplete"]
    technical = per_seed.get("technical_pass", pd.Series(False, index=per_seed.index)).all()
    if not technical:
        reasons.append("technical_check_failed")
        return NOT_READY, reasons
    base = baseline.set_index("seed")
    candidate_pass = False
    for candidate, group in per_seed.groupby("candidate"):
        improvements = 0
        positive_signal = 0
        for _, row in group.iterrows():
            if int(row.seed) not in base.index:
                continue
            ref = base.loc[int(row.seed)]
            behavior = (row.red_launch > ref.red_launch) or (row.blue_loss > ref.blue_loss)
            geometry = row.get("geometry_rate", 0) > ref.get("geometry_rate", 0)
            angle_or_distance = (
                (row.uav_angle > ref.uav_angle and row.uav_distance >= ref.uav_distance - 0.05)
                or (row.uav_distance > ref.uav_distance and row.uav_angle >= ref.uav_angle - 0.05)
            )
            if behavior or geometry: positive_signal += 1
            if angle_or_distance: improvements += 1
        if positive_signal >= 2 and improvements >= 2 and not (
            (group.red_launch == 0).all() and (group.blue_loss == 0).all()
        ):
            candidate_pass = True
    if not candidate_pass:
        reasons.append("no_candidate_has_2_of_3_seed_consistent_behavior_and_geometry_signal")
        return NOT_READY, reasons
    return READY, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--baseline-csv", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    calibration = Path(args.calibration_dir)
    candidates_path = calibration / "v5_unknown_constants_candidates.csv"
    expected = pd.read_csv(candidates_path).candidate.astype(str).tolist() if candidates_path.exists() else []
    rows, failures = [], []
    for candidate in expected:
        for seed in sorted(EXPECTED_SEEDS):
            run = Path(args.outputs_root) / f"tam_v5_{candidate}_20k_probe_s{seed}"
            retry = Path(args.outputs_root) / f"tam_v5_{candidate}_20k_probe_s{seed}_retry"
            if retry.exists():
                retry_status = retry / "runner_status.json"
                if retry_status.exists() and json.loads(retry_status.read_text(encoding="utf-8")).get("runner_completed_normally"):
                    run = retry
            row, reasons = inspect_run(run, candidate, seed)
            evidence_reasons = [reason for reason in reasons if reason.startswith("evidence_")]
            technical_reasons = [reason for reason in reasons if not reason.startswith("evidence_")]
            row["technical_pass"] = not technical_reasons
            row["evidence_pass"] = not evidence_reasons
            rows.append(row)
            failures.extend(f"{candidate}/s{seed}:{reason}" for reason in reasons)
    data = pd.DataFrame(rows)
    baseline = pd.read_csv(args.baseline_csv) if args.baseline_csv and Path(args.baseline_csv).exists() else pd.DataFrame()
    status, readiness_failures = readiness(data, expected, baseline)
    failures.extend(readiness_failures)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "v5_probe_per_seed.csv", index=False)
    baseline.to_csv(output / "v5_matched_v3_baseline.csv", index=False)
    numeric = [column for column in data.columns if column not in {"candidate", "run_dir"}]
    summary = data.groupby("candidate")[numeric].agg(["mean", "std"]) if not data.empty else pd.DataFrame()
    summary.to_csv(output / "v5_probe_mean_std.csv")
    funnel_columns = [column for column in data.columns if "gate" in column or column in {
        "candidate", "seed", "run_dir", "red_launch", "red_hit", "red_kill", "geometry_rate",
    }]
    data[funnel_columns].to_csv(output / "v5_probe_launch_gate_funnel.csv", index=False)
    outcome_rows = []
    contribution_rows = []
    for _, probe in data.iterrows():
        run = Path(probe["run_dir"])
        terminal_path = run / "terminal_episode_audit.csv"
        episode_path = run / "rich_logs/episode_reward_components.csv"
        if not terminal_path.exists() or not episode_path.exists():
            continue
        terminal = pd.read_csv(terminal_path)
        episode = pd.read_csv(episode_path)
        if "episode_id" in terminal and "episode_id" in episode:
            team_return = episode.groupby("episode_id")["episode_return"].mean()
            for outcome, group in terminal.groupby("winner"):
                ids = pd.to_numeric(group["episode_id"], errors="coerce").dropna().astype(int)
                values = team_return.reindex(ids).dropna()
                outcome_rows.append({"candidate": probe.candidate, "seed": probe.seed,
                                     "outcome": outcome, "episodes": len(group),
                                     "mean_team_return": values.mean() if not values.empty else np.nan})
        v5_columns = [column for column in episode.columns if column.endswith(("_raw_sum", "_scaled_total_sum"))]
        for column in v5_columns:
            values = pd.to_numeric(episode[column], errors="coerce").dropna()
            contribution_rows.append({"candidate": probe.candidate, "seed": probe.seed,
                                      "component": column, "sum": values.sum(),
                                      "mean": values.mean() if not values.empty else np.nan})
    pd.DataFrame(outcome_rows).to_csv(output / "v5_probe_outcome_conditioned_return.csv", index=False)
    pd.DataFrame(contribution_rows).to_csv(output / "v5_probe_reward_component_contributions.csv", index=False)
    checklist = {"status": status, "expected_candidates": expected, "expected_runs": len(expected) * 3,
                 "observed_runs": len(data), "failures": failures}
    (output / "v5_readiness_checklist.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    (output / "v5_probe_summary.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    (output / "v5_probe_summary.md").write_text(
        "# TAM v5 probe summary\n\n" + f"`{status}`\n\n"
        + (summary.to_markdown() if not summary.empty else "No complete probe evidence.")
        + "\n\n## Failures\n\n" + "\n".join(f"- {item}" for item in failures), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
