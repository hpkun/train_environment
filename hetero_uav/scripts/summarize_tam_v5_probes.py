from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


READY = "TAM_HAPPO_PAPER_FORMULA_V5_READY_FOR_200K_PROBE"
NOT_READY = "TAM_HAPPO_PAPER_FORMULA_V5_NOT_READY"
INSUFFICIENT = "TAM_HAPPO_PAPER_FORMULA_V5_INSUFFICIENT_EVIDENCE"
EXPECTED_SEEDS = {0, 1, 2}


def _finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all())


def _event_counts(run: Path) -> tuple[int, int, int]:
    path = run / "rich_logs/missile_events.csv"
    if not path.exists():
        return 0, 0, 0
    frame = pd.read_csv(path)
    if frame.empty:
        return 0, 0, 0
    red = frame[frame.get("shooter_id", "").astype(str).str.startswith("red_")].copy()
    key = [column for column in ("episode_id", "missile_id") if column in red.columns]
    launches = red.drop_duplicates(key).shape[0] if key else 0
    hits = red[red.get("raw_termination_reason", "").astype(str).eq("hit")]
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
    row.update({
        "total_steps": int(status.get("total_env_steps_actual", last.get("total_env_steps_actual", 0))),
        "runner_normal": bool(status.get("runner_completed_normally", status.get("status") == "normal")),
        "finite": _finite_frame(train) and _finite_frame(episodes), "identity_max": identity,
        "paper_contract": invariant, "episodes": len(terminal),
        "avg_return": float(pd.to_numeric(terminal.get("episode_return", np.nan), errors="coerce").mean()),
        "timeout_rate": float((terminal.get("end_reason", "").astype(str) == "timeout").mean()),
        "mav_survival": float(pd.to_numeric(terminal.get("mav_alive_env", 0), errors="coerce").mean()),
        "blue_loss": float((2.0 - pd.to_numeric(terminal.get("blue_alive_env", 2), errors="coerce")).sum()),
        "red_launch": launches, "red_hit": hits, "red_kill": kills,
        "uav_angle": float(pd.to_numeric(train.get("v5_uav_angle_mean", 0), errors="coerce").mean()),
        "uav_distance": float(pd.to_numeric(train.get("v5_uav_distance_mean", 0), errors="coerce").mean()),
        "mav_safety": float(pd.to_numeric(train.get("v5_mav_safety_mean", 0), errors="coerce").mean()),
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
            row, reasons = inspect_run(run, candidate, seed)
            row["technical_pass"] = not reasons
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
