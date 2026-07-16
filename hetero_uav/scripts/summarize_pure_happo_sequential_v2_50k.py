"""Summarize the fixed three-seed sequential-v2 50K experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RUN = "hetero_3v2_pure_happo_sequential_v2_seed{seed}_50k"
STAGES = (("10k", 10_240), ("25k", 25_088), ("50k", 50_176))
COUNT_FIELDS = (
    "episodes_completed", "red_launches", "blue_launches", "red_hits",
    "blue_hits", "red_kills", "blue_kills", "red_win", "blue_win",
    "mutual_elimination", "timeout", "terminated_count", "truncation_count",
    "episode_boundary_count", "flight_failures", "out_of_zone_deaths",
    "missile_deaths",
)
MEAN_FIELDS = (
    "avg_role_reward_mav", "avg_role_reward_uav", "mav_dense", "mav_safety",
    "mav_support_position", "mav_shared_information", "uav_dense", "uav_flight",
    "uav_speed", "uav_angle", "uav_distance", "uav_dodge", "team_event_reward",
    "actor_loss", "entropy", "critic_loss", "value_explained_variance",
    "value_explained_variance_old", "approx_kl_abs", "approx_kl_abs_mav",
    "approx_kl_abs_uav", "final_approx_kl_abs_mav", "final_approx_kl_abs_uav",
    "clip_fraction_mav", "clip_fraction_uav", "ratio_p95_mav", "ratio_p95_uav",
    "ratio_p99_mav", "ratio_p99_uav", "policy_update_norm_mav",
    "policy_update_norm_uav", "actor_grad_norm_mav", "actor_grad_norm_uav",
    "final_ratio_p95_mav", "final_ratio_p95_uav", "final_ratio_p99_mav",
    "final_ratio_p99_uav", "factor_final_mean", "factor_final_std",
    "factor_final_min", "factor_final_max", "critic_grad_norm",
    "critic_update_norm", "advantage_raw_mean", "advantage_raw_std",
    "advantage_norm_mean", "advantage_norm_std", "return_mean", "return_std",
    "action_log_std_mav_mean", "action_log_std_uav_mean", "action_saturation",
)
for _role in ("mav", "uav"):
    for _dimension in ("pitch", "heading", "speed"):
        MEAN_FIELDS += (
            f"{_role}_action_mean_{_dimension}",
            f"{_role}_action_std_{_dimension}",
            f"{_role}_action_saturation_{_dimension}",
        )


def _number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite summary input: {value!r}")
    return result


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_stage(seed: int, stage: str, rows: list[dict]) -> dict:
    result = {"seed": seed, "stage": stage, "actual_steps": int(rows[-1]["total_steps"])}
    for key in COUNT_FIELDS:
        result[key] = sum(_number(row[key]) for row in rows)
    episode_count = result["episodes_completed"]
    for key in ("mav_survival", "red_alive_final", "blue_alive_final"):
        numerator = sum(_number(row[key]) * _number(row["episodes_completed"]) for row in rows)
        result[key] = numerator / episode_count if episode_count else 0.0
    for key in MEAN_FIELDS:
        values = [_number(row[key]) for row in rows]
        result[f"{key}_mean"] = mean(values)
        if key in ("approx_kl_abs", "factor_final_max", "action_saturation"):
            result[f"{key}_max"] = max(values)
    samples = sum(_number(row["red_geometry_samples"]) for row in rows)
    result["red_geometry_samples"] = samples
    for key in ("red_range_rate", "red_ata_rate", "red_ta_rate", "red_geometry_rate"):
        numerator = sum(_number(row[key]) * _number(row["red_geometry_samples"]) for row in rows)
        result[key] = numerator / samples if samples else 0.0
    result["finite"] = int(all(int(float(row["finite"])) == 1 for row in rows))
    return result


def _eval_row(seed: int, stage: str, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {"seed": seed, "stage": stage}
    for key, value in data.items():
        if key != "episodes":
            row[key] = value
    return row


def _aggregate_eval(rows: list[dict]) -> list[dict]:
    result = []
    for stage, _ in STAGES:
        selected = [row for row in rows if row["stage"] == stage]
        aggregate = {"stage": stage, "seed_count": len(selected)}
        numeric = [key for key, value in selected[0].items()
                   if key not in ("seed", "stage", "finite") and isinstance(value, (int, float))]
        for key in numeric:
            values = [float(row[key]) for row in selected]
            aggregate[f"{key}_mean"] = mean(values)
            aggregate[f"{key}_std"] = pstdev(values)
        aggregate["all_finite"] = int(all(row["finite"] for row in selected))
        result.append(aggregate)
    return result


def main() -> None:
    output = ROOT / "outputs" / "hetero_3v2_pure_happo_sequential_v2_3seed_50k_summary"
    output.mkdir(parents=True, exist_ok=True)
    training_rows = []
    eval_rows = []
    for seed in range(3):
        run = ROOT / "outputs" / RUN.format(seed=seed)
        rows = _read_csv(run / "train_log.csv")
        for stage, actual in STAGES:
            selected = [row for row in rows if int(row["total_steps"]) <= actual]
            training_rows.append(_training_stage(seed, stage, selected))
            eval_rows.append(_eval_row(
                seed, stage, run / f"eval_step_{int(stage[:-1]) * 1000:06d}_10ep.json"))
    aggregates = _aggregate_eval(eval_rows)
    _write_csv(output / "training_stage_summary.csv", training_rows)
    _write_csv(output / "checkpoint_eval_summary.csv", eval_rows)
    _write_csv(output / "checkpoint_eval_aggregate.csv", aggregates)
    payload = {"training": training_rows, "evaluation": eval_rows, "aggregate": aggregates}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Pure HAPPO Sequential-v2 Three-seed 50K Summary", "",
        "All runs use the frozen `hetero_3v2_pure_happo_v1` environment and "
        "`pure_happo_sequential_v2` algorithm contract. Requested 50K corresponds "
        "to 50,176 actual steps (196 complete rollouts).", "",
        "## Deterministic checkpoint evaluation", "",
        "Each checkpoint uses the same 10 episode seeds, 2000 through 2009.", "",
        "| Seed | Stage | Red launch/ep | Red hit/ep | Blue alive | Geometry | MAV survival | Red win |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_rows:
        lines.append(
            f"| {row['seed']} | {row['stage']} | {row['red_launches_mean']:.3f} | "
            f"{row['red_hits_mean']:.3f} | {row['blue_alive_final_mean']:.3f} | "
            f"{row['geometry_rate_mean']:.5f} | {row['mav_survival_rate']:.3f} | "
            f"{row['red_win_rate']:.3f} |")
    lines += ["", "## Three-seed aggregate", "",
              "| Stage | Launch mean +/- std | Hit mean +/- std | Blue alive mean +/- std | Geometry mean +/- std |",
              "|:---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(
            f"| {row['stage']} | {row['red_launches_mean_mean']:.3f} +/- "
            f"{row['red_launches_mean_std']:.3f} | {row['red_hits_mean_mean']:.3f} +/- "
            f"{row['red_hits_mean_std']:.3f} | {row['blue_alive_final_mean_mean']:.3f} +/- "
            f"{row['blue_alive_final_mean_std']:.3f} | {row['geometry_rate_mean_mean']:.5f} +/- "
            f"{row['geometry_rate_mean_std']:.5f} |")
    lines += [
        "", "## Interpretation", "",
        "- All three runs and all nine deterministic evaluations are finite.",
        "- Seed 0 improves at 25K and loses the behavior at 50K.",
        "- Seed 1 first shows deterministic launch geometry at 50K, without hits.",
        "- Seed 2 strongly improves at 25K and partially regresses at 50K.",
        "- The baseline therefore demonstrates learnable attack behavior, but not stable "
        "convergence, robust MAV survival, or consistent wins within 50K.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
