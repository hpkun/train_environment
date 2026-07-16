"""Summarize the fixed three-seed Pure HAPPO sequential-v2 200K experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RUN = "hetero_3v2_pure_happo_sequential_v2_seed{seed}_200k"
CHECKPOINTS = (
    ("initial", 0, 0),
    ("25k", 25_000, 25_088),
    ("50k", 50_000, 50_176),
    ("100k", 100_000, 100_096),
    ("150k", 150_000, 150_016),
    ("200k", 200_000, 200_192),
)
INTERVALS = (
    ("0-25k", 0, 25_088),
    ("25-50k", 25_088, 50_176),
    ("50-100k", 50_176, 100_096),
    ("100-150k", 100_096, 150_016),
    ("150-200k", 150_016, 200_192),
)
COUNT_FIELDS = (
    "episodes_completed", "red_launches", "blue_launches", "red_hits", "blue_hits",
    "red_kills", "blue_kills", "red_win", "blue_win", "mutual_elimination", "timeout",
    "flight_failures", "out_of_zone_deaths", "missile_deaths", "terminated_count",
    "truncation_count", "episode_boundary_count",
)
MEAN_FIELDS = (
    "avg_role_reward_mav", "avg_role_reward_uav", "mav_dense", "mav_safety",
    "mav_support_position", "mav_shared_information", "uav_dense", "uav_flight",
    "uav_speed", "uav_angle", "uav_distance", "uav_dodge", "team_event_reward",
    "actor_loss", "entropy", "approx_kl", "approx_kl_abs", "approx_kl_mav",
    "approx_kl_uav", "approx_kl_abs_mav", "approx_kl_abs_uav",
    "final_approx_kl_abs_mav", "final_approx_kl_abs_uav", "clip_fraction_mav",
    "clip_fraction_uav", "ratio_p95_mav", "ratio_p95_uav", "ratio_p99_mav",
    "ratio_p99_uav", "policy_update_norm_mav", "policy_update_norm_uav",
    "actor_grad_norm_mav", "actor_grad_norm_uav", "final_ratio_p95_mav",
    "final_ratio_p95_uav", "final_ratio_p99_mav", "final_ratio_p99_uav",
    "factor_final_mean", "factor_final_std", "factor_final_min", "factor_final_max",
    "critic_loss", "critic_grad_norm", "critic_update_norm",
    "value_explained_variance", "value_explained_variance_old", "advantage_raw_mean",
    "advantage_raw_std", "advantage_norm_mean", "advantage_norm_std", "return_mean",
    "return_std", "action_log_std_mav_mean", "action_log_std_uav_mean",
    "action_saturation",
)
for _role in ("mav", "uav"):
    for _dimension in ("pitch", "heading", "speed"):
        MEAN_FIELDS += (
            f"{_role}_action_mean_{_dimension}", f"{_role}_action_std_{_dimension}",
            f"{_role}_action_saturation_{_dimension}",
        )


def _number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite input: {value!r}")
    return result


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval_summary(seed: int, label: str, rows: list[dict]) -> dict:
    result = {"seed": seed, "stage": label, "start_step_exclusive": 0, "end_step": 0}
    if rows:
        result["start_step_exclusive"] = int(rows[0]["total_steps"]) - 256
        result["end_step"] = int(rows[-1]["total_steps"])
    for key in COUNT_FIELDS:
        result[key] = sum(_number(row[key]) for row in rows)
    episode_count = result["episodes_completed"]
    for key in ("mav_survival", "red_alive_final", "blue_alive_final"):
        numerator = sum(_number(row[key]) * _number(row["episodes_completed"]) for row in rows)
        result[key] = numerator / episode_count if episode_count else 0.0
    for key in MEAN_FIELDS:
        values = [_number(row[key]) for row in rows]
        result[f"{key}_mean"] = mean(values) if values else 0.0
        result[f"{key}_max"] = max(values) if values else 0.0
        result[f"{key}_final"] = values[-1] if values else 0.0
    samples = sum(_number(row["red_geometry_samples"]) for row in rows)
    result["red_geometry_samples"] = samples
    for key in ("red_range_rate", "red_ata_rate", "red_ta_rate", "red_geometry_rate"):
        numerator = sum(_number(row[key]) * _number(row["red_geometry_samples"]) for row in rows)
        result[key] = numerator / samples if samples else 0.0
    result["finite"] = int(all(int(float(row["finite"])) == 1 for row in rows))
    return result


def _eval_path(run: Path, label: str, requested: int) -> Path:
    return run / ("eval_initial_20ep.json" if label == "initial"
                  else f"eval_step_{requested:06d}_20ep.json")


def _eval_summary(seed: int, label: str, requested: int, actual: int, run: Path) -> dict:
    path = _eval_path(run, label, requested)
    data = json.loads(path.read_text(encoding="utf-8"))
    if label == "initial":
        meta_path = run / "initial" / "meta.json"
    else:
        meta_path = run / "checkpoints" / f"step_{requested:06d}" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    row = {
        "seed": seed, "checkpoint": label,
        "requested_checkpoint_step": requested,
        "actual_checkpoint_step": int(meta.get("total_env_steps_actual", actual)),
    }
    for key, value in data.items():
        if key not in ("episodes", "episode_metric_std"):
            row[key] = value
    for key, value in data.get("episode_metric_std", {}).items():
        row[f"{key}_std"] = value
    return row


def _aggregate(rows: list[dict]) -> list[dict]:
    result = []
    for label, _, _ in CHECKPOINTS:
        selected = [row for row in rows if row["checkpoint"] == label]
        aggregate = {"checkpoint": label, "seed_count": len(selected)}
        numeric = [key for key, value in selected[0].items()
                   if key not in ("seed", "checkpoint", "finite")
                   and isinstance(value, (int, float))]
        for key in numeric:
            values = [float(row[key]) for row in selected]
            aggregate[f"{key}_mean"] = mean(values)
            aggregate[f"{key}_std"] = pstdev(values)
            aggregate[f"{key}_min"] = min(values)
            aggregate[f"{key}_max"] = max(values)
        aggregate["all_finite"] = int(all(row["finite"] for row in selected))
        result.append(aggregate)
    return result


def _improvements(rows: list[dict]) -> list[dict]:
    metrics = (
        "red_launches_mean", "red_hits_mean", "red_kills_mean", "red_win_rate",
        "mav_survival_rate", "red_alive_final_mean", "blue_alive_final_mean",
        "range_rate_mean", "ata_rate_mean", "ta_rate_mean", "geometry_rate_mean",
    )
    result = []
    for seed in range(3):
        selected = {row["checkpoint"]: row for row in rows if row["seed"] == seed}
        for label, _, _ in CHECKPOINTS:
            row = {"seed": seed, "checkpoint": label}
            for key in metrics:
                row[f"{key}_delta_from_initial"] = (
                    float(selected[label][key]) - float(selected["initial"][key]))
                row[f"{key}_delta_from_50k"] = (
                    float(selected[label][key]) - float(selected["50k"][key]))
            result.append(row)
    return result


def _learning_status(training: list[dict], evaluations: list[dict]) -> dict:
    reachable = any(row["red_launches"] > 0 or row["red_hits"] > 0 or row["red_kills"] > 0
                    for row in training) or any(
                        row["red_launches_mean"] > 0 or row["red_hits_mean"] > 0
                        or row["red_kills_mean"] > 0 for row in evaluations)
    final = {row["seed"]: row for row in evaluations if row["checkpoint"] == "200k"}
    initial = {row["seed"]: row for row in evaluations if row["checkpoint"] == "initial"}
    improved_attack = sum(
        final[seed]["red_launches_mean"] > initial[seed]["red_launches_mean"]
        or final[seed]["geometry_rate_mean"] > initial[seed]["geometry_rate_mean"]
        for seed in range(3))
    effective_attack = sum(
        final[seed]["red_hits_mean"] > 0 or final[seed]["red_kills_mean"] > 0
        for seed in range(3))
    if improved_attack >= 2 and effective_attack >= 2:
        conclusion = "Pure HAPPO learnability established for the formal environment"
    elif any(row["red_launches_mean"] > 0 or row["geometry_rate_mean"] > 0
             for row in evaluations):
        conclusion = "learnable but optimization is unstable"
    elif reachable:
        conclusion = "exploration-driven attack signal, deterministic policy learning not established"
    else:
        conclusion = "no attack behavior reached"
    return {
        "behavior_reachability": bool(reachable),
        "seeds_with_200k_attack_or_geometry_improvement": int(improved_attack),
        "seeds_with_200k_deterministic_hit_or_kill": int(effective_attack),
        "conclusion": conclusion,
    }


def main() -> None:
    output = ROOT / "outputs" / "hetero_3v2_pure_happo_sequential_v2_3seed_200k_summary"
    output.mkdir(parents=True, exist_ok=True)
    training = []
    evaluations = []
    for seed in range(3):
        run = ROOT / "outputs" / RUN.format(seed=seed)
        rows = [row for row in _read_csv(run / "train_log.csv")
                if row.get("record_type", "update") == "update"]
        for label, start, end in INTERVALS:
            selected = [row for row in rows if start < int(row["total_steps"]) <= end]
            training.append(_interval_summary(seed, label, selected))
        for label, requested, actual in CHECKPOINTS:
            evaluations.append(_eval_summary(seed, label, requested, actual, run))
    aggregates = _aggregate(evaluations)
    improvements = _improvements(evaluations)
    status = _learning_status(training, evaluations)
    _write_csv(output / "training_stage_summary.csv", training)
    _write_csv(output / "checkpoint_eval_summary.csv", evaluations)
    _write_csv(output / "checkpoint_eval_aggregate.csv", aggregates)
    _write_csv(output / "checkpoint_improvement_from_initial.csv", improvements)
    payload = {
        "training_stage_summary": training,
        "checkpoint_evaluation": evaluations,
        "checkpoint_aggregate": aggregates,
        "checkpoint_improvement": improvements,
        "learnability": status,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Pure HAPPO Sequential-v2 Three-seed 200K Report", "",
        "All runs use the frozen formal environment and identical hyperparameters except seed.", "",
        "## Deterministic checkpoint evaluation", "",
        "| Seed | Checkpoint | Actual | Red launch/ep | Red hit/ep | Blue alive | Geometry | MAV survival | Red win |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        lines.append(
            f"| {row['seed']} | {row['checkpoint']} | {row['actual_checkpoint_step']} | "
            f"{row['red_launches_mean']:.3f} | {row['red_hits_mean']:.3f} | "
            f"{row['blue_alive_final_mean']:.3f} | {row['geometry_rate_mean']:.5f} | "
            f"{row['mav_survival_rate']:.3f} | {row['red_win_rate']:.3f} |")
    lines += ["", "## Learnability classification", "",
              f"- Behavior reachable: {status['behavior_reachability']}",
              f"- Seeds with 200K attack/geometry improvement: "
              f"{status['seeds_with_200k_attack_or_geometry_improvement']}/3",
              f"- Seeds with deterministic 200K hit/kill: "
              f"{status['seeds_with_200k_deterministic_hit_or_kill']}/3",
              f"- Conclusion: `{status['conclusion']}`", "",
              "All interpretations above are computed from the checkpoint evaluation data."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
