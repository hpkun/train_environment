#!/usr/bin/env python3
"""Detailed checkpoint-trajectory analysis for TAM paper v4 512k runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std_or_none(values: list[float], ddof: int = 0) -> float | None:
    if not values:
        return None
    if len(values) <= ddof:
        return 0.0
    return float(np.std(values, ddof=ddof))


def student_t_ci95(values: list[float]) -> list[float | None]:
    array = np.asarray(values, dtype=float)
    n = len(array)

    if n == 0:
        return [None, None]
    if n == 1:
        value = float(array[0])
        return [value, value]

    critical = {
        1: 12.706204736,
        2: 4.302652730,
        3: 3.182446305,
        4: 2.776445105,
        5: 2.570581836,
        6: 2.446911851,
        7: 2.364624252,
        8: 2.306004135,
        9: 2.262157163,
        10: 2.228138852,
    }

    df = n - 1
    t_value = critical.get(df, 1.959963985)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(n))
    half_width = t_value * standard_error
    return [mean - half_width, mean + half_width]


def parse_distribution(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed = json.loads(text)
        if parsed is None:
            return None
        return [float(item) for item in parsed]
    return None


def checkpoint_map(panel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {
        int(row["environment_steps"]): row
        for row in panel["checkpoints"]
    }
    if 0 not in rows:
        raise ValueError("Checkpoint panel does not contain step 0")
    return rows


def action_statistics(row: dict[str, Any]) -> tuple[float | None, float | None]:
    entropies: list[float] = []
    maximum_bins: list[float] = []

    for head in range(4):
        entropy = as_float(row.get(f"active_action_head_{head}_entropy"))
        if entropy is not None:
            entropies.append(entropy)

        distribution = parse_distribution(
            row.get(f"active_action_head_{head}_distribution")
        )
        if distribution:
            maximum_bins.append(max(distribution))

    return mean_or_none(entropies), max(maximum_bins) if maximum_bins else None


def combat_changes(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, float | int]:
    win_change = float(candidate["win_rate"] - reference["win_rate"])
    survival_change = float(
        candidate["red_survival_rate"]
        - reference["red_survival_rate"]
    )
    hit_change = float(
        candidate["red_hit_rate"]
        - reference["red_hit_rate"]
    )
    crash_change = float(
        candidate["red_crash_rate"]
        - reference["red_crash_rate"]
    )
    boundary_change = float(
        candidate["red_boundary_rate"]
        - reference["red_boundary_rate"]
    )

    improvement_flags = {
        "win": win_change > 0,
        "survival": survival_change > 0,
        "hit": hit_change > 0,
        "crash": crash_change < 0,
        "boundary": boundary_change < 0,
    }

    # Positive means combat outcomes improved overall.
    combat_score = (
        win_change
        + survival_change
        + hit_change
        - crash_change
        - boundary_change
    )

    return {
        "win_rate_change": win_change,
        "survival_change": survival_change,
        "hit_rate_change": hit_change,
        "crash_rate_change": crash_change,
        "boundary_rate_change": boundary_change,
        "combat_improvement_count": int(sum(improvement_flags.values())),
        "combat_direction_score": float(combat_score),
    }


def nearest_training_row(
    rows: list[dict[str, str]],
    target_step: int,
) -> dict[str, str] | None:
    candidates: list[tuple[int, dict[str, str]]] = []

    for row in rows:
        step = as_float(row.get("environment_steps"))
        if step is None:
            continue
        integer_step = int(step)
        if integer_step <= target_step:
            candidates.append((integer_step, row))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def mean_prefixed_values(
    row: dict[str, str] | None,
    prefixes: tuple[str, ...],
) -> float | None:
    if row is None:
        return None

    values: list[float] = []
    for key, value in row.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            parsed = as_float(value)
            if parsed is not None:
                values.append(parsed)

    return mean_or_none(values)


def training_snapshot(
    training_rows: list[dict[str, str]],
    step: int,
) -> dict[str, Any]:
    row = nearest_training_row(training_rows, step)
    if row is None:
        return {
            "training_snapshot_step": None,
            "training_actor_entropy": None,
            "training_explained_variance": None,
            "training_approx_kl": None,
            "training_clip_fraction": None,
        }

    return {
        "training_snapshot_step": int(float(row["environment_steps"])),
        "training_actor_entropy": mean_prefixed_values(
            row,
            ("entropy/",),
        ),
        "training_explained_variance": mean_prefixed_values(
            row,
            ("explained_variance/",),
        ),
        "training_approx_kl": mean_prefixed_values(
            row,
            ("approx_kl/",),
        ),
        "training_clip_fraction": mean_prefixed_values(
            row,
            ("clip_fraction/",),
        ),
    }


def reward_component_rows(
    training_rows: list[dict[str, str]],
    seed: int,
    step: int,
) -> list[dict[str, Any]]:
    row = nearest_training_row(training_rows, step)
    if row is None:
        return []

    result: list[dict[str, Any]] = []
    prefixes = (
        "reward_component/controlled_mean/",
        "reward_component/role/",
    )

    for key, raw in row.items():
        if not key.startswith(prefixes):
            continue
        value = as_float(raw)
        if value is None:
            continue
        result.append({
            "training_seed": seed,
            "checkpoint_steps": step,
            "training_snapshot_step": int(float(row["environment_steps"])),
            "reward_component": key,
            "value": value,
        })

    return result


def per_seed_analysis(
    run_directory: Path,
    selection_subdir: str,
    holdout_subdir: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config = read_json(run_directory / "config_snapshot.json")
    run_summary = read_json(run_directory / "summary.json")
    selection = read_json(
        run_directory
        / selection_subdir
        / "checkpoint_panel_summary.json"
    )
    holdout = read_json(
        run_directory
        / holdout_subdir
        / "checkpoint_panel_summary.json"
    )
    training_rows = read_csv(run_directory / "training.csv")

    seed = int(config["seed"])
    selection_rows = checkpoint_map(selection)
    holdout_rows = checkpoint_map(holdout)

    if set(selection_rows) != set(holdout_rows):
        raise ValueError(
            f"Selection/holdout steps differ for seed {seed}"
        )

    trained_steps = sorted(step for step in selection_rows if step > 0)
    final_step = max(trained_steps)

    selected_step = max(
        trained_steps,
        key=lambda step: selection_rows[step]["stochastic_mean_return"],
    )

    # Descriptive only. It must not be treated as independently selected.
    oracle_holdout_step = max(
        trained_steps,
        key=lambda step: holdout_rows[step]["stochastic_mean_return"],
    )

    holdout_zero = holdout_rows[0]
    holdout_random = holdout["uniform_random"]

    trajectory_rows: list[dict[str, Any]] = []
    rewards: list[dict[str, Any]] = []

    for step in sorted(holdout_rows):
        selection_row = selection_rows[step]
        holdout_row = holdout_rows[step]
        entropy, maximum_bin = action_statistics(holdout_row)
        changes = (
            combat_changes(holdout_row, holdout_zero)
            if step > 0
            else {
                "win_rate_change": 0.0,
                "survival_change": 0.0,
                "hit_rate_change": 0.0,
                "crash_rate_change": 0.0,
                "boundary_rate_change": 0.0,
                "combat_improvement_count": 0,
                "combat_direction_score": 0.0,
            }
        )

        paired_step0 = holdout_row["paired_vs_step_0"]["return"]
        paired_random = holdout_row["paired_vs_uniform_random"]["return"]

        row = {
            "training_seed": seed,
            "checkpoint_steps": step,
            "is_step_0": step == 0,
            "is_selection_chosen": step == selected_step,
            "is_final": step == final_step,
            "is_oracle_holdout_best": step == oracle_holdout_step,
            "selection_mean_return": float(
                selection_row["stochastic_mean_return"]
            ),
            "holdout_mean_return": float(
                holdout_row["stochastic_mean_return"]
            ),
            "holdout_std_return": float(
                holdout_row["stochastic_std_return"]
            ),
            "selection_to_holdout_return_change": float(
                holdout_row["stochastic_mean_return"]
                - selection_row["stochastic_mean_return"]
            ),
            "holdout_delta_vs_step_0": float(paired_step0["mean"]),
            "holdout_delta_vs_step_0_ci_low": float(
                paired_step0["bootstrap_ci95"][0]
            ),
            "holdout_delta_vs_step_0_ci_high": float(
                paired_step0["bootstrap_ci95"][1]
            ),
            "holdout_positive_fraction_vs_step_0": float(
                paired_step0["positive_fraction"]
            ),
            "holdout_delta_vs_random": float(paired_random["mean"]),
            "holdout_beats_random": bool(paired_random["mean"] > 0),
            "holdout_win_rate": float(holdout_row["win_rate"]),
            "holdout_survival_rate": float(
                holdout_row["red_survival_rate"]
            ),
            "holdout_hit_rate": float(holdout_row["red_hit_rate"]),
            "holdout_crash_rate": float(
                holdout_row["red_crash_rate"]
            ),
            "holdout_boundary_rate": float(
                holdout_row["red_boundary_rate"]
            ),
            "deterministic_winner": holdout_row[
                "deterministic_winner"
            ],
            "deterministic_return": float(
                holdout_row["deterministic_return"]
            ),
            "deterministic_crash_rate": float(
                holdout_row["deterministic_red_crash_rate"]
            ),
            "deterministic_boundary_rate": float(
                holdout_row["deterministic_red_boundary_rate"]
            ),
            "panel_mean_action_entropy": entropy,
            "panel_maximum_action_bin_probability": maximum_bin,
            "uniform_random_mean_return": float(
                holdout_random["stochastic_mean_return"]
            ),
            **changes,
            **training_snapshot(training_rows, step),
        }

        trajectory_rows.append(row)

        if step > 0:
            rewards.extend(
                reward_component_rows(training_rows, seed, step)
            )

    trained_holdout_returns = [
        holdout_rows[step]["stochastic_mean_return"]
        for step in trained_steps
    ]

    decreases = sum(
        current < previous
        for previous, current in zip(
            trained_holdout_returns,
            trained_holdout_returns[1:],
        )
    )

    monotonic_non_decreasing = all(
        current >= previous
        for previous, current in zip(
            trained_holdout_returns,
            trained_holdout_returns[1:],
        )
    )

    selected_holdout = holdout_rows[selected_step]
    final_holdout = holdout_rows[final_step]
    oracle_holdout = holdout_rows[oracle_holdout_step]

    selected_changes = combat_changes(selected_holdout, holdout_zero)
    final_changes = combat_changes(final_holdout, holdout_zero)

    runtime_valid = bool(
        run_summary.get("OPTIMIZATION_PIPELINE_ACTIVE")
        and run_summary.get("HAPPO_RUNTIME_INVARIANTS_VALID")
        and int(run_summary["actual_final_environment_steps"]) == final_step
    )

    selected_pair = selected_holdout["paired_vs_step_0"]["return"]
    final_pair = final_holdout["paired_vs_step_0"]["return"]

    seed_summary = {
        "training_seed": seed,
        "run_directory": str(run_directory),
        "runtime_valid": runtime_valid,
        "final_step": final_step,
        "selection_chosen_step": selected_step,
        "oracle_holdout_best_step_descriptive_only": oracle_holdout_step,
        "selection_chosen_holdout_return": float(
            selected_holdout["stochastic_mean_return"]
        ),
        "oracle_holdout_best_return_descriptive_only": float(
            oracle_holdout["stochastic_mean_return"]
        ),
        "final_holdout_return": float(
            final_holdout["stochastic_mean_return"]
        ),
        "step_0_holdout_return": float(
            holdout_zero["stochastic_mean_return"]
        ),
        "uniform_random_holdout_return": float(
            holdout_random["stochastic_mean_return"]
        ),
        "selected_delta_vs_step_0": float(selected_pair["mean"]),
        "selected_ci_low": float(selected_pair["bootstrap_ci95"][0]),
        "selected_ci_high": float(selected_pair["bootstrap_ci95"][1]),
        "selected_beats_random": bool(
            selected_holdout["stochastic_mean_return"]
            > holdout_random["stochastic_mean_return"]
        ),
        "selected_combat_improvement_count": int(
            selected_changes["combat_improvement_count"]
        ),
        "final_delta_vs_step_0": float(final_pair["mean"]),
        "final_ci_low": float(final_pair["bootstrap_ci95"][0]),
        "final_ci_high": float(final_pair["bootstrap_ci95"][1]),
        "final_beats_random": bool(
            final_holdout["stochastic_mean_return"]
            > holdout_random["stochastic_mean_return"]
        ),
        "final_combat_improvement_count": int(
            final_changes["combat_improvement_count"]
        ),
        "final_minus_selection_chosen_holdout": float(
            final_holdout["stochastic_mean_return"]
            - selected_holdout["stochastic_mean_return"]
        ),
        "final_minus_oracle_holdout_best": float(
            final_holdout["stochastic_mean_return"]
            - oracle_holdout["stochastic_mean_return"]
        ),
        "peak_occurs_before_final": oracle_holdout_step < final_step,
        "holdout_decrease_transition_count": int(decreases),
        "holdout_monotonic_non_decreasing": bool(
            monotonic_non_decreasing
        ),
    }

    return seed_summary, trajectory_rows, rewards


def cross_seed_rows(
    trajectory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    steps = sorted({
        int(row["checkpoint_steps"])
        for row in trajectory_rows
    })

    result: list[dict[str, Any]] = []

    for step in steps:
        rows = [
            row for row in trajectory_rows
            if int(row["checkpoint_steps"]) == step
        ]

        deltas = [
            float(row["holdout_delta_vs_step_0"])
            for row in rows
        ]
        returns = [
            float(row["holdout_mean_return"])
            for row in rows
        ]
        combat_scores = [
            float(row["combat_direction_score"])
            for row in rows
        ]
        entropies = [
            float(row["panel_mean_action_entropy"])
            for row in rows
            if row["panel_mean_action_entropy"] is not None
        ]
        maximum_bins = [
            float(row["panel_maximum_action_bin_probability"])
            for row in rows
            if row["panel_maximum_action_bin_probability"] is not None
        ]

        result.append({
            "checkpoint_steps": step,
            "seed_count": len(rows),
            "mean_holdout_return": float(np.mean(returns)),
            "std_holdout_return": float(
                np.std(returns, ddof=1)
                if len(returns) > 1 else 0.0
            ),
            "mean_delta_vs_step_0": float(np.mean(deltas)),
            "std_delta_vs_step_0": float(
                np.std(deltas, ddof=1)
                if len(deltas) > 1 else 0.0
            ),
            "delta_t_ci95_low": student_t_ci95(deltas)[0],
            "delta_t_ci95_high": student_t_ci95(deltas)[1],
            "positive_delta_seed_count": int(sum(
                value > 0 for value in deltas
            )),
            "positive_paired_ci_seed_count": int(sum(
                row["holdout_delta_vs_step_0_ci_low"] > 0
                for row in rows
            )),
            "beats_random_seed_count": int(sum(
                row["holdout_beats_random"] for row in rows
            )),
            "combat_improved_seed_count": int(sum(
                row["combat_improvement_count"] >= 2
                for row in rows
            )),
            "mean_combat_direction_score": float(
                np.mean(combat_scores)
            ),
            "mean_panel_action_entropy": mean_or_none(entropies),
            "mean_panel_maximum_action_bin_probability": (
                mean_or_none(maximum_bins)
            ),
            "maximum_action_bin_probability_any_seed": (
                max(maximum_bins) if maximum_bins else None
            ),
        })

    return result


def pearson_or_none(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) != len(x):
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_report(
    seed_summaries: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trained_cross = [
        row for row in cross_rows
        if row["checkpoint_steps"] > 0
    ]
    final_step = max(row["checkpoint_steps"] for row in trained_cross)
    final_cross = next(
        row for row in trained_cross
        if row["checkpoint_steps"] == final_step
    )
    best_cross = max(
        trained_cross,
        key=lambda row: row["mean_delta_vs_step_0"],
    )

    peak_before_final_count = sum(
        row["peak_occurs_before_final"]
        for row in seed_summaries
    )
    monotonic_count = sum(
        row["holdout_monotonic_non_decreasing"]
        for row in seed_summaries
    )
    final_below_selected_count = sum(
        row["final_minus_selection_chosen_holdout"] < 0
        for row in seed_summaries
    )
    final_below_oracle_count = sum(
        row["final_minus_oracle_holdout_best"] < 0
        for row in seed_summaries
    )

    trained_trajectory = [
        row for row in trajectory_rows
        if row["checkpoint_steps"] > 0
    ]
    return_deltas = [
        float(row["holdout_delta_vs_step_0"])
        for row in trained_trajectory
    ]
    combat_scores = [
        float(row["combat_direction_score"])
        for row in trained_trajectory
    ]

    all_runtime_valid = all(
        row["runtime_valid"] for row in seed_summaries
    )

    environment_learnability_supported = bool(
        all_runtime_valid
        and sum(row["selected_delta_vs_step_0"] > 0
                for row in seed_summaries) >= 2
    )

    final_policy_stability_supported = bool(
        all_runtime_valid
        and final_cross["positive_delta_seed_count"] >= 2
        and final_cross["beats_random_seed_count"] >= 2
        and final_cross["delta_t_ci95_low"] is not None
        and final_cross["delta_t_ci95_low"] > 0
    )

    systematic_late_degradation = bool(
        peak_before_final_count >= 2
        and final_below_oracle_count >= 2
    )

    if not all_runtime_valid:
        diagnosis = "RUNTIME_INVALID"
    elif (
        environment_learnability_supported
        and systematic_late_degradation
    ):
        diagnosis = (
            "LEARNABLE_WITH_LATE_STAGE_DEGRADATION_PATTERN"
        )
    elif (
        environment_learnability_supported
        and not final_policy_stability_supported
    ):
        diagnosis = (
            "LEARNABLE_BUT_FINAL_POLICY_STABILITY_NOT_CONFIRMED"
        )
    elif final_policy_stability_supported:
        diagnosis = "STABLE_MULTI_SEED_FINAL_IMPROVEMENT"
    else:
        diagnosis = "INSUFFICIENT_LEARNING_EVIDENCE"

    maximum_bin = max(
        (
            row["panel_maximum_action_bin_probability"]
            for row in trajectory_rows
            if row["panel_maximum_action_bin_probability"] is not None
        ),
        default=None,
    )

    return {
        "diagnosis": diagnosis,
        "all_runtime_valid": all_runtime_valid,
        "environment_learnability_supported": (
            environment_learnability_supported
        ),
        "final_policy_stability_supported": (
            final_policy_stability_supported
        ),
        "systematic_late_stage_degradation_detected": (
            systematic_late_degradation
        ),
        "seed_count": len(seed_summaries),
        "peak_before_final_seed_count": int(
            peak_before_final_count
        ),
        "final_below_selection_chosen_seed_count": int(
            final_below_selected_count
        ),
        "final_below_oracle_holdout_best_seed_count": int(
            final_below_oracle_count
        ),
        "monotonic_holdout_improvement_seed_count": int(
            monotonic_count
        ),
        "cross_seed_best_mean_checkpoint_steps": int(
            best_cross["checkpoint_steps"]
        ),
        "cross_seed_best_mean_delta_vs_step_0": float(
            best_cross["mean_delta_vs_step_0"]
        ),
        "cross_seed_best_mean_delta_t_ci95": [
            best_cross["delta_t_ci95_low"],
            best_cross["delta_t_ci95_high"],
        ],
        "final_checkpoint_steps": int(final_step),
        "final_mean_delta_vs_step_0": float(
            final_cross["mean_delta_vs_step_0"]
        ),
        "final_delta_t_ci95": [
            final_cross["delta_t_ci95_low"],
            final_cross["delta_t_ci95_high"],
        ],
        "final_positive_delta_seed_count": int(
            final_cross["positive_delta_seed_count"]
        ),
        "final_beats_random_seed_count": int(
            final_cross["beats_random_seed_count"]
        ),
        "final_combat_improved_seed_count": int(
            final_cross["combat_improved_seed_count"]
        ),
        "return_delta_combat_score_pearson": pearson_or_none(
            return_deltas,
            combat_scores,
        ),
        "maximum_active_action_bin_probability": maximum_bin,
        "complete_action_collapse_detected": bool(
            maximum_bin is not None and maximum_bin >= 0.999
        ),
        "interpretation_limits": {
            "selection_and_holdout_action_seeds_disjoint": True,
            "environment_initial_state_fixed_nominal": True,
            "table_8_generalization_tested": False,
            "algorithm": "vanilla_happo",
            "tam_tested": False,
            "stable_convergence_claim_permitted": (
                final_policy_stability_supported
            ),
        },
    }


def build_markdown(
    report: dict[str, Any],
    seed_summaries: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# TAM paper v4 512k checkpoint trajectory diagnosis",
        "",
        f"- Diagnosis: `{report['diagnosis']}`",
        f"- All runtime valid: `{report['all_runtime_valid']}`",
        f"- Environment learnability supported: "
        f"`{report['environment_learnability_supported']}`",
        f"- Final policy stability supported: "
        f"`{report['final_policy_stability_supported']}`",
        f"- Systematic late-stage degradation detected: "
        f"`{report['systematic_late_stage_degradation_detected']}`",
        f"- Peak before final: "
        f"`{report['peak_before_final_seed_count']}/{report['seed_count']}`",
        f"- Monotonic holdout improvement: "
        f"`{report['monotonic_holdout_improvement_seed_count']}/"
        f"{report['seed_count']}`",
        f"- Cross-seed best mean checkpoint: "
        f"`{report['cross_seed_best_mean_checkpoint_steps']}`",
        f"- Final checkpoint: `{report['final_checkpoint_steps']}`",
        f"- Complete action collapse: "
        f"`{report['complete_action_collapse_detected']}`",
        "",
        "## Per-seed diagnosis",
        "",
        "| Seed | Selection step | Oracle holdout step* | "
        "Selected holdout | Final holdout | Final-selected | "
        "Final-oracle | Decreases | Monotonic |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in seed_summaries:
        lines.append(
            f"| {row['training_seed']} "
            f"| {row['selection_chosen_step']} "
            f"| {row['oracle_holdout_best_step_descriptive_only']} "
            f"| {row['selection_chosen_holdout_return']:.3f} "
            f"| {row['final_holdout_return']:.3f} "
            f"| {row['final_minus_selection_chosen_holdout']:.3f} "
            f"| {row['final_minus_oracle_holdout_best']:.3f} "
            f"| {row['holdout_decrease_transition_count']} "
            f"| {row['holdout_monotonic_non_decreasing']} |"
        )

    lines.extend([
        "",
        "\\* Oracle holdout step is descriptive only and must not be used "
        "as an independently validated model-selection result.",
        "",
        "## Cross-seed checkpoint trajectory",
        "",
        "| Step | Mean holdout return | Mean delta | t CI95 | "
        "Positive seeds | Beats random | Combat improved | "
        "Mean entropy | Max bin |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])

    for row in cross_rows:
        ci = (
            f"[{row['delta_t_ci95_low']:.3f}, "
            f"{row['delta_t_ci95_high']:.3f}]"
            if row["delta_t_ci95_low"] is not None
            else "N/A"
        )
        entropy = (
            f"{row['mean_panel_action_entropy']:.4f}"
            if row["mean_panel_action_entropy"] is not None
            else "N/A"
        )
        maximum_bin = (
            f"{row['maximum_action_bin_probability_any_seed']:.4f}"
            if row["maximum_action_bin_probability_any_seed"] is not None
            else "N/A"
        )
        lines.append(
            f"| {row['checkpoint_steps']} "
            f"| {row['mean_holdout_return']:.3f} "
            f"| {row['mean_delta_vs_step_0']:.3f} "
            f"| {ci} "
            f"| {row['positive_delta_seed_count']}/3 "
            f"| {row['beats_random_seed_count']}/3 "
            f"| {row['combat_improved_seed_count']}/3 "
            f"| {entropy} "
            f"| {maximum_bin} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Selection and holdout panels use disjoint policy-action seeds.",
        "- All evaluations use the same nominal initial environment state.",
        "- A mid-training peak followed by a weaker final checkpoint indicates "
        "training instability or policy drift, not environment failure.",
        "- Return improvement without combat-score improvement indicates a "
        "possible reward-objective mismatch and requires component-level review.",
        "- This report does not validate TAM, recurrence, attention, 5v4 "
        "generalization, or Table-8 perturbation robustness.",
        "",
    ])

    return "\n".join(lines)


def create_plots(
    output: Path,
    trajectory_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped plots")
        return

    seeds = sorted({
        int(row["training_seed"])
        for row in trajectory_rows
    })

    for metric, filename, ylabel in (
        (
            "holdout_mean_return",
            "holdout_return_trajectory.png",
            "Holdout mean return",
        ),
        (
            "holdout_delta_vs_step_0",
            "holdout_delta_vs_step0_trajectory.png",
            "Paired holdout delta vs step 0",
        ),
        (
            "panel_mean_action_entropy",
            "action_entropy_trajectory.png",
            "Mean active action-head entropy",
        ),
    ):
        plt.figure(figsize=(9, 5))

        for seed in seeds:
            rows = sorted(
                (
                    row for row in trajectory_rows
                    if int(row["training_seed"]) == seed
                ),
                key=lambda row: int(row["checkpoint_steps"]),
            )
            x = [int(row["checkpoint_steps"]) for row in rows]
            y = [row[metric] for row in rows]

            valid = [
                (x_value, float(y_value))
                for x_value, y_value in zip(x, y)
                if y_value is not None
            ]
            if valid:
                plt.plot(
                    [item[0] for item in valid],
                    [item[1] for item in valid],
                    marker="o",
                    label=str(seed),
                )

        plt.xlabel("Environment steps")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True)
        plt.legend(title="Training seed")
        plt.tight_layout()
        plt.savefig(output / filename, dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-directories",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--selection-subdir",
        default="panel_selection_500k",
    )
    parser.add_argument(
        "--holdout-subdir",
        default="panel_holdout_500k",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_directories = [
        Path(path).resolve()
        for path in args.run_directories
    ]
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []

    for run_directory in run_directories:
        summary, rows, rewards = per_seed_analysis(
            run_directory,
            args.selection_subdir,
            args.holdout_subdir,
        )
        seed_summaries.append(summary)
        trajectory_rows.extend(rows)
        reward_rows.extend(rewards)

    seed_summaries.sort(key=lambda row: row["training_seed"])
    trajectory_rows.sort(
        key=lambda row: (
            row["training_seed"],
            row["checkpoint_steps"],
        )
    )

    cross_rows = cross_seed_rows(trajectory_rows)
    report = build_report(
        seed_summaries,
        trajectory_rows,
        cross_rows,
    )

    payload = {
        "summary": report,
        "seed_summaries": seed_summaries,
        "cross_seed_checkpoint_trajectory": cross_rows,
    }

    (output / "trajectory_diagnostic_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output / "trajectory_diagnostic_report.md").write_text(
        build_markdown(report, seed_summaries, cross_rows),
        encoding="utf-8",
    )

    write_csv(
        output / "checkpoint_trajectory_by_seed.csv",
        trajectory_rows,
    )
    write_csv(
        output / "checkpoint_trajectory_cross_seed.csv",
        cross_rows,
    )
    write_csv(
        output / "seed_trajectory_summary.csv",
        seed_summaries,
    )
    write_csv(
        output / "reward_component_snapshots.csv",
        reward_rows,
    )

    if not args.skip_plots:
        create_plots(output, trajectory_rows)

    print(json.dumps(report, indent=2))
    print()
    print("Detailed report:")
    print(output / "trajectory_diagnostic_report.md")
    print()
    print("Per-seed trajectory:")
    print(output / "checkpoint_trajectory_by_seed.csv")
    print()
    print("Cross-seed trajectory:")
    print(output / "checkpoint_trajectory_cross_seed.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
