#!/usr/bin/env python3
"""Analyze independent selection and holdout panels for multi-seed 500k runs."""

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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 91260717,
    samples: int = 10000,
) -> list[float | None]:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return [None, None]

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(array),
        size=(samples, len(array)),
    )
    means = array[indices].mean(axis=1)

    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def student_t_mean_ci(values: list[float]) -> list[float | None]:
    """Two-sided 95% t interval without requiring scipy."""

    array = np.asarray(values, dtype=float)
    n = len(array)

    if n == 0:
        return [None, None]

    mean = float(array.mean())

    if n == 1:
        return [mean, mean]

    # Two-sided 95% Student-t critical values.
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
        11: 2.200985160,
        12: 2.178812830,
        13: 2.160368656,
        14: 2.144786688,
        15: 2.131449546,
        16: 2.119905299,
        17: 2.109815578,
        18: 2.100922040,
        19: 2.093024054,
        20: 2.085963447,
        21: 2.079613845,
        22: 2.073873068,
        23: 2.068657610,
        24: 2.063898562,
        25: 2.059538553,
        26: 2.055529439,
        27: 2.051830516,
        28: 2.048407142,
        29: 2.045229642,
        30: 2.042272456,
    }

    df = n - 1
    t_value = critical.get(df, 1.959963985)
    standard_error = float(array.std(ddof=1) / math.sqrt(n))
    half_width = t_value * standard_error

    return [mean - half_width, mean + half_width]


def checkpoint_map(panel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {
        int(row["environment_steps"]): row
        for row in panel["checkpoints"]
    }

    if 0 not in result:
        raise ValueError("Panel is missing reconstructed step-0 checkpoint")

    return result


def combat_improvements(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, bool]:
    return {
        "win_rate_increased": (
            candidate["win_rate"] > reference["win_rate"]
        ),
        "survival_increased": (
            candidate["red_survival_rate"]
            > reference["red_survival_rate"]
        ),
        "hit_rate_increased": (
            candidate["red_hit_rate"]
            > reference["red_hit_rate"]
        ),
        "crash_rate_decreased": (
            candidate["red_crash_rate"]
            < reference["red_crash_rate"]
        ),
        "boundary_rate_decreased": (
            candidate["red_boundary_rate"]
            < reference["red_boundary_rate"]
        ),
    }


def analyze_seed(
    run_directory: Path,
    selection_subdir: str,
    holdout_subdir: str,
    expected_steps: int,
) -> dict[str, Any]:
    config = read_json(run_directory / "config_snapshot.json")
    summary = read_json(run_directory / "summary.json")

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

    selection_checkpoints = checkpoint_map(selection)
    holdout_checkpoints = checkpoint_map(holdout)

    if set(selection_checkpoints) != set(holdout_checkpoints):
        raise ValueError(
            f"Selection/holdout checkpoint mismatch in {run_directory}"
        )

    trained_steps = sorted(
        step for step in selection_checkpoints if step > 0
    )
    if not trained_steps:
        raise ValueError(f"No trained checkpoints in {run_directory}")

    # Checkpoint selection uses only the selection panel.
    selected_step = max(
        trained_steps,
        key=lambda step: selection_checkpoints[step][
            "stochastic_mean_return"
        ],
    )

    final_step = max(trained_steps)

    selection_best = selection_checkpoints[selected_step]
    holdout_zero = holdout_checkpoints[0]
    holdout_selected = holdout_checkpoints[selected_step]
    holdout_final = holdout_checkpoints[final_step]
    holdout_random = holdout["uniform_random"]

    selected_pair = holdout_selected[
        "paired_vs_step_0"
    ]["return"]
    final_pair = holdout_final[
        "paired_vs_step_0"
    ]["return"]

    selected_vs_random = holdout_selected[
        "paired_vs_uniform_random"
    ]["return"]
    final_vs_random = holdout_final[
        "paired_vs_uniform_random"
    ]["return"]

    selected_combat = combat_improvements(
        holdout_selected,
        holdout_zero,
    )
    final_combat = combat_improvements(
        holdout_final,
        holdout_zero,
    )

    selected_deterministic_degraded = bool(
        holdout_selected["deterministic_red_crash_rate"]
        > holdout_zero["deterministic_red_crash_rate"]
        or holdout_selected["deterministic_red_boundary_rate"]
        > holdout_zero["deterministic_red_boundary_rate"]
    )

    runtime_valid = bool(
        summary.get("OPTIMIZATION_PIPELINE_ACTIVE")
        and summary.get("HAPPO_RUNTIME_INVARIANTS_VALID")
        and int(summary["actual_final_environment_steps"])
        == expected_steps
        and int(summary["expected_target_environment_steps"])
        == expected_steps
    )

    selected_holdout_confirmed = bool(
        selected_pair["mean"] > 0
        and selected_pair["bootstrap_ci95"][0] > 0
        and selected_vs_random["mean"] > 0
        and sum(selected_combat.values()) >= 2
    )

    if not runtime_valid:
        seed_verdict = "RUNTIME_INVALID"
    elif selected_holdout_confirmed:
        seed_verdict = "HOLDOUT_CONFIRMED_EARLY_SIGNAL"
    elif (
        selected_pair["mean"] > 0
        or selected_vs_random["mean"] > 0
        or sum(selected_combat.values()) > 0
    ):
        seed_verdict = "MIXED_OR_WEAK_HOLDOUT_SIGNAL"
    else:
        seed_verdict = "NO_HOLDOUT_SIGNAL"

    row: dict[str, Any] = {
        "training_seed": int(config["seed"]),
        "run_directory": str(run_directory),
        "environment_revision": config[
            "environment_fidelity_revision"
        ],
        "runtime_valid": runtime_valid,
        "actual_environment_steps": int(
            summary["actual_final_environment_steps"]
        ),
        "total_updates": int(summary["total_updates"]),
        "total_completed_episodes": int(
            summary["total_completed_episodes"]
        ),
        "selected_checkpoint_steps": selected_step,
        "final_checkpoint_steps": final_step,
        "selection_best_mean_return": float(
            selection_best["stochastic_mean_return"]
        ),
        "holdout_step_0_mean_return": float(
            holdout_zero["stochastic_mean_return"]
        ),
        "holdout_selected_mean_return": float(
            holdout_selected["stochastic_mean_return"]
        ),
        "holdout_final_mean_return": float(
            holdout_final["stochastic_mean_return"]
        ),
        "holdout_random_mean_return": float(
            holdout_random["stochastic_mean_return"]
        ),
        "holdout_selected_delta_vs_step_0": float(
            selected_pair["mean"]
        ),
        "holdout_selected_delta_ci_low": float(
            selected_pair["bootstrap_ci95"][0]
        ),
        "holdout_selected_delta_ci_high": float(
            selected_pair["bootstrap_ci95"][1]
        ),
        "holdout_selected_positive_fraction": float(
            selected_pair["positive_fraction"]
        ),
        "holdout_selected_delta_vs_random": float(
            selected_vs_random["mean"]
        ),
        "holdout_selected_beats_random": bool(
            selected_vs_random["mean"] > 0
        ),
        "holdout_selected_combat_improvement_count": int(
            sum(selected_combat.values())
        ),
        "holdout_selected_win_rate_change": float(
            holdout_selected["win_rate"]
            - holdout_zero["win_rate"]
        ),
        "holdout_selected_survival_change": float(
            holdout_selected["red_survival_rate"]
            - holdout_zero["red_survival_rate"]
        ),
        "holdout_selected_hit_rate_change": float(
            holdout_selected["red_hit_rate"]
            - holdout_zero["red_hit_rate"]
        ),
        "holdout_selected_crash_rate_change": float(
            holdout_selected["red_crash_rate"]
            - holdout_zero["red_crash_rate"]
        ),
        "holdout_selected_boundary_rate_change": float(
            holdout_selected["red_boundary_rate"]
            - holdout_zero["red_boundary_rate"]
        ),
        "holdout_selected_deterministic_return": float(
            holdout_selected["deterministic_return"]
        ),
        "holdout_step_0_deterministic_return": float(
            holdout_zero["deterministic_return"]
        ),
        "holdout_selected_deterministic_degraded": (
            selected_deterministic_degraded
        ),
        "holdout_final_delta_vs_step_0": float(
            final_pair["mean"]
        ),
        "holdout_final_delta_ci_low": float(
            final_pair["bootstrap_ci95"][0]
        ),
        "holdout_final_delta_ci_high": float(
            final_pair["bootstrap_ci95"][1]
        ),
        "holdout_final_delta_vs_random": float(
            final_vs_random["mean"]
        ),
        "holdout_final_beats_random": bool(
            final_vs_random["mean"] > 0
        ),
        "holdout_final_combat_improvement_count": int(
            sum(final_combat.values())
        ),
        "seed_verdict": seed_verdict,
    }

    for name, improved in selected_combat.items():
        row[f"selected_{name}"] = improved

    for name, improved in final_combat.items():
        row[f"final_{name}"] = improved

    return row


def build_markdown(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# TAM paper v4 2v2 512k holdout report",
        "",
        f"- Multi-seed verdict: `{aggregate['verdict']}`",
        f"- Runtime-valid seeds: "
        f"`{aggregate['runtime_valid_seed_count']}/{len(rows)}`",
        f"- Selection-chosen checkpoints with positive holdout delta: "
        f"`{aggregate['selected_positive_delta_seed_count']}/{len(rows)}`",
        f"- Selection-chosen checkpoints with positive holdout CI: "
        f"`{aggregate['selected_positive_ci_seed_count']}/{len(rows)}`",
        f"- Selection-chosen checkpoints beating random: "
        f"`{aggregate['selected_beats_random_seed_count']}/{len(rows)}`",
        f"- Selection-chosen checkpoints with at least two combat gains: "
        f"`{aggregate['selected_combat_improved_seed_count']}/{len(rows)}`",
        f"- Final checkpoints with positive holdout delta: "
        f"`{aggregate['final_positive_delta_seed_count']}/{len(rows)}`",
        "",
        "## Cross-seed return statistics",
        "",
        f"- Selected-checkpoint mean holdout delta: "
        f"`{aggregate['selected_mean_holdout_delta']:.6f}`",
        f"- Selected-checkpoint t CI95: "
        f"`{aggregate['selected_t_ci95']}`",
        f"- Selected-checkpoint bootstrap CI95: "
        f"`{aggregate['selected_bootstrap_ci95']}`",
        f"- Final-checkpoint mean holdout delta: "
        f"`{aggregate['final_mean_holdout_delta']:.6f}`",
        f"- Final-checkpoint t CI95: "
        f"`{aggregate['final_t_ci95']}`",
        f"- Final-checkpoint bootstrap CI95: "
        f"`{aggregate['final_bootstrap_ci95']}`",
        "",
        "## Per-seed results",
        "",
        "| Seed | Selected step | Step-0 | Selected holdout | "
        "Delta | CI95 | Final holdout | Final delta | Verdict |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]

    for row in rows:
        ci = (
            f"[{row['holdout_selected_delta_ci_low']:.3f}, "
            f"{row['holdout_selected_delta_ci_high']:.3f}]"
        )
        lines.append(
            f"| {row['training_seed']} "
            f"| {row['selected_checkpoint_steps']} "
            f"| {row['holdout_step_0_mean_return']:.3f} "
            f"| {row['holdout_selected_mean_return']:.3f} "
            f"| {row['holdout_selected_delta_vs_step_0']:.3f} "
            f"| {ci} "
            f"| {row['holdout_final_mean_return']:.3f} "
            f"| {row['holdout_final_delta_vs_step_0']:.3f} "
            f"| {row['seed_verdict']} |"
        )

    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Checkpoint selection and holdout evaluation use disjoint policy-action seeds.",
        "- The environment initial state remains nominal and fixed.",
        "- This validates nominal early learnability, not Table-8 perturbation generalization.",
        "- This is Vanilla HAPPO, not TAM-HAPPO.",
        "- A positive best-checkpoint result does not imply monotonic convergence.",
        "- Final-checkpoint statistics remain the primary convergence indicator.",
        "",
    ])

    return "\n".join(lines)


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
        "--expected-steps",
        type=int,
        default=512000,
    )
    parser.add_argument(
        "--output-directory",
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runs = [Path(path).resolve() for path in args.run_directories]
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = [
        analyze_seed(
            run,
            args.selection_subdir,
            args.holdout_subdir,
            args.expected_steps,
        )
        for run in runs
    ]

    selected_deltas = [
        row["holdout_selected_delta_vs_step_0"]
        for row in rows
    ]
    final_deltas = [
        row["holdout_final_delta_vs_step_0"]
        for row in rows
    ]

    runtime_count = sum(row["runtime_valid"] for row in rows)
    selected_positive = sum(
        row["holdout_selected_delta_vs_step_0"] > 0
        for row in rows
    )
    selected_ci_positive = sum(
        row["holdout_selected_delta_ci_low"] > 0
        for row in rows
    )
    selected_beats_random = sum(
        row["holdout_selected_beats_random"]
        for row in rows
    )
    selected_combat = sum(
        row["holdout_selected_combat_improvement_count"] >= 2
        for row in rows
    )
    final_positive = sum(
        row["holdout_final_delta_vs_step_0"] > 0
        for row in rows
    )
    final_ci_positive = sum(
        row["holdout_final_delta_ci_low"] > 0
        for row in rows
    )
    final_beats_random = sum(
        row["holdout_final_beats_random"]
        for row in rows
    )

    selected_t_ci = student_t_mean_ci(selected_deltas)
    final_t_ci = student_t_mean_ci(final_deltas)

    if runtime_count != len(rows):
        verdict = "MULTISEED_RUNTIME_INVALID"
    elif (
        selected_positive >= 2
        and selected_ci_positive >= 2
        and selected_beats_random >= 2
        and selected_combat >= 2
        and selected_t_ci[0] is not None
        and selected_t_ci[0] > 0
    ):
        verdict = "ROBUST_HOLDOUT_MULTI_SEED_EARLY_SIGNAL"
    elif (
        selected_positive > 0
        or selected_beats_random > 0
        or selected_combat > 0
        or final_positive > 0
    ):
        verdict = "MIXED_HOLDOUT_MULTI_SEED_SIGNAL"
    else:
        verdict = "NO_HOLDOUT_MULTI_SEED_SIGNAL"

    aggregate = {
        "seed_count": len(rows),
        "runtime_valid_seed_count": runtime_count,
        "selected_positive_delta_seed_count": selected_positive,
        "selected_positive_ci_seed_count": selected_ci_positive,
        "selected_beats_random_seed_count": selected_beats_random,
        "selected_combat_improved_seed_count": selected_combat,
        "final_positive_delta_seed_count": final_positive,
        "final_positive_ci_seed_count": final_ci_positive,
        "final_beats_random_seed_count": final_beats_random,
        "selected_mean_holdout_delta": float(
            np.mean(selected_deltas)
        ),
        "selected_std_holdout_delta": float(
            np.std(selected_deltas, ddof=1)
            if len(selected_deltas) > 1
            else 0.0
        ),
        "selected_t_ci95": selected_t_ci,
        "selected_bootstrap_ci95": bootstrap_mean_ci(
            selected_deltas,
            seed=91260717,
        ),
        "final_mean_holdout_delta": float(
            np.mean(final_deltas)
        ),
        "final_std_holdout_delta": float(
            np.std(final_deltas, ddof=1)
            if len(final_deltas) > 1
            else 0.0
        ),
        "final_t_ci95": final_t_ci,
        "final_bootstrap_ci95": bootstrap_mean_ci(
            final_deltas,
            seed=92260717,
        ),
        "verdict": verdict,
        "interpretation": {
            "environment_protocol": "paper_nominal",
            "environment_revision": "published_rules_simplified_v4",
            "algorithm": "vanilla_happo",
            "checkpoint_selection_panel_is_independent_of_holdout": True,
            "nominal_environment_learnability_only": True,
            "table_8_generalization_validated": False,
            "convergence_validated": False,
        },
    }

    payload = {
        "aggregate": aggregate,
        "seeds": rows,
    }

    (output / "holdout_multiseed_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output / "holdout_multiseed_seed_summary.csv",
        rows,
    )

    (output / "holdout_multiseed_report.md").write_text(
        build_markdown(rows, aggregate),
        encoding="utf-8",
    )

    print(json.dumps(aggregate, indent=2))
    print(
        "\nReport:",
        output / "holdout_multiseed_report.md",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
