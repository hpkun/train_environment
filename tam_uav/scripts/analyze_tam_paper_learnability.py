"""Analyze one TAM paper v4 Vanilla-HAPPO early-learnability run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tam_output_paths import resolve_tam_output


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_episode_windows(records):
    """Return early 20%, middle 60%, late 20% in episode order."""
    n = len(records)
    if n == 0:
        return [], [], []
    edge = max(1, int(math.ceil(0.2 * n)))
    early = records[:edge]
    late = records[-edge:]
    middle_start, middle_end = min(edge, n), max(n - edge, edge)
    middle = records[middle_start:middle_end]
    return early, middle, late


def trend_statistics(records, key, transform=None):
    values = []
    for record in records:
        value = transform(record) if transform else _number(record.get(key))
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"early_mean": None, "late_mean": None, "late_minus_early": None,
                "relative_change": None, "linear_slope": None, "sample_count": 0}
    early, _middle, late = split_episode_windows(values)
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    delta = late_mean - early_mean
    relative = delta / abs(early_mean) if abs(early_mean) > 1e-12 else None
    if len(values) > 1:
        x_mean = (len(values) - 1) / 2.0
        denominator = sum((index - x_mean) ** 2
                          for index in range(len(values)))
        slope = sum((index - x_mean) * (value - sum(values) / len(values))
                    for index, value in enumerate(values)) / denominator
    else:
        slope = 0.0
    return {"early_mean": early_mean, "late_mean": late_mean,
            "late_minus_early": delta, "relative_change": relative,
            "linear_slope": slope, "sample_count": len(values)}


def determine_verdict(runtime_valid, positive_training_trend,
                      positive_combat_trend, positive_deterministic_eval):
    if not runtime_valid:
        return "RUNTIME_INVALID"
    count = sum((positive_training_trend, positive_combat_trend,
                 positive_deterministic_eval))
    if count >= 2:
        return "CLEAR_EARLY_LEARNING_SIGNAL"
    if count == 1:
        return "MIXED_OR_WEAK_EARLY_SIGNAL"
    return "NO_EARLY_SIGNAL_AT_102400_STEPS"


def _evaluation_delta(reference, candidate):
    keys = ("red_team_episode_return", "red_survival_rate", "red_hit_rate",
            "episode_length", "red_boundary_deaths", "red_crashes")
    return {key: (_number(candidate.get(key)) - _number(reference.get(key)))
            for key in keys}


def analyze_run(run_directory: Path) -> dict:
    config = json.loads((run_directory / "config_snapshot.json").read_text(
        encoding="utf-8"))
    summary = json.loads((run_directory / "summary.json").read_text(
        encoding="utf-8"))
    baseline = json.loads((run_directory / "baseline_reference.json").read_text(
        encoding="utf-8"))
    episodes = _read_csv(run_directory / "episodes.csv")
    training = _read_csv(run_directory / "training.csv")
    evaluations = _read_csv(run_directory / "evaluation_history.csv")
    expected_steps = int(config["total_environment_steps"])
    actual_steps = int(summary["actual_final_environment_steps"])
    target_violations = int(sum(_number(row.get("target_consistency_violation")) or 0
                                for row in episodes))
    structural_failures = int(sum(_number(row.get("structural_failures")) or 0
                                  for row in episodes))
    updates_nonfinite = int(sum((_number(row.get("nan_inf_count")) or 0) > 0
                                for row in training))
    failed_contracts = int(sum(
        str(row.get("optimization_update_contract_valid")).lower() != "true"
        for row in training))
    horizon_mismatches = int(sum(
        _number(row.get("rollout_planned_horizon"))
        != _number(row.get("rollout_collected_steps")) for row in training))
    all_episode_finite = bool(episodes and all(
        str(row.get("finite")).lower() == "true" for row in episodes))
    runtime_valid = bool(
        actual_steps == expected_steps and all_episode_finite
        and target_violations == 0 and structural_failures == 0
        and updates_nonfinite == 0 and failed_contracts == 0
        and horizon_mismatches == 0)
    runtime = {
        "all_reward_and_state_finite": all_episode_finite,
        "target_consistency_violation_total": target_violations,
        "structural_failures_total": structural_failures,
        "updates_with_failed_contract": failed_contracts,
        "updates_with_nonfinite": updates_nonfinite,
        "rollout_horizon_mismatch_count": horizon_mismatches,
        "checkpoint_revision": summary["checkpoint_environment_fidelity_revision"],
        "completed_target_steps": actual_steps == expected_steps,
        "actual_environment_steps": actual_steps,
        "expected_environment_steps": expected_steps,
        "runtime_valid": runtime_valid,
    }

    metrics = {
        "red_team_episode_return": ("red_team_episode_return", None),
        "red_win_rate": (None, lambda row: float(row.get("winner") == "red")),
        "draw_rate": (None, lambda row: float(row.get("winner") == "draw")),
        "episode_length": ("episode_length", None),
        "red_survival_rate": ("red_survival_rate", None),
        "red_hit_rate": ("red_hit_rate", None),
        "red_missiles_fired": ("red_missiles_fired", None),
        "red_boundary_rate": (None, lambda row: (_number(
            row.get("red_boundary_deaths")) or 0.0) / 2.0),
        "red_crash_rate": (None, lambda row: (_number(
            row.get("red_crashes")) or 0.0) / 2.0),
    }
    trends = {name: trend_statistics(episodes, key, transform)
              for name, (key, transform) in metrics.items()}

    evaluation_progression = None
    if evaluations:
        ordered = sorted(evaluations,
                         key=lambda row: int(float(row["environment_steps"])))
        step_zero, final = ordered[0], ordered[-1]
        best = max(ordered, key=lambda row: _number(
            row.get("red_team_episode_return")) or -float("inf"))
        evaluation_progression = {
            "step_0": step_zero,
            "best": best,
            "final": final,
            "best_minus_step_0": _evaluation_delta(step_zero, best),
            "final_minus_step_0": _evaluation_delta(step_zero, final),
            "winner_changed_step_0_to_final": step_zero["winner"] != final["winner"],
        }

    def column_values(prefix):
        return [float(value) for row in training for key, raw in row.items()
                if key.startswith(prefix) and (value := _number(raw)) is not None]

    actor_entropies = column_values("entropy/")
    approx_kls = column_values("approx_kl/")
    clip_fractions = column_values("clip_fraction/")
    explained = column_values("explained_variance/")
    actor_gradients = column_values("gradient_norm/")
    critic_gradients = column_values("critic_gradient_norm")
    head_entropies = column_values("action_head_")
    distribution_peaks = []
    for row in training:
        for head in range(4):
            raw = row.get(f"action_head_{head}_distribution")
            if raw:
                distribution_peaks.append(max(json.loads(raw)))
    optimization = {
        "actor_entropy": trend_statistics(
            [{"v": value} for value in actor_entropies], "v"),
        "approx_kl": trend_statistics([{"v": value} for value in approx_kls], "v"),
        "clip_fraction": trend_statistics(
            [{"v": value} for value in clip_fractions], "v"),
        "explained_variance": trend_statistics(
            [{"v": value} for value in explained], "v"),
        "actor_gradient_norm": trend_statistics(
            [{"v": value} for value in actor_gradients], "v"),
        "critic_gradient_norm": trend_statistics(
            [{"v": value} for value in critic_gradients], "v"),
        "action_head_entropy": trend_statistics(
            [{"v": value} for value in head_entropies], "v"),
        "maximum_action_bin_probability": max(distribution_peaks, default=None),
        "action_distribution_completely_collapsed": bool(
            distribution_peaks and max(distribution_peaks) >= 0.999),
        "all_actors_failed_to_update_count": int(sum(
            str(row.get("all_actors_changed")).lower() != "true" for row in training)),
        "updates_with_zero_active_controlled_agent": int(sum(
            (_number(row.get("minimum_active_sample_count")) or 0) == 0
            for row in training)),
    }

    return_trend = trends["red_team_episode_return"]
    positive_training = bool(
        return_trend["late_minus_early"] is not None
        and return_trend["late_minus_early"] > 0
        and return_trend["linear_slope"] > 0)
    def changed(name, direction):
        value = trends[name]["late_minus_early"]
        return value is not None and (value > 0 if direction == "up" else value < 0)

    combat_improvements = {
        "red_win_rate_increased": changed("red_win_rate", "up"),
        "red_survival_increased": changed("red_survival_rate", "up"),
        "red_hit_rate_increased": changed("red_hit_rate", "up"),
        "red_boundary_rate_decreased": changed("red_boundary_rate", "down"),
        "red_crash_rate_decreased": changed("red_crash_rate", "down"),
    }
    positive_combat = sum(combat_improvements.values()) >= 2
    positive_eval = False
    if evaluation_progression:
        step_zero = evaluation_progression["step_0"]
        candidates = [evaluation_progression["best"], evaluation_progression["final"]]
        positive_eval = any(
            _number(candidate["red_team_episode_return"])
            > _number(step_zero["red_team_episode_return"])
            and _number(candidate["red_boundary_deaths"])
            <= _number(step_zero["red_boundary_deaths"])
            and _number(candidate["red_crashes"])
            <= _number(step_zero["red_crashes"])
            for candidate in candidates)

    insufficient = len(episodes) < 5 or len(evaluations) < 2
    verdict = None if insufficient else determine_verdict(
        runtime_valid, positive_training, positive_combat, positive_eval)
    status = "INSUFFICIENT_DATA" if insufficient else "COMPLETE"
    report = {
        "analysis_status": status,
        "run_directory": str(run_directory),
        "runtime_validity": runtime,
        "training_outcome_trends": trends,
        "deterministic_evaluation_progression": evaluation_progression,
        "optimization_health": optimization,
        "learnability_verdict": verdict,
        "verdict_rule": {
            "positive_training_trend": positive_training,
            "positive_combat_trend": positive_combat,
            "combat_improvements": combat_improvements,
            "positive_deterministic_eval": positive_eval,
            "positive_category_count": sum(
                (positive_training, positive_combat, positive_eval)),
            "clear_requires": "runtime valid and at least two positive categories",
            "mixed_requires": "runtime valid and exactly one positive category",
            "no_signal_requires": "runtime valid and zero positive categories",
            "runtime_invalid_overrides_learning_verdict": True,
        },
        "baseline_reference": baseline,
        "limitations": [
            "102400 steps tests early signal only; the paper uses 10 million steps.",
            "No early signal does not imply that the environment is unlearnable.",
            "One seed cannot establish statistical robustness.",
            "This is Vanilla HAPPO, not TAM-HAPPO.",
        ],
    }
    if insufficient:
        report["insufficient_data_reason"] = (
            f"requires at least 5 completed episodes and 2 evaluations; got "
            f"{len(episodes)} episodes and {len(evaluations)} evaluations")
    return report


def _markdown(report):
    runtime = report["runtime_validity"]
    lines = [
        "# TAM paper Vanilla-HAPPO learnability report", "",
        f"- Analysis status: `{report['analysis_status']}`",
        f"- Learnability verdict: `{report['learnability_verdict']}`",
        f"- Runtime valid: `{runtime['runtime_valid']}`",
        f"- Environment steps: `{runtime['actual_environment_steps']}` / "
        f"`{runtime['expected_environment_steps']}`",
        f"- Target consistency violations: `{runtime['target_consistency_violation_total']}`",
        f"- Structural failures: `{runtime['structural_failures_total']}`",
        f"- Failed update contracts: `{runtime['updates_with_failed_contract']}`",
        "", "## Verdict evidence", "",
        f"- Positive training trend: `{report['verdict_rule']['positive_training_trend']}`",
        f"- Positive combat trend: `{report['verdict_rule']['positive_combat_trend']}`",
        f"- Positive deterministic evaluation: "
        f"`{report['verdict_rule']['positive_deterministic_eval']}`",
        "", "## Limitations", "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_directory = resolve_tam_output(ROOT, args.run_directory)
    report = analyze_run(run_directory)
    (run_directory / "learnability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (run_directory / "learnability_report.md").write_text(
        _markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
