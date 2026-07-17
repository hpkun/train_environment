"""Aggregate three TAM paper v4 checkpoint-panel learnability reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_tam_paper_checkpoint_panel import paired_bootstrap
from scripts.tam_output_paths import resolve_tam_output


def multiseed_verdict(runtime_all, return_improved, combat_improved,
                      beats_random, mean_delta):
    if not runtime_all:
        return "MULTISEED_RUNTIME_INVALID"
    if (return_improved >= 2 and combat_improved >= 2 and beats_random >= 2
            and mean_delta > 0):
        return "ROBUST_MULTI_SEED_EARLY_LEARNING_SIGNAL"
    if return_improved or combat_improved or beats_random:
        return "MIXED_MULTI_SEED_EARLY_SIGNAL"
    return "NO_ROBUST_MULTI_SEED_SIGNAL"


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze_multiseed(run_directories):
    seeds = []
    checkpoint_rows = []
    for run in run_directories:
        config = json.loads((run / "config_snapshot.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        robust = run / "robust_evaluation_v2"
        panel = json.loads((robust / "checkpoint_panel_summary.json").read_text(
            encoding="utf-8"))
        single = json.loads((robust / "learnability_report.json").read_text(
            encoding="utf-8"))
        checkpoints = panel["checkpoints"]
        zero = next(row for row in checkpoints if row["environment_steps"] == 0)
        final = max(checkpoints, key=lambda row: row["environment_steps"])
        best = max((row for row in checkpoints if row["environment_steps"] > 0),
                   key=lambda row: row["stochastic_mean_return"])
        random_row = panel["uniform_random"]
        combat = sum((
            best["win_rate"] > zero["win_rate"],
            best["red_survival_rate"] > zero["red_survival_rate"],
            best["red_hit_rate"] > zero["red_hit_rate"],
            best["red_crash_rate"] < zero["red_crash_rate"],
            best["red_boundary_rate"] < zero["red_boundary_rate"],
        ))
        delta = best["paired_vs_step_0"]["return"]["mean"]
        deterministic_ok = (
            best["deterministic_red_crash_rate"] <=
            zero["deterministic_red_crash_rate"]
            and best["deterministic_red_boundary_rate"] <=
            zero["deterministic_red_boundary_rate"])
        seed_row = {
            "training_seed": int(config["seed"]),
            "run_directory": str(run),
            "runtime_valid": single["runtime_validity"]["runtime_valid"],
            "training_return_early": single["training_outcome_trends"][
                "red_team_episode_return"]["early_mean"],
            "training_return_late": single["training_outcome_trends"][
                "red_team_episode_return"]["late_mean"],
            "step_0_panel_mean_return": zero["stochastic_mean_return"],
            "best_checkpoint_steps": best["environment_steps"],
            "best_panel_mean_return": best["stochastic_mean_return"],
            "final_panel_mean_return": final["stochastic_mean_return"],
            "paired_return_delta_vs_step_0": delta,
            "paired_return_ci_low": best["paired_vs_step_0"]["return"][
                "bootstrap_ci95"][0],
            "paired_return_ci_high": best["paired_vs_step_0"]["return"][
                "bootstrap_ci95"][1],
            "combat_improvement_count": combat,
            "beats_uniform_random": best["stochastic_mean_return"] >
                                    random_row["stochastic_mean_return"],
            "deterministic_not_degraded": deterministic_ok,
            "single_seed_verdict": single["learnability_verdict"],
            "action_entropy_step_0": zero["active_action_head_0_entropy"],
            "action_entropy_best": best["active_action_head_0_entropy"],
            "critic_explained_variance_early": single["optimization_health"][
                "explained_variance"]["early_mean"],
            "critic_explained_variance_late": single["optimization_health"][
                "explained_variance"]["late_mean"],
        }
        seeds.append(seed_row)
        for row in checkpoints:
            checkpoint_rows.append({"training_seed": int(config["seed"]), **row})

    deltas = np.asarray([row["paired_return_delta_vs_step_0"] for row in seeds])
    improved = sum(row["paired_return_delta_vs_step_0"] > 0 for row in seeds)
    combat_improved = sum(row["combat_improvement_count"] >= 2 for row in seeds)
    beats_random = sum(row["beats_uniform_random"] for row in seeds)
    deterministic_ok = sum(row["deterministic_not_degraded"] for row in seeds)
    runtime_all = all(row["runtime_valid"] for row in seeds)
    mean_delta = float(deltas.mean())
    report = {
        "training_seed_count": len(seeds),
        "seed_summaries": seeds,
        "cross_seed": {
            "step_0_mean": float(np.mean(
                [row["step_0_panel_mean_return"] for row in seeds])),
            "step_0_std": float(np.std(
                [row["step_0_panel_mean_return"] for row in seeds])),
            "best_mean": float(np.mean(
                [row["best_panel_mean_return"] for row in seeds])),
            "best_std": float(np.std(
                [row["best_panel_mean_return"] for row in seeds])),
            "final_mean": float(np.mean(
                [row["final_panel_mean_return"] for row in seeds])),
            "final_std": float(np.std(
                [row["final_panel_mean_return"] for row in seeds])),
            "mean_paired_return_delta": mean_delta,
            "seed_level_mean_delta_bootstrap_ci95": paired_bootstrap(
                deltas, seed=80260717, samples=10000),
            "between_seed_delta_variance": float(deltas.var()),
            "return_improved_seed_count": int(improved),
            "combat_improved_seed_count": int(combat_improved),
            "beats_random_seed_count": int(beats_random),
            "deterministic_not_degraded_seed_count": int(deterministic_ok),
            "all_runtime_valid": bool(runtime_all),
        },
    }
    report["multiseed_verdict"] = multiseed_verdict(
        runtime_all, improved, combat_improved, beats_random, mean_delta)
    return report, checkpoint_rows, seeds


def _markdown(report):
    cross = report["cross_seed"]
    return "\n".join([
        "# TAM paper v4 multi-seed learnability", "",
        f"- Verdict: `{report['multiseed_verdict']}`",
        f"- Runtime valid: `{cross['all_runtime_valid']}`",
        f"- Mean paired return delta: `{cross['mean_paired_return_delta']}`",
        f"- Improved seeds: `{cross['return_improved_seed_count']}/3`",
        f"- Combat-improved seeds: `{cross['combat_improved_seed_count']}/3`",
        f"- Beats random: `{cross['beats_random_seed_count']}/3`", "",
        "This is an early 102400-step Vanilla-HAPPO result, not convergence proof.",
    ]) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directories", nargs="+", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = [resolve_tam_output(ROOT, path) for path in args.run_directories]
    output = resolve_tam_output(ROOT, args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report, checkpoints, seeds = analyze_multiseed(runs)
    (output / "multiseed_learnability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output / "multiseed_learnability_report.md").write_text(
        _markdown(report), encoding="utf-8")
    _write_csv(output / "multiseed_checkpoint_summary.csv",
               [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                 for key, value in row.items()} for row in checkpoints])
    _write_csv(output / "multiseed_seed_summary.csv", seeds)
    print(json.dumps(report["cross_seed"] | {
        "multiseed_verdict": report["multiseed_verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
