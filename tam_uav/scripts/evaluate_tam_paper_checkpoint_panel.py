"""Paired deterministic and stochastic checkpoint evaluation for TAM paper v4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo.vanilla_happo import VanillaHAPPOTrainer
from algorithms.happo.vanilla_happo_checkpoint import (
    FORMAT, load_vanilla_happo_checkpoint, save_vanilla_happo_checkpoint)
from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import (
    evaluate_policy_panel, infer_policy, make_paper_env, seed_all)
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, NOMINAL_PERTURBATION,
    PAPER_NOMINAL_PROTOCOL, PAPER_SILENT_ASSUMPTIONS_PRESENT)


DEFAULT_STEPS = (0, 20480, 40960, 61440, 81920, 102400)
PAIR_FIELDS = {
    "return": lambda row: row["red_episode_return"],
    "win_indicator": lambda row: float(row["winner"] == "red"),
    "survival": lambda row: row["red_survival_rate"],
    "hit_rate": lambda row: row["red_hit_rate"],
    "crash_rate": lambda row: row["red_crash_rate"],
    "boundary_rate": lambda row: row["red_boundary_rate"],
    "episode_length": lambda row: row["episode_steps"],
}


def paired_bootstrap(values, *, seed=60260717, samples=10000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def paired_delta_summary(candidate, reference, *, bootstrap_seed=60260717,
                         bootstrap_samples=10000):
    if len(candidate) != len(reference):
        raise ValueError("paired panels must have equal lengths")
    result = {}
    for offset, (name, accessor) in enumerate(PAIR_FIELDS.items()):
        delta = np.asarray([accessor(a) - accessor(b)
                            for a, b in zip(candidate, reference)], dtype=float)
        result[name] = {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
            "std": float(delta.std()),
            "positive_fraction": float(np.mean(delta > 0)),
            "negative_fraction": float(np.mean(delta < 0)),
            "zero_fraction": float(np.mean(delta == 0)),
            "bootstrap_ci95": paired_bootstrap(
                delta, seed=bootstrap_seed + offset,
                samples=bootstrap_samples),
        }
    return result


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_row(row):
    return {key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()}


def _panel_summary(step, path, metadata, canonical, result):
    records = result["episodes_detail"]
    returns = np.asarray([row["red_episode_return"] for row in records])
    summary = {
        "environment_steps": int(step),
        "checkpoint_path": str(path) if path else None,
        "checkpoint_type": metadata.get("checkpoint_type", "reconstructed_step_0"),
        "policy_version": int(metadata.get("policy_version", 0)),
        "deterministic_winner": canonical["winner"],
        "deterministic_return": float(canonical["red_episode_return"]),
        "deterministic_episode_length": int(canonical["episode_steps"]),
        "deterministic_red_crash_rate": float(canonical["red_crash_rate"]),
        "deterministic_red_boundary_rate": float(canonical["red_boundary_rate"]),
        "stochastic_panel_episode_count": len(records),
        "stochastic_mean_return": float(returns.mean()),
        "stochastic_std_return": float(returns.std()),
        "stochastic_median_return": float(np.median(returns)),
        "win_rate": float(np.mean([row["winner"] == "red" for row in records])),
        "draw_rate": float(np.mean([row["winner"] == "draw" for row in records])),
        "red_survival_rate": float(np.mean(
            [row["red_survival_rate"] for row in records])),
        "red_combat_survival_rate": float(np.mean(
            [row["red_combat_survival_rate"] for row in records])),
        "red_hit_rate": float(np.mean([row["red_hit_rate"] for row in records])),
        "red_missiles_fired": float(np.mean(
            [row["red_missiles_fired"] for row in records])),
        "red_kills": float(np.mean([row["red_kills"] for row in records])),
        "red_crash_rate": float(np.mean([row["red_crash_rate"] for row in records])),
        "red_boundary_rate": float(np.mean(
            [row["red_boundary_rate"] for row in records])),
        "mean_episode_length": float(np.mean(
            [row["episode_steps"] for row in records])),
        "finite": bool(all(row["finite"] for row in records)),
        "target_consistency_violations": int(sum(
            row["target_consistency_violations"] for row in records)),
        "active_action_sample_count": int(
            result["summary"]["active_action_sample_count"]),
    }
    for head in range(4):
        summary[f"active_action_head_{head}_entropy"] = result["summary"][
            f"active_action_head_{head}_entropy"]
        summary[f"active_action_head_{head}_distribution"] = result["summary"][
            f"active_action_head_{head}_distribution"]
    return summary


def _load_config(run_directory):
    return json.loads((run_directory / "config_snapshot.json").read_text(
        encoding="utf-8"))


def _make_initial_policy(config, device):
    seed_all(int(config["seed"]))
    env = make_paper_env(ROOT, config["scenario"])
    policy, _, _ = infer_policy(
        env, config["actor_sharing"], int(config["hidden_dim"]), device)
    return env, policy


def evaluate_run(run_directory, steps, panel_size, environment_seed,
                 policy_action_seed_base, device, output, bootstrap_samples=10000):
    config = _load_config(run_directory)
    if config["environment_fidelity_revision"] != ENVIRONMENT_FIDELITY_REVISION:
        raise ValueError("run environment revision is not current frozen v4")
    output.mkdir(parents=True, exist_ok=True)
    action_seeds = [policy_action_seed_base + index
                    for index in range(panel_size)]
    environment_seeds = [environment_seed] * panel_size
    panel_results, summaries, canonicals = {}, [], []
    started = time.perf_counter()
    for step in steps:
        env, policy = _make_initial_policy(config, device)
        checkpoint_path, metadata = None, {
            "checkpoint_type": "reconstructed_step_0", "policy_version": 0}
        try:
            if step == 0:
                original = json.loads((run_directory / "evaluation_0.json").read_text(
                    encoding="utf-8"))["episodes_detail"][0]
            else:
                checkpoint_path = run_directory / f"checkpoint_{step}.pt"
                metadata = load_vanilla_happo_checkpoint(
                    checkpoint_path, policy, restore_rng=False,
                    expected_scenario=config["scenario"],
                    expected_environment_fidelity_revision=(
                        ENVIRONMENT_FIDELITY_REVISION),
                    expected_experiment_protocol=PAPER_NOMINAL_PROTOCOL,
                    expected_initial_perturbation=NOMINAL_PERTURBATION,
                    expected_dynamics_backend="jsbsim",
                    expected_paper_silent_assumptions_present=(
                        PAPER_SILENT_ASSUMPTIONS_PRESENT))
                if metadata["format"] != FORMAT:
                    raise ValueError("checkpoint format mismatch")
            canonical_result = evaluate_policy_panel(
                env, policy, 1, environment_seed,
                environment_seeds=[environment_seed],
                policy_action_seeds=[policy_action_seed_base],
                deterministic=True)
            canonical = canonical_result["episodes_detail"][0]
            if step == 0:
                if (canonical["winner"] != original["winner"]
                        or canonical["episode_steps"] != original["episode_steps"]
                        or not math.isclose(
                            canonical["red_episode_return"],
                            original["red_episode_return"],
                            rel_tol=1e-9, abs_tol=1e-6)):
                    raise RuntimeError("reconstructed step-0 policy does not match evaluation_0")
                trainer = VanillaHAPPOTrainer(policy)
                reconstructed = output / "reconstructed_checkpoint_0.pt"
                save_vanilla_happo_checkpoint(
                    reconstructed, policy, trainer, environment_steps=0, episodes=0,
                    config=config, numpy_rng=np.random.default_rng(config["seed"]),
                    checkpoint_type="evaluation_weights", policy_version=0,
                    extra_metadata={
                        "reconstructed_from_training_seed": True,
                        "source_environment_steps": 0,
                    })
                checkpoint_path = reconstructed
            panel = evaluate_policy_panel(
                env, policy, panel_size, environment_seed,
                environment_seeds=environment_seeds,
                policy_action_seeds=action_seeds, deterministic=False)
        finally:
            env.close()
        panel_results[int(step)] = panel["episodes_detail"]
        canonical_row = {
            "environment_steps": int(step),
            "checkpoint_path": str(checkpoint_path),
            **{key: value for key, value in canonical.items()
               if not key.startswith("active_action_head_")},
        }
        canonicals.append(canonical_row)
        summaries.append(_panel_summary(
            step, checkpoint_path, metadata, canonical, panel))

    random_env, initial_policy = _make_initial_policy(config, device)
    try:
        random_panel = evaluate_policy_panel(
            random_env, initial_policy, panel_size, environment_seed,
            baseline="random", environment_seeds=environment_seeds,
            policy_action_seeds=action_seeds, deterministic=False)
    finally:
        random_env.close()
    random_records = random_panel["episodes_detail"]
    random_summary = _panel_summary(
        -1, None, {"checkpoint_type": "uniform_random", "policy_version": 0},
        random_records[0], random_panel)
    random_summary["baseline"] = "uniform_random"

    step_zero = panel_results[0]
    for summary in summaries:
        step = summary["environment_steps"]
        summary["paired_vs_step_0"] = paired_delta_summary(
            panel_results[step], step_zero,
            bootstrap_seed=60260717 + step,
            bootstrap_samples=bootstrap_samples)
        summary["paired_vs_uniform_random"] = paired_delta_summary(
            panel_results[step], random_records,
            bootstrap_seed=70260717 + step,
            bootstrap_samples=bootstrap_samples)

    episode_rows = []
    for step, records in panel_results.items():
        for index, record in enumerate(records):
            episode_rows.append({"environment_steps": step, "panel_index": index,
                                 "policy_kind": "checkpoint", **record})
    for index, record in enumerate(random_records):
        episode_rows.append({"environment_steps": -1, "panel_index": index,
                             "policy_kind": "uniform_random", **record})
    csv_episode_rows = [_jsonable_row(row) for row in episode_rows]
    _write_csv(output / "checkpoint_panel_episode_results.csv", csv_episode_rows)
    with (output / "checkpoint_panel_episode_results.jsonl").open(
            "w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row) + "\n")
    _write_csv(output / "checkpoint_panel_summary.csv",
               [_jsonable_row(row) for row in summaries + [random_summary]])
    payload = {
        "run_directory": str(run_directory),
        "panel_size": panel_size,
        "environment_seed": environment_seed,
        "policy_action_seeds": action_seeds,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": 60260717,
        "checkpoints": summaries,
        "uniform_random": random_summary,
        "panel_wall_time_s": time.perf_counter() - started,
    }
    (output / "checkpoint_panel_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output / "deterministic_canonical_history.csv",
               [_jsonable_row(row) for row in canonicals])
    (output / "deterministic_canonical_history.json").write_text(
        json.dumps(canonicals, indent=2), encoding="utf-8")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+",
                        default=list(DEFAULT_STEPS))
    parser.add_argument("--panel-size", type=int, default=50)
    parser.add_argument("--environment-seed", type=int, default=30260717)
    parser.add_argument("--policy-action-seed-base", type=int, default=40260717)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run = resolve_tam_output(ROOT, args.run_directory)
    output = (resolve_tam_output(ROOT, args.output) if args.output else
              run / "robust_evaluation_v2")
    report = evaluate_run(
        run, args.checkpoint_steps, args.panel_size, args.environment_seed,
        args.policy_action_seed_base, torch.device(args.device), output,
        args.bootstrap_samples)
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "panel_wall_time_s": report["panel_wall_time_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
