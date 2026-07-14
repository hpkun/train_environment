"""Run the bounded TAM v6 Pure HAPPO learnability protocol.

This is an experiment runner, not a new training implementation. It delegates
all updates and checkpointing to ``train_happo_reference.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_contract_corrected_v6.yaml"
)


def training_command(args, seed: int, seed_dir: Path) -> list[str]:
    return [
        sys.executable, "-u", "scripts/train_happo_reference.py",
        "--config", args.config,
        "--output-dir", str(seed_dir),
        "--total-env-steps", str(args.train_steps),
        "--rollout-length", "256", "--num-envs", "1",
        "--max-steps", "1000", "--device", args.device,
        "--policy-arch", "pure_happo", "--opponent-policy", "tam_greedy_rule",
        "--credit-mode", args.credit_mode,
        "--seed", str(seed),
        "--checkpoint-interval-steps", str(args.checkpoint_interval_steps),
        "--keep-checkpoints", str(args.keep_checkpoints),
        "--save-initial-checkpoint-dir", str(seed_dir / "init"),
        "--enable-rich-logging", "--rich-log-mode", "summary",
        "--rich-log-dir", str(seed_dir / "rich_logs"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="outputs/tam_v6_contract_corrected_learnability")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int,
                        default=list(range(91000, 91020)))
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=10_000)
    parser.add_argument("--keep-checkpoints", type=int, default=10)
    parser.add_argument("--credit-mode", default="shared_alive_team_mean",
                        choices=["shared_alive_team_mean", "role_local_vector_critic"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.train_steps < 1:
        raise ValueError("train_steps must be positive")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_eval_seeds.txt").write_text(
        "\n".join(map(str, args.eval_seeds)) + "\n", encoding="utf-8")
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        command = training_command(args, seed, seed_dir)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if args.dry_run:
        return

    from scripts.run_tam_v5_fast_learnability import METRICS, classify, evaluate_checkpoint
    rows = []
    completed = []
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        initial = evaluate_checkpoint(
            seed_dir / "init" / "model.pt", config=args.config,
            episode_seeds=args.eval_seeds, device_name=args.device, max_steps=1000)
        final = evaluate_checkpoint(
            seed_dir / "latest" / "model.pt", config=args.config,
            episode_seeds=args.eval_seeds, device_name=args.device, max_steps=1000)
        row = {"seed": seed}
        for metric in METRICS:
            row[f"init_{metric}"] = initial[metric]
            row[f"final_{metric}"] = final[metric]
            row[f"delta_{metric}"] = final[metric] - initial[metric]
        train_rows = list(csv.DictReader((seed_dir / "train_log.csv").open(encoding="utf-8")))
        if train_rows:
            last = train_rows[-1]
            for metric in (
                "action_override_rate", "range_rate", "ata_rate", "ta_rate",
                "geometry_rate", "lock_mature_rate", "actual_launch_rate",
                "reward_target_matches_lock_given_lock",
                "reward_target_matches_launch_given_launch", "target_switches_per_episode",
                "dense_event_abs_ratio", "mav_uav_sign_conflict_rate",
                "approx_kl_mav", "approx_kl_uav", "clip_fraction_mav",
                "clip_fraction_uav", "value_explained_variance",
                "mav_action_saturation_rate", "uav_action_saturation_rate",
                "actor_effective_sample_fraction",
            ):
                value = last.get(metric, "")
                row[f"train_final_{metric}"] = float(value) if value not in ("", None) else float("nan")
        rows.append(row)
        status_path = seed_dir / "runner_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        completed.append(bool(status.get("runner_completed_normally", False)))
    with (output_dir / "per_seed_init_final.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for metric in METRICS:
        initial = np.asarray([row[f"init_{metric}"] for row in rows], dtype=np.float64)
        final = np.asarray([row[f"final_{metric}"] for row in rows], dtype=np.float64)
        delta = final - initial
        aggregate.append({
            "metric": metric,
            "init_mean": float(initial.mean()), "init_std": float(initial.std(ddof=0)),
            "final_mean": float(final.mean()), "final_std": float(final.std(ddof=0)),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std(ddof=0)),
        })
    with (output_dir / "aggregate_init_final.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    status, evidence = classify(rows, completed)
    summary = {
        "status": status,
        "contract": "tam_happo_contract_corrected_v6",
        "credit_mode": args.credit_mode,
        "seeds": args.seeds,
        "eval_seeds": args.eval_seeds,
        "completed": completed,
        "evidence": evidence,
        "rows": rows,
    }
    (output_dir / "learnability_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    lines = [
        "# TAM v6 Contract-Corrected Learnability",
        "",
        f"**Status:** `{status}`",
        "",
        f"Credit mode: `{args.credit_mode}`.",
        f"Training seeds: `{args.seeds}`; fixed evaluation seeds: `{args.eval_seeds}`.",
        "",
        "This runner preserves the frozen v5 reward and evaluates the v6 action-authority, "
        "incoming-missile observation, engagement-target, and credit contracts.",
    ]
    (output_dir / "learnability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
