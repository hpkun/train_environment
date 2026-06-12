"""Fast link check for oracle dataset, pretrain, fine-tune, and eval scripts.

This is a runtime sanity check only. It is not a paper experiment and should
not be reported as a training result.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"
DEFAULT_INIT = "outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/best/model.pt"


def _commands(output_dir: str, device: str) -> list[tuple[str, list[str]]]:
    out = Path(output_dir)
    dataset = out / "dataset" / "direct_chase_oracle_fast.npz"
    summary = out / "dataset" / "direct_chase_oracle_fast_summary.json"
    pretrained = out / "pretrained" / "model.pt"
    pretrained_meta = out / "pretrained" / "meta.json"
    finetune = out / "finetune"
    return [
        ("collect", [
            "python", "scripts/collect_direct_chase_oracle_dataset.py",
            "--config", DEFAULT_CONFIG,
            "--output", str(dataset),
            "--summary-json", str(summary),
            "--episodes", "3",
            "--max-samples", "5000",
            "--stop-when-samples-reached",
            "--opponent-policy", "zero",
        ]),
        ("pretrain", [
            "python", "scripts/pretrain_uav_actor_from_oracle.py",
            "--dataset", str(dataset),
            "--init-checkpoint", DEFAULT_INIT,
            "--output-checkpoint", str(pretrained),
            "--output-meta", str(pretrained_meta),
            "--epochs", "3",
            "--max-train-samples", "5000",
            "--device", device,
        ]),
        ("finetune", [
            "python", "scripts/train_happo_reference.py",
            "--config", DEFAULT_CONFIG,
            "--reward-mode", "happo_ref_v0",
            "--opponent-policy", "brma_rule",
            "--total-env-steps", "4096",
            "--rollout-length", "256",
            "--max-steps", "1000",
            "--eval-during-training",
            "--eval-interval-steps", "4096",
            "--train-eval-episodes", "1",
            "--init-checkpoint", str(pretrained),
            "--output-dir", str(finetune),
            "--device", device,
        ]),
        ("eval", [
            "python", "scripts/evaluate_happo_3v2_reference_checkpoints.py",
            "--experiment-dir", str(finetune),
            "--episodes", "5",
            "--checkpoint-mode", "latest_only",
            "--fast",
            "--device", device,
        ]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a very small oracle-pretrain HAPPO link check")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="outputs/oracle_pretrain_fast_check")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for label, cmd in _commands(args.output_dir, args.device):
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
