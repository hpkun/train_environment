"""Run oracle dataset collection, UAV actor pretrain, and 200k HAPPO fine-tune."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _cmds(args) -> list[tuple[str, list[str]]]:
    dataset = _rel(args.dataset)
    cmds: list[tuple[str, list[str]]] = []
    if args.force_collect or not dataset.exists():
        cmds.append(("collect", [
            "python", "scripts/collect_direct_chase_oracle_dataset.py",
            "--config", args.config,
            "--output", args.dataset,
            "--summary-json", args.dataset_summary,
            "--episodes", str(args.dataset_episodes),
            "--max-steps", str(args.max_steps),
            "--opponent-policy", args.opponent_policy,
        ]))
    cmds.append(("pretrain", [
        "python", "scripts/pretrain_uav_actor_from_oracle.py",
        "--dataset", args.dataset,
        "--init-checkpoint", args.init_checkpoint,
        "--output-checkpoint", args.pretrained_checkpoint,
        "--output-meta", args.pretrained_meta,
        "--epochs", str(args.pretrain_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.pretrain_lr),
        "--device", args.device,
    ]))
    cmds.append(("finetune", [
        "python", "scripts/train_happo_reference.py",
        "--config", args.config,
        "--reward-mode", "happo_ref_v0",
        "--opponent-policy", args.opponent_policy,
        "--total-env-steps", str(args.total_env_steps),
        "--rollout-length", str(args.rollout_length),
        "--max-steps", str(args.max_steps),
        "--ppo-epochs", str(args.ppo_epochs),
        "--entropy-coef", str(args.entropy_coef),
        "--actor-lr", str(args.actor_lr),
        "--critic-lr", str(args.critic_lr),
        "--eval-during-training",
        "--eval-interval-steps", str(args.eval_interval_steps),
        "--train-eval-episodes", str(args.train_eval_episodes),
        "--init-checkpoint", args.pretrained_checkpoint,
        "--output-dir", args.output_dir,
        "--device", args.device,
    ]))
    return cmds


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle-pretrain + HAPPO 200k fine-tune runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml")
    parser.add_argument("--dataset", default="outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2.npz")
    parser.add_argument("--dataset-summary", default="outputs/direct_chase_oracle_dataset/direct_chase_oracle_3v2_summary.json")
    parser.add_argument("--dataset-episodes", type=int, default=50)
    parser.add_argument("--init-checkpoint", default="outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast/best/model.pt")
    parser.add_argument("--pretrained-checkpoint", default="outputs/oracle_pretrain/uav_actor_oracle_pretrained/model.pt")
    parser.add_argument("--pretrained-meta", default="outputs/oracle_pretrain/uav_actor_oracle_pretrained/meta.json")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/happo_oracle_pretrain_finetune_200k")
    parser.add_argument("--total-env-steps", type=int, default=200000)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--eval-interval-steps", type=int, default=25000)
    parser.add_argument("--train-eval-episodes", type=int, default=5)
    parser.add_argument("--opponent-policy", choices=["brma_rule", "zero"], default="brma_rule")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    for label, cmd in _cmds(args):
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
