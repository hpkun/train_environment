"""Run short opt-in BRMA recurrent masked policy smoke tests."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _safe_output_dir(path: str) -> Path:
    out = ROOT / path
    if not out.exists() or not any(out.iterdir()):
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out.with_name(f"{out.name}_{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml")
    parser.add_argument("--total-env-steps", type=int, default=1024)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/debug_brma_nomask_smoke")
    parser.add_argument("--biased-mask", action="store_true")
    parser.add_argument("--random-scale-mask", action=argparse.BooleanOptionalAction, default=False,
                        help="Deprecated unsafe path; training entrypoints reject this flag.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.random_scale_mask:
        raise SystemExit(
            "--random-scale-mask is disabled for main training smoke runs; "
            "use no-random-mask, or implement rollout mask replay / full BRMA mask objective first."
        )

    output_dir = _safe_output_dir(args.output_dir)
    cmd = [
        "python",
        "-u",
        str((ROOT / "scripts" / "train_happo_reference.py").relative_to(ROOT)),
        "--config",
        args.config,
        "--output-dir",
        str(output_dir.relative_to(ROOT)),
        "--total-env-steps",
        str(args.total_env_steps),
        "--rollout-length",
        str(args.rollout_length),
        "--num-envs",
        str(args.num_envs),
        "--max-steps",
        str(args.max_steps),
        "--device",
        args.device,
        "--policy-arch",
        "brma_recurrent_masked",
        "--opponent-policy",
        "brma_rule",
        "--eval-during-training",
        "--eval-interval-steps",
        "5000",
        "--train-eval-episodes",
        "2",
    ]
    if args.biased_mask:
        cmd.append("--brma-biased-mask")
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
