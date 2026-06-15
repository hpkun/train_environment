"""Print final paper experiment commands.

This script is intentionally dry-run only by default. It does not launch long
training unless a future explicit implementation adds that behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_happo_reference.py"


CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def _base_cmd(output_dir: str, total_env_steps: int) -> list[str]:
    return [
        "python",
        "-u",
        str(TRAIN.relative_to(ROOT)),
        "--config",
        CONFIG,
        "--output-dir",
        output_dir,
        "--total-env-steps",
        str(total_env_steps),
        "--rollout-length",
        "256",
        "--num-envs",
        "1",
        "--max-steps",
        "1000",
        "--device",
        "cuda",
        "--policy-arch",
        "brma_recurrent_masked",
        "--opponent-policy",
        "brma_rule",
        "--eval-during-training",
        "--eval-interval-steps",
        "50000",
        "--train-eval-episodes",
        "5",
        "--enable-rich-logging",
        "--rich-log-dir",
        f"{output_dir}/rich_logs",
    ]


def build_commands(total_env_steps: int = 500000) -> dict[str, list[str]]:
    random_cmd = _base_cmd(
        "outputs/final_brma_recurrent_random_mask_500k_probe",
        total_env_steps,
    )
    random_cmd.append("--brma-random-scale-mask")
    biased_cmd = _base_cmd(
        "outputs/final_brma_recurrent_biased_mask_500k_probe",
        total_env_steps,
    )
    biased_cmd.extend(["--brma-biased-mask"])
    return {
        "B_brma_recurrent_masked_500k_probe": random_cmd,
        "C_brma_recurrent_masked_biased_500k_probe": biased_cmd,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-env-steps", type=int, default=500000)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.dry_run:
        raise SystemExit("This script currently supports --dry-run only; launch commands manually after review.")
    print("# Final paper experiment commands", flush=True)
    print("# Existing baseline: outputs/full_10m_normal_geometry_max1000_env1", flush=True)
    for name, cmd in build_commands(args.total_env_steps).items():
        print(f"\n## {name}", flush=True)
        print(" ".join(cmd), flush=True)


if __name__ == "__main__":
    main()
