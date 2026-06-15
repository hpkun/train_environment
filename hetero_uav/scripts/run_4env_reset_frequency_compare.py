"""Print or run reset-frequency comparison commands for parallel env debugging."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CASES = [
    ("max64_4env_100k", 4, 64, 100000, "outputs/debug_4env_max64_100k"),
    ("max1000_4env_500k", 4, 1000, 500000, "outputs/debug_4env_max1000_500k"),
    ("max1000_2env_500k", 2, 1000, 500000, "outputs/debug_2env_max1000_500k"),
]


def _cmd(num_envs: int, max_steps: int, total_steps: int, output_dir: str) -> list[str]:
    return [
        sys.executable,
        "scripts/run_4env_stability_debug.py",
        "--num-envs", str(num_envs),
        "--max-steps", str(max_steps),
        "--total-env-steps", str(total_steps),
        "--output-dir", output_dir,
        "--heartbeat-stall-timeout-sec", "300",
        "--exit-on-heartbeat-stall",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true",
                        help="Run all cases sequentially. Default only prints commands.")
    args = parser.parse_args()

    for label, num_envs, max_steps, total_steps, output_dir in CASES:
        cmd = _cmd(num_envs, max_steps, total_steps, output_dir)
        if not args.execute:
            cmd.append("--dry-run")
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
