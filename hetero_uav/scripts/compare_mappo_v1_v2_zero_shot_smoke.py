"""V1/V2 zero-shot smoke comparison. Not a formal experiment."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHOT = ROOT / "scripts" / "eval_mappo_zero_shot.py"

V1_MODEL = "outputs/compare_mappo_v1_v2/v1/latest/model.pt"
V2_MODEL = "outputs/compare_mappo_v1_v2/v2/latest/model.pt"
V1_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml",
    "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml",
]
V2_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    for version, model, configs in [
        ("v1", V1_MODEL, V1_CONFIGS),
        ("v2", V2_MODEL, V2_CONFIGS),
    ]:
        if not Path(model).exists():
            print(f"SKIP {version}: model not found at {model}")
            continue
        print(f"=== {version} zero-shot smoke ===")
        result = subprocess.run(
            [sys.executable, str(ZERO_SHOT),
             "--model", model,
             "--obs-adapter-version", version,
             "--episodes", str(args.episodes),
             "--device", args.device,
             "--opponent-policy", args.opponent_policy,
             "--configs", *configs],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(ROOT), timeout=600, env=env,
        )
        print(result.stdout[-600:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"FAILED: {result.stderr[-300:]}")
        print()

    print("Note: this is smoke, not formal zero-shot experiment.")


if __name__ == "__main__":
    main()
