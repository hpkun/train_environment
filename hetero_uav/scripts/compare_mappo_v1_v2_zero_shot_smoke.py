"""V1/V2 zero-shot smoke. Subprocess failure = script failure."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHOT = ROOT / "scripts" / "eval_mappo_zero_shot.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir", default="outputs/compare_mappo_v1_v2")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    summary_json = args.summary_json or f"{out_dir}/zero_shot_smoke_summary.json"

    runs = [
        ("v1", f"{out_dir}/v1/latest/model.pt", [
            "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml",
            "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml",
        ]),
        ("v2", f"{out_dir}/v2/latest/model.pt", [
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
        ]),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    summaries = []

    for version, model, configs in runs:
        if not Path(model).exists():
            if args.allow_missing:
                print(f"SKIP {version}: model not found")
                continue
            raise RuntimeError(f"Model not found: {model}")

        stdout_path = f"{out_dir}/{version}_zero_shot_stdout.txt"
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
        os.makedirs(out_dir, exist_ok=True)
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        print(result.stdout[-500:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            raise RuntimeError(
                f"Zero-shot {version} failed rc={result.returncode}: "
                f"{result.stderr[-300:]}")

        nan_found = "nan_detected: True" in result.stdout
        dim_ok = ("actor_dim_ok: True" in result.stdout
                  and "critic_dim_ok: True" in result.stdout)

        summaries.append({
            "version": version, "model": model,
            "configs": configs, "episodes": args.episodes,
            "returncode": result.returncode,
            "stdout_path": stdout_path,
            "nan_detected": nan_found,
            "actor_dim_ok": "actor_dim_ok: True" in result.stdout,
            "critic_dim_ok": "critic_dim_ok: True" in result.stdout,
        })

        if nan_found:
            raise RuntimeError(f"{version} zero-shot: NaN detected")
        if not dim_ok:
            raise RuntimeError(f"{version} zero-shot: actor/critic dim mismatch")
        print()

    with open(summary_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"summary_json: {summary_json}")
    print("Note: this is smoke, not formal zero-shot experiment.")


if __name__ == "__main__":
    main()
