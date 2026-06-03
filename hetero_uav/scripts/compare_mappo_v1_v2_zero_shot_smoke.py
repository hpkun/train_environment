"""V1/V2 zero-shot smoke diagnostics.

This is not a formal zero-shot experiment. It checks that saved V1/V2 models
can be evaluated on their matching config families without NaNs or observation
dimension mismatches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHOT = ROOT / "scripts" / "eval_mappo_zero_shot.py"


def _tail(text: str, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    return text[-limit:]


def _run_zero_shot(version: str, model: Path, configs: list[str],
                   stdout_path: Path, args: argparse.Namespace,
                   env: dict[str, str]) -> dict:
    cmd = [
        "python", str(ZERO_SHOT),
        "--model", str(model),
        "--obs-adapter-version", version,
        "--episodes", str(args.episodes),
        "--device", args.device,
        "--opponent-policy", args.opponent_policy,
        "--configs", *configs,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=600,
        env=env,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    print(_tail(result.stdout, 1200))

    if result.returncode != 0:
        print(f"ZERO-SHOT FAILED for {version} with returncode={result.returncode}")
        print("stdout tail:")
        print(_tail(result.stdout))
        print("stderr tail:")
        print(_tail(result.stderr))
        raise RuntimeError(
            f"Zero-shot {version} failed with rc={result.returncode}"
        )

    nan_found = "nan_detected: True" in result.stdout
    actor_bad = "actor_dim_ok: False" in result.stdout
    critic_bad = "critic_dim_ok: False" in result.stdout
    summary = {
        "version": version,
        "model": str(model),
        "configs": configs,
        "episodes": args.episodes,
        "returncode": result.returncode,
        "stdout_path": str(stdout_path),
        "nan_detected_found_in_stdout": nan_found,
        "actor_dim_ok_found_in_stdout": "actor_dim_ok: True" in result.stdout,
        "critic_dim_ok_found_in_stdout": "critic_dim_ok: True" in result.stdout,
    }

    if nan_found:
        raise RuntimeError(f"{version} zero-shot stdout contains nan_detected: True")
    if actor_bad:
        raise RuntimeError(f"{version} zero-shot stdout contains actor_dim_ok: False")
    if critic_bad:
        raise RuntimeError(f"{version} zero-shot stdout contains critic_dim_ok: False")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="rule_nearest")
    parser.add_argument("--output-dir", default="outputs/compare_mappo_v1_v2")
    parser.add_argument("--summary-json",
                        default="outputs/compare_mappo_v1_v2/zero_shot_smoke_summary.json")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)

    runs = [
        ("v1", out_dir / "v1" / "latest" / "model.pt", [
            "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml",
            "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml",
        ]),
        ("v2", out_dir / "v2" / "latest" / "model.pt", [
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
        ]),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    summaries = []

    for version, model, configs in runs:
        if not model.exists():
            if args.allow_missing:
                print(f"SKIP {version}: model not found: {model}")
                continue
            raise FileNotFoundError(f"Model not found: {model}")

        stdout_path = out_dir / f"{version}_zero_shot_stdout.txt"
        print(f"=== {version} zero-shot smoke ===")
        summaries.append(_run_zero_shot(
            version, model, configs, stdout_path, args, env
        ))
        print()

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"summary_json: {summary_json}")
    print("this is smoke, not a formal zero-shot experiment.")


if __name__ == "__main__":
    main()
