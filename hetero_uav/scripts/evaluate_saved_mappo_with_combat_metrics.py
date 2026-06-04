"""Posthoc combat-metrics evaluation for a saved MAPPO baseline model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_mappo_zero_shot.py"
DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]


def _tail(text: str, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    return text[-limit:]


def _finite(value, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{name} is not finite: {value}")
    return out


def _run_eval(args: argparse.Namespace) -> None:
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python", "-u", str(EVAL_SCRIPT),
        "--model", args.model,
        "--obs-adapter-version", args.obs_adapter_version,
        "--episodes", str(args.episodes),
        "--device", args.device,
        "--opponent-policy", args.opponent_policy,
        "--configs", *args.configs,
        "--summary-json", args.output_json,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=7200,
        env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("stderr tail:")
        print(_tail(result.stderr))
        raise RuntimeError(f"posthoc eval failed with rc={result.returncode}")


def _validate_and_write_csv(json_path: Path, csv_path: Path) -> list[dict]:
    if not json_path.exists():
        raise RuntimeError(f"missing eval summary json: {json_path}")
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"invalid eval summary: {json_path}")

    rows = []
    for rec in records:
        if rec.get("nan_detected"):
            raise RuntimeError(f"NaN detected in eval: {rec['config']}")
        if not rec.get("actor_dim_ok"):
            raise RuntimeError(f"actor_dim_ok False: {rec['config']}")
        if not rec.get("critic_dim_ok"):
            raise RuntimeError(f"critic_dim_ok False: {rec['config']}")
        row = dict(rec)
        row["avg_return"] = _finite(row["avg_return"], "avg_return")
        row["avg_length"] = _finite(row["avg_length"], "avg_length")
        if row["avg_length"] <= 0:
            raise RuntimeError(f"avg_length <= 0: {rec['config']}")
        row["episode_end_reason_counts"] = json.dumps(
            row.get("episode_end_reason_counts", {}), sort_keys=True)
        row["winner_counts"] = json.dumps(row.get("winner_counts", {}), sort_keys=True)
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="outputs/mappo_balanced_baseline_500k/seed_0/latest/model.pt",
    )
    parser.add_argument("--obs-adapter-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy",
                        choices=["zero", "random", "rule_nearest"],
                        default="rule_nearest")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--output-json",
        default="outputs/mappo_balanced_baseline_500k/seed_0/combat_metrics_eval.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/mappo_balanced_baseline_500k/seed_0/combat_metrics_eval.csv",
    )
    args = parser.parse_args()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    _run_eval(args)
    _validate_and_write_csv(Path(args.output_json), Path(args.output_csv))
    print(f"output_json: {args.output_json}")
    print(f"output_csv: {args.output_csv}")
    print("combat metrics evaluation completed")
    print("This is diagnostic evaluation, not a formal zero-shot claim.")


if __name__ == "__main__":
    main()
