"""Paper-aligned protocol smoke runner.

Runs very short MAPPO training + eval for each opponent policy,
verifying the frozen environment protocol can be exercised end-to-end.
This is NOT a long-run baseline and NOT a zero-shot claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], cwd: Path, label: str, timeout: int = 600) -> subprocess.CompletedProcess:
    print(f"[smoke] {label}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"[smoke] FAIL {label} rc={result.returncode}", flush=True)
        print(f"  stdout last 500: {result.stdout[-500:]}", flush=True)
        print(f"  stderr last 500: {result.stderr[-500:]}", flush=True)
    else:
        print(f"[smoke] OK {label}", flush=True)
    return result


def _assert(condition, msg: str):
    if not condition:
        raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-env-steps", type=int, default=512)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/paper_aligned_protocol_smoke")
    parser.add_argument(
        "--opponent-policies",
        nargs="*",
        default=["rule_nearest", "greedy_fsm"],
    )
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()

    train_config = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
    eval_configs = [
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_records: list[dict] = []
    all_passed = True

    for opponent_policy in args.opponent_policies:
        print(f"\n{'='*60}", flush=True)
        print(f"[smoke] opponent_policy={opponent_policy}", flush=True)
        print(f"{'='*60}", flush=True)

        policy_dir = out_dir / opponent_policy

        # ---- 1. Train ----
        train_cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "train_mappo_baseline.py"),
            "--config", train_config,
            "--obs-adapter-version", "v2",
            "--total-env-steps", str(args.total_env_steps),
            "--rollout-length", str(args.rollout_length),
            "--max-steps", str(args.max_steps),
            "--seed", "0",
            "--device", args.device,
            "--output-dir", str(policy_dir),
            "--log-csv", str(policy_dir / "train_log.csv"),
            "--opponent-policy", opponent_policy,
            "--save-interval", str(args.save_interval),
        ]
        result = _run(train_cmd, ROOT, f"train {opponent_policy}")
        _assert(result.returncode == 0,
                f"train failed for {opponent_policy}: {result.stderr[-500:]}")

        # Check train outputs
        model_pt = policy_dir / "latest" / "model.pt"
        meta_json = policy_dir / "latest" / "meta.json"
        train_csv = policy_dir / "train_log.csv"
        _assert(model_pt.exists(), f"missing {model_pt}")
        _assert(meta_json.exists(), f"missing {meta_json}")
        _assert(train_csv.exists(), f"missing {train_csv}")

        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        _assert(meta.get("obs_adapter_version") == "v2",
                f"meta obs_adapter_version={meta.get('obs_adapter_version')}")
        actor_dim = int(meta.get("actor_obs_dim", 0))
        critic_dim = int(meta.get("critic_state_dim", 0))
        _assert(actor_dim == 96, f"actor_dim={actor_dim}")
        _assert(critic_dim == 480, f"critic_dim={critic_dim}")

        # Check train log for NaN
        train_nan_detected = False
        with open(train_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("nan_detected", "1")) != 0:
                    train_nan_detected = True
                    break
        _assert(not train_nan_detected,
                f"train_log.csv has nan_detected for {opponent_policy}")

        # ---- 2. Eval ----
        eval_cmd = [
            sys.executable, "-u",
            str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
            "--model", str(model_pt),
            "--obs-adapter-version", "v2",
            "--episodes", str(args.eval_episodes),
            "--device", args.device,
            "--opponent-policy", opponent_policy,
            "--configs", *eval_configs,
            "--summary-json", str(policy_dir / "eval_summary.json"),
        ]
        result = _run(eval_cmd, ROOT, f"eval {opponent_policy}")
        _assert(result.returncode == 0,
                f"eval failed for {opponent_policy}: {result.stderr[-500:]}")

        # Check eval outputs
        eval_json = policy_dir / "eval_summary.json"
        _assert(eval_json.exists(), f"missing {eval_json}")

        eval_data = json.loads(eval_json.read_text(encoding="utf-8"))
        _assert(isinstance(eval_data, list) and len(eval_data) > 0,
                "eval_summary.json is empty")

        eval_nan = any(rec.get("nan_detected", True) for rec in eval_data)
        eval_actor_ok = all(rec.get("actor_dim_ok", False) for rec in eval_data)
        eval_critic_ok = all(rec.get("critic_dim_ok", False) for rec in eval_data)

        _assert(not eval_nan, f"eval nan_detected for {opponent_policy}")
        _assert(eval_actor_ok, f"eval actor_dim_ok=False for {opponent_policy}")
        _assert(eval_critic_ok, f"eval critic_dim_ok=False for {opponent_policy}")

        # Aggregate combat metrics across eval configs
        combat_keys = [
            "red_win_rate", "blue_win_rate", "draw_rate",
            "timeout_rate", "mav_survival_rate",
        ]
        agg_metrics: dict[str, float] = {}
        for key in combat_keys:
            values = [rec.get(key, 0.0) for rec in eval_data if key in rec]
            agg_metrics[key] = float(sum(values) / len(values)) if values else 0.0

        # Per-eval-config records
        for rec in eval_data:
            cfg_name = Path(rec.get("config", "")).name
            record = {
                "opponent_policy": opponent_policy,
                "train_config": train_config,
                "eval_config": rec.get("config", ""),
                "total_env_steps": args.total_env_steps,
                "eval_episodes": args.eval_episodes,
                "actor_dim": actor_dim,
                "critic_dim": critic_dim,
                "train_nan_detected": train_nan_detected,
                "eval_nan_detected": rec.get("nan_detected", True),
                "actor_dim_ok": rec.get("actor_dim_ok", False),
                "critic_dim_ok": rec.get("critic_dim_ok", False),
                "red_win_rate": rec.get("red_win_rate", 0.0),
                "blue_win_rate": rec.get("blue_win_rate", 0.0),
                "draw_rate": rec.get("draw_rate", 0.0),
                "timeout_rate": rec.get("timeout_rate", 0.0),
                "mav_survival_rate": rec.get("mav_survival_rate", 0.0),
                "status": "passed",
            }
            for ck in combat_keys:
                _assert(ck in rec, f"eval record missing combat metric {ck} for {cfg_name}")

            summary_records.append(record)

        print(f"[smoke] {opponent_policy}: PASSED", flush=True)

    # ---- Save summary ----
    summary_json_path = out_dir / "protocol_smoke_summary.json"
    summary_csv_path = out_dir / "protocol_smoke_summary.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    if summary_records:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)

    print(f"\n{'='*60}", flush=True)
    print(f"output_json: {summary_json_path}", flush=True)
    print(f"output_csv:  {summary_csv_path}", flush=True)
    print(f"records: {len(summary_records)}", flush=True)

    if all_passed:
        print("\npaper-aligned protocol smoke passed", flush=True)
        print("This is not a long-run baseline and not a zero-shot claim.", flush=True)
    else:
        print("\npaper-aligned protocol smoke FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
