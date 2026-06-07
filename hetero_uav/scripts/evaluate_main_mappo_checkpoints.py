"""Evaluate selected MAPPO checkpoints from a training run.

Default mode samples a small subset of checkpoints so the main experiment can
move forward quickly. Use ``--selection-mode all`` only for explicit slow audits.
This script does not modify the environment or algorithm.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
OBS_ADAPTER = "v2"


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r"iter_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _discover_checkpoints(exp_dir: Path, include_latest: bool) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    ckpt_dir = exp_dir / "checkpoints"
    if ckpt_dir.exists():
        for pt in sorted(ckpt_dir.glob("iter_*.pt"), key=_checkpoint_iteration):
            checkpoints.append({
                "checkpoint": pt.name,
                "iteration": _checkpoint_iteration(pt),
                "path": str(pt),
                "is_latest": False,
            })
    latest_pt = exp_dir / "latest" / "model.pt"
    if include_latest and latest_pt.exists():
        checkpoints.append({
            "checkpoint": "latest",
            "iteration": -1,
            "path": str(latest_pt),
            "is_latest": True,
        })
    return checkpoints


def _limit_preserve_latest(
    selected: list[dict[str, Any]], max_checkpoints: int
) -> list[dict[str, Any]]:
    if len(selected) <= max_checkpoints:
        return selected
    latest = [item for item in selected if item.get("is_latest")]
    non_latest = [item for item in selected if not item.get("is_latest")]
    keep_count = max_checkpoints - len(latest)
    if keep_count <= 0:
        return latest[:max_checkpoints]
    return non_latest[:keep_count] + latest


def _select_sampled(
    checkpoints: list[dict[str, Any]], stride: int, max_checkpoints: int
) -> list[dict[str, Any]]:
    regular = [item for item in checkpoints if not item.get("is_latest")]
    latest = [item for item in checkpoints if item.get("is_latest")]
    sampled = regular[::max(1, stride)] + latest
    return _limit_preserve_latest(sampled, max_checkpoints)


def _select_top_train(
    exp_dir: Path, checkpoints: list[dict[str, Any]], max_checkpoints: int
) -> list[dict[str, Any]]:
    train_csv = exp_dir / "train_log.csv"
    if not train_csv.exists():
        raise SystemExit(f"[ckpt] ERROR: missing train_log.csv for top-train: {train_csv}")
    by_iter = {
        int(item["iteration"]): item
        for item in checkpoints
        if not item.get("is_latest") and int(item["iteration"]) >= 0
    }
    ranked: list[tuple[float, int]] = []
    with open(train_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                iteration = int(row.get("iteration", "-1"))
                value = float(row.get("average_team_return", "-inf"))
            except ValueError:
                continue
            if iteration in by_iter:
                ranked.append((value, iteration))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = [by_iter[it] for _, it in ranked]
    selected.extend(item for item in checkpoints if item.get("is_latest"))
    return _limit_preserve_latest(selected, max_checkpoints)


def _select_checkpoints(
    exp_dir: Path,
    checkpoints: list[dict[str, Any]],
    mode: str,
    max_checkpoints: int,
    stride: int,
) -> list[dict[str, Any]]:
    if mode == "all":
        print("[ckpt] WARNING: ALL checkpoint eval may be very slow", flush=True)
        return checkpoints
    if mode == "top-train":
        return _select_top_train(exp_dir, checkpoints, max_checkpoints)
    return _select_sampled(checkpoints, stride, max_checkpoints)


def _run_eval(
    checkpoint: dict[str, Any],
    episode_count: int,
    device: str,
    opponent_policy: str,
) -> list[dict[str, Any]] | None:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint["checkpoint"])
    output_json = ROOT / "outputs" / "_checkpoint_eval_tmp" / f"{safe_name}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "eval_mappo_zero_shot.py"),
        "--model",
        checkpoint["path"],
        "--obs-adapter-version",
        OBS_ADAPTER,
        "--episodes",
        str(episode_count),
        "--device",
        device,
        "--opponent-policy",
        opponent_policy,
        "--configs",
        *EVAL_CONFIGS,
        "--summary-json",
        str(output_json),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )
    if result.returncode != 0:
        print(f"[ckpt] FAIL {checkpoint['checkpoint']}", flush=True)
        print(result.stdout[-500:], flush=True)
        print(result.stderr[-500:], flush=True)
        return None
    return json.loads(output_json.read_text(encoding="utf-8"))


def _score(records: list[dict[str, Any]]) -> float:
    for record in records:
        if "3v2" in record.get("config", ""):
            return (
                float(record.get("red_win_rate", 0.0))
                + 0.1 * float(record.get("mav_survival_rate", 0.0))
                + 0.01 * float(record.get("avg_return", 0.0))
            )
    return 0.0


def _summary_line(records: list[dict[str, Any]]) -> str:
    by_name = {Path(r.get("config", "")).name: r for r in records}
    r3 = by_name.get("hetero_mav_shared_geo_3v2.yaml", {})
    r5 = by_name.get("hetero_mav_shared_geo_5v4.yaml", {})
    return (
        f"3v2_return={float(r3.get('avg_return', 0.0)):.3f} "
        f"3v2_red_win={float(r3.get('red_win_rate', 0.0)):.3f} "
        f"3v2_mav_survival={float(r3.get('mav_survival_rate', 0.0)):.3f} "
        f"5v4_return={float(r5.get('avg_return', 0.0)):.3f} "
        f"5v4_red_win={float(r5.get('red_win_rate', 0.0)):.3f}"
    )


def _record_from(records: list[dict[str, Any]], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "checkpoint": checkpoint["checkpoint"],
            "iteration": checkpoint["iteration"],
            "eval_config": Path(record.get("config", "")).name,
            "avg_return": record.get("avg_return", 0.0),
            "avg_length": record.get("avg_length", 0.0),
            "red_win_rate": record.get("red_win_rate", 0.0),
            "blue_win_rate": record.get("blue_win_rate", 0.0),
            "draw_rate": record.get("draw_rate", 0.0),
            "timeout_rate": record.get("timeout_rate", 0.0),
            "mav_survival_rate": record.get("mav_survival_rate", 0.0),
            "red_alive_final_mean": record.get("red_alive_final_mean", 0.0),
            "blue_alive_final_mean": record.get("blue_alive_final_mean", 0.0),
            "nan_detected": record.get("nan_detected", True),
            "actor_dim_ok": record.get("actor_dim_ok", False),
            "critic_dim_ok": record.get("critic_dim_ok", False),
        })
    return rows


def _load_existing_records(output_json: Path) -> list[dict[str, Any]]:
    if not output_json.exists():
        return []
    data = json.loads(output_json.read_text(encoding="utf-8"))
    return list(data.get("records", []))


def _write_outputs(output_json: Path, output_csv: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    if records:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="outputs/main_mappo_experiment_f22_100k")
    parser.add_argument("--selection-mode", choices=["sampled", "top-train", "all"], default="sampled")
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--checkpoint-stride", type=int, default=10)
    parser.add_argument("--quick-eval-episodes", type=int, default=2)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-latest", action="store_true", default=True)
    parser.add_argument("--no-include-latest", action="store_false", dest="include_latest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="greedy_fsm")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    if args.max_checkpoints <= 0:
        raise SystemExit("--max-checkpoints must be positive")
    if args.checkpoint_stride <= 0:
        raise SystemExit("--checkpoint-stride must be positive")
    eval_episodes = args.eval_episodes or args.quick_eval_episodes

    exp_dir = Path(args.experiment_dir)
    all_checkpoints = _discover_checkpoints(exp_dir, args.include_latest)
    if not all_checkpoints:
        print("[ckpt] ERROR: no checkpoints found", flush=True)
        raise SystemExit(1)
    selected = _select_checkpoints(
        exp_dir,
        all_checkpoints,
        args.selection_mode,
        args.max_checkpoints,
        args.checkpoint_stride,
    )
    if not selected:
        print("[ckpt] ERROR: no checkpoints selected", flush=True)
        raise SystemExit(1)

    output_json = Path(args.output_json or str(exp_dir / "checkpoint_eval_summary.json"))
    output_csv = Path(args.output_csv or str(exp_dir / "checkpoint_eval_summary.csv"))
    records = _load_existing_records(output_json) if args.resume else []
    completed = {row["checkpoint"] for row in records}
    ranked: list[tuple[float, str, int]] = []

    print(f"[ckpt] total checkpoints found: {len(all_checkpoints)}", flush=True)
    print(f"[ckpt] selected checkpoints: {len(selected)}", flush=True)
    print(f"[ckpt] selection_mode={args.selection_mode} episodes={eval_episodes}", flush=True)

    for idx, checkpoint in enumerate(selected, start=1):
        name = checkpoint["checkpoint"]
        if args.resume and name in completed:
            print(f"[ckpt] {idx}/{len(selected)} checkpoint={name} skipped(resume)", flush=True)
            continue
        print(f"[ckpt] {idx}/{len(selected)} checkpoint={name} episodes={eval_episodes}", flush=True)
        eval_records = _run_eval(checkpoint, eval_episodes, args.device, args.opponent_policy)
        if eval_records is None:
            continue
        score = _score(eval_records)
        ranked.append((score, name, int(checkpoint["iteration"])))
        records.extend(_record_from(eval_records, checkpoint))
        print(f"[ckpt] {name} score={score:.4f} {_summary_line(eval_records)}", flush=True)

        partial_summary = {
            "total_checkpoints_found": len(all_checkpoints),
            "selected_checkpoints": [item["checkpoint"] for item in selected],
            "selection_mode": args.selection_mode,
            "eval_episodes": eval_episodes,
            "no_effective_checkpoint_found": False,
        }
        _write_outputs(output_json, output_csv, records, partial_summary)

    if not records:
        print("[ckpt] ERROR: all selected checkpoint evaluations failed", flush=True)
        raise SystemExit(1)

    if not ranked:
        ranked = [
            (0.0, name, -1)
            for name in sorted({row["checkpoint"] for row in records})
        ]
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_items = ranked[: args.top_k]
    any_red_win = any(float(row["red_win_rate"]) > 0.0 for row in records)
    any_mav_surv = any(float(row["mav_survival_rate"]) > 0.0 for row in records)
    all_blue_win = all(float(row["blue_win_rate"]) == 1.0 for row in records)

    summary = {
        "total_checkpoints_found": len(all_checkpoints),
        "selected_checkpoints": [item["checkpoint"] for item in selected],
        "selection_mode": args.selection_mode,
        "eval_episodes": eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "top_k_requested": args.top_k,
        "any_red_win_gt_zero": any_red_win,
        "any_mav_survival_gt_zero": any_mav_surv,
        "all_selected_blue_win_rate_1": all_blue_win,
        "no_effective_checkpoint_found": not any_red_win and all_blue_win,
        "top_checkpoints": [
            {"checkpoint": name, "iteration": iteration, "score": round(score, 4)}
            for score, name, iteration in top_items
        ],
    }
    _write_outputs(output_json, output_csv, records, summary)

    print(f"output_json: {output_json}", flush=True)
    print(f"output_csv:  {output_csv}", flush=True)
    print(f"selected/evaluated records: {len(records)}", flush=True)
    print(f"any_red_win>0: {any_red_win}", flush=True)
    print(f"any_mav_surv>0: {any_mav_surv}", flush=True)
    print(f"no_effective_checkpoint_found: {summary['no_effective_checkpoint_found']}", flush=True)


if __name__ == "__main__":
    main()
