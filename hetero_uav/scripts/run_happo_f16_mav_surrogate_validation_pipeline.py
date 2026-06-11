"""F-16 MAV surrogate validation pipeline: 200k, gate, optional 1M."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"
OUTPUT_200K = "outputs/happo_3v2_reference_f16_mav_surrogate_200k"
OUTPUT_1M = "outputs/happo_3v2_reference_f16_mav_surrogate_1m_fast"
PIPELINE_DIR = "outputs/happo_3v2_reference_f16_mav_surrogate_pipeline"
GATE_CONDITIONS = [
    "mav_survival_rate >= 0.3",
    "red_missile_hits_mean > 0 or blue_dead_mean > 0",
    "blue_elimination_win_rate < 0.9",
    "not all timeout draw",
    "nan_detected is false",
]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _run(cmd: list[str], label: str, timeout: int | None = None) -> None:
    print(f"[pipeline] {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=timeout)


def _write_json(path: str | Path, data: dict | list) -> Path:
    out = _rel(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _write_md(path: str | Path, lines: list[str]) -> Path:
    out = _rel(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def train_200k_command() -> list[str]:
    return ["python", "-u", "scripts/run_happo_3v2_reference_f16_mav_surrogate_200k.py"]


def train_1m_command() -> list[str]:
    return ["python", "-u", "scripts/run_happo_3v2_reference_f16_mav_surrogate_1m_fast.py"]


def eval_command(exp_dir: str, episodes: int, output_json: str, output_md: str) -> list[str]:
    return [
        "python", "-u", "scripts/evaluate_happo_3v2_reference_checkpoints.py",
        "--experiment-dir", exp_dir,
        "--episodes", str(episodes),
        "--opponent-policy", "brma_rule",
        "--checkpoint-mode", "all",
        "--configs", CONFIG,
        "--output-json", output_json,
        "--output-md", output_md,
    ]


def acmi_command(exp_dir: str, output_acmi: str, output_summary: str) -> list[str]:
    return [
        "python", "-u", "scripts/export_happo_reference_acmi.py",
        "--experiment-dir", exp_dir,
        "--checkpoint", "best",
        "--config", CONFIG,
        "--output", output_acmi,
        "--summary-json", output_summary,
        "--opponent-policy", "brma_rule",
    ]


def _best_3v2(records: list[dict]) -> dict:
    for record in records:
        if record.get("checkpoint") == "best" and "3v2" in str(record.get("config", "")):
            return record
    return {}


def gate_decision(records: list[dict]) -> dict:
    best = _best_3v2(records)
    mav = float(best.get("mav_survival_rate", 0.0) or 0.0)
    hits = float(best.get("red_missile_hits_mean", 0.0) or 0.0)
    blue_dead = float(best.get("blue_dead_mean", 0.0) or 0.0)
    blue_elim = float(best.get("blue_elimination_win_rate", 1.0) or 0.0)
    draw = float(best.get("draw_rate", 1.0) or 0.0)
    timeout = float(best.get("timeout_rate", 0.0) or 0.0)
    nan = bool(best.get("nan_detected", True))
    checks = {
        "mav_survival_rate >= 0.3": mav >= 0.3,
        "red_missile_hits_mean > 0 or blue_dead_mean > 0": hits > 0.0 or blue_dead > 0.0,
        "blue_elimination_win_rate < 0.9": blue_elim < 0.9,
        "not all timeout draw": not (draw >= 0.95 and timeout >= 0.95),
        "nan_detected is false": not nan,
    }
    return {
        "dry_run": False,
        "gate_conditions": GATE_CONDITIONS,
        "checks": checks,
        "passed": all(checks.values()),
        "best_3v2": best,
    }


def write_eval_md(path: str | Path, title: str, records: list[dict]) -> None:
    lines = [f"# {title}", ""]
    for record in records:
        lines.extend([
            f"## {record.get('checkpoint')} - {Path(record.get('config', '')).name}",
            f"- red_win_rate: {record.get('red_win_rate')}",
            f"- blue_win_rate: {record.get('blue_win_rate')}",
            f"- draw_rate: {record.get('draw_rate')}",
            f"- timeout_rate: {record.get('timeout_rate')}",
            f"- red_elimination_win_rate: {record.get('red_elimination_win_rate')}",
            f"- blue_elimination_win_rate: {record.get('blue_elimination_win_rate')}",
            f"- red_timeout_alive_advantage_rate: {record.get('red_timeout_alive_advantage_rate')}",
            f"- blue_timeout_alive_advantage_rate: {record.get('blue_timeout_alive_advantage_rate')}",
            f"- mav_survival_rate: {record.get('mav_survival_rate')}",
            f"- red_alive_final_mean: {record.get('red_alive_final_mean')}",
            f"- blue_alive_final_mean: {record.get('blue_alive_final_mean')}",
            f"- blue_dead_mean: {record.get('blue_dead_mean')}",
            f"- red_missile_hits_mean: {record.get('red_missile_hits_mean')}",
            f"- red_missiles_fired_mean: {record.get('red_missiles_fired_mean')}",
            f"- blue_missiles_fired_mean: {record.get('blue_missiles_fired_mean')}",
            f"- kill_death_ratio: {record.get('kill_death_ratio')}",
            "",
        ])
    _write_md(path, lines)


def write_gate_md(path: str | Path, decision: dict) -> None:
    lines = ["# F-16 MAV Surrogate Gate Decision", "", f"- passed: {decision['passed']}", ""]
    for condition, ok in decision["checks"].items():
        lines.append(f"- {condition}: {ok}")
    _write_md(path, lines)


def dry_run(args) -> int:
    decision = {
        "dry_run": True,
        "gate_conditions": GATE_CONDITIONS,
        "train_200k_command": train_200k_command(),
        "train_1m_command": train_1m_command(),
        "eval_200k_command": eval_command(
            OUTPUT_200K, 50,
            str(Path(args.output_dir) / "surrogate_200k_eval.json"),
            str(Path(args.output_dir) / "surrogate_200k_eval.md"),
        ),
    }
    print("F-16 MAV surrogate validation dry-run", flush=True)
    print("200k command:", " ".join(decision["train_200k_command"]), flush=True)
    print("1M command:", " ".join(decision["train_1m_command"]), flush=True)
    for condition in GATE_CONDITIONS:
        print(f"gate: {condition}", flush=True)
    _write_json(args.decision_json, decision)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="F-16 MAV surrogate HAPPO reference validation pipeline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=PIPELINE_DIR)
    parser.add_argument("--decision-json", default=None)
    args = parser.parse_args()
    pipeline_dir = _rel(args.output_dir)
    decision_json = args.decision_json or str(pipeline_dir / "surrogate_gate_decision.json")
    if args.dry_run:
        return dry_run(argparse.Namespace(**{**vars(args), "decision_json": decision_json}))

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    eval_200k_json = pipeline_dir / "surrogate_200k_eval.json"
    eval_200k_md = pipeline_dir / "surrogate_200k_eval.md"
    gate_json = pipeline_dir / "surrogate_gate_decision.json"
    gate_md = pipeline_dir / "surrogate_gate_decision.md"

    _run(train_200k_command(), "train 200k", timeout=28800)
    _run(eval_command(OUTPUT_200K, 50, str(eval_200k_json), str(eval_200k_md)), "eval 200k", timeout=7200)
    records = json.loads(eval_200k_json.read_text(encoding="utf-8"))
    write_eval_md(eval_200k_md, "F-16 MAV Surrogate 200k Eval", records)
    decision = gate_decision(records)
    _write_json(gate_json, decision)
    write_gate_md(gate_md, decision)
    _run(
        acmi_command(
            OUTPUT_200K,
            f"{OUTPUT_200K}/acmi/best_3v2_episode0.acmi",
            f"{OUTPUT_200K}/acmi/best_3v2_episode0_summary.json",
        ),
        "export 200k best ACMI",
        timeout=1800,
    )

    if not decision["passed"]:
        print("gate_failed: skip 1M", flush=True)
        return 0

    _run(train_1m_command(), "train 1M", timeout=86400)
    eval_1m_json = pipeline_dir / "surrogate_1m_eval.json"
    eval_1m_md = pipeline_dir / "surrogate_1m_eval.md"
    _run(eval_command(OUTPUT_1M, 100, str(eval_1m_json), str(eval_1m_md)), "eval 1M", timeout=14400)
    records_1m = json.loads(eval_1m_json.read_text(encoding="utf-8"))
    write_eval_md(eval_1m_md, "F-16 MAV Surrogate 1M Eval", records_1m)
    _run(
        acmi_command(
            OUTPUT_1M,
            f"{OUTPUT_1M}/acmi/best_3v2_episode0.acmi",
            f"{OUTPUT_1M}/acmi/best_3v2_episode0_summary.json",
        ),
        "export 1M best ACMI",
        timeout=1800,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
