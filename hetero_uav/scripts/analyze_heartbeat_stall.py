"""Analyze heartbeat logs after a stalled HAPPO run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in line.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key] = value
    return out


def _read_last_train_iteration(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _infer_stall_event(last: dict[str, str]) -> str:
    event = last.get("event", "unknown")
    if event.startswith("before_"):
        return event.removeprefix("before_")
    if event == "after_policy_act":
        return "between_policy_and_opponent"
    if event == "after_opponent_act":
        return "between_opponent_and_env_step"
    if event == "after_env_step":
        return "after_env_step_or_next_transition"
    if event == "after_reset":
        return "after_reset_or_next_transition"
    if event == "after_logging":
        return "after_logging_or_next_rollout"
    return "unknown"


def analyze(output_dir: Path) -> dict:
    heartbeat_path = output_dir / "heartbeat.log"
    lines = heartbeat_path.read_text(encoding="utf-8").splitlines() if heartbeat_path.exists() else []
    events = [_parse_line(line) for line in lines if line.strip()]
    last = events[-1] if events else {}
    report_path = output_dir / "heartbeat_stall_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    reset_count = sum(1 for event in events if event.get("event") == "before_reset")
    max_steps = last.get("max_steps") or report.get("last_event", {}).get("max_steps")
    num_envs = last.get("num_envs") or report.get("last_event", {}).get("num_envs")
    total_steps = int(float(last.get("total_env_steps_actual", 0) or 0))
    reset_frequency = reset_count / max(total_steps, 1)
    event_counts = Counter(event.get("event", "unknown") for event in events)
    analysis = {
        "output_dir": str(output_dir),
        "last_train_iteration": _read_last_train_iteration(output_dir / "train_log.csv"),
        "last_heartbeat": last,
        "last_20_heartbeats": lines[-20:],
        "stall_event_guess": _infer_stall_event(last),
        "env_idx": last.get("env_idx"),
        "rollout_local_step": last.get("rollout_local_step"),
        "env_episode_step": last.get("env_episode_step"),
        "reset_count": reset_count,
        "reset_frequency_per_env_step": reset_frequency,
        "max_steps": max_steps,
        "num_envs": num_envs,
        "event_counts": dict(event_counts),
        "heartbeat_stall_report_present": report_path.exists(),
        "heartbeat_stall_report": report,
        "high_frequency_reset_likely": str(max_steps) == "64" or reset_frequency > 0.01,
    }
    return analysis


def _write_markdown(path: Path, data: dict) -> None:
    last = data.get("last_heartbeat", {})
    md = [
        "# Heartbeat Stall Analysis",
        "",
        f"- output_dir: `{data['output_dir']}`",
        f"- stall_event_guess: `{data['stall_event_guess']}`",
        f"- env_idx: `{data.get('env_idx')}`",
        f"- rollout_local_step: `{data.get('rollout_local_step')}`",
        f"- env_episode_step: `{data.get('env_episode_step')}`",
        f"- max_steps: `{data.get('max_steps')}`",
        f"- num_envs: `{data.get('num_envs')}`",
        f"- reset_count: `{data.get('reset_count')}`",
        f"- reset_frequency_per_env_step: `{data.get('reset_frequency_per_env_step'):.6f}`",
        f"- high_frequency_reset_likely: `{data.get('high_frequency_reset_likely')}`",
        "",
        "## Last Heartbeat",
        "",
        "```json",
        json.dumps(last, indent=2),
        "```",
        "",
        "## Last 20 Heartbeats",
        "",
        "```text",
        *data.get("last_20_heartbeats", []),
        "```",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/debug_4env_max1000_500k")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    data = analyze(output_dir)
    out_json = output_dir / "heartbeat_stall_analysis.json"
    out_md = output_dir / "heartbeat_stall_analysis.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(out_md, data)
    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)
    print(f"stall_event_guess: {data['stall_event_guess']}", flush=True)


if __name__ == "__main__":
    main()
