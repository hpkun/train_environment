"""Audit terminal reward semantics from existing episode/train logs.

The script estimates whether a run's terminal outcomes are closer to all-red
count difference, attack-UAV-only difference, or a weighted-MAV count.  It is a
read-only diagnostic and does not infer exact reward implementation when fields
are missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key not in row:
            continue
        try:
            value = row.get(key)
            if value in ("", None):
                continue
            out = float(value)
            return out if math.isfinite(out) else default
        except (TypeError, ValueError):
            continue
    return default


def _bool_like(row: dict, *keys: str) -> bool:
    for key in keys:
        if key not in row:
            continue
        value = str(row.get(key, "")).strip().lower()
        if value in {"1", "true", "yes", "red", "red_win"}:
            return True
        if value in {"0", "false", "no", "blue", "draw", "timeout"}:
            return False
        try:
            return float(value) > 0.5
        except ValueError:
            continue
    return False


def _episode_source(run_dir: Path) -> tuple[str, list[dict]]:
    candidates = [
        ("terminal_episode_audit", run_dir / "terminal_episode_audit.csv"),
        ("rich_episode_reward_components", run_dir / "rich_logs" / "episode_reward_components.csv"),
        ("eval_episode_metrics", run_dir / "rich_logs" / "eval_episode_metrics.csv"),
        ("train_log_recent_window", run_dir / "train_log.csv"),
    ]
    for name, path in candidates:
        rows = _read_rows(path)
        if rows:
            return name, rows
    return "missing", []


def build_terminal_semantics_audit(output_dir: str | Path, mav_weight: float = 1.0) -> dict:
    run_dir = Path(output_dir)
    out_dir = run_dir / "degradation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_name, rows = _episode_source(run_dir)
    audit_rows: list[dict] = []
    for idx, row in enumerate(rows):
        red_alive = _num(row, "red_alive_final", "red_alive", "red_alive_env", default=0.0)
        blue_alive = _num(row, "blue_alive_final", "blue_alive", "blue_alive_env", default=0.0)
        mav_alive = _num(row, "mav_survival", "mav_alive", "mav_alive_final", default=0.0)
        red_fire = _num(
            row, "red_missiles_fired", "red_launch_count",
            "red_episode_missiles_fired_mean", default=0.0)
        red_hits = _num(
            row, "red_missile_hits", "red_hit_count",
            "red_episode_missile_hits_mean", default=0.0)
        timeout = _bool_like(row, "timeout") or "timeout" in str(row.get("outcome", "")).lower()
        red_win = _bool_like(row, "red_win") or str(row.get("winner", "")).lower() == "red"
        all_red_count_diff = red_alive - blue_alive
        attack_uav_alive = max(red_alive - mav_alive, 0.0)
        attack_uav_only_diff = attack_uav_alive - blue_alive
        weighted_mav_diff = attack_uav_alive + float(mav_weight) * mav_alive - blue_alive
        audit_rows.append({
            "source": source_name,
            "row_index": idx,
            "episode": row.get("episode", row.get("episode_id", idx)),
            "red_win": int(red_win),
            "timeout": int(timeout),
            "mav_alive": mav_alive,
            "red_alive_final": red_alive,
            "blue_alive_final": blue_alive,
            "red_fire": red_fire,
            "red_hits": red_hits,
            "all_red_count_diff": all_red_count_diff,
            "attack_uav_only_diff": attack_uav_only_diff,
            "weighted_mav_diff": weighted_mav_diff,
            "timeout_alive_advantage": int(timeout and all_red_count_diff > 0),
            "kill_win": int(red_win and red_hits > 0),
            "no_kill_red_win": int(red_win and red_hits <= 0),
            "all_red_positive_attack_uav_nonpositive": int(
                all_red_count_diff > 0 and attack_uav_only_diff <= 0),
        })

    def rate(key: str) -> float:
        return mean([float(r[key]) for r in audit_rows]) if audit_rows else 0.0

    summary = {
        "output_dir": str(run_dir),
        "source": source_name,
        "rows": len(audit_rows),
        "mav_weight": float(mav_weight),
        "timeout_alive_advantage_rate": rate("timeout_alive_advantage"),
        "kill_win_rate": rate("kill_win"),
        "no_kill_red_win_rate": rate("no_kill_red_win"),
        "all_red_positive_attack_uav_nonpositive_rate": rate("all_red_positive_attack_uav_nonpositive"),
        "all_red_count_diff_mean": mean([r["all_red_count_diff"] for r in audit_rows]) if audit_rows else 0.0,
        "attack_uav_only_diff_mean": mean([r["attack_uav_only_diff"] for r in audit_rows]) if audit_rows else 0.0,
        "weighted_mav_diff_mean": mean([r["weighted_mav_diff"] for r in audit_rows]) if audit_rows else 0.0,
        "notes": [
            "This audit estimates terminal semantics from available logs; it is not a proof of reward code.",
            "If source is train_log_recent_window, rows are recent-window aggregates rather than individual episodes.",
        ],
    }
    _write_csv(out_dir / "terminal_reward_semantics.csv", audit_rows)
    (out_dir / "terminal_reward_semantics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(out_dir / "terminal_reward_semantics.md", summary)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, summary: dict) -> None:
    lines = [
        "# Terminal Reward Semantics Audit",
        "",
        f"- Output dir: `{summary['output_dir']}`",
        f"- Source: `{summary['source']}`",
        f"- Rows: {summary['rows']}",
        f"- MAV weight: {summary['mav_weight']}",
        "",
        "## Rates",
        "",
        f"- timeout_alive_advantage_rate: {summary['timeout_alive_advantage_rate']:.4f}",
        f"- kill_win_rate: {summary['kill_win_rate']:.4f}",
        f"- no_kill_red_win_rate: {summary['no_kill_red_win_rate']:.4f}",
        f"- all_red_positive_attack_uav_nonpositive_rate: {summary['all_red_positive_attack_uav_nonpositive_rate']:.4f}",
        "",
        "## Difference Means",
        "",
        f"- all_red_count_diff_mean: {summary['all_red_count_diff_mean']:.4f}",
        f"- attack_uav_only_diff_mean: {summary['attack_uav_only_diff_mean']:.4f}",
        f"- weighted_mav_diff_mean: {summary['weighted_mav_diff_mean']:.4f}",
        "",
    ]
    lines.extend(f"- {note}" for note in summary["notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mav-weight", type=float, default=1.0)
    args = parser.parse_args()
    summary = build_terminal_semantics_audit(args.output_dir, mav_weight=args.mav_weight)
    print(f"wrote {Path(args.output_dir) / 'degradation_audit' / 'terminal_reward_semantics.json'}")
    print(f"source={summary['source']} rows={summary['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
