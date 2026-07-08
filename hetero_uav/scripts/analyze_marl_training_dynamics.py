"""Analyze MARL/PPO learning dynamics from existing training logs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "avg_return", "red_win", "blue_win", "draw", "timeout", "mav_survival",
    "red_alive_final", "blue_alive_final", "actor_loss_mav", "actor_loss_uav",
    "critic_loss", "critic_loss_unscaled", "critic_loss_scaled", "entropy_mav",
    "entropy_uav", "approx_kl_mav", "approx_kl_uav", "approx_kl_abs_mav",
    "approx_kl_abs_uav", "clip_fraction_mav", "clip_fraction_uav",
    "ratio_mean_mav", "ratio_mean_uav", "ratio_std_mav", "ratio_std_uav",
    "ratio_p95_mav", "ratio_p95_uav", "ratio_p99_mav", "ratio_p99_uav",
    "actor_grad_norm_mav", "actor_grad_norm_uav", "critic_grad_norm",
    "policy_update_norm_mav", "policy_update_norm_uav", "critic_update_norm",
    "value_explained_variance", "value_pred_mean", "value_pred_std",
    "return_mean", "return_std", "advantage_raw_mean", "advantage_raw_std",
    "advantage_raw_min", "advantage_raw_max", "advantage_norm_mean",
    "advantage_norm_std", "advantage_norm_min", "advantage_norm_max",
    "action_log_std_mav_mean", "action_log_std_uav_mean",
    "mav_action_saturation_rate", "uav_action_saturation_rate",
    "mav_action_saturation_pitch", "mav_action_saturation_heading",
    "mav_action_saturation_speed", "uav_action_saturation_pitch",
    "uav_action_saturation_heading", "uav_action_saturation_speed",
    "mav_action_mean_pitch", "mav_action_mean_heading", "mav_action_mean_speed",
    "uav_action_mean_pitch", "uav_action_mean_heading", "uav_action_mean_speed",
    "mav_action_std_pitch", "mav_action_std_heading", "mav_action_std_speed",
    "uav_action_std_pitch", "uav_action_std_heading", "uav_action_std_speed",
    "mav_active_sample_count", "uav_active_sample_count",
    "paper_v1_uav_flight_sum", "paper_v1_uav_adv_sum",
    "paper_v1_uav_end_sum", "paper_v1_uav_total_sum",
    "paper_v1_mav_flight_sum", "paper_v1_mav_safety_sum",
    "paper_v1_mav_support_sum", "paper_v1_mav_event_raw_sum",
    "paper_v1_mav_scaled_tam_sum", "paper_v1_mav_total_sum",
    "actor_obs_mean", "actor_obs_std", "actor_obs_abs_max", "actor_obs_nan_count",
    "critic_state_mean", "critic_state_std", "critic_state_abs_max", "critic_state_nan_count",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return vals[idx]


def _parse_bins(text: str | None, rows: list[dict]) -> list[tuple[int, int]]:
    if text:
        nums = [int(float(part)) for part in text.split(",") if part.strip()]
    else:
        max_step = max([int(_num(row, "total_steps")) for row in rows], default=0)
        step = max(max_step // 5, 1)
        nums = list(range(0, max_step + step, step))
    if len(nums) < 2:
        nums = [0, max(nums[0] if nums else 1, 1)]
    return list(zip(nums[:-1], nums[1:]))


def _phase_label(start: int, end: int) -> str:
    return f"{start}-{end}"


def _rows_in_phase(rows: list[dict], start: int, end: int) -> list[dict]:
    out = []
    for row in rows:
        step = int(_num(row, "total_steps", _num(row, "step", 0.0)))
        if start < step <= end or (start == 0 and step == 0):
            out.append(row)
    return out


def _summarize_terminal(rows: list[dict], start: int, end: int) -> dict:
    part = _rows_in_phase(rows, start, end)
    out = {"terminal_rows": len(part)}
    for key in ("winner", "end_reason"):
        counts = Counter(str(row.get(key, "")) for row in part)
        for name, count in counts.items():
            if name:
                out[f"{key}_{name}_count"] = count
                out[f"{key}_{name}_rate"] = count / max(len(part), 1)
    lengths = [_num(row, "episode_length") for row in part if "episode_length" in row]
    if lengths:
        out.update({
            "episode_length_mean": _mean(lengths),
            "episode_length_std": _std(lengths),
            "episode_length_min": min(lengths),
            "episode_length_max": max(lengths),
        })
    return out


def _flag_phases(phase_rows: list[dict]) -> dict:
    flags = defaultdict(list)
    kl_values = [
        max(row.get("approx_kl_mav", 0.0), row.get("approx_kl_uav", 0.0))
        for row in phase_rows
    ]
    sorted_kl = sorted(kl_values)
    median_kl = sorted_kl[len(sorted_kl) // 2] if sorted_kl else 0.0
    critic_losses = sorted(row.get("critic_loss", 0.0) for row in phase_rows)
    critic_median = critic_losses[len(critic_losses) // 2] if critic_losses else 0.0
    critic_p95 = _p95(critic_losses)
    first_entropy = phase_rows[0].get("entropy_uav", 0.0) if phase_rows else 0.0
    first_logstd = phase_rows[0].get("action_log_std_uav_mean", 0.0) if phase_rows else 0.0
    for row in phase_rows:
        label = row["phase"]
        if max(row.get("approx_kl_mav", 0.0), row.get("approx_kl_uav", 0.0)) > max(0.03, 5.0 * median_kl):
            flags["ppo_kl_spike"].append(label)
        if max(row.get("clip_fraction_mav", 0.0), row.get("clip_fraction_uav", 0.0)) > 0.3:
            flags["ppo_clip_high"].append(label)
        if critic_p95 > 10.0 * max(critic_median, 1e-6) or row.get("value_explained_variance", 0.0) < 0.0:
            flags["critic_instability"].append(label)
        if row.get("critic_loss", 0.0) <= max(critic_median, 1e-6) and row.get("red_win", 0.0) < 0.05 and row.get("timeout", 0.0) > 0.7:
            flags["critic_low_loss_but_bad_policy"].append(label)
        if (first_entropy > 0 and row.get("entropy_uav", 0.0) < 0.5 * first_entropy) or (
            abs(first_logstd) > 1e-9 and abs(row.get("action_log_std_uav_mean", 0.0)) < 0.5 * abs(first_logstd)
        ):
            flags["entropy_collapse"].append(label)
        if row.get("entropy_uav", 0.0) >= first_entropy and row.get("red_win", 0.0) < 0.05:
            flags["entropy_high_no_improvement"].append(label)
        sat = max(
            row.get("uav_action_saturation_rate", 0.0),
            row.get("uav_action_saturation_pitch", 0.0),
            row.get("uav_action_saturation_heading", 0.0),
            row.get("uav_action_saturation_speed", 0.0),
        )
        if sat > 0.4:
            flags["action_saturation_high"].append(label)
        if row.get("mav_active_sample_count", 0.0) < 0.5 * max(row.get("rollout_transitions", 1.0), 1.0):
            flags["active_sample_imbalance"].append(label)
        if row.get("red_win", 0.0) < 0.05 and row.get("timeout", 0.0) > 0.7 and row.get("mav_survival", 0.0) > 0.7:
            flags["survival_timeout_local_optimum"].append(label)
        if (
            row.get("red_win", 0.0) < 0.05
            and row.get("timeout", 0.0) > 0.5
            and row.get("paper_v1_mav_safety_sum", 0.0) > abs(row.get("paper_v1_uav_adv_sum", 0.0))
        ):
            flags["reward_misalignment"].append(label)
    return {key: sorted(set(value)) for key, value in flags.items()}


def analyze(output_dir: str | Path, phase_bins: str | None = None) -> dict:
    out_dir = Path(output_dir)
    train_rows = _read_csv(out_dir / "train_log.csv")
    terminal_rows = _read_csv(out_dir / "terminal_episode_audit.csv")
    update_rows = _read_jsonl(out_dir / "update_diagnostics.jsonl")
    reward_rows = _read_csv(out_dir / "rich_logs" / "episode_reward_components.csv")
    phases = _parse_bins(phase_bins, train_rows)
    summary_rows = []
    for start, end in phases:
        part = _rows_in_phase(train_rows, start, end)
        row = {"phase": _phase_label(start, end), "phase_start": start, "phase_end": end, "train_rows": len(part)}
        for field in SUMMARY_FIELDS:
            row[field] = _mean([_num(item, field) for item in part]) if part else 0.0
        row.update(_summarize_terminal(terminal_rows, start, end))
        summary_rows.append(row)
    flags = _flag_phases(summary_rows)
    _write_csv(out_dir / "learning_dynamics_summary.csv", summary_rows)
    payload = {
        "output_dir": str(out_dir),
        "train_rows": len(train_rows),
        "terminal_episode_rows": len(terminal_rows),
        "update_diagnostics_rows": len(update_rows),
        "reward_component_rows": len(reward_rows),
        "phases": summary_rows,
        "diagnostic_flags": flags,
        "notes": [
            "Missing fields are treated as 0.0 for backward-compatible old logs.",
            "Flags are heuristic screening signals, not causal proof.",
        ],
    }
    (out_dir / "learning_dynamics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_report(out_dir / "learning_dynamics_report.md", payload)
    return payload


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict) -> None:
    lines = [
        "# MARL/PPO Training Dynamics Report",
        "",
        f"- output_dir: `{payload['output_dir']}`",
        f"- train_log rows: {payload['train_rows']}",
        f"- update_diagnostics rows: {payload['update_diagnostics_rows']}",
        f"- terminal_episode_audit rows: {payload['terminal_episode_rows']}",
        "",
        "## Heuristic Flags",
        "",
    ]
    if payload["diagnostic_flags"]:
        for name, phases in payload["diagnostic_flags"].items():
            lines.append(f"- {name}: {', '.join(phases)}")
    else:
        lines.append("- No major heuristic risk flag was triggered.")
    lines.extend([
        "",
        "## Reading Order",
        "",
        "- Check KL, clip_fraction, and grad norms first to screen for overly aggressive PPO updates.",
        "- Check explained_variance and critic_loss next to decide whether the critic target is usable.",
        "- Check entropy, log_std, and action saturation to identify actor collapse or boundary actions.",
        "- Combine timeout, MAV survival, and reward components to detect survival/timeout local optima.",
        "",
        "## Phase Summary",
        "",
        "| phase | return | red_win | timeout | mav_surv | KL_uav | clip_uav | critic | EV | uav_sat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["phases"]:
        lines.append(
            f"| {row['phase']} | {row['avg_return']:.3f} | {row['red_win']:.3f} | "
            f"{row['timeout']:.3f} | {row['mav_survival']:.3f} | "
            f"{row['approx_kl_uav']:.5f} | {row['clip_fraction_uav']:.3f} | "
            f"{row['critic_loss']:.3f} | {row['value_explained_variance']:.3f} | "
            f"{row['uav_action_saturation_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-bins", default=None)
    args = parser.parse_args()
    payload = analyze(args.output_dir, phase_bins=args.phase_bins)
    print(f"wrote {Path(args.output_dir) / 'learning_dynamics_report.md'}")
    print(f"phases={len(payload['phases'])} flags={len(payload['diagnostic_flags'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
