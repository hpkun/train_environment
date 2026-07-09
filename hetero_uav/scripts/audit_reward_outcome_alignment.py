"""Read-only reward/outcome alignment audit for pure HAPPO logs.

The script reads existing output CSV/JSONL files and writes audit artifacts. It
does not import or mutate the environment, reward implementation, missile model,
PID, blue policy, action space, observation space, or training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TAM_MAV_FIELDS = [
    "paper_v1_mav_safety_sum",
    "paper_v1_mav_support_sum",
    "paper_v1_mav_event_raw_sum",
    "paper_v1_mav_scaled_tam_sum",
    "paper_v1_mav_total_sum",
]
BRMA_UAV_FIELDS = [
    "paper_v1_uav_flight_sum",
    "paper_v1_uav_adv_sum",
    "paper_v1_uav_end_sum",
    "paper_v1_uav_total_sum",
]
DIAGNOSTIC_FIELDS = [
    "paper_v1_mav_removed_r_adv_sum",
    "paper_v1_mav_removed_r_end_sum",
    "paper_v1_mav_r_death_log_sum",
    "paper_v1_uav_r_death_log_sum",
]
COMPONENT_FIELDS = TAM_MAV_FIELDS + BRMA_UAV_FIELDS + DIAGNOSTIC_FIELDS


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


def _str(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return str(value)
    return ""


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = _mean(vals)
    return math.sqrt(sum((value - mean) ** 2 for value in vals) / len(vals))


def _finite(value) -> float | str:
    try:
        out = float(value)
        return out if math.isfinite(out) else ""
    except (TypeError, ValueError):
        return ""


def _csv_value(value) -> float | str:
    if isinstance(value, str):
        return value
    return _finite(value)


def _phase_bins(text: str | None) -> list[tuple[int, int]]:
    if not text:
        return []
    nums = [int(float(part)) for part in text.split(",") if part.strip()]
    if len(nums) < 2:
        return []
    return list(zip(nums[:-1], nums[1:]))


def _outcome_group(winner: str, end_reason: str) -> str:
    winner = (winner or "").lower()
    end_reason = (end_reason or "").lower()
    if winner == "red" and "blue_eliminated" in end_reason:
        return "red_win_blue_eliminated"
    if winner == "red" and "timeout" in end_reason:
        return "red_win_timeout"
    if winner == "blue" and "red_eliminated" in end_reason:
        return "blue_win_red_eliminated"
    if winner == "blue" and "timeout" in end_reason:
        return "blue_win_timeout"
    if winner == "draw" and "timeout" in end_reason:
        return "draw_timeout"
    return "other"


def _group_by_episode(component_rows: list[dict], terminal_rows: list[dict]) -> tuple[list[dict], dict]:
    terminal_by_ep = {
        str(row.get("episode_id")): row
        for row in terminal_rows
        if row.get("episode_id") not in ("", None)
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in component_rows:
        grouped[str(row.get("episode_id", ""))].append(row)

    episodes = []
    exact_matches = 0
    for episode_id, rows in sorted(grouped.items(), key=lambda kv: (str(kv[0]))):
        red_rows = [row for row in rows if _str(row, "team") in ("", "red")]
        if not red_rows:
            red_rows = rows
        terminal = terminal_by_ep.get(episode_id)
        if terminal:
            exact_matches += 1
        first = red_rows[0]
        roles = [_str(row, "role") for row in red_rows]
        mav_rows = [row for row in red_rows if _str(row, "role") == "mav" or _str(row, "agent_id") == "red_0"]
        uav_rows = [row for row in red_rows if row not in mav_rows]
        red_count = max(len(red_rows), 1)
        uav_count = len(uav_rows)
        winner = _str(terminal or {}, "winner") or _str(first, "outcome")
        end_reason = _str(terminal or {}, "end_reason") or _str(first, "end_reason")
        red_alive = _num(terminal or {}, "red_alive_info", _num(first, "red_alive_final", 0.0))
        blue_alive = _num(terminal or {}, "blue_alive_info", _num(first, "blue_alive_final", 0.0))
        mav_alive = _num(terminal or {}, "mav_alive_info", _num(first, "mav_alive_final", 0.0))
        episode = {
            "episode_id": episode_id,
            "episode_length": _num(first, "episode_length", _num(terminal or {}, "episode_length", 0.0)),
            "outcome": winner,
            "end_reason": end_reason,
            "outcome_group": _outcome_group(winner, end_reason),
            "red_alive_final": red_alive,
            "blue_alive_final": blue_alive,
            "mav_alive_final": mav_alive,
            "red_count_observed": red_count,
            "uav_count_observed": uav_count,
            "team_return": _mean([_num(row, "episode_return") for row in red_rows]),
            "mav_return": _mean([_num(row, "episode_return") for row in mav_rows]),
            "uav_return_mean": _mean([_num(row, "episode_return") for row in uav_rows]),
            "uav_return_sum": sum(_num(row, "episode_return") for row in uav_rows),
            "red_launch_count": _num(terminal or {}, "red_launch_count", _num(first, "red_launch_count", 0.0)),
            "red_hit_count": _num(terminal or {}, "red_hit_count", _num(first, "red_hit_count", 0.0)),
            "blue_launch_count": _num(terminal or {}, "blue_launch_count", _num(first, "blue_launch_count", 0.0)),
            "blue_hit_count": _num(terminal or {}, "blue_hit_count", _num(first, "blue_hit_count", 0.0)),
        }
        for field in COMPONENT_FIELDS:
            episode[field] = sum(_num(row, field) for row in red_rows)
        _add_proxies(episode)
        episodes.append(episode)

    merge_info = {
        "component_episode_count": len(grouped),
        "terminal_episode_count": len(terminal_by_ep),
        "terminal_exact_match_count": exact_matches,
        "terminal_unmatched_count": max(len(terminal_by_ep) - exact_matches, 0),
    }
    return episodes, merge_info


def _add_proxies(ep: dict) -> None:
    uav_count = int(max(ep.get("uav_count_observed", 0), 0))
    red_alive = _finite(ep.get("red_alive_final")) or 0.0
    mav_alive = 1.0 if float(ep.get("mav_alive_final", 0.0)) > 0.5 else 0.0
    uav_alive = max(float(red_alive) - mav_alive, 0.0)
    red_uav_death = max(float(uav_count) - uav_alive, 0.0)
    red_out_of_zone = _finite(ep.get("red_out_of_zone_count", ""))
    missing_out_of_zone = 0 if red_out_of_zone != "" else 1
    red_out_of_zone = 0.0 if red_out_of_zone == "" else float(red_out_of_zone)
    red_hits = float(ep.get("red_hit_count", 0.0))
    mav_death = 1.0 - mav_alive
    tam_uav = 200.0 * red_hits - 200.0 * red_uav_death - 100.0 * red_out_of_zone
    tam_mav = -200.0 * mav_death + min(100.0 * red_hits, 200.0)
    tam_total = tam_uav + tam_mav
    brma_team = 30.0 * (float(ep.get("red_alive_final", 0.0)) - float(ep.get("blue_alive_final", 0.0)))
    red_count = max(float(ep.get("red_count_observed", 1.0)), 1.0)
    ep.update({
        "tam_proxy_uav_event": tam_uav,
        "tam_proxy_mav_event": tam_mav,
        "tam_proxy_event_total": tam_total,
        "current_team_return_minus_tam_proxy_event": float(ep.get("team_return", 0.0)) - tam_total,
        "proxy_death_inferred": 1,
        "red_out_of_zone_count": red_out_of_zone,
        "missing_out_of_zone": missing_out_of_zone,
        "proxy_missing_flags": "missing_out_of_zone" if missing_out_of_zone else "",
        "brma_proxy_terminal_team": brma_team,
        "brma_proxy_terminal_per_agent": brma_team / red_count,
    })


def _positive(value: float) -> float:
    return max(float(value), 0.0)


def _negative_abs(value: float) -> float:
    return abs(min(float(value), 0.0))


def _summary_for_group(group: str, rows: list[dict]) -> dict:
    out = {
        "outcome_group": group,
        "episode_count": len(rows),
        "mean_team_return": _mean([float(row["team_return"]) for row in rows]),
        "std_team_return": _std([float(row["team_return"]) for row in rows]),
        "mean_mav_return": _mean([float(row["mav_return"]) for row in rows]),
        "mean_uav_return": _mean([float(row["uav_return_mean"]) for row in rows]),
        "mean_episode_length": _mean([float(row["episode_length"]) for row in rows]),
        "mean_red_alive_final": _mean([float(row["red_alive_final"]) for row in rows]),
        "mean_blue_alive_final": _mean([float(row["blue_alive_final"]) for row in rows]),
        "mean_mav_alive_final": _mean([float(row["mav_alive_final"]) for row in rows]),
        "mean_red_launch_count": _mean([float(row["red_launch_count"]) for row in rows]),
        "mean_red_hit_count": _mean([float(row["red_hit_count"]) for row in rows]),
        "mean_blue_launch_count": _mean([float(row["blue_launch_count"]) for row in rows]),
        "mean_blue_hit_count": _mean([float(row["blue_hit_count"]) for row in rows]),
    }
    for field in COMPONENT_FIELDS:
        vals = [float(row.get(field, 0.0)) for row in rows]
        out[f"{field}_mean"] = _mean(vals)
        out[f"{field}_std"] = _std(vals)
    dense_vals = []
    neg_vals = []
    for row in rows:
        dense = (
            _positive(row.get("paper_v1_uav_flight_sum", 0.0))
            + _positive(row.get("paper_v1_uav_adv_sum", 0.0))
            + _positive(row.get("paper_v1_mav_safety_sum", 0.0))
            + _positive(row.get("paper_v1_mav_support_sum", 0.0))
        )
        neg = (
            _negative_abs(row.get("paper_v1_uav_end_sum", 0.0))
            + _negative_abs(row.get("paper_v1_mav_scaled_tam_sum", 0.0))
            + _negative_abs(row.get("paper_v1_mav_r_death_log_sum", 0.0))
            + _negative_abs(row.get("paper_v1_uav_r_death_log_sum", 0.0))
        )
        dense_vals.append(dense)
        neg_vals.append(neg)
    dense_mean = _mean(dense_vals)
    neg_mean = _mean(neg_vals)
    out["dense_positive_sum_mean"] = dense_mean
    out["terminal_or_event_negative_sum_mean"] = neg_mean
    out["dense_to_terminal_abs_ratio"] = dense_mean / max(neg_mean, 1e-6)
    total_abs = sum(abs(out.get(f"{field}_mean", 0.0)) for field in COMPONENT_FIELDS)
    for field in COMPONENT_FIELDS:
        out[f"{field}_abs_contribution_ratio"] = abs(out.get(f"{field}_mean", 0.0)) / max(total_abs, 1e-6)
    for field in (
        "tam_proxy_uav_event", "tam_proxy_mav_event", "tam_proxy_event_total",
        "current_team_return_minus_tam_proxy_event", "brma_proxy_terminal_team",
        "brma_proxy_terminal_per_agent",
    ):
        out[f"{field}_mean"] = _mean([float(row.get(field, 0.0)) for row in rows])
    return out


def _group_summaries(episodes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["outcome_group"]].append(episode)
    order = [
        "red_win_blue_eliminated",
        "red_win_timeout",
        "blue_win_red_eliminated",
        "blue_win_timeout",
        "draw_timeout",
        "other",
    ]
    return [_summary_for_group(group, groups.get(group, [])) for group in order]


def _misalignment_flags(episodes: list[dict], summaries: list[dict]) -> dict:
    by_group = {row["outcome_group"]: row for row in summaries}
    blue_red_elim = [ep for ep in episodes if ep["outcome_group"] == "blue_win_red_eliminated"]
    no_kill = [ep for ep in episodes if ep["red_hit_count"] <= 0 and ep["blue_alive_final"] > 0]
    mav_surv = [
        ep for ep in episodes
        if ep["mav_alive_final"] > 0.5 and ep["red_hit_count"] <= 0
        and (ep["uav_return_mean"] <= 0 or ep["red_alive_final"] <= 1)
    ]
    uav_adv_vals = sorted(float(ep.get("paper_v1_uav_adv_sum", 0.0)) for ep in episodes)
    mav_support_vals = sorted(float(ep.get("paper_v1_mav_support_sum", 0.0)) for ep in episodes)
    med_uav_adv = uav_adv_vals[len(uav_adv_vals) // 2] if uav_adv_vals else 0.0
    med_mav_support = mav_support_vals[len(mav_support_vals) // 2] if mav_support_vals else 0.0
    pos_dense = _mean([
        _positive(ep.get("paper_v1_uav_flight_sum", 0.0))
        + _positive(ep.get("paper_v1_uav_adv_sum", 0.0))
        + _positive(ep.get("paper_v1_mav_safety_sum", 0.0))
        + _positive(ep.get("paper_v1_mav_support_sum", 0.0))
        for ep in episodes
    ])
    end_neg = _mean([
        _negative_abs(ep.get("paper_v1_uav_end_sum", 0.0))
        + _negative_abs(ep.get("paper_v1_mav_scaled_tam_sum", 0.0))
        for ep in episodes
    ])
    return {
        "blue_win_red_eliminated_positive_return": bool(
            blue_red_elim and _mean([float(ep["team_return"]) for ep in blue_red_elim]) > 0.0
        ),
        "timeout_dominant_high_return": bool(
            by_group["red_win_timeout"]["mean_team_return"] >= by_group["red_win_blue_eliminated"]["mean_team_return"]
            or (
                by_group["red_win_blue_eliminated"]["episode_count"] == 0
                and by_group["red_win_timeout"]["mean_team_return"] > 0.0
            )
        ),
        "no_kill_high_return": bool(no_kill and _mean([float(ep["team_return"]) for ep in no_kill]) > 0.0),
        "mav_survival_overdominance": bool(
            mav_surv and _mean([float(ep["team_return"]) for ep in mav_surv]) > 0.0
        ),
        "end_penalty_underweighted": bool(end_neg < pos_dense),
        "uav_adv_not_converted_to_kill": bool(
            any(float(ep.get("paper_v1_uav_adv_sum", 0.0)) > med_uav_adv and ep["red_hit_count"] <= 0 for ep in episodes)
        ),
        "mav_support_not_converted_to_kill": bool(
            any(
                float(ep.get("paper_v1_mav_support_sum", 0.0)) > med_mav_support
                and (ep["red_hit_count"] <= 0 or ep["blue_alive_final"] > 0)
                for ep in episodes
            )
        ),
        "mean_positive_dense_shaping": pos_dense,
        "mean_negative_terminal_or_event": end_neg,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _write_report(path: Path, payload: dict) -> None:
    flags = payload["misalignment_flags"]
    lines = [
        "# Reward-Outcome Alignment Audit",
        "",
        "## Executive Summary",
        "",
        f"- episodes: {payload['episode_count']}",
        f"- terminal exact matches: {payload['merge_info']['terminal_exact_match_count']}",
        f"- terminal unmatched count: {payload['merge_info']['terminal_unmatched_count']}",
        f"- missile_events rows: {payload['missile_event_rows']}",
        f"- update_diagnostics rows: {payload['update_diagnostics_rows']}",
        "",
        "## Paper-grounded reward mapping",
        "",
        "- TAM-HAPPO purpose: role-aware MAV support/survivability and UAV combat effectiveness.",
        "- BRMA-MAPPO purpose: homogeneous UAV flight, tactical advantage, and win/loss terminal shaping.",
        "- Current tam_brma_paper_aligned_v1 mapping audited here: MAV safety/support/event logs and BRMA-style UAV flight/adv/end components.",
        "",
        "## Outcome-group return table",
        "",
        "| group | n | team_return | mav_return | uav_return | red_hits | blue_alive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['outcome_group']} | {row['episode_count']} | "
            f"{row['mean_team_return']:.3f} | {row['mean_mav_return']:.3f} | "
            f"{row['mean_uav_return']:.3f} | {row['mean_red_hit_count']:.3f} | "
            f"{row['mean_blue_alive_final']:.3f} |"
        )
    lines.extend([
        "",
        "## Reward component decomposition by outcome",
        "",
        "See `reward_outcome_alignment_summary.csv` for component means, stds, and absolute contribution ratios.",
        "",
        "## TAM-HAPPO log-only proxy comparison",
        "",
        "- `tam_proxy_uav_event = 200 * red_hit_count - 200 * red_uav_death_count - 100 * red_out_of_zone_count_if_available`.",
        "- `tam_proxy_mav_event = -200 * mav_death + min(100 * red_hit_count, 200)`.",
        "- These are audit-only proxies and are not training rewards.",
        "",
        "## BRMA-MAPPO log-only terminal proxy comparison",
        "",
        "- `brma_proxy_terminal_team = 30 * (red_alive_final - blue_alive_final)`.",
        "- This only checks terminal magnitude and remaining-number alignment; it is not a heterogeneous TAM-HAPPO objective.",
        "",
        "## Misalignment flags",
        "",
    ])
    for key, value in flags.items():
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Training stability should be read from PPO/critic diagnostics, not from this reward audit alone.",
        "- Task effectiveness should be judged by red hits, blue alive, outcome, and missile logs.",
        "- Reward-outcome alignment is weak when high return appears in blue wins, no-kill episodes, or timeout-only outcomes.",
        "",
        "## Limitations",
        "",
    ])
    for note in payload["limitations"]:
        lines.append(f"- {note}")
    lines.extend([
        "",
        "## Do-not-change note",
        "",
        "This script does not modify reward, environment dynamics, missile logic, PID, blue policy, action space, observation space, or training logic. It only provides evidence for whether a future paper-grounded reward redesign may be needed.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(output_dir: str | Path, phase_bins: str | None = None,
            report_name: str = "reward_outcome_alignment_report.md",
            summary_name: str = "reward_outcome_alignment_summary.csv",
            episode_name: str = "reward_outcome_by_episode.csv") -> dict:
    out_dir = Path(output_dir)
    component_rows = _read_csv(out_dir / "rich_logs" / "episode_reward_components.csv")
    terminal_rows = _read_csv(out_dir / "terminal_episode_audit.csv")
    missile_rows = _read_csv(out_dir / "rich_logs" / "missile_events.csv")
    update_rows = _read_jsonl(out_dir / "update_diagnostics.jsonl")
    _ = _read_csv(out_dir / "train_log.csv")
    episodes, merge_info = _group_by_episode(component_rows, terminal_rows)
    summaries = _group_summaries(episodes)
    flags = _misalignment_flags(episodes, summaries)
    limitations = []
    if not missile_rows:
        limitations.append("rich_logs/missile_events.csv missing or empty; missile source details are unavailable.")
    if merge_info["terminal_unmatched_count"]:
        limitations.append(f"{merge_info['terminal_unmatched_count']} terminal episodes were not exact-matched by episode_id.")
    if not update_rows:
        limitations.append("update_diagnostics.jsonl missing; value/advantage diagnostics unavailable.")
    if not component_rows:
        limitations.append("rich_logs/episode_reward_components.csv missing or empty; no episode reward decomposition available.")
    payload = {
        "output_dir": str(out_dir),
        "phase_bins": _phase_bins(phase_bins),
        "episode_count": len(episodes),
        "merge_info": merge_info,
        "missile_event_rows": len(missile_rows),
        "update_diagnostics_rows": len(update_rows),
        "summary_rows": summaries,
        "misalignment_flags": flags,
        "limitations": limitations,
    }
    _write_csv(out_dir / episode_name, episodes)
    _write_csv(out_dir / summary_name, summaries)
    _write_report(out_dir / report_name, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-bins", default=None)
    parser.add_argument("--report-name", default="reward_outcome_alignment_report.md")
    parser.add_argument("--summary-name", default="reward_outcome_alignment_summary.csv")
    parser.add_argument("--episode-name", default="reward_outcome_by_episode.csv")
    args = parser.parse_args()
    payload = analyze(
        args.output_dir,
        phase_bins=args.phase_bins,
        report_name=args.report_name,
        summary_name=args.summary_name,
        episode_name=args.episode_name,
    )
    print(f"wrote {Path(args.output_dir) / args.report_name}")
    print(f"episodes={payload['episode_count']} groups={len(payload['summary_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
