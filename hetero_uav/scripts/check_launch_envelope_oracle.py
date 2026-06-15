"""Check whether scripted red policies can enter the launch envelope.

This is a diagnostic script only. It does not modify reward, missile dynamics,
PID, aircraft XML, action space, observation, or blue-rule logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from red_attack_audit_utils import (
    DEFAULT_CONFIG,
    alive_counts,
    blue_actions,
    collect_step_counts,
    make_env,
    red_oracle_actions,
    team_done,
)


ROOT = Path(__file__).resolve().parents[1]


DIAG_FIELDS = [
    "episode_id",
    "step",
    "red_id",
    "target_id",
    "range_m",
    "ao_rad",
    "ta_rad",
    "has_missile",
    "cooldown_ready",
    "lock_timer",
    "lock_ready",
    "deconflict_ok",
    "target_alive",
    "range_ok",
    "ao_ok",
    "ta_ok",
    "launch_allowed_predicted",
    "launch_block_reason",
    "missiles_fired_this_step",
    "red_hits_total",
    "blue_dead",
    "terminal_reason",
]


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_output_dir(path: str | Path) -> Path:
    out = _as_path(path)
    if out.exists() and any(out.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out.with_name(f"{out.name}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _all_sims(env) -> dict[str, Any]:
    sims = {}
    sims.update(getattr(env, "red_planes", {}))
    sims.update(getattr(env, "blue_planes", {}))
    return sims


def _geometry_pair(shooter, target) -> tuple[float, float, float]:
    from uav_env.JSBSim.utils import get2d_AO_TA_R

    spos = np.asarray(shooter.get_position(), dtype=np.float64)
    svel = np.asarray(shooter.get_velocity(), dtype=np.float64)
    tpos = np.asarray(target.get_position(), dtype=np.float64)
    tvel = np.asarray(target.get_velocity(), dtype=np.float64)
    sfeat = np.array([spos[0], spos[1], -spos[2], svel[0], svel[1], -svel[2]], dtype=np.float64)
    tfeat = np.array([tpos[0], tpos[1], -tpos[2], tvel[0], tvel[1], -tvel[2]], dtype=np.float64)
    ao, ta, rng = get2d_AO_TA_R(sfeat, tfeat)
    return float(rng), float(ao), float(ta)


def _terminal_reason(env, terminated: dict, truncated: dict) -> str:
    counts = alive_counts(env)
    if counts["blue_alive"] <= 0 and counts["red_alive"] > 0:
        return "red_win_elimination"
    if counts["red_alive"] <= 0 and counts["blue_alive"] > 0:
        return "blue_win_elimination"
    if all(bool(v) for v in truncated.values()):
        return "timeout"
    if all(bool(v) for v in terminated.values()):
        return "terminated"
    return ""


def _diagnose_red_shooter(env, red_id: str) -> dict[str, Any]:
    sim = getattr(env, "red_planes", {}).get(red_id)
    if sim is None or not bool(getattr(sim, "is_alive", False)):
        return {
            "red_id": red_id,
            "target_id": "",
            "range_m": "",
            "ao_rad": "",
            "ta_rad": "",
            "has_missile": False,
            "cooldown_ready": False,
            "lock_timer": 0,
            "lock_ready": False,
            "deconflict_ok": False,
            "target_alive": False,
            "range_ok": False,
            "ao_ok": False,
            "ta_ok": False,
            "launch_allowed_predicted": False,
            "launch_block_reason": "shooter_dead",
        }

    has_missile = int(getattr(sim, "num_left_missiles", 0)) > 0
    cooldown = int(getattr(env, "_missile_cooldown", {}).get(red_id, 0))
    cooldown_ready = cooldown == 0
    lock_timer = int(getattr(env, "_lock_timer", {}).get(red_id, 0))
    lock_target = getattr(env, "_lock_target", {}).get(red_id)
    lock_delay = int(getattr(env, "missile_lock_delay_frames", 0))
    on_kill_cooldown = red_id in getattr(env, "_agents_deny_kill", set())
    engaged = getattr(env, "_engaged_targets", set())

    best_any = None
    best_any_geom = (math.inf, math.nan, math.nan)
    best_unengaged = None
    best_unengaged_geom = (math.inf, math.nan, math.nan)
    for target in getattr(env, "blue_planes", {}).values():
        if not bool(getattr(target, "is_alive", False)):
            continue
        try:
            rng, ao, ta = _geometry_pair(sim, target)
        except Exception:
            continue
        if rng < best_any_geom[0]:
            best_any = target
            best_any_geom = (rng, ao, ta)
        if target.uid in engaged:
            continue
        if rng < best_unengaged_geom[0]:
            best_unengaged = target
            best_unengaged_geom = (rng, ao, ta)

    target = best_unengaged or best_any
    if target is None:
        return {
            "red_id": red_id,
            "target_id": "",
            "range_m": "",
            "ao_rad": "",
            "ta_rad": "",
            "has_missile": has_missile,
            "cooldown_ready": cooldown_ready,
            "lock_timer": lock_timer,
            "lock_ready": False,
            "deconflict_ok": False,
            "target_alive": False,
            "range_ok": False,
            "ao_ok": False,
            "ta_ok": False,
            "launch_allowed_predicted": False,
            "launch_block_reason": "no_alive_target",
        }

    rng, ao, ta = best_unengaged_geom if best_unengaged is not None else best_any_geom
    deconflict_ok = target.uid not in engaged
    range_ok = bool(env.MISSILE_LAUNCH_MIN_RANGE < rng < env.MISSILE_LAUNCH_RANGE_THRESH)
    ao_ok = bool(ao < env.MISSILE_LAUNCH_AO_THRESH)
    ta_ok = bool(ta > env.MISSILE_LAUNCH_TA_THRESH)
    lock_ready = bool(lock_target == target.uid and lock_timer >= lock_delay)
    target_alive = bool(getattr(target, "is_alive", False))
    allowed = bool(
        has_missile
        and cooldown_ready
        and not on_kill_cooldown
        and deconflict_ok
        and target_alive
        and range_ok
        and ao_ok
        and ta_ok
        and lock_ready
    )

    if not has_missile:
        reason = "no_missile"
    elif not target_alive:
        reason = "target_dead"
    elif not deconflict_ok:
        reason = "engaged_deconflict"
    elif not range_ok:
        reason = "out_of_range"
    elif not ao_ok:
        reason = "ao_blocked"
    elif not ta_ok:
        reason = "ta_blocked"
    elif not cooldown_ready:
        reason = "cooldown"
    elif on_kill_cooldown:
        reason = "kill_cooldown"
    elif not lock_ready:
        reason = "lock_delay"
    else:
        reason = "allowed"

    return {
        "red_id": red_id,
        "target_id": target.uid,
        "range_m": rng,
        "ao_rad": ao,
        "ta_rad": ta,
        "has_missile": has_missile,
        "cooldown_ready": cooldown_ready,
        "lock_timer": lock_timer,
        "lock_ready": lock_ready,
        "deconflict_ok": deconflict_ok,
        "target_alive": target_alive,
        "range_ok": range_ok,
        "ao_ok": ao_ok,
        "ta_ok": ta_ok,
        "launch_allowed_predicted": allowed,
        "launch_block_reason": reason,
    }


def _episode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str(r["launch_block_reason"]) for r in records)
    finite_ranges = [float(r["range_m"]) for r in records if r["range_m"] not in ("", None) and np.isfinite(float(r["range_m"]))]
    return {
        "rows": len(records),
        "block_reason_counts": dict(reasons),
        "min_range_m": float(np.min(finite_ranges)) if finite_ranges else None,
        "mean_range_m": float(np.mean(finite_ranges)) if finite_ranges else None,
        "range_ok_rate": float(np.mean([bool(r["range_ok"]) for r in records])) if records else 0.0,
        "ao_ok_rate": float(np.mean([bool(r["ao_ok"]) for r in records])) if records else 0.0,
        "ta_ok_rate": float(np.mean([bool(r["ta_ok"]) for r in records])) if records else 0.0,
        "predicted_allowed_rate": float(np.mean([bool(r["launch_allowed_predicted"]) for r in records])) if records else 0.0,
    }


def run_case(
    *,
    config: str,
    red_policy: str,
    blue_policy: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    for ep in range(episodes):
        env = make_env(config)
        try:
            obs, _info = env.reset(seed=seed + ep)
            red_fired = blue_fired = red_hits = blue_hits = 0
            terminated = truncated = {}
            episode_diag_rows: list[dict[str, Any]] = []
            for step in range(1, max_steps + 1):
                actions = {}
                actions.update(red_oracle_actions(env, obs, red_policy))
                actions.update(blue_actions(env, obs, blue_policy))

                before_rows = []
                for rid in env.red_ids:
                    if env.agent_roles.get(rid) == "mav":
                        continue
                    row = _diagnose_red_shooter(env, rid)
                    row.update({"episode_id": ep, "step": step})
                    before_rows.append(row)

                obs, _rewards, terminated, truncated, info = env.step(actions)
                counts = collect_step_counts(info)
                red_fired += counts["red_fired"]
                blue_fired += counts["blue_fired"]
                red_hits = max(red_hits, counts["red_hits_total"])
                blue_hits = max(blue_hits, counts["blue_hits_total"])
                blue_dead = alive_counts(env)["blue_dead"]
                terminal = _terminal_reason(env, terminated, truncated)
                fired_by_red = {
                    aid: int(agent_info.get("missiles_fired_this_step", 0) or 0)
                    for aid, agent_info in info.items()
                    if isinstance(agent_info, dict) and aid.startswith("red_")
                }
                for row in before_rows:
                    row.update({
                        "missiles_fired_this_step": fired_by_red.get(str(row["red_id"]), 0),
                        "red_hits_total": counts["red_hits_total"],
                        "blue_dead": blue_dead,
                        "terminal_reason": terminal,
                    })
                    step_records.append(row)
                    episode_diag_rows.append(row)
                if team_done(terminated, truncated):
                    break
            alive = alive_counts(env)
            summary = _episode_summary(episode_diag_rows)
            episode_records.append({
                "episode_id": ep,
                "steps": step,
                "red_policy": red_policy,
                "blue_policy": blue_policy,
                "red_missiles_fired": red_fired,
                "blue_missiles_fired": blue_fired,
                "red_missile_hits": red_hits,
                "blue_missile_hits": blue_hits,
                "blue_dead": alive["blue_dead"],
                "red_dead": alive["red_dead"],
                "terminal_reason": _terminal_reason(env, terminated, truncated),
                **summary,
            })
        finally:
            env.close()
    return step_records, episode_records


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def _build_summary(config: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "config": config,
        "cases": cases,
        "overall_oracle_can_fire": any(c["red_missiles_fired_mean"] > 0 for c in cases),
        "overall_oracle_can_hit": any(c["red_missile_hits_mean"] > 0 for c in cases),
        "overall_min_range_m": min(
            (c["min_range_m"] for c in cases if c["min_range_m"] is not None),
            default=None,
        ),
    }


def _case_summary(red_policy: str, blue_policy: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for ep in episodes:
        reason_counts.update(ep.get("block_reason_counts", {}))
    min_ranges = [ep["min_range_m"] for ep in episodes if ep.get("min_range_m") is not None]
    return {
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "episodes": len(episodes),
        "red_missiles_fired_mean": _mean(episodes, "red_missiles_fired"),
        "blue_missiles_fired_mean": _mean(episodes, "blue_missiles_fired"),
        "red_missile_hits_mean": _mean(episodes, "red_missile_hits"),
        "blue_dead_mean": _mean(episodes, "blue_dead"),
        "range_ok_rate_mean": _mean(episodes, "range_ok_rate"),
        "ao_ok_rate_mean": _mean(episodes, "ao_ok_rate"),
        "ta_ok_rate_mean": _mean(episodes, "ta_ok_rate"),
        "predicted_allowed_rate_mean": _mean(episodes, "predicted_allowed_rate"),
        "min_range_m": float(np.min(min_ranges)) if min_ranges else None,
        "launch_block_reason_counts": dict(reason_counts),
        "episodes_detail": episodes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DIAG_FIELDS})


def _write_md(path: Path, summary: dict[str, Any], csv_path: Path) -> None:
    lines = [
        "# Launch Envelope Oracle Check",
        "",
        "This diagnostic uses scripted red policies and reads runtime launch geometry.",
        "It does not change missile decisions or training code.",
        "",
        f"- config: `{summary['config']}`",
        f"- launch diagnostics csv: `{csv_path}`",
        f"- oracle can fire: `{summary['overall_oracle_can_fire']}`",
        f"- oracle can hit: `{summary['overall_oracle_can_hit']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.extend([
            f"### red={case['red_policy']} blue={case['blue_policy']}",
            f"- red_missiles_fired_mean: `{case['red_missiles_fired_mean']}`",
            f"- red_missile_hits_mean: `{case['red_missile_hits_mean']}`",
            f"- blue_dead_mean: `{case['blue_dead_mean']}`",
            f"- min_range_m: `{case['min_range_m']}`",
            f"- range_ok_rate_mean: `{case['range_ok_rate_mean']}`",
            f"- ao_ok_rate_mean: `{case['ao_ok_rate_mean']}`",
            f"- ta_ok_rate_mean: `{case['ta_ok_rate_mean']}`",
            f"- predicted_allowed_rate_mean: `{case['predicted_allowed_rate_mean']}`",
            f"- launch_block_reason_counts: `{case['launch_block_reason_counts']}`",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check scripted red launch envelope entry")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/debug_launch_envelope_oracle")
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "direct_chase_vs_zero",
            "direct_chase_vs_brma_rule",
            "brma_rule_vs_zero",
            "brma_rule_vs_brma_rule",
        ],
        help="Case to run. Defaults to direct_chase_vs_zero and direct_chase_vs_brma_rule.",
    )
    args = parser.parse_args()

    cases = args.case or ["direct_chase_vs_zero", "direct_chase_vs_brma_rule"]
    case_modes = {
        "direct_chase_vs_zero": ("direct_chase", "zero"),
        "direct_chase_vs_brma_rule": ("direct_chase", "brma_rule"),
        "brma_rule_vs_zero": ("brma_rule", "zero"),
        "brma_rule_vs_brma_rule": ("brma_rule", "brma_rule"),
    }
    out_dir = _safe_output_dir(args.output_dir)
    all_step_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case in cases:
        red_policy, blue_policy = case_modes[case]
        step_rows, episode_rows = run_case(
            config=args.config,
            red_policy=red_policy,
            blue_policy=blue_policy,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed + 1000 * len(summaries),
        )
        all_step_rows.extend(step_rows)
        summaries.append(_case_summary(red_policy, blue_policy, episode_rows))

    summary = _build_summary(args.config, summaries)
    csv_path = out_dir / "launch_diagnostics.csv"
    json_path = out_dir / "launch_envelope_oracle_summary.json"
    md_path = out_dir / "launch_envelope_oracle_summary.md"
    _write_csv(csv_path, all_step_rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_md(md_path, summary, csv_path)

    print(f"output_dir: {out_dir}")
    print(f"output_json: {json_path}")
    print(f"launch_diagnostics_csv: {csv_path}")
    print(f"oracle_can_fire: {summary['overall_oracle_can_fire']}")
    print(f"oracle_can_hit: {summary['overall_oracle_can_hit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
