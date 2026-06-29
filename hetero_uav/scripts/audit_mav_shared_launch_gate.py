"""Audit whether MAV-shared tracks help red UAVs pass launch gates.

The script observes the existing BRMA-style launch chain. It does not relax
AO/range/TA, lock delay, cooldown, deconfliction, missile dynamics or reward.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_mav_shared_observation_quality import POLICIES, _policy_actions  # noqa: E402
from uav_env import make_env  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lock_ready(env, shooter_id: str, target_id: str) -> bool:
    """Use frame-based lock timer matching the real environment gate (env.py L1263).

    The environment tracks ``_lock_timer`` in physics frames and compares against
    ``missile_lock_delay_frames``.  Seconds-based constants like
    ``MISSILE_LOCK_DELAY_SEC`` are NOT used in the launch gate.
    """
    lock_target = getattr(env, "_lock_target", {}).get(shooter_id)
    lock_timer = int(getattr(env, "_lock_timer", {}).get(shooter_id, 0))
    lock_delay_frames = int(getattr(env, "missile_lock_delay_frames", 15))
    return bool(lock_target == target_id and lock_timer >= lock_delay_frames)


def _cooldown_ready(env, shooter_id: str) -> bool:
    cooldown = getattr(env, "_missile_cooldown", {})
    return float(cooldown.get(shooter_id, 0.0)) <= 1e-6


def _candidate_rows(env, policy: str, episode: int, step: int) -> list[dict[str, Any]]:
    rows = []
    for rid in env.red_ids:
        if env.agent_roles.get(rid) == "mav":
            continue
        shooter = env.red_planes.get(rid)
        if shooter is None or not shooter.is_alive:
            continue
        for bid in env.blue_ids:
            target = env.blue_planes.get(bid)
            if target is None or not target.is_alive:
                continue
            has_track, track_source = env._has_launch_track(rid, bid)
            metrics = env._missile_candidate_metrics(shooter, target)
            engaged = bid in getattr(env, "_engaged_targets", set())
            ammo_ready = int(getattr(shooter, "num_left_missiles", 0) > 0)
            cooldown_ready = int(_cooldown_ready(env, rid))
            lock_ready = int(_lock_ready(env, rid, bid))
            range_ok = bool(metrics.get("range_ok", False))
            ao_ok = bool(metrics.get("ao_ok", False))
            ta_ok = bool(metrics.get("ta_ok", False))
            boresight_ok = bool(metrics.get("boresight_ok_3d", False))
            use_boresight = bool(getattr(env, "use_boresight_launch_gate", False))
            if use_boresight:
                geometry_ok = bool(range_ok and ao_ok and ta_ok and boresight_ok)
            else:
                geometry_ok = bool(range_ok and ao_ok and ta_ok)
            launch_candidate = bool(has_track and not engaged and ammo_ready and geometry_ok)
            launch_allowed_now = bool(launch_candidate and cooldown_ready and lock_ready)
            if not has_track:
                block = track_source
            elif engaged:
                block = "engaged_target"
            elif not ammo_ready:
                block = "no_ammo"
            elif not range_ok:
                block = "range_blocked"
            elif not ao_ok:
                block = "ao_blocked"
            elif not ta_ok:
                block = "ta_blocked"
            elif not cooldown_ready:
                block = "cooldown"
            elif not lock_ready:
                block = "lock_delay"
            else:
                block = "launch_allowed"
            rows.append({
                "policy": policy,
                "episode": episode,
                "step": step,
                "red_id": rid,
                "blue_id": bid,
                "track_source": track_source,
                "has_track": int(has_track),
                "range_m": float(metrics.get("range_m", np.nan)),
                "AO_rad": float(metrics.get("AO_rad", np.nan)),
                "TA_rad": float(metrics.get("TA_rad", np.nan)),
                "boresight_3d_rad": float(metrics.get("boresight_3d_rad", np.nan)),
                "range_ok": int(range_ok),
                "ao_ok": int(ao_ok),
                "ta_ok": int(ta_ok),
                "boresight_ok": int(boresight_ok),
                "use_boresight_gate": int(use_boresight),
                "geometry_ok": int(geometry_ok),
                "cooldown_ready": cooldown_ready,
                "lock_ready": lock_ready,
                "ammo_ready": ammo_ready,
                "engaged": int(engaged),
                "launch_candidate": int(launch_candidate),
                "launch_allowed_now": int(launch_allowed_now),
                "block_reason": block,
            })
    return rows


def run_audit(config: str, episodes: int, max_steps: int, output_dir: Path) -> None:
    gate_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for ep in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
            try:
                obs, _info = env.reset(seed=2000 + ep)
                for step in range(max_steps):
                    env._last_step_obs = obs
                    gate_rows.extend(_candidate_rows(env, policy, ep, step))
                    obs, _rew, terminated, truncated, info = env.step(_policy_actions(env, obs, policy))
                    for rec in info.get("__launch_quality_step__", []) or []:
                        shooter_id = str(rec.get("shooter_id", rec.get("shooter", "")))
                        if not shooter_id.startswith("red"):
                            continue
                        track_source = rec.get("launch_track_source", rec.get("track_source", "unknown"))
                        event_rows.append({
                            "policy": policy,
                            "episode": ep,
                            "step": step,
                            "event": "launch",
                            "shooter_id": shooter_id,
                            "target_id": rec.get("target_id", ""),
                            "track_source": track_source,
                            "launch_track_ok": rec.get("launch_track_ok", ""),
                            "launch_track_block_reason": rec.get("launch_track_block_reason", ""),
                            "launch_geometry_ok_3d": rec.get("launch_geometry_ok_3d", ""),
                            "range_ok_3d": rec.get("range_ok_3d", ""),
                            "ata_ok_3d": rec.get("ata_ok_3d", ""),
                            "ta_ok_3d": rec.get("ta_ok_3d", ""),
                            "boresight_ok_3d": rec.get("boresight_ok_3d", ""),
                            "range_m": rec.get("range_m", rec.get("range_3d_m", "")),
                            "AO_rad": rec.get("AO_rad", rec.get("AO_3d_rad", "")),
                            "TA_rad": rec.get("TA_rad", rec.get("TA_3d_rad", "")),
                            "raw_termination_reason": "",
                            "hit": "",
                        })
                    for rec in info.get("__launch_quality_done__", []) or []:
                        shooter_id = str(rec.get("shooter_id", rec.get("shooter", "")))
                        if not shooter_id.startswith("red"):
                            continue
                        track_source = rec.get("launch_track_source", rec.get("track_source", "unknown"))
                        event_rows.append({
                            "policy": policy,
                            "episode": ep,
                            "step": step,
                            "event": "termination",
                            "shooter_id": shooter_id,
                            "target_id": rec.get("target_id", ""),
                            "track_source": track_source,
                            "launch_track_ok": rec.get("launch_track_ok", ""),
                            "launch_track_block_reason": rec.get("launch_track_block_reason", ""),
                            "launch_geometry_ok_3d": rec.get("launch_geometry_ok_3d", ""),
                            "range_ok_3d": rec.get("range_ok_3d", ""),
                            "ata_ok_3d": rec.get("ata_ok_3d", ""),
                            "ta_ok_3d": rec.get("ta_ok_3d", ""),
                            "boresight_ok_3d": rec.get("boresight_ok_3d", ""),
                            "range_m": rec.get("range_m", rec.get("range_3d_m", "")),
                            "AO_rad": rec.get("AO_rad", rec.get("AO_3d_rad", "")),
                            "TA_rad": rec.get("TA_rad", rec.get("TA_3d_rad", "")),
                            "raw_termination_reason": rec.get("raw_termination_reason", rec.get("termination_reason", "")),
                            "hit": int(bool(rec.get("hit", False))),
                        })
                    if all(terminated.values()) or all(truncated.values()):
                        break
            finally:
                env.close()

    block_rows = _aggregate_blocks(gate_rows)
    quality_rows = _aggregate_quality(gate_rows)
    lock_rows = _aggregate_lock(gate_rows)
    mav_rows = [r for r in gate_rows if r["track_source"] == "mav_shared"]
    event_summary = _aggregate_events(event_rows)

    _write_csv(output_dir / "launch_gate_by_track_source.csv", gate_rows, [
        "policy", "episode", "step", "red_id", "blue_id", "track_source",
        "has_track", "range_m", "AO_rad", "TA_rad", "boresight_3d_rad",
        "range_ok", "ao_ok", "ta_ok", "boresight_ok", "use_boresight_gate",
        "geometry_ok", "cooldown_ready", "lock_ready",
        "ammo_ready", "engaged", "launch_candidate", "launch_allowed_now",
        "block_reason",
    ])
    _write_csv(output_dir / "launch_block_reason_by_track_source.csv", block_rows, [
        "policy", "track_source", "block_reason", "count",
    ])
    _write_csv(output_dir / "launch_quality_by_track_source.csv", quality_rows, [
        "policy", "track_source", "samples", "range_ok_rate", "ao_ok_rate",
        "ta_ok_rate", "launch_candidate_rate", "launch_allowed_rate",
        "mean_range_m", "mean_AO_rad", "mean_TA_rad",
    ])
    _write_csv(output_dir / "lock_continuity_by_track_source.csv", lock_rows, [
        "policy", "track_source", "samples", "lock_ready_rate",
        "max_lock_not_ready_gap_steps",
    ])
    _write_csv(output_dir / "mav_shared_launch_candidates.csv", mav_rows, [
        "policy", "episode", "step", "red_id", "blue_id", "track_source",
        "range_m", "AO_rad", "TA_rad", "range_ok", "ao_ok", "ta_ok",
        "lock_ready", "cooldown_ready", "launch_candidate", "launch_allowed_now",
        "block_reason",
    ])
    _write_csv(output_dir / "launch_events_by_track_source.csv", event_summary, [
        "policy", "track_source", "launch_count", "termination_count", "hit_count",
        "launch_track_ok_rate", "geometry_ok_3d_rate",
    ])
    _write_report(output_dir, quality_rows, event_summary)
    _write_alignment_note(output_dir)


def _aggregate_blocks(rows):
    counts = Counter((r["policy"], r["track_source"], r["block_reason"]) for r in rows)
    return [
        {"policy": p, "track_source": s, "block_reason": b, "count": c}
        for (p, s, b), c in sorted(counts.items())
    ]


def _mean(rows, field):
    vals = [float(r[field]) for r in rows if str(r.get(field, "")) != "" and np.isfinite(float(r[field]))]
    return float(np.mean(vals)) if vals else 0.0


def _aggregate_quality(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["track_source"])].append(row)
    out = []
    for (policy, source), items in sorted(grouped.items()):
        out.append({
            "policy": policy,
            "track_source": source,
            "samples": len(items),
            "range_ok_rate": _mean(items, "range_ok"),
            "ao_ok_rate": _mean(items, "ao_ok"),
            "ta_ok_rate": _mean(items, "ta_ok"),
            "launch_candidate_rate": _mean(items, "launch_candidate"),
            "launch_allowed_rate": _mean(items, "launch_allowed_now"),
            "mean_range_m": _mean(items, "range_m"),
            "mean_AO_rad": _mean(items, "AO_rad"),
            "mean_TA_rad": _mean(items, "TA_rad"),
        })
    return out


def _aggregate_lock(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["track_source"])].append(int(row["lock_ready"]))
    out = []
    for (policy, source), vals in sorted(grouped.items()):
        gap = cur = 0
        for ready in vals:
            cur = 0 if ready else cur + 1
            gap = max(gap, cur)
        out.append({
            "policy": policy,
            "track_source": source,
            "samples": len(vals),
            "lock_ready_rate": float(np.mean(vals)) if vals else 0.0,
            "max_lock_not_ready_gap_steps": gap,
        })
    return out


def _aggregate_events(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row.get("track_source") or "unknown")].append(row)
    out = []
    for (policy, source), items in sorted(grouped.items()):
        launch_items = [r for r in items if r["event"] == "launch"]
        out.append({
            "policy": policy,
            "track_source": source,
            "launch_count": len(launch_items),
            "termination_count": sum(1 for r in items if r["event"] == "termination"),
            "hit_count": sum(int(r.get("hit") == 1) for r in items),
            "launch_track_ok_rate": float(np.mean([
                int(r.get("launch_track_ok") == 1 or r.get("launch_track_ok") == "True")
                for r in launch_items])) if launch_items else 0.0,
            "geometry_ok_3d_rate": float(np.mean([
                int(r.get("launch_geometry_ok_3d") == 1 or r.get("launch_geometry_ok_3d") == "True")
                for r in launch_items])) if launch_items else 0.0,
        })
    return out


def _write_report(output_dir: Path, quality_rows, event_rows) -> None:
    lines = [
        "# MAV Shared Launch Gate Audit",
        "",
        "> **Correction note:** 上一版 audit 的 `lock_ready` / event `track_source` 统计存在脚本层误差，本版已修正。",
        "> - `lock_ready` 现在使用 `env._lock_timer` (physics frames) 与 `env.missile_lock_delay_frames` 比较，不再误用秒数 `MISSILE_LOCK_DELAY_SEC`。",
        "> - launch/termination event 现在读取 `launch_track_source` 字段，不再误读 `track_source`。",
        "> - `geometry_ok` 现在尊重 `use_boresight_launch_gate` 配置。",
        "",
        "This audit keeps the existing BRMA-style fire-control gates unchanged and only groups candidate quality by direct vs MAV-shared track source.",
        "",
        "## Candidate Quality",
        "",
        "| policy | source | samples | range ok | AO ok | TA ok | candidate | allowed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in quality_rows:
        lines.append(
            f"| {r['policy']} | {r['track_source']} | {r['samples']} | "
            f"{r['range_ok_rate']:.3f} | {r['ao_ok_rate']:.3f} | "
            f"{r['ta_ok_rate']:.3f} | {r['launch_candidate_rate']:.3f} | "
            f"{r['launch_allowed_rate']:.3f} |"
        )
    lines.extend(["", "## Launch Events", "", "| policy | source | launches | terminations | hits |", "|---|---|---:|---:|---:|"])
    for r in event_rows:
        lines.append(f"| {r['policy']} | {r['track_source']} | {r['launch_count']} | {r['termination_count']} | {r['hit_count']} |")
    _write(output_dir / "mav_shared_launch_gate_report.md", "\n".join(lines) + "\n")


def _write_alignment_note(output_dir: Path) -> None:
    text = """# Launch Interval Paper Alignment

This audit does not change launch timing.

- BRMA-style automatic launch gates remain AO, range, TA, lock delay, cooldown, target alive, ammo and deconfliction.
- Current code keeps the existing lock delay and launch cooldown values from the environment.
- MAV-shared observation only affects whether a red UAV has a valid target track and which tracked target can be ranked; it does not loosen the launch gate.
- If a paper/report mentions launch intervals, report the configured environment values used by the experiment rather than inferring a new interval from this audit.
"""
    _write(output_dir / "launch_interval_paper_alignment.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/diagnostic_mav_shared_geo_3v2.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/mav_shared_launch_gate_auto")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit(args.config, args.episodes, args.max_steps, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
