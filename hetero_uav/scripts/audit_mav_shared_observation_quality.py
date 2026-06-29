"""Audit canonical MAV-shared full-geometry observation quality.

This is a rollout-only diagnostic. It does not train and does not modify
reward, missile, PID, blue rule, action space or observation dimensions.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402


POLICIES = (
    "level_flight",
    "blue_rule_pressure",
    "obs_limited_chase_red_vs_blue_rule",
    "obs_limited_chase_red_vs_blue_zero",
    "true_oracle_launch_window_red_vs_blue_rule",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _safe_arr(value, shape=None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None and arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return arr


def _target_action_to_position(sim, target_pos: np.ndarray, speed_cmd: float = 0.4) -> np.ndarray:
    pos = np.asarray(sim.get_position(), dtype=np.float64)
    delta = np.asarray(target_pos, dtype=np.float64) - pos
    horizontal = float(np.linalg.norm(delta[:2]))
    heading = math.atan2(float(delta[1]), float(delta[0])) / math.pi
    pitch = math.atan2(float(delta[2]), max(horizontal, 1e-6)) / (math.pi / 2.0)
    return np.asarray([np.clip(pitch, -0.4, 0.4), heading, speed_cmd], dtype=np.float32)


def _nearest_alive_enemy(env, aid: str):
    sim = env._get_sim(aid)
    if sim is None or not sim.is_alive:
        return None
    enemies = env.blue_ids if aid.startswith("red_") else env.red_ids
    best = None
    best_dist = float("inf")
    for eid in enemies:
        enemy = env._get_sim(eid)
        if enemy is None or not enemy.is_alive:
            continue
        dist = float(np.linalg.norm(sim.get_position() - enemy.get_position()))
        if dist < best_dist:
            best = enemy
            best_dist = dist
    return best


def _policy_actions(env, obs: dict, policy_name: str) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    for aid in env.agent_ids:
        sim = env._get_sim(aid)
        if sim is None or not sim.is_alive:
            actions[aid] = np.zeros(3, dtype=np.float32)
            continue
        if aid.startswith("blue") and policy_name in {
            "blue_rule_pressure",
            "obs_limited_chase_red_vs_blue_rule",
            "true_oracle_launch_window_red_vs_blue_rule",
        }:
            target = _nearest_alive_enemy(env, aid)
            actions[aid] = _target_action_to_position(sim, target.get_position(), 0.4) if target else np.zeros(3, dtype=np.float32)
            continue
        if aid.startswith("blue") and policy_name == "obs_limited_chase_red_vs_blue_zero":
            actions[aid] = np.zeros(3, dtype=np.float32)
            continue
        if aid.startswith("red") and "chase" in policy_name:
            target = _nearest_alive_enemy(env, aid)
            actions[aid] = _target_action_to_position(sim, target.get_position(), 0.5) if target else np.zeros(3, dtype=np.float32)
            continue
        actions[aid] = np.asarray([0.0, float(sim.get_rpy()[2]) / math.pi, 0.2], dtype=np.float32)
    return actions


def _reconstruction_error(env, rid: str, bid: str, robs: dict, index: int) -> dict[str, float]:
    red = env.red_planes[rid]
    blue = env.blue_planes[bid]
    rel_pos = _safe_arr(robs.get("enemy_relative_pos_xyz", []), (len(env.blue_ids), 3))
    rel_vel = _safe_arr(robs.get("enemy_relative_vel_xyz", []), (len(env.blue_ids), 3))
    bearing_el = _safe_arr(robs.get("enemy_bearing_elevation", []), (len(env.blue_ids), 2))
    true_rel_pos = (np.asarray(blue.get_position(), dtype=np.float64) - np.asarray(red.get_position(), dtype=np.float64)) / 40000.0
    true_rel_vel = (np.asarray(blue.get_velocity(), dtype=np.float64) - np.asarray(red.get_velocity(), dtype=np.float64)) / 600.0
    horizontal = float(np.linalg.norm(true_rel_pos[:2]))
    true_bearing = math.atan2(float(true_rel_pos[1]), float(true_rel_pos[0])) / math.pi
    true_elev = math.atan2(float(true_rel_pos[2]), max(horizontal, 1e-6)) / math.pi
    return {
        "rel_pos_error": float(np.linalg.norm(rel_pos[index] - true_rel_pos)),
        "rel_vel_error": float(np.linalg.norm(rel_vel[index] - true_rel_vel)),
        "bearing_error": float(abs(_wrap_pi((bearing_el[index, 0] - true_bearing) * math.pi))),
        "elevation_error": float(abs((bearing_el[index, 1] - true_elev) * math.pi)),
    }


def run_audit(config: str, episodes: int, max_steps: int, output_dir: Path) -> None:
    coverage_rows: list[dict[str, Any]] = []
    uav_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for policy in POLICIES:
        for ep in range(episodes):
            env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
            try:
                obs, _info = env.reset(seed=1000 + ep)
                for step in range(max_steps):
                    mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), env.red_ids[0])
                    mav_obs = obs.get(mav_id, {})
                    mav = env.red_planes.get(mav_id)
                    for bi, bid in enumerate(env.blue_ids):
                        blue = env.blue_planes.get(bid)
                        if mav is None or blue is None:
                            continue
                        dist = float(np.linalg.norm(mav.get_position() - blue.get_position()))
                        observed = _safe_arr(mav_obs.get("enemy_observed_mask", []))
                        valid = _safe_arr(mav_obs.get("enemy_full_geo_valid_mask", []))
                        coverage_rows.append({
                            "policy": policy,
                            "episode": ep,
                            "step": step,
                            "blue_id": bid,
                            "mav_alive": int(mav.is_alive),
                            "mav_to_blue_distance_m": dist,
                            "blue_in_mav_observation_range": int(dist <= env.mav_observation_range_m),
                            "mav_enemy_observed": float(observed[bi]) if bi < observed.size else 0.0,
                            "mav_full_geo_valid": float(valid[bi]) if bi < valid.size else 0.0,
                        })
                    for rid in env.red_ids:
                        if env.agent_roles.get(rid) == "mav":
                            continue
                        robs = obs.get(rid, {})
                        source = _safe_arr(robs.get("enemy_track_source", []), (len(env.blue_ids), 2))
                        valid = _safe_arr(robs.get("enemy_full_geo_valid_mask", []))
                        observed = _safe_arr(robs.get("enemy_observed_mask", []))
                        for bi, bid in enumerate(env.blue_ids):
                            direct = bool(bi < source.shape[0] and source[bi, 0] > 0.5)
                            shared = bool(bi < source.shape[0] and source[bi, 1] > 0.5)
                            full_valid = bool(bi < valid.size and valid[bi] > 0.5)
                            uav_rows.append({
                                "policy": policy,
                                "episode": ep,
                                "step": step,
                                "red_uav_id": rid,
                                "blue_id": bid,
                                "direct_track": int(direct),
                                "mav_shared_track": int(shared),
                                "observed": float(observed[bi]) if bi < observed.size else 0.0,
                                "full_geo_valid": int(full_valid),
                            })
                            if full_valid:
                                errs = _reconstruction_error(env, rid, bid, robs, bi)
                                error_rows.append({
                                    "policy": policy,
                                    "episode": ep,
                                    "step": step,
                                    "red_uav_id": rid,
                                    "blue_id": bid,
                                    "track_source": "direct" if direct else "mav_shared" if shared else "unknown",
                                    **errs,
                                })
                    obs, _rew, terminated, truncated, _info = env.step(_policy_actions(env, obs, policy))
                    if all(terminated.values()) or all(truncated.values()):
                        break
            finally:
                env.close()

    continuity_rows = _continuity_rows(uav_rows)
    policy_rows = _policy_rows(coverage_rows, uav_rows, error_rows)
    _write_csv(output_dir / "mav_coverage_quality.csv", coverage_rows, [
        "policy", "episode", "step", "blue_id", "mav_alive",
        "mav_to_blue_distance_m", "blue_in_mav_observation_range",
        "mav_enemy_observed", "mav_full_geo_valid",
    ])
    _write_csv(output_dir / "uav_shared_track_quality.csv", uav_rows, [
        "policy", "episode", "step", "red_uav_id", "blue_id",
        "direct_track", "mav_shared_track", "observed", "full_geo_valid",
    ])
    _write_csv(output_dir / "shared_track_continuity.csv", continuity_rows, [
        "policy", "red_uav_id", "blue_id", "track_source", "samples",
        "observed_rate", "max_unobserved_gap_steps",
    ])
    _write_csv(output_dir / "shared_geo_reconstruction_error.csv", error_rows, [
        "policy", "episode", "step", "red_uav_id", "blue_id", "track_source",
        "rel_pos_error", "rel_vel_error", "bearing_error", "elevation_error",
    ])
    _write_csv(output_dir / "shared_obs_quality_by_policy.csv", policy_rows, [
        "policy", "mav_coverage_rate", "uav_mav_shared_rate",
        "uav_full_geo_valid_rate", "mean_rel_pos_error", "mean_bearing_error",
    ])
    _write_report(output_dir, policy_rows)


def _continuity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        source = "mav_shared" if row["mav_shared_track"] else "direct" if row["direct_track"] else "none"
        key = (row["policy"], row["red_uav_id"], row["blue_id"], source)
        grouped[key].append(int(row["observed"] > 0.5))
    out = []
    for (policy, rid, bid, source), vals in grouped.items():
        gap = cur = 0
        for v in vals:
            cur = 0 if v else cur + 1
            gap = max(gap, cur)
        out.append({
            "policy": policy,
            "red_uav_id": rid,
            "blue_id": bid,
            "track_source": source,
            "samples": len(vals),
            "observed_rate": float(np.mean(vals)) if vals else 0.0,
            "max_unobserved_gap_steps": gap,
        })
    return out


def _mean(vals: list[float]) -> float:
    finite = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else 0.0


def _policy_rows(coverage, uav_rows, error_rows):
    out = []
    for policy in POLICIES:
        cov = [r for r in coverage if r["policy"] == policy]
        uav = [r for r in uav_rows if r["policy"] == policy]
        err = [r for r in error_rows if r["policy"] == policy]
        out.append({
            "policy": policy,
            "mav_coverage_rate": _mean([r["mav_enemy_observed"] for r in cov]),
            "uav_mav_shared_rate": _mean([r["mav_shared_track"] for r in uav]),
            "uav_full_geo_valid_rate": _mean([r["full_geo_valid"] for r in uav]),
            "mean_rel_pos_error": _mean([r["rel_pos_error"] for r in err]),
            "mean_bearing_error": _mean([r["bearing_error"] for r in err]),
        })
    return out


def _write_report(output_dir: Path, policy_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MAV Shared Observation Quality Audit",
        "",
        "Canonical `mav_shared_geo` now exposes full relative position, relative velocity, bearing/elevation, target speed/heading and a full-geometry valid mask.",
        "",
        "| policy | MAV coverage | UAV MAV-shared rate | full-geo valid | mean rel-pos error | mean bearing error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {row['mav_coverage_rate']:.3f} | "
            f"{row['uav_mav_shared_rate']:.3f} | {row['uav_full_geo_valid_rate']:.3f} | "
            f"{row['mean_rel_pos_error']:.6f} | {row['mean_bearing_error']:.6f} |"
        )
    _write(output_dir / "mav_shared_observation_quality_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/diagnostic_mav_shared_geo_3v2.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/mav_shared_obs_quality_auto")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_audit(args.config, args.episodes, args.max_steps, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
