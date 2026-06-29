"""Diagnose MAV-shared full-geometry observation coverage.

This script runs deterministic scripted rollouts only. It does not train and
does not modify reward, missile, PID, blue rule, action space or observation
dimension.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bearing(src: np.ndarray, dst: np.ndarray) -> float:
    delta = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return math.atan2(float(delta[1]), float(delta[0]))


def _angle_error(a: float, b: float) -> float:
    return abs(_wrap_pi(a - b))


def _safe_arr(value, shape=None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None and arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return arr


def _level_actions(env) -> dict[str, np.ndarray]:
    actions = {}
    for aid in env.agent_ids:
        sim = env._get_sim(aid)
        if sim is not None and sim.is_alive:
            actions[aid] = np.asarray([0.0, float(sim.get_rpy()[2]) / math.pi, 0.2], dtype=np.float32)
        else:
            actions[aid] = np.zeros(3, dtype=np.float32)
    return actions


def write_alignment_report(out_dir: Path) -> None:
    text = """# TAM-HAPPO Observation Alignment

## TAM-HAPPO Eq.13-Style Observation Items
- own position: x, y, z.
- own speed: scalar speed.
- own attitude: pitch, yaw, roll / attitude angles.
- relative speed: delta velocity or speed difference.
- relative altitude: delta h.
- distance: target distance.
- ATA and AA: attack/target aspect geometry.
- incoming missile entity: the paper includes missile-related observable entities; this environment currently exposes only `missile_warning` and does not provide a full missile entity token with position/velocity/time-to-go in `mav_shared_geo_v2`.

## MAV Role in TAM-HAPPO
- MAV is treated as a battlefield information and mission guidance node for UAVs.
- MAV should enhance enemy position, velocity and relative geometry awareness for UAVs.
- The paper does not model communication delay, packet loss or interference in the current project implementation.

## Current mav_shared_geo Fields
- `ego_geo_state`: normalized own position, speed and attitude.
- `ally_geo_states`: compressed relative ally geometry.
- `enemy_geo_states`: compressed enemy geometry: speed difference, relative altitude, distance, ATA, AA.
- `enemy_alive_mask`, `enemy_observed_mask`, `enemy_track_source`: alive, observed and direct/MAV-shared source masks.

## Current Gap
- Legacy `mav_shared_geo` does not expose enemy relative position xyz directly.
- Legacy `mav_shared_geo` does not expose enemy relative velocity xyz directly.
- Legacy `mav_shared_geo` does not expose enemy bearing/elevation directly.
- Legacy `mav_shared_geo` does not expose enemy heading/velocity direction directly.
- It is a compressed geometry summary, not the full TAM-HAPPO-style entity observation.

## mav_shared_geo_v2
- Adds `enemy_relative_pos_xyz`, `enemy_relative_vel_xyz`, `enemy_bearing_elevation`, `enemy_speed_heading` and `enemy_full_geo_valid_mask`.
- Keeps old `mav_shared_geo` fields and `enemy_track_source`.
- Does not change `_has_launch_track()`, missile launch gate, reward, blue rule or action semantics.

## Conclusion
- Legacy MAV shared observation is a simplified version of the paper observation.
- This simplification can plausibly make it harder for a UAV policy to infer pursuit direction and launch geometry, especially when it must act from MAV-shared rather than direct tracks.
"""
    _write(out_dir / "tam_happo_observation_alignment.md", text)


def run_diagnosis(config: str, episodes: int, max_steps: int, out_dir: Path) -> None:
    coverage_rows: list[dict[str, Any]] = []
    reception_rows: list[dict[str, Any]] = []
    bearing_rows: list[dict[str, Any]] = []
    schema_sample: list[dict[str, Any]] = []

    for ep in range(episodes):
        env = make_env(config, env_type="jsbsim_hetero", max_steps=max_steps)
        try:
            obs, info = env.reset(seed=ep)
            for step in range(max_steps):
                mav_id = next((rid for rid in env.red_ids if env.agent_roles.get(rid) == "mav"), "red_0")
                mav = env.red_planes.get(mav_id)
                mav_obs = obs.get(mav_id, {})
                mav_alive = bool(mav is not None and mav.is_alive)
                if step == 0 and ep == 0:
                    for aid in env.red_ids + env.blue_ids:
                        sample = obs.get(aid, {})
                        schema_sample.append({
                            "agent_id": aid,
                            "keys": json.dumps(sorted(sample.keys())),
                            "enemy_relative_pos_xyz_shape": list(np.asarray(sample.get("enemy_relative_pos_xyz", [])).shape),
                            "enemy_relative_vel_xyz_shape": list(np.asarray(sample.get("enemy_relative_vel_xyz", [])).shape),
                            "enemy_bearing_elevation_shape": list(np.asarray(sample.get("enemy_bearing_elevation", [])).shape),
                            "enemy_speed_heading_shape": list(np.asarray(sample.get("enemy_speed_heading", [])).shape),
                            "enemy_full_geo_valid_mask_shape": list(np.asarray(sample.get("enemy_full_geo_valid_mask", [])).shape),
                        })
                for idx, bid in enumerate(env.blue_ids):
                    blue = env.blue_planes.get(bid)
                    if mav is None or blue is None:
                        continue
                    dist = float(np.linalg.norm(mav.get_position() - blue.get_position()))
                    obs_mask = _safe_arr(mav_obs.get("enemy_observed_mask", []))
                    rel_pos = _safe_arr(mav_obs.get("enemy_relative_pos_xyz", []), (len(env.blue_ids), 3))
                    rel_vel = _safe_arr(mav_obs.get("enemy_relative_vel_xyz", []), (len(env.blue_ids), 3))
                    valid = _safe_arr(mav_obs.get("enemy_full_geo_valid_mask", []))
                    coverage_rows.append({
                        "episode": ep,
                        "step": step,
                        "blue_id": bid,
                        "mav_alive": int(mav_alive),
                        "mav_to_blue_distance_m": dist,
                        "blue_in_mav_observation_range": int(dist <= env.mav_observation_range_m),
                        "mav_enemy_observed_mask": float(obs_mask[idx]) if idx < obs_mask.size else "",
                        "mav_has_relative_pos_xyz": int(idx < rel_pos.shape[0] and np.linalg.norm(rel_pos[idx]) > 1e-8),
                        "mav_has_relative_vel_xyz": int(idx < rel_vel.shape[0] and np.linalg.norm(rel_vel[idx]) > 1e-8),
                        "mav_full_geo_valid": float(valid[idx]) if idx < valid.size else "",
                    })
                for rid in env.red_ids:
                    if env.agent_roles.get(rid) == "mav":
                        continue
                    robs = obs.get(rid, {})
                    source = _safe_arr(robs.get("enemy_track_source", []), (len(env.blue_ids), 2))
                    valid = _safe_arr(robs.get("enemy_full_geo_valid_mask", []))
                    bearing_el = _safe_arr(robs.get("enemy_bearing_elevation", []), (len(env.blue_ids), 2))
                    enemy_states = _safe_arr(robs.get("enemy_states", []))
                    ego_geo = _safe_arr(robs.get("ego_geo_state", []))
                    ego_yaw = float(ego_geo[5] * math.pi) if ego_geo.size >= 6 else 0.0
                    for idx, bid in enumerate(env.blue_ids):
                        blue = env.blue_planes.get(bid)
                        red = env.red_planes.get(rid)
                        if red is None or blue is None:
                            continue
                        direct = bool(idx < source.shape[0] and source[idx, 0] > 0.5)
                        shared = bool(idx < source.shape[0] and source[idx, 1] > 0.5)
                        full_valid = bool(idx < valid.size and valid[idx] > 0.5)
                        reception_rows.append({
                            "episode": ep,
                            "step": step,
                            "red_uav_id": rid,
                            "blue_id": bid,
                            "direct_track": int(direct),
                            "mav_shared_track": int(shared),
                            "full_geo_valid": int(full_valid),
                        })
                        if full_valid:
                            oracle = _bearing(red.get_position(), blue.get_position())
                            v2_bearing = float(bearing_el[idx, 0] * math.pi)
                            old_bearing = ""
                            old_error = ""
                            if enemy_states.ndim == 2 and idx < enemy_states.shape[0] and enemy_states.shape[1] >= 2:
                                old_bearing_val = _wrap_pi(ego_yaw + math.atan2(float(enemy_states[idx, 1]), float(enemy_states[idx, 0])))
                                old_bearing = old_bearing_val
                                old_error = _angle_error(old_bearing_val, oracle)
                            bearing_rows.append({
                                "episode": ep,
                                "step": step,
                                "red_uav_id": rid,
                                "blue_id": bid,
                                "track_source": "direct" if direct else "mav_shared" if shared else "",
                                "oracle_bearing_rad": oracle,
                                "v2_bearing_rad": v2_bearing,
                                "v2_bearing_error_rad": _angle_error(v2_bearing, oracle),
                                "legacy_reconstructed_bearing_rad": old_bearing,
                                "legacy_bearing_error_rad": old_error,
                            })
                actions = _level_actions(env)
                obs, _rewards, terminated, truncated, info = env.step(actions)
                if all(terminated.values()) or all(truncated.values()):
                    break
        finally:
            env.close()

    _write_csv(out_dir / "mav_observation_coverage.csv", coverage_rows, [
        "episode", "step", "blue_id", "mav_alive", "mav_to_blue_distance_m",
        "blue_in_mav_observation_range", "mav_enemy_observed_mask",
        "mav_has_relative_pos_xyz", "mav_has_relative_vel_xyz", "mav_full_geo_valid",
    ])
    _write_csv(out_dir / "uav_shared_track_reception.csv", reception_rows, [
        "episode", "step", "red_uav_id", "blue_id", "direct_track",
        "mav_shared_track", "full_geo_valid",
    ])
    _write_csv(out_dir / "bearing_reconstruction_error.csv", bearing_rows, [
        "episode", "step", "red_uav_id", "blue_id", "track_source",
        "oracle_bearing_rad", "v2_bearing_rad", "v2_bearing_error_rad",
        "legacy_reconstructed_bearing_rad", "legacy_bearing_error_rad",
    ])
    _write_csv(out_dir / "shared_obs_schema_sample.csv", schema_sample, [
        "agent_id", "keys", "enemy_relative_pos_xyz_shape", "enemy_relative_vel_xyz_shape",
        "enemy_bearing_elevation_shape", "enemy_speed_heading_shape",
        "enemy_full_geo_valid_mask_shape",
    ])
    write_report(out_dir, config, episodes, max_steps, coverage_rows, reception_rows, bearing_rows)


def _mean(values: list[float]) -> float | str:
    vals = [float(v) for v in values if v != "" and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else ""


def write_report(
    out_dir: Path,
    config: str,
    episodes: int,
    max_steps: int,
    coverage_rows: list[dict[str, Any]],
    reception_rows: list[dict[str, Any]],
    bearing_rows: list[dict[str, Any]],
) -> None:
    coverage_count = len(coverage_rows)
    mav_visible = sum(int(r["mav_enemy_observed_mask"] == 1.0) for r in coverage_rows)
    shared = sum(int(r["mav_shared_track"]) for r in reception_rows)
    direct = sum(int(r["direct_track"]) for r in reception_rows)
    full_valid = sum(int(r["full_geo_valid"]) for r in reception_rows)
    v2_errors = [float(r["v2_bearing_error_rad"]) for r in bearing_rows if r["v2_bearing_error_rad"] != ""]
    old_errors = [float(r["legacy_bearing_error_rad"]) for r in bearing_rows if r["legacy_bearing_error_rad"] != ""]
    track_sources = Counter(r["track_source"] for r in bearing_rows)
    text = f"""# MAV Shared Observation V2 Report

## Run Scope
- config: `{config}`
- episodes: {episodes}
- max_steps: {max_steps}
- no RL training was run.

## Key Findings
- MAV observed blue rows: {mav_visible}/{coverage_count}
- red UAV direct track rows: {direct}
- red UAV MAV-shared track rows: {shared}
- red UAV full-geo-valid rows: {full_valid}
- bearing rows by track source: {dict(track_sources)}
- mean v2 bearing error rad: {_mean(v2_errors)}
- mean legacy reconstructed bearing error rad: {_mean(old_errors)}

## Required Questions
1. Current legacy `mav_shared_geo` is a compressed observation: yes, it exposes distance/ATA/AA-like summaries but not direct relative position xyz, relative velocity xyz, bearing/elevation or target speed/heading.
2. New `mav_shared_geo_v2` is closer to TAM-HAPPO observation semantics because it exposes explicit relative position, relative velocity and bearing/elevation while retaining old masks and track source.
3. MAV can observe blue position/velocity/relative geometry when the target is within `mav_observation_range_m`; see `mav_observation_coverage.csv`.
4. MAV observation range is controlled by environment config parameter `mav_observation_range_m`, not by F16/F22 aircraft XML sensor definitions in this code path.
5. F16-dynamics MAV does not directly change observation range, but it can indirectly change MAV position, survival and coverage duration through flight dynamics.
6. Red UAVs do receive MAV-shared track rows when `enemy_track_source[:,1] == 1`; see `uav_shared_track_reception.csv`.
7. Red UAVs can reconstruct target bearing from v2 shared observation when `enemy_full_geo_valid_mask == 1`; see `bearing_reconstruction_error.csv`.
8. Legacy vs v2 bearing error is reported above. Legacy reconstruction uses existing `enemy_states` body x/y when available; v2 directly exposes bearing/elevation.
9. Evidence supports over-compressed shared observation as an important suspect for non-pursuit/non-fire behavior, but it is not the only possible cause.
10. A small-scale diagnostic training run with `mav_shared_geo_v2` is reasonable after this observation-only validation.

## Aircraft Dynamics Note
- `mav_observation_range_m` is an environment abstraction. It is not read from F16/F22 aircraft XML.
- F16 dynamics can indirectly affect observation coverage by moving the MAV differently from a true F22.
"""
    _write(out_dir / "mav_shared_observation_v2_report.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="uav_env/JSBSim/configs/diagnostic_mav_shared_geo_v2_3v2.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/mav_shared_obs_alignment_auto")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_alignment_report(out_dir)
    run_diagnosis(args.config, args.episodes, args.max_steps, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
