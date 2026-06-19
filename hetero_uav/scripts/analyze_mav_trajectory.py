"""Analyze whether the learned F22 MAV follows a rear-support trajectory.

This is an evaluation-only diagnostic. Fixed MAV behaviors override red_0's
runtime action inside this script; they do not alter the environment or policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT = (
    "outputs/brma_recurrent_nomask_nonfinitecrash_500k_probe_f22/latest/model.pt"
)
DEFAULT_CONFIG = (
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
)
DEFAULT_OUTPUT_DIR = "outputs/f22_mav_trajectory_audit"
FIXED_BEHAVIORS = ("level_flight", "rear_retreat", "gentle_loiter", "climb_safe")
_WORKER_POLICY = None
_WORKER_DEVICE = None
_WORKER_ADAPTER = None


def _rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _safe_float(value, default=float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float], default=float("nan")) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else default


def _minimum(values: list[float], default=float("nan")) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else default


def _maximum(values: list[float], default=float("nan")) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else default


def fixed_mav_action(
    behavior: str,
    *,
    current_heading: float,
    initial_heading: float,
    sim_time: float,
    mav_position: np.ndarray,
    blue_positions: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a finite normalized high-level action for a diagnostic behavior."""

    heading = float(initial_heading)
    pitch = 0.0
    speed = 0.0  # midpoint of the environment's 102--408 m/s target range
    if behavior == "rear_retreat":
        if blue_positions:
            blue_center = np.mean(np.stack(list(blue_positions.values())), axis=0)
            toward_blue = math.atan2(
                float(blue_center[1] - mav_position[1]),
                float(blue_center[0] - mav_position[0]),
            )
            heading = _wrap_pi(toward_blue + math.pi)
        else:
            heading = _wrap_pi(current_heading + math.pi)
    elif behavior == "gentle_loiter":
        heading = _wrap_pi(initial_heading + math.radians(3.0) * sim_time)
    elif behavior == "climb_safe":
        pitch = 0.08  # normalized command, about 7.2 degrees target pitch
    elif behavior != "level_flight":
        raise ValueError(f"unsupported fixed MAV behavior: {behavior}")
    return np.clip(
        np.asarray([pitch, heading / math.pi, speed], dtype=np.float32),
        -1.0,
        1.0,
    ).astype(np.float32)


def predeath_window_stats(
    rows: list[dict], death_time_sec: float | None, window_sec: float = 10.0
) -> dict:
    if death_time_sec is None or not math.isfinite(float(death_time_sec)):
        return {"predeath_sample_count": 0}
    start = float(death_time_sec) - float(window_sec)
    selected = [
        row for row in rows
        if start <= _safe_float(row.get("sim_time")) <= float(death_time_sec)
        and bool(row.get("mav_alive", True))
    ]
    out = {"predeath_sample_count": len(selected)}
    fields = {
        "altitude": "mav_altitude_m",
        "speed": "mav_speed_mps",
        "roll": "mav_roll_deg",
        "pitch": "mav_pitch_deg",
        "yaw": "mav_yaw_deg",
    }
    for label, field in fields.items():
        values = [_safe_float(row.get(field)) for row in selected]
        suffix = "m" if label == "altitude" else "mps" if label == "speed" else "deg"
        out[f"predeath_{label}_mean_{suffix}"] = _mean(values)
        out[f"predeath_{label}_min_{suffix}"] = _minimum(values)
        out[f"predeath_{label}_max_{suffix}"] = _maximum(values)
    return out


def _load_policy(checkpoint: Path, device_name: str):
    import torch
    from scripts.eval_happo_reference import _build_policy_from_meta, _load_meta

    meta = _load_meta(checkpoint)
    device = torch.device(device_name)
    policy = _build_policy_from_meta(meta, device)
    policy.load(checkpoint, map_location=device)
    policy.eval()
    return policy, device, meta


def _worker_init(checkpoint: str, device_name: str) -> None:
    global _WORKER_POLICY, _WORKER_DEVICE, _WORKER_ADAPTER
    import torch
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    torch.set_num_threads(1)
    _WORKER_POLICY, _WORKER_DEVICE, _meta = _load_policy(
        Path(checkpoint), device_name
    )
    _WORKER_ADAPTER = HeteroObsAdapterV2()


def _worker_run_episode(job: dict):
    if _WORKER_POLICY is None or _WORKER_ADAPTER is None:
        raise RuntimeError("trajectory worker was not initialized")
    return _run_episode(
        _WORKER_POLICY,
        _WORKER_DEVICE,
        _WORKER_ADAPTER,
        job["config"],
        job["opponent_policy"],
        job["seed"],
        job["episode"],
        job["behavior"],
        job["max_steps"],
    )


def _active_mask(info: dict, env) -> np.ndarray:
    return np.asarray(
        [1.0 if bool(info.get(rid, {}).get("alive", False)) else 0.0 for rid in env.red_ids],
        dtype=np.float32,
    )


def _policy_actions(policy, device, adapter, obs, info, env, hidden):
    import torch
    from algorithms.happo.rollout_safety import (
        sanitize_policy_inputs,
        zero_inactive_actions,
        zero_inactive_hidden,
    )

    adapted = adapter.adapt_all(
        obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids
    )
    actor_obs = np.stack([adapted["actor_obs"][rid] for rid in env.red_ids])
    active = _active_mask(info, env)
    sanitized = sanitize_policy_inputs(
        actor_obs,
        active,
        critic_state=adapted["critic_state"],
        rnn_hidden=hidden,
        context={"env_idx": "trajectory", "total_steps": int(env.current_step)},
    )
    kwargs = {}
    if sanitized["rnn_hidden"] is not None:
        kwargs["rnn_hidden"] = torch.as_tensor(
            sanitized["rnn_hidden"], dtype=torch.float32, device=device
        )
    roles = [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]
    with torch.no_grad():
        result = policy.act(
            torch.as_tensor(sanitized["actor_obs"], dtype=torch.float32, device=device),
            roles=roles,
            critic_state=torch.as_tensor(
                sanitized["critic_state"], dtype=torch.float32, device=device
            ),
            deterministic=True,
            **kwargs,
        )
    raw = result["action"].detach().cpu().numpy().astype(np.float32)
    actions = zero_inactive_actions(raw, active)
    next_hidden = hidden
    if hidden is not None and "rnn_hidden" in result:
        next_hidden = zero_inactive_hidden(
            result["rnn_hidden"].detach().cpu().numpy(), active
        )
    return raw, actions, next_hidden


def _alive_positions(planes: dict) -> dict[str, np.ndarray]:
    return {
        aid: np.asarray(sim.get_position(), dtype=np.float64)
        for aid, sim in planes.items()
        if sim is not None and sim.is_alive
    }


def _nearest_distance(position: np.ndarray, positions: dict[str, np.ndarray]) -> float:
    if not positions:
        return float("nan")
    return min(float(np.linalg.norm(position - target)) for target in positions.values())


def _incoming_missile_distance(sim) -> float:
    if sim is None:
        return float("nan")
    distances = []
    for missile in getattr(sim, "under_missiles", []):
        if missile.is_alive:
            distances.append(
                float(np.linalg.norm(missile.get_position() - sim.get_position()))
            )
    return min(distances) if distances else float("nan")


def _formation_axis(env) -> np.ndarray:
    red = list(_alive_positions(env.red_planes).values())
    blue = list(_alive_positions(env.blue_planes).values())
    if not red or not blue:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    delta = np.mean(np.stack(blue), axis=0) - np.mean(np.stack(red), axis=0)
    horizontal = np.asarray([delta[0], delta[1], 0.0], dtype=np.float64)
    return horizontal / max(float(np.linalg.norm(horizontal)), 1e-8)


def _mav_relation(env, axis: np.ndarray) -> tuple[bool, bool, bool]:
    mav = env.red_planes.get("red_0")
    if mav is None or not mav.is_alive:
        return False, False, False
    uav_positions = [
        np.asarray(sim.get_position(), dtype=np.float64)
        for rid, sim in env.red_planes.items()
        if rid != "red_0" and sim is not None and sim.is_alive
    ]
    if not uav_positions:
        return False, False, False
    mav_progress = float(np.dot(mav.get_position(), axis))
    uav_progress = float(np.mean([np.dot(position, axis) for position in uav_positions]))
    retreating = float(np.dot(mav.get_velocity(), axis)) < 0.0
    return mav_progress > uav_progress, mav_progress < uav_progress, retreating


def _state_row(env, episode: int, behavior: str, raw_action, applied_action, axis) -> dict:
    mav = env.red_planes.get("red_0")
    alive = bool(mav is not None and mav.is_alive)
    geo = np.full(3, np.nan)
    pos = np.full(3, np.nan)
    vel = np.full(3, np.nan)
    rpy = np.full(3, np.nan)
    if alive:
        geo = np.asarray(mav.get_geodetic(), dtype=np.float64)
        pos = np.asarray(mav.get_position(), dtype=np.float64)
        vel = np.asarray(mav.get_velocity(), dtype=np.float64)
        rpy = np.asarray(mav.get_rpy(), dtype=np.float64)
    blue_positions = _alive_positions(env.blue_planes)
    frontmost, rear, retreating = _mav_relation(env, axis)
    incoming_distance = _incoming_missile_distance(mav)
    uav_states = {}
    for rid, sim in env.red_planes.items():
        if rid == "red_0" or sim is None:
            continue
        uav_states[rid] = {
            "alive": bool(sim.is_alive),
            "speed_mps": float(np.linalg.norm(sim.get_velocity())) if sim.is_alive else None,
            "altitude_m": float(sim.get_geodetic()[2]) if sim.is_alive else None,
        }
    blue_nearest = Counter()
    red_positions = _alive_positions(env.red_planes)
    for blue_pos in blue_positions.values():
        if red_positions:
            nearest = min(
                red_positions, key=lambda rid: np.linalg.norm(red_positions[rid] - blue_pos)
            )
            blue_nearest[nearest] += 1
    return {
        "behavior": behavior,
        "episode": episode,
        "env_step": int(env.current_step),
        "sim_time": float(getattr(env, "_sim_time", 0.0)),
        "mav_alive": alive,
        "mav_death_reason": getattr(env, "_death_reasons", {}).get("red_0", ""),
        "mav_lon_deg": geo[0], "mav_lat_deg": geo[1], "mav_altitude_m": geo[2],
        "mav_north_m": pos[0], "mav_east_m": pos[1], "mav_up_m": pos[2],
        "mav_vn_mps": vel[0], "mav_ve_mps": vel[1], "mav_vu_mps": vel[2],
        "mav_speed_mps": float(np.linalg.norm(vel)) if alive else float("nan"),
        "mav_roll_deg": math.degrees(rpy[0]) if alive else float("nan"),
        "mav_pitch_deg": math.degrees(rpy[1]) if alive else float("nan"),
        "mav_yaw_deg": math.degrees(rpy[2]) if alive else float("nan"),
        "mav_action_raw_pitch": float(raw_action[0]),
        "mav_action_raw_heading": float(raw_action[1]),
        "mav_action_raw_speed": float(raw_action[2]),
        "mav_action_applied_pitch": float(applied_action[0]),
        "mav_action_applied_heading": float(applied_action[1]),
        "mav_action_applied_speed": float(applied_action[2]),
        "mav_nearest_blue_distance_m": _nearest_distance(pos, blue_positions) if alive else float("nan"),
        "mav_missile_warning": bool(alive and mav.check_missile_warning() is not None),
        "mav_nearest_incoming_missile_distance_m": incoming_distance,
        "mav_frontmost": frontmost,
        "mav_rear_of_uav": rear,
        "mav_retreating": retreating,
        "red_alive_count": sum(sim.is_alive for sim in env.red_planes.values()),
        "blue_alive_count": sum(sim.is_alive for sim in env.blue_planes.values()),
        "blue_nearest_red_counts_json": json.dumps(dict(blue_nearest), sort_keys=True),
        "red_uav_states_json": json.dumps(uav_states, sort_keys=True),
    }


def _episode_summary(rows: list[dict], env, episode: int, behavior: str,
                     missile_rows: list[dict]) -> dict:
    death_rows = [row for row in rows if row.get("mav_death_reason")]
    death_time = _safe_float(death_rows[0]["sim_time"], None) if death_rows else None
    alive_rows = [row for row in rows if row["mav_alive"]]
    red_launches = [row for row in missile_rows if row.get("shooter_team") == "red"]
    uav_launches = [row for row in red_launches if row.get("shooter_id") != "red_0"]
    result = {
        "behavior": behavior,
        "episode": episode,
        "steps": int(env.current_step),
        "sim_time_end_sec": float(getattr(env, "_sim_time", 0.0)),
        "mav_alive_final": bool(env.red_planes["red_0"].is_alive),
        "mav_death_time_sec": death_time,
        "mav_death_reason": death_rows[0]["mav_death_reason"] if death_rows else "survived",
        "mav_frontmost_rate": _mean([float(row["mav_frontmost"]) for row in alive_rows], 0.0),
        "mav_rear_of_uav_rate": _mean([float(row["mav_rear_of_uav"]) for row in alive_rows], 0.0),
        "mav_retreating_rate": _mean([float(row["mav_retreating"]) for row in alive_rows], 0.0),
        "mav_nearest_blue_distance_mean_m": _mean([row["mav_nearest_blue_distance_m"] for row in alive_rows]),
        "mav_nearest_blue_distance_min_m": _minimum([row["mav_nearest_blue_distance_m"] for row in alive_rows]),
        "mav_warning_rate": _mean([float(row["mav_missile_warning"]) for row in alive_rows], 0.0),
        "mav_incoming_missile_distance_min_m": _minimum([row["mav_nearest_incoming_missile_distance_m"] for row in alive_rows]),
        "mav_altitude_mean_m": _mean([row["mav_altitude_m"] for row in alive_rows]),
        "mav_altitude_min_m": _minimum([row["mav_altitude_m"] for row in alive_rows]),
        "mav_speed_mean_mps": _mean([row["mav_speed_mps"] for row in alive_rows]),
        "mav_speed_min_mps": _minimum([row["mav_speed_mps"] for row in alive_rows]),
        "mav_roll_abs_mean_deg": _mean([abs(row["mav_roll_deg"]) for row in alive_rows]),
        "mav_roll_abs_max_deg": _maximum([abs(row["mav_roll_deg"]) for row in alive_rows]),
        "mav_pitch_mean_deg": _mean([row["mav_pitch_deg"] for row in alive_rows]),
        "mav_pitch_min_deg": _minimum([row["mav_pitch_deg"] for row in alive_rows]),
        "mav_pitch_max_deg": _maximum([row["mav_pitch_deg"] for row in alive_rows]),
        "mav_yaw_mean_deg": _mean([row["mav_yaw_deg"] for row in alive_rows]),
        "red_alive_final": sum(sim.is_alive for sim in env.red_planes.values()),
        "blue_alive_final": sum(sim.is_alive for sim in env.blue_planes.values()),
        "red_missiles_fired": len(red_launches),
        "red_missile_hits": sum(str(row.get("termination_reason")) == "hit" for row in red_launches),
        "red_uav_launch_speed_mean_mps": _mean([_safe_float(row.get("shooter_speed_mps")) for row in uav_launches]),
        "red_uav_launch_speed_min_mps": _minimum([_safe_float(row.get("shooter_speed_mps")) for row in uav_launches]),
        "mav_action_raw_pitch_mean": _mean([row["mav_action_raw_pitch"] for row in alive_rows]),
        "mav_action_raw_heading_mean": _mean([row["mav_action_raw_heading"] for row in alive_rows]),
        "mav_action_raw_speed_mean": _mean([row["mav_action_raw_speed"] for row in alive_rows]),
        "mav_action_applied_pitch_mean": _mean([row["mav_action_applied_pitch"] for row in alive_rows]),
        "mav_action_applied_heading_mean": _mean([row["mav_action_applied_heading"] for row in alive_rows]),
        "mav_action_applied_speed_mean": _mean([row["mav_action_applied_speed"] for row in alive_rows]),
    }
    result.update(predeath_window_stats(rows, death_time, 10.0))
    return result


def _run_episode(policy, device, adapter, config: str, opponent_mode: str,
                 seed: int, episode: int, behavior: str, max_steps: int | None):
    from algorithms.happo.rollout_safety import zero_inactive_actions
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env

    env = make_env(config, env_type="jsbsim_hetero")
    if max_steps is not None:
        env.max_steps = int(max_steps)
    opponent = OpponentPolicy(mode=opponent_mode, seed=seed + 97)
    rows: list[dict] = []
    launches: dict[str, dict] = {}
    launch_without_id: list[dict] = []
    try:
        obs, info = env.reset(seed=seed)
        axis = _formation_axis(env)
        initial_heading = float(env.red_planes["red_0"].get_rpy()[2])
        hidden_size = int(getattr(policy, "rnn_hidden_size", 0))
        hidden = (
            np.zeros((len(env.red_ids), hidden_size), dtype=np.float32)
            if hidden_size > 0 else None
        )
        while True:
            raw, actions, hidden = _policy_actions(
                policy, device, adapter, obs, info, env, hidden
            )
            applied = zero_inactive_actions(actions, _active_mask(info, env))
            if behavior != "policy" and env.red_planes["red_0"].is_alive:
                mav = env.red_planes["red_0"]
                applied[0] = fixed_mav_action(
                    behavior,
                    current_heading=float(mav.get_rpy()[2]),
                    initial_heading=initial_heading,
                    sim_time=float(getattr(env, "_sim_time", 0.0)),
                    mav_position=np.asarray(mav.get_position(), dtype=np.float64),
                    blue_positions=_alive_positions(env.blue_planes),
                )
            rows.append(_state_row(env, episode, behavior, raw[0], applied[0], axis))
            action_dict = {
                rid: applied[index].astype(np.float32)
                for index, rid in enumerate(env.red_ids)
            }
            action_dict.update(opponent.act(obs, env.blue_ids, env=env))
            obs, _rewards, terminated, truncated, info = env.step(action_dict)
            if rows[-1]["mav_alive"] and not env.red_planes["red_0"].is_alive:
                rows.append(
                    _state_row(env, episode, behavior, raw[0], applied[0], axis)
                )
            for record in info.get("__launch_quality_step__", []):
                enriched = {"behavior": behavior, "episode": episode, **dict(record)}
                missile_id = str(enriched.get("missile_id", ""))
                if missile_id:
                    launches[missile_id] = enriched
                else:
                    launch_without_id.append(enriched)
            for record in info.get("__launch_quality_done__", []):
                missile_id = str(record.get("missile_id", ""))
                if missile_id in launches:
                    launches[missile_id].update(dict(record))
            if all(terminated.values()) or all(truncated.values()):
                break
            if env.current_step >= env.max_steps:
                break
        missile_rows = list(launches.values()) + launch_without_id
        return rows, _episode_summary(rows, env, episode, behavior, missile_rows), missile_rows
    finally:
        env.close()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_fixed(summaries: list[dict]) -> list[dict]:
    output = []
    for behavior in FIXED_BEHAVIORS:
        records = [row for row in summaries if row["behavior"] == behavior]
        if not records:
            continue
        reasons = Counter(row["mav_death_reason"] for row in records)
        output.append({
            "behavior": behavior,
            "episodes": len(records),
            "mav_survival_rate": _mean([float(row["mav_alive_final"]) for row in records], 0.0),
            "mav_mean_death_time_sec": _mean([_safe_float(row["mav_death_time_sec"]) for row in records]),
            "mav_death_reason_counts_json": json.dumps(dict(reasons), sort_keys=True),
            "mav_altitude_mean_m": _mean([row["mav_altitude_mean_m"] for row in records]),
            "mav_altitude_min_m": _minimum([row["mav_altitude_min_m"] for row in records]),
            "mav_speed_mean_mps": _mean([row["mav_speed_mean_mps"] for row in records]),
            "mav_speed_min_mps": _minimum([row["mav_speed_min_mps"] for row in records]),
            "mav_roll_abs_mean_deg": _mean([row["mav_roll_abs_mean_deg"] for row in records]),
            "mav_roll_abs_max_deg": _maximum([row["mav_roll_abs_max_deg"] for row in records]),
            "mav_pitch_mean_deg": _mean([row["mav_pitch_mean_deg"] for row in records]),
            "mav_pitch_min_deg": _minimum([row["mav_pitch_min_deg"] for row in records]),
            "mav_pitch_max_deg": _maximum([row["mav_pitch_max_deg"] for row in records]),
            "mav_frontmost_rate": _mean([row["mav_frontmost_rate"] for row in records], 0.0),
            "mav_rear_of_uav_rate": _mean([row["mav_rear_of_uav_rate"] for row in records], 0.0),
            "mav_retreating_rate": _mean([row["mav_retreating_rate"] for row in records], 0.0),
            "mav_action_applied_pitch_mean": _mean([row["mav_action_applied_pitch_mean"] for row in records]),
            "mav_action_applied_heading_mean": _mean([row["mav_action_applied_heading_mean"] for row in records]),
            "mav_action_applied_speed_mean": _mean([row["mav_action_applied_speed_mean"] for row in records]),
        })
    return output


def _markdown(checkpoint: Path, config: str, policy_summaries: list[dict],
              missile_rows: list[dict], fixed_rows: list[dict],
              nearest_counts: Counter) -> str:
    n = max(len(policy_summaries), 1)
    death_reasons = Counter(row["mav_death_reason"] for row in policy_summaries)
    red_launches = [row for row in missile_rows if row.get("shooter_team") == "red"]
    blue_launches = [row for row in missile_rows if row.get("shooter_team") == "blue"]
    blue_mav_targets = sum(row.get("target_id") == "red_0" for row in blue_launches)
    uav_launches = [row for row in red_launches if row.get("shooter_id") != "red_0"]
    red_termination_reasons = Counter(
        str(row.get("termination_reason") or "unfinished") for row in red_launches
    )
    front = _mean([row["mav_frontmost_rate"] for row in policy_summaries], 0.0)
    rear = _mean([row["mav_rear_of_uav_rate"] for row in policy_summaries], 0.0)
    survival = _mean([float(row["mav_alive_final"]) for row in policy_summaries], 0.0)
    if rear > 0.5 and front < 0.35:
        behavior = "rear_support_like" if survival > 0.8 else "unsafe_rear"
    elif front > 0.5:
        behavior = "unsafe_forward"
    else:
        behavior = "inconclusive"
    lines = [
        "# F22 MAV Trajectory Analysis",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Config: `{config}`",
        f"- Deterministic policy episodes: {len(policy_summaries)}",
        f"- MAV survival rate: {survival:.3f}",
        f"- MAV mean death time: {_mean([_safe_float(row['mav_death_time_sec']) for row in policy_summaries]):.2f} s",
        f"- MAV death reasons: `{dict(death_reasons)}`",
        f"- MAV frontmost rate: {front:.3f}",
        f"- MAV rear-of-UAV rate: {rear:.3f}",
        f"- Trajectory classification: **{behavior}**",
        "",
        "## Targeting and missile context",
        "",
        f"- Blue launches targeting MAV: {blue_mav_targets}/{len(blue_launches)}",
        f"- Blue nearest-red target counts (inferred): `{dict(nearest_counts)}`",
        f"- Red UAV launches: {len(uav_launches)}",
        f"- Red UAV launch speed mean/min: {_mean([_safe_float(row.get('shooter_speed_mps')) for row in uav_launches]):.2f} / {_minimum([_safe_float(row.get('shooter_speed_mps')) for row in uav_launches]):.2f} m/s",
        f"- Red missile hits: {sum(str(row.get('termination_reason')) == 'hit' for row in red_launches)}",
        f"- Red missile termination reasons: `{dict(red_termination_reasons)}`",
        "",
        "Blue target preference is exact for missile launch targets and inferred from nearest-red geometry otherwise.",
        "",
        "## Fixed MAV behavior sanity check",
        "",
        "| behavior | episodes | survival | mean death s | altitude mean m | speed mean m/s | frontmost | rear |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        lines.append(
            f"| {row['behavior']} | {row['episodes']} | {row['mav_survival_rate']:.3f} | "
            f"{row['mav_mean_death_time_sec']:.2f} | {row['mav_altitude_mean_m']:.1f} | "
            f"{row['mav_speed_mean_mps']:.1f} | {row['mav_frontmost_rate']:.3f} | "
            f"{row['mav_rear_of_uav_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Fixed actions are diagnostic controls, not trained policies or formal results. "
        "A rear-support claim requires sustained rear positioning and survivability; survival alone is insufficient.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fixed-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Independent evaluation worker processes; workers>1 use CPU inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.fixed_episodes < 0 or args.workers < 1:
        raise ValueError(
            "--episodes and --workers must be >= 1; --fixed-episodes must be >= 0"
        )
    checkpoint = _rel(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir = _rel(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((checkpoint.parent / "meta.json").read_text(encoding="utf-8"))
    timeseries: list[dict] = []
    summaries: list[dict] = []
    missile_rows: list[dict] = []
    jobs = [
        {
            "config": args.config, "opponent_policy": args.opponent_policy,
            "seed": args.seed + episode, "episode": episode,
            "behavior": "policy", "max_steps": args.max_steps,
        }
        for episode in range(args.episodes)
    ]
    for behavior_index, behavior in enumerate(FIXED_BEHAVIORS):
        jobs.extend({
            "config": args.config, "opponent_policy": args.opponent_policy,
            "seed": args.seed + 10000 + behavior_index * 1000 + episode,
            "episode": episode, "behavior": behavior, "max_steps": args.max_steps,
        } for episode in range(args.fixed_episodes))

    results = []
    if args.workers == 1:
        from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
        policy, device, _meta = _load_policy(checkpoint, args.device)
        adapter = HeteroObsAdapterV2()
        for job in jobs:
            results.append((job, _run_episode(
                policy, device, adapter, job["config"], job["opponent_policy"],
                job["seed"], job["episode"], job["behavior"], job["max_steps"],
            )))
            print(f"[{job['behavior']}] episode={job['episode'] + 1} complete", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(checkpoint), "cpu"),
        ) as pool:
            future_jobs = {pool.submit(_worker_run_episode, job): job for job in jobs}
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                results.append((job, future.result()))
                print(f"[{job['behavior']}] episode={job['episode'] + 1} complete", flush=True)

    fixed_summaries: list[dict] = []
    fixed_timeseries: list[dict] = []
    for job, (rows, summary, launches) in sorted(
        results, key=lambda item: (item[0]["behavior"], item[0]["episode"])
    ):
        if job["behavior"] == "policy":
            timeseries.extend(rows)
            summaries.append(summary)
            missile_rows.extend(launches)
        else:
            fixed_timeseries.extend(rows)
            fixed_summaries.append(summary)

    fixed_rows = _aggregate_fixed(fixed_summaries)
    nearest_counts = Counter()
    for row in timeseries:
        nearest_counts.update(json.loads(row["blue_nearest_red_counts_json"]))
    _write_csv(output_dir / "mav_timeseries.csv", timeseries)
    _write_csv(output_dir / "episode_summary.csv", summaries)
    _write_csv(output_dir / "missile_launch_context.csv", missile_rows)
    _write_csv(output_dir / "fixed_mav_behavior_summary.csv", fixed_rows)
    _write_csv(output_dir / "fixed_mav_timeseries.csv", fixed_timeseries)
    (output_dir / "analysis_summary.md").write_text(
        _markdown(
            checkpoint, args.config, summaries, missile_rows, fixed_rows,
            nearest_counts,
        ),
        encoding="utf-8",
    )
    (output_dir / "analysis_meta.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "config": args.config,
        "episodes": args.episodes,
        "fixed_episodes_per_behavior": args.fixed_episodes,
        "policy_arch": meta.get("policy_arch"),
        "deterministic": True,
        "workers": args.workers,
        "fixed_behaviors_are_diagnostic_only": True,
    }, indent=2), encoding="utf-8")
    print(f"output_dir: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
