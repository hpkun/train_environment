"""Audit 5v4 zero-shot behavior consistency from an existing checkpoint.

Read-only diagnostic: no training and no changes to reward, missile, PID,
aircraft XML, action space, or observation dimensions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from happo_mav_audit_common import load_policy, rel, role_ids, team_done


DEFAULT_MODEL = "outputs/happo_geometry_curriculum_100k/normal_50k/best/model.pt"
DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_happo_ref_v0_f16_mav_surrogate.yaml"
DEFAULT_OUTPUT_DIR = "outputs/happo_geometry_curriculum_100k/normal_50k/behavior_audit_5v4"


def _wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _mean(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _rate(flags: list[bool]) -> float:
    return float(np.mean(np.asarray(flags, dtype=np.float32))) if flags else 0.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _bearing(src: np.ndarray, dst: np.ndarray) -> float:
    delta = np.asarray(dst, dtype=np.float64)[:2] - np.asarray(src, dtype=np.float64)[:2]
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _heading(sim) -> float:
    return float(sim.get_rpy()[2])


def _speed(sim) -> float:
    return float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))


def _alive_positions(planes: dict) -> dict[str, np.ndarray]:
    return {
        aid: np.asarray(sim.get_position(), dtype=np.float64)
        for aid, sim in planes.items()
        if sim is not None and sim.is_alive
    }


def _nearest(src_pos: np.ndarray, targets: dict[str, np.ndarray]) -> tuple[str | None, float]:
    best_id, best_dist = None, float("inf")
    for aid, pos in targets.items():
        distance = _dist(src_pos, pos)
        if distance < best_dist:
            best_id, best_dist = aid, distance
    return best_id, best_dist


def _heading_aligned_target(blue_sim, red_positions: dict[str, np.ndarray]) -> tuple[str | None, float]:
    if not red_positions:
        return None, 0.0
    blue_pos = np.asarray(blue_sim.get_position(), dtype=np.float64)
    blue_heading = _heading(blue_sim)
    best_id, best_error = None, float("inf")
    for rid, red_pos in red_positions.items():
        error = abs(_wrap_pi(_bearing(blue_pos, red_pos) - blue_heading))
        if error < best_error:
            best_id, best_error = rid, error
    return best_id, best_error


def _policy_actions(policy, device, adapter, obs, info, env) -> np.ndarray:
    import torch

    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
        for rid in env.red_ids
    ])
    with torch.no_grad():
        out = policy.act(
            torch.as_tensor(actor_obs, device=device),
            roles=role_ids(env),
            critic_state=torch.as_tensor(adapted["critic_state"], device=device),
            deterministic=True,
        )
    return out["action"].detach().cpu().numpy()


def _classify_mav_behavior(frontmost_rate: float, mean_center_dist: float, survival_rate: float) -> str:
    if frontmost_rate > 0.6 and survival_rate > 0.8:
        return "forward_decoy_like"
    if frontmost_rate > 0.6 and survival_rate <= 0.8:
        return "unsafe_forward"
    if frontmost_rate < 0.35 and mean_center_dist < 20000.0 and survival_rate > 0.8:
        return "support_like"
    return "inconclusive"


def _classify_blue_targeting(mav_missiles: int, uav_missiles: int, nearest_mav_rate: float) -> str:
    total = mav_missiles + uav_missiles
    if total > 0:
        if mav_missiles / total > 0.5:
            return "mav_targeted"
        return "uav_targeted"
    if nearest_mav_rate > 0.5:
        return "nearest_target_only"
    return "inconclusive"


def _classify_blue_pursuit(heading_to_rate: float, heading_error_mean: float, saturation_rate: float) -> str:
    if heading_to_rate > 0.55 and heading_error_mean < math.radians(50):
        return "responsive"
    if saturation_rate > 0.55 and heading_error_mean > math.radians(60):
        return "slow_turning"
    if heading_to_rate < 0.35:
        return "target_selection_weak"
    return "inconclusive"


def _run_episode(policy, device, adapter, args, seed: int) -> dict[str, Any]:
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env

    env = make_env(args.config, env_type="jsbsim_hetero")
    opponent = OpponentPolicy(mode=args.opponent_policy, seed=seed + 17)

    nearest_counts = Counter()
    inferred_counts = Counter()
    missile_targets = Counter()
    nearest_is_mav_flags: list[bool] = []
    inferred_is_mav_flags: list[bool] = []
    blue_heading_to_nearest_flags: list[bool] = []
    blue_heading_errors: list[float] = []
    blue_turn_rates: list[float] = []
    blue_action_saturation: list[float] = []
    blue_speeds: list[float] = []
    mav_nearest_blue_distances: list[float] = []
    mav_to_uav_center_distances: list[float] = []
    mav_frontmost_flags: list[bool] = []
    mav_heading_to_blue_flags: list[bool] = []
    mav_turn_back_flags: list[bool] = []
    mav_actions: list[np.ndarray] = []
    prev_blue_headings: dict[str, float] = {}
    prev_mav_heading: float | None = None
    nan_detected = False

    try:
        obs, info = env.reset(seed=seed)
        terminated = {aid: False for aid in env.agent_ids}
        truncated = {aid: False for aid in env.agent_ids}
        steps = 0
        while True:
            red_actions = _policy_actions(policy, device, adapter, obs, info, env)
            actions = {rid: red_actions[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            blue_actions = opponent.act(obs, env.blue_ids, env=env)
            actions.update(blue_actions)

            red_positions = _alive_positions(env.red_planes)
            blue_positions = _alive_positions(env.blue_planes)
            red_uav_positions = {
                rid: pos for rid, pos in red_positions.items() if rid != "red_0"
            }

            for bid, blue_sim in env.blue_planes.items():
                if blue_sim is None or not blue_sim.is_alive:
                    continue
                blue_pos = np.asarray(blue_sim.get_position(), dtype=np.float64)
                nearest_id, _nearest_dist = _nearest(blue_pos, red_positions)
                if nearest_id:
                    nearest_counts[nearest_id] += 1
                    nearest_is_mav_flags.append(nearest_id == "red_0")
                    error = abs(_wrap_pi(_bearing(blue_pos, red_positions[nearest_id]) - _heading(blue_sim)))
                    blue_heading_errors.append(error)
                    blue_heading_to_nearest_flags.append(error < math.radians(45.0))

                inferred_id, _error = _heading_aligned_target(blue_sim, red_positions)
                if inferred_id:
                    inferred_counts[inferred_id] += 1
                    inferred_is_mav_flags.append(inferred_id == "red_0")

                heading_now = _heading(blue_sim)
                if bid in prev_blue_headings:
                    blue_turn_rates.append(abs(_wrap_pi(heading_now - prev_blue_headings[bid])))
                prev_blue_headings[bid] = heading_now
                blue_speeds.append(_speed(blue_sim))
                act = np.asarray(blue_actions.get(bid, np.zeros(3)), dtype=np.float32)
                blue_action_saturation.append(float(np.any(np.abs(act) > 0.95)))

            red0 = env.red_planes.get("red_0")
            if red0 is not None and red0.is_alive:
                red0_pos = np.asarray(red0.get_position(), dtype=np.float64)
                if blue_positions:
                    nearest_blue_id, nearest_blue_dist = _nearest(red0_pos, blue_positions)
                    del nearest_blue_id
                    mav_nearest_blue_distances.append(nearest_blue_dist)
                    nearest_blue_pos = min(blue_positions.values(), key=lambda p: _dist(red0_pos, p))
                    error = abs(_wrap_pi(_bearing(red0_pos, nearest_blue_pos) - _heading(red0)))
                    mav_heading_to_blue_flags.append(error < math.radians(45.0))
                if red_uav_positions:
                    uav_center = np.mean(np.stack(list(red_uav_positions.values())), axis=0)
                    mav_to_uav_center_distances.append(_dist(red0_pos, uav_center))
                    nearest_blue_center = (
                        np.mean(np.stack(list(blue_positions.values())), axis=0)
                        if blue_positions else None
                    )
                    if nearest_blue_center is not None:
                        mav_dist_to_blue_center = _dist(red0_pos, nearest_blue_center)
                        uav_dists = [_dist(pos, nearest_blue_center) for pos in red_uav_positions.values()]
                        mav_frontmost_flags.append(mav_dist_to_blue_center <= min(uav_dists))
                mav_heading = _heading(red0)
                if prev_mav_heading is not None:
                    mav_turn_back_flags.append(abs(_wrap_pi(mav_heading - prev_mav_heading)) > math.radians(30.0))
                prev_mav_heading = mav_heading
                if red_actions.shape[0] > 0:
                    mav_actions.append(red_actions[0].astype(np.float32))

            obs, rewards, terminated, truncated, info = env.step(actions)
            for record in info.get("__launch_quality_step__", []):
                if str(record.get("shooter_id", "")).startswith("blue_"):
                    target_id = str(record.get("target_id", ""))
                    if target_id:
                        missile_targets[target_id] += 1
            steps += 1
            if any(np.any(~np.isfinite(np.asarray(action))) for action in actions.values()):
                nan_detected = True
            if team_done(terminated, truncated):
                break
            if steps >= int(getattr(env, "max_steps", 1000)):
                break
        mav_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
        return {
            "steps": steps,
            "nearest_counts": dict(nearest_counts),
            "inferred_counts": dict(inferred_counts),
            "missile_targets": dict(missile_targets),
            "nearest_is_mav_flags": nearest_is_mav_flags,
            "inferred_is_mav_flags": inferred_is_mav_flags,
            "blue_heading_to_nearest_flags": blue_heading_to_nearest_flags,
            "blue_heading_errors": blue_heading_errors,
            "blue_turn_rates": blue_turn_rates,
            "blue_action_saturation": blue_action_saturation,
            "blue_speeds": blue_speeds,
            "mav_nearest_blue_distances": mav_nearest_blue_distances,
            "mav_to_uav_center_distances": mav_to_uav_center_distances,
            "mav_frontmost_flags": mav_frontmost_flags,
            "mav_heading_to_blue_flags": mav_heading_to_blue_flags,
            "mav_turn_back_flags": mav_turn_back_flags,
            "mav_actions": [a.tolist() for a in mav_actions],
            "mav_alive": mav_alive,
            "nan_detected": nan_detected,
        }
    finally:
        env.close()


def build_audit(args) -> dict[str, Any]:
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    model = rel(args.checkpoint)
    if not model.exists():
        raise FileNotFoundError(f"checkpoint not found: {model}")
    policy, device = load_policy(model, args.device)
    adapter = HeteroObsAdapterV2()
    episodes = [
        _run_episode(policy, device, adapter, args, seed=args.seed + ep)
        for ep in range(args.episodes)
    ]

    nearest_counts = Counter()
    inferred_counts = Counter()
    missile_targets = Counter()
    nearest_is_mav_flags: list[bool] = []
    inferred_is_mav_flags: list[bool] = []
    blue_heading_to_flags: list[bool] = []
    blue_heading_errors: list[float] = []
    blue_turn_rates: list[float] = []
    blue_action_saturation: list[float] = []
    blue_speeds: list[float] = []
    mav_nearest_blue_distances: list[float] = []
    mav_to_uav_center_distances: list[float] = []
    mav_frontmost_flags: list[bool] = []
    mav_heading_to_blue_flags: list[bool] = []
    mav_turn_back_flags: list[bool] = []
    mav_actions: list[list[float]] = []

    for episode in episodes:
        nearest_counts.update(episode["nearest_counts"])
        inferred_counts.update(episode["inferred_counts"])
        missile_targets.update(episode["missile_targets"])
        nearest_is_mav_flags.extend(episode["nearest_is_mav_flags"])
        inferred_is_mav_flags.extend(episode["inferred_is_mav_flags"])
        blue_heading_to_flags.extend(episode["blue_heading_to_nearest_flags"])
        blue_heading_errors.extend(episode["blue_heading_errors"])
        blue_turn_rates.extend(episode["blue_turn_rates"])
        blue_action_saturation.extend(episode["blue_action_saturation"])
        blue_speeds.extend(episode["blue_speeds"])
        mav_nearest_blue_distances.extend(episode["mav_nearest_blue_distances"])
        mav_to_uav_center_distances.extend(episode["mav_to_uav_center_distances"])
        mav_frontmost_flags.extend(episode["mav_frontmost_flags"])
        mav_heading_to_blue_flags.extend(episode["mav_heading_to_blue_flags"])
        mav_turn_back_flags.extend(episode["mav_turn_back_flags"])
        mav_actions.extend(episode["mav_actions"])

    mav_missile_targets = missile_targets.get("red_0", 0)
    uav_missile_targets = sum(
        count for rid, count in missile_targets.items()
        if str(rid).startswith("red_") and rid != "red_0"
    )
    mav_action_arr = np.asarray(mav_actions, dtype=np.float32) if mav_actions else np.zeros((0, 3), dtype=np.float32)
    nearest_mav_rate = _rate(nearest_is_mav_flags)
    inferred_mav_rate = _rate(inferred_is_mav_flags)
    mav_survival_rate = _rate([bool(ep["mav_alive"]) for ep in episodes])
    mav_center_dist_mean = _mean(mav_to_uav_center_distances)
    blue_heading_error_mean = _mean(blue_heading_errors)
    blue_saturation_rate = _mean(blue_action_saturation)
    data = {
        "checkpoint": str(model),
        "config": args.config,
        "episodes": args.episodes,
        "blue_policy": args.opponent_policy,
        "blue_target_selection_code": {
            "contains_mav": True,
            "red_slots_treated_equally_by_role": True,
            "priority": "score alive unengaged red tracks; greedy deconfliction; no explicit MAV or armed-UAV priority",
            "chosen_target_id_available": False,
            "chosen_target_inference": "nearest red, heading-aligned red, and missile launch target records",
        },
        "nearest_red_id_distribution": dict(nearest_counts),
        "inferred_chosen_target_id_distribution": dict(inferred_counts),
        "missile_target_id_distribution": dict(missile_targets),
        "time_blue_nearest_target_is_mav_rate": nearest_mav_rate,
        "time_blue_chosen_target_is_mav_rate": inferred_mav_rate,
        "blue_missile_target_mav_count": int(mav_missile_targets),
        "blue_missile_target_uav_count": int(uav_missile_targets),
        "mav_nearest_blue_distance_mean": _mean(mav_nearest_blue_distances),
        "mav_nearest_blue_distance_min": min(mav_nearest_blue_distances) if mav_nearest_blue_distances else None,
        "mav_to_uav_center_distance_mean": mav_center_dist_mean,
        "mav_frontmost_rate": _rate(mav_frontmost_flags),
        "mav_turn_back_rate": _rate(mav_turn_back_flags),
        "mav_heading_to_blue_rate": _rate(mav_heading_to_blue_flags),
        "mav_action_mean": mav_action_arr.mean(axis=0).tolist() if mav_action_arr.size else [0.0, 0.0, 0.0],
        "mav_action_saturation_rate": float(np.mean(np.abs(mav_action_arr) > 0.95)) if mav_action_arr.size else 0.0,
        "mav_survival_rate": mav_survival_rate,
        "blue_heading_to_nearest_red_rate": _rate(blue_heading_to_flags),
        "blue_turn_rate_mean_rad_per_step": _mean(blue_turn_rates),
        "blue_turn_rate_mean_deg_per_step": math.degrees(_mean(blue_turn_rates)),
        "blue_heading_error_to_nearest_red_mean_rad": blue_heading_error_mean,
        "blue_heading_error_to_nearest_red_mean_deg": math.degrees(blue_heading_error_mean),
        "blue_action_saturation_rate": blue_saturation_rate,
        "blue_speed_mean": _mean(blue_speeds),
        "nan_detected": any(bool(ep["nan_detected"]) for ep in episodes),
    }
    data["mav_role_behavior"] = _classify_mav_behavior(
        data["mav_frontmost_rate"],
        data["mav_to_uav_center_distance_mean"],
        data["mav_survival_rate"],
    )
    data["blue_targeting_behavior"] = _classify_blue_targeting(
        data["blue_missile_target_mav_count"],
        data["blue_missile_target_uav_count"],
        data["time_blue_nearest_target_is_mav_rate"],
    )
    data["blue_pursuit_behavior"] = _classify_blue_pursuit(
        data["blue_heading_to_nearest_red_rate"],
        data["blue_heading_error_to_nearest_red_mean_rad"],
        data["blue_action_saturation_rate"],
    )
    return data


def _write_outputs(data: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "behavior_consistency_5v4.json"
    md_path = output_dir / "behavior_consistency_5v4.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "# 5v4 Behavior Consistency Audit",
        "",
        "## Blue Targeting",
        "",
        f"- target selection contains MAV: {data['blue_target_selection_code']['contains_mav']}",
        f"- role priority: {data['blue_target_selection_code']['priority']}",
        f"- nearest_red_id_distribution: {data['nearest_red_id_distribution']}",
        f"- inferred_chosen_target_id_distribution: {data['inferred_chosen_target_id_distribution']}",
        f"- missile_target_id_distribution: {data['missile_target_id_distribution']}",
        f"- time_blue_nearest_target_is_mav_rate: {data['time_blue_nearest_target_is_mav_rate']}",
        f"- time_blue_chosen_target_is_mav_rate: {data['time_blue_chosen_target_is_mav_rate']}",
        f"- blue_targeting_behavior: {data['blue_targeting_behavior']}",
        "",
        "## MAV Behavior",
        "",
        f"- mav_frontmost_rate: {data['mav_frontmost_rate']}",
        f"- mav_to_uav_center_distance_mean: {data['mav_to_uav_center_distance_mean']}",
        f"- mav_heading_to_blue_rate: {data['mav_heading_to_blue_rate']}",
        f"- mav_turn_back_rate: {data['mav_turn_back_rate']}",
        f"- mav_role_behavior: {data['mav_role_behavior']}",
        "",
        "## Blue Pursuit",
        "",
        f"- blue_heading_to_nearest_red_rate: {data['blue_heading_to_nearest_red_rate']}",
        f"- blue_heading_error_to_nearest_red_mean_deg: {data['blue_heading_error_to_nearest_red_mean_deg']}",
        f"- blue_turn_rate_mean_deg_per_step: {data['blue_turn_rate_mean_deg_per_step']}",
        f"- blue_action_saturation_rate: {data['blue_action_saturation_rate']}",
        f"- blue_pursuit_behavior: {data['blue_pursuit_behavior']}",
        "",
        "## Paper Consistency Note",
        "",
        "The heterogeneous paper motivates MAV battlefield information, mission guidance, and self-safety roles.",
        "It does not specify an exact trajectory constraint that forces the MAV to stay behind the UAVs.",
        "Current behavior should therefore be described as a behavior-level limitation, not as a full paper-aligned MAV trajectory.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit 5v4 behavior consistency")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = build_audit(args)
    json_path, md_path = _write_outputs(data, rel(args.output_dir))
    print(f"output_json: {json_path}")
    print(f"output_md: {md_path}")
    print(f"blue_targeting_behavior: {data['blue_targeting_behavior']}")
    print(f"mav_role_behavior: {data['mav_role_behavior']}")
    print(f"blue_pursuit_behavior: {data['blue_pursuit_behavior']}")
    print(f"mav_frontmost_rate: {data['mav_frontmost_rate']}")
    print(f"time_blue_nearest_target_is_mav_rate: {data['time_blue_nearest_target_is_mav_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
