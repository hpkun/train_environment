"""Deterministic episode-level contract audit for the scripted composite reward."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402


ACTIVE_KEYS = (
    "brma_pitch", "brma_roll", "brma_vel", "tam_speed_weighted",
    "tam_angle_weighted", "tam_distance_weighted", "uav_event_total",
    "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
    "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
)


def _zero_actions(env) -> dict:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _finite_number(value) -> bool:
    return not isinstance(value, (int, float, np.integer, np.floating)) or math.isfinite(float(value))


def _active_values(role: str, comp: dict) -> dict[str, float]:
    keys = ["brma_pitch", "brma_roll", "brma_vel"]
    if role == "mav":
        keys += [
            "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
            "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
        ]
    else:
        keys += ["tam_speed_weighted", "tam_angle_weighted", "tam_distance_weighted", "uav_event_total"]
    return {key: float(comp.get(key, 0.0) or 0.0) for key in keys}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _outcome(red_alive: int, blue_alive: int, timed_out: bool) -> tuple[str, str]:
    if timed_out:
        return "draw", "timeout"
    if red_alive > 0 and blue_alive == 0:
        return "red", "red_elimination"
    if blue_alive > 0 and red_alive == 0:
        return "blue", "blue_elimination"
    return "draw", "mutual_or_non_elimination"


def run_audit(args) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.config, max_steps=args.max_steps)
    step_rows: list[dict] = []
    target_rows: list[dict] = []
    episode_rows: list[dict] = []
    env_steps_total = 0
    target_row_warnings: list[str] = []
    try:
        initial_red_count = len(env.red_ids)
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            team_effective: list[float] = []
            episode_agent_reward_sum = 0.0
            initial_team_mean_sum = 0.0
            component_totals: defaultdict[str, float] = defaultdict(float)
            target_seen = 0
            target_observed = target_direct = target_shared = 0
            target_match_lock = target_match_launch = 0
            target_switch_last: dict[str, float] = {}
            red_launch = blue_launch = red_hit = blue_hit = 0
            previous_hits = {"red": 0, "blue": 0}
            episode_length = 0
            last_terminated = last_truncated = {}

            for step in range(args.max_steps):
                alive_before = {
                    aid: bool(getattr(env._get_sim(aid), "is_alive", False))
                    for aid in env.agent_ids
                }
                _obs, rewards, terminated, truncated, info = env.step(_zero_actions(env))
                episode_length += 1
                env_steps_total += 1
                last_terminated, last_truncated = terminated, truncated
                comps = info.get("reward_components", {}) if isinstance(info, dict) else {}
                diagnostic_by_agent = {
                    str(rec.get("agent_id", "")): rec
                    for rec in info.get("__reward_target_diagnostics__", []) or []
                    if isinstance(rec, dict)
                }
                expected_diag = sum(
                    1 for rid in env.red_ids
                    if alive_before.get(rid, False) and env.agent_roles.get(rid) == "attack_uav"
                )
                if len(diagnostic_by_agent) != expected_diag:
                    target_row_warnings.append(
                        f"episode={ep} step={step} expected={expected_diag} actual={len(diagnostic_by_agent)}"
                    )
                target_rows.extend(
                    {"episode_id": ep, "env_step": step, **dict(rec)}
                    for rec in diagnostic_by_agent.values()
                )

                active_rewards = []
                active_component_sums = []
                step_all_red_sum = 0.0
                for rid in env.red_ids:
                    role = env.agent_roles.get(rid, "")
                    comp = dict(comps.get(rid, {}) or {})
                    reward = float(rewards.get(rid, 0.0) or 0.0)
                    active = _active_values(role, comp)
                    active_sum = float(sum(active.values()))
                    if not math.isclose(active_sum, reward, rel_tol=1e-7, abs_tol=1e-6):
                        raise ValueError(
                            f"active total mismatch episode={ep} step={step} agent={rid}: "
                            f"components={active_sum} reward={reward}"
                        )
                    if not all(_finite_number(v) for v in [reward, *comp.values()]):
                        raise ValueError(f"non-finite reward diagnostic episode={ep} step={step} agent={rid}")
                    diag = diagnostic_by_agent.get(rid, {})
                    row = {
                        "episode_id": ep, "env_step": step, "agent_id": rid, "role": role,
                        "alive_before": int(alive_before.get(rid, False)),
                        "alive_after": int(bool(getattr(env._get_sim(rid), "is_alive", False))),
                        "raw_agent_reward": reward,
                        "active_component_sum": active_sum,
                        **comp,
                    }
                    for key, value in diag.items():
                        row[f"diag_{key}"] = value
                    step_rows.append(row)
                    episode_agent_reward_sum += reward
                    step_all_red_sum += reward
                    for key, value in comp.items():
                        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                            component_totals[key] += float(value)
                    if alive_before.get(rid, False):
                        active_rewards.append(reward)
                        active_component_sums.append(active_sum)
                    if diag:
                        target_seen += 1
                        target_observed += int(float(diag.get("reward_target_observed", 0.0) or 0.0) > 0.5)
                        target_direct += int(float(diag.get("reward_target_direct_visible", 0.0) or 0.0) > 0.5)
                        target_shared += int(float(diag.get("reward_target_mav_shared_visible", 0.0) or 0.0) > 0.5)
                        target_match_lock += int(float(diag.get("reward_target_matches_lock", 0.0) or 0.0) > 0.5)
                        target_match_launch += int(float(diag.get("reward_target_matches_launch", 0.0) or 0.0) > 0.5)
                        target_switch_last[rid] = float(diag.get("reward_target_switch_count", 0.0) or 0.0)

                effective = float(np.mean(active_rewards)) if active_rewards else 0.0
                effective_components = float(np.mean(active_component_sums)) if active_component_sums else 0.0
                if not math.isclose(effective, effective_components, rel_tol=1e-7, abs_tol=1e-6):
                    raise ValueError(f"trainer-effective mismatch episode={ep} step={step}")
                team_effective.append(effective)
                initial_team_mean_sum += step_all_red_sum / max(initial_red_count, 1)

                for aid in env.agent_ids:
                    fired = int(info.get(aid, {}).get("missiles_fired_this_step", 0) or 0)
                    if aid.startswith("red_"):
                        red_launch += fired
                    else:
                        blue_launch += fired
                mt = info.get("__missile_term__", {}) if isinstance(info, dict) else {}
                if isinstance(mt, dict):
                    rh = int(mt.get("red", {}).get("hit", 0) or 0)
                    bh = int(mt.get("blue", {}).get("hit", 0) or 0)
                    red_hit += max(rh - previous_hits["red"], 0)
                    blue_hit += max(bh - previous_hits["blue"], 0)
                    previous_hits = {"red": rh, "blue": bh}
                if all(terminated.values()) or all(truncated.values()):
                    break

            red_alive = sum(int(getattr(env.red_planes.get(rid), "is_alive", False)) for rid in env.red_ids)
            blue_alive = sum(int(getattr(env.blue_planes.get(bid), "is_alive", False)) for bid in env.blue_ids)
            timed_out = bool(last_truncated and all(last_truncated.values()))
            winner, end_reason = _outcome(red_alive, blue_alive, timed_out)
            discounted = sum((0.99 ** idx) * value for idx, value in enumerate(team_effective))
            dense = sum(component_totals[k] for k in (
                "brma_pitch", "brma_roll", "brma_vel", "tam_speed_weighted",
                "tam_angle_weighted", "tam_distance_weighted", "mav_dist_weighted",
                "mav_threat_weighted", "mav_aspect_weighted", "mav_pos_weighted", "mav_aware_weighted",
            ))
            event = component_totals["uav_event_total"] + component_totals["mav_event_total"]
            episode_rows.append({
                "episode_id": ep, "seed": args.seed + ep, "episode_length": episode_length,
                "winner": winner, "end_reason": end_reason, "red_alive_final": red_alive,
                "blue_alive_final": blue_alive,
                "mav_alive_final": int(bool(getattr(env.red_planes.get("red_0"), "is_alive", False))),
                "episode_reward_sum_all_red": episode_agent_reward_sum,
                "episode_reward_initial_team_mean_sum": initial_team_mean_sum,
                "trainer_effective_team_reward_sum": sum(team_effective),
                "trainer_effective_team_reward_per_step": float(np.mean(team_effective)) if team_effective else 0.0,
                "discounted_team_return_gamma_0_99": discounted,
                "dense_reward_sum": dense, "dense_reward_per_step": dense / max(episode_length, 1),
                "event_reward_sum": event,
                "brma_flight_sum": sum(component_totals[k] for k in ("brma_pitch", "brma_roll", "brma_vel")),
                "uav_speed_sum": component_totals["tam_speed_weighted"],
                "uav_angle_sum": component_totals["tam_angle_weighted"],
                "uav_distance_sum": component_totals["tam_distance_weighted"],
                "uav_event_sum": component_totals["uav_event_total"],
                "mav_safety_sum": sum(component_totals[k] for k in ("mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted")),
                "mav_support_sum": component_totals["mav_pos_weighted"] + component_totals["mav_aware_weighted"],
                "mav_event_sum": component_totals["mav_event_total"],
                "mav_dist_sum": component_totals["mav_dist_raw"],
                "mav_threat_sum": component_totals["mav_threat_raw"],
                "mav_aspect_sum": component_totals["mav_aspect_raw_sum"],
                "mav_pos_sum": component_totals["mav_pos_raw"],
                "mav_aware_sum": component_totals["mav_aware_raw_sum"],
                "evasion_override_steps": component_totals["evasion_override_active"],
                "reward_target_rows": target_seen,
                "reward_target_observed_rate": target_observed / max(target_seen, 1),
                "reward_target_direct_visible_rate": target_direct / max(target_seen, 1),
                "reward_target_mav_shared_visible_rate": target_shared / max(target_seen, 1),
                "reward_target_matches_lock_rate": target_match_lock / max(target_seen, 1),
                "reward_target_matches_launch_rate": target_match_launch / max(target_seen, 1),
                "reward_target_switch_count": sum(target_switch_last.values()),
                "all_attack_uav_dead_steps": component_totals["all_attack_uav_dead"],
                "mav_reward_after_all_attack_uav_dead": component_totals["mav_reward_after_all_attack_uav_dead"],
                "above_altitude_max_steps": component_totals["above_altitude_max_steps"],
                "max_altitude_m": max((float(row.get("max_altitude_m", 0.0) or 0.0) for row in step_rows if row["episode_id"] == ep), default=0.0),
                "above_altitude_max_episode_flag": int(component_totals["above_altitude_max_episode_flag"] > 0.0),
                "red_launch_count": red_launch, "red_hit_count": red_hit,
                "blue_launch_count": blue_launch, "blue_hit_count": blue_hit,
            })
    finally:
        env.close()

    for row in episode_rows:
        if not all(_finite_number(value) for value in row.values()):
            raise ValueError(f"non-finite episode summary: episode={row['episode_id']}")
    _write_csv(out_dir / "brma_tam_scripted_composite_v1_components.csv", step_rows)
    _write_csv(out_dir / "reward_target_diagnostics.csv", target_rows)
    _write_csv(out_dir / "episode_summary.csv", episode_rows)

    grouped = {}
    for label in ("red_win", "blue_win", "draw", "timeout"):
        selected = [
            row for row in episode_rows
            if (label == "red_win" and row["winner"] == "red")
            or (label == "blue_win" and row["winner"] == "blue")
            or (label == "draw" and row["winner"] == "draw" and row["end_reason"] != "timeout")
            or (label == "timeout" and row["end_reason"] == "timeout")
        ]
        rewards = [row["trainer_effective_team_reward_sum"] for row in selected]
        returns = [row["discounted_team_return_gamma_0_99"] for row in selected]
        reward_mean, reward_std = _mean_std(rewards)
        return_mean, return_std = _mean_std(returns)
        grouped[label] = {
            "episode_count": len(selected), "team_reward_mean": reward_mean,
            "team_reward_std": reward_std, "discounted_return_mean": return_mean,
            "discounted_return_std": return_std,
            "mav_survival": float(np.mean([r["mav_alive_final"] for r in selected])) if selected else 0.0,
            "red_alive_mean": float(np.mean([r["red_alive_final"] for r in selected])) if selected else 0.0,
            "blue_alive_mean": float(np.mean([r["blue_alive_final"] for r in selected])) if selected else 0.0,
            "red_hit_mean": float(np.mean([r["red_hit_count"] for r in selected])) if selected else 0.0,
            "blue_hit_mean": float(np.mean([r["blue_hit_count"] for r in selected])) if selected else 0.0,
            "dense_reward_mean": float(np.mean([r["dense_reward_sum"] for r in selected])) if selected else 0.0,
            "event_reward_mean": float(np.mean([r["event_reward_sum"] for r in selected])) if selected else 0.0,
        }
    summary = {
        "episodes": int(args.episodes), "steps_recorded": env_steps_total,
        "agent_step_rows": len(step_rows), "target_diagnostic_rows": len(target_rows),
        "target_diagnostic_warnings": target_row_warnings,
        "reward_contract_revision": 2, "outcome_groups": grouped,
        "episode_summaries": episode_rows,
        "note": "Fixed zero-action rollout audit only; this is not learned-policy performance.",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# BRMA/TAM scripted composite v1 contract audit", "", f"- episodes: {args.episodes}",
             f"- environment decision steps: {env_steps_total}", f"- agent-step rows: {len(step_rows)}",
             f"- target diagnostic warnings: {len(target_row_warnings)}", "", "## Outcome groups", ""]
    for label, values in grouped.items():
        lines.append(f"- {label}: {json.dumps(values, ensure_ascii=False)}")
    (out_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/audit_brma_tam_scripted_composite_v1")
    args = parser.parse_args()
    print(json.dumps(run_audit(args), indent=2))


if __name__ == "__main__":
    main()
