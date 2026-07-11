"""Strict read-only reward protocol audit for brma_tam_scale_aligned_v1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy  # noqa: E402
from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv  # noqa: E402

DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v1.yaml"
RED_MODES = ("zero_absolute", "hold_current_kinematics", "random_full_range")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _discounted(values, gamma):
    return float(sum((gamma ** i) * float(v) for i, v in enumerate(values)))


def _path_row(name: str, distances_km: list[float]) -> tuple[dict, list[dict]]:
    phi = HeteroUavCombatEnv._scale_v1_distance_potential
    steps = []
    raw_sum = active_sum = 0.0
    for index, (prev, cur) in enumerate(zip(distances_km, distances_km[1:]), 1):
        prev_phi, cur_phi = phi(prev * 1000), phi(cur * 1000)
        raw = 5.0 * (cur_phi - prev_phi)
        clipped = float(np.clip(raw, -0.5, 0.5))
        raw_sum += raw; active_sum += clipped
        steps.append({
            "path": name, "step": index, "previous_km": prev, "current_km": cur,
            "previous_potential": prev_phi, "current_potential": cur_phi,
            "raw_delta_reward": raw, "clipped_reward": clipped,
            "cumulative_raw_reward": raw_sum, "cumulative_active_reward": active_sum,
        })
    endpoint = 5.0 * (phi(distances_km[-1] * 1000) - phi(distances_km[0] * 1000))
    clip_count = sum(abs(r["raw_delta_reward"] - r["clipped_reward"]) > 1e-12 for r in steps)
    return {
        "path": name, "segments": len(steps),
        "endpoint_unclipped_potential_difference": endpoint,
        "stepwise_raw_sum": raw_sum, "stepwise_clipped_active_sum": active_sum,
        "clip_count": clip_count, "clip_ratio": clip_count / max(len(steps), 1),
    }, steps


def _progress_audits(gamma: float):
    paths = {
        "forward_fine": [22.3, 20, 17.5, 15, 12.5, 10, 7.5, 5],
        "reverse_fine": [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.3],
        "forward_coarse": [22.3, 15, 10, 5],
        "reverse_coarse": [5, 15, 22.3],
    }
    summaries, step_rows = [], []
    for name, sequence in paths.items():
        summary, rows = _path_row(name, sequence)
        summaries.append(summary); step_rows.extend(rows)
    by_name = {row["path"]: row for row in summaries}
    for forward, reverse in (("forward_fine", "reverse_fine"), ("forward_coarse", "reverse_coarse")):
        by_name[forward]["forward_reverse_active_sum"] = (
            by_name[forward]["stepwise_clipped_active_sum"] + by_name[reverse]["stepwise_clipped_active_sum"]
        )

    forward_rewards = [r["clipped_reward"] for r in step_rows if r["path"] == "forward_fine"]
    reverse_rewards = [r["clipped_reward"] for r in step_rows if r["path"] == "reverse_fine"]
    cycle_rewards = forward_rewards + reverse_rewards
    cycles = [{
        "cycle": "fine_round_trip", "steps": len(cycle_rewards),
        "undiscounted_active_cycle_return": sum(cycle_rewards),
        "discounted_active_cycle_return_gamma_0_99": _discounted(cycle_rewards, .99),
        "training_gamma": gamma,
        "discounted_active_cycle_return_training_gamma": _discounted(cycle_rewards, gamma),
        "potential_form": "Phi(s_next)-Phi(s)",
        "gamma_potential_form_used": False,
    }]

    discretization = []
    for segments in (2, 4, 8, 16, 32):
        forward = np.linspace(22.3, 5.0, segments + 1).tolist()
        reverse = list(reversed(forward))
        f, _ = _path_row(f"forward_{segments}", forward)
        r, _ = _path_row(f"reverse_{segments}", reverse)
        discretization.append({
            "segments": segments, "forward_active_return": f["stepwise_clipped_active_sum"],
            "reverse_active_return": r["stepwise_clipped_active_sum"],
            "round_trip_active_return": f["stepwise_clipped_active_sum"] + r["stepwise_clipped_active_sum"],
            "forward_clip_count": f["clip_count"], "reverse_clip_count": r["clip_count"],
        })
    return summaries, step_rows, cycles, discretization


def _hold_action(env, aid: str) -> np.ndarray:
    sim = env._get_sim(aid)
    rpy = np.asarray(sim.get_rpy(), dtype=np.float64)
    pitch = float(rpy[1]) if rpy.size > 1 else 0.0
    heading = float(rpy[2]) if rpy.size > 2 else 0.0
    speed = float(np.linalg.norm(np.asarray(sim.get_velocity(), dtype=np.float64)))
    action = np.asarray([
        pitch / np.deg2rad(float(env.PITCH_DEG)),
        heading / np.pi,
        2.0 * (speed - float(env.VELOCITY_MIN)) / max(float(env.VELOCITY_MAX - env.VELOCITY_MIN), 1e-6) - 1.0,
    ], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def _red_actions(env, mode: str, rng) -> dict:
    if mode == "zero_absolute":
        return {rid: np.zeros(3, np.float32) for rid in env.red_ids}
    if mode == "hold_current_kinematics":
        return {rid: _hold_action(env, rid) for rid in env.red_ids}
    if mode == "random_full_range":
        return {rid: rng.uniform(-1.0, 1.0, 3).astype(np.float32) for rid in env.red_ids}
    raise ValueError(mode)


UAV_KEYS = (
    "scale_v1_flight_pitch", "scale_v1_flight_roll", "scale_v1_flight_altitude",
    "scale_v1_flight_boundary", "scale_v1_flight_velocity", "scale_v1_flight_total",
    "scale_v1_progress_raw", "scale_v1_progress_clipped", "scale_v1_uav_event_kill",
    "scale_v1_uav_event_death", "scale_v1_uav_event_oob", "scale_v1_terminal",
    "scale_v1_uav_total",
)
MAV_KEYS = (
    "scale_v1_flight_pitch", "scale_v1_flight_roll", "scale_v1_flight_altitude",
    "scale_v1_flight_boundary", "scale_v1_flight_velocity", "scale_v1_flight_total",
    "scale_v1_mav_role_raw", "scale_v1_mav_role", "scale_v1_mav_dist_raw",
    "scale_v1_mav_threat_raw", "scale_v1_mav_aspect_raw_sum", "scale_v1_mav_aspect_mean",
    "scale_v1_mav_pos_raw", "scale_v1_mav_aware_raw_sum", "scale_v1_mav_aware_mean",
    "scale_v1_mav_event_death", "scale_v1_mav_team_credit_delta", "scale_v1_terminal",
    "scale_v1_mav_total",
)


def _episode(config, mode, opponent_mode, seed, max_steps):
    env = make_env(config, max_steps=max_steps)
    opponent = OpponentPolicy(mode=opponent_mode, seed=seed + 10000)
    rng = np.random.default_rng(seed)
    uav_sums = defaultdict(float); mav_sums = defaultdict(float)
    uav_rows = mav_rows = 0; team_return = fixed_return = 0.0
    action_values = []; identity_max = 0.0; launches = {"red": 0, "blue": 0}; hits = {"red": 0, "blue": 0}
    prior_hits = {"red": 0, "blue": 0}; first_obs = None
    try:
        obs, _ = env.reset(seed=seed); first_obs = obs
        opponent.reset_memory()
        for step in range(max_steps):
            red = _red_actions(env, mode, rng)
            blue = opponent.act(obs, env.blue_ids, deterministic=True, env=env)
            action_values.extend(float(v) for a in red.values() for v in a)
            obs, rewards, terminated, truncated, info = env.step({**red, **blue})
            comps = info.get("reward_components", {})
            active = [rid for rid in env.red_ids if env._scale_v1_alive_before_step.get(rid, False)]
            if active:
                team_return += float(np.mean([rewards[rid] for rid in active]))
            fixed_return += float(sum(rewards[rid] for rid in env.red_ids)) / len(env.red_ids)
            for rid in active:
                comp = comps.get(rid, {})
                identity_max = max(identity_max, abs(float(comp.get("scale_v1_identity_error", 0.0))))
                if env.agent_roles.get(rid) == "mav":
                    mav_rows += 1
                    for key in MAV_KEYS: mav_sums[key] += float(comp.get(key, 0.0) or 0.0)
                else:
                    uav_rows += 1
                    for key in UAV_KEYS: uav_sums[key] += float(comp.get(key, 0.0) or 0.0)
            for aid in env.agent_ids:
                launches["red" if aid.startswith("red_") else "blue"] += int(info.get(aid, {}).get("missiles_fired_this_step", 0) or 0)
            term = info.get("__missile_term__", {})
            for side in ("red", "blue"):
                total_hit = int(term.get(side, {}).get("hit", prior_hits[side]) or 0)
                hits[side] += max(total_hit - prior_hits[side], 0); prior_hits[side] = total_hit
            if all(terminated.values()) or all(truncated.values()): break
        red_alive = sum(sim.is_alive for sim in env.red_planes.values()); blue_alive = sum(sim.is_alive for sim in env.blue_planes.values())
        death_reasons = dict(getattr(env, "_death_reasons", {}) or {})
        mav_id = next((r for r in env.red_ids if env.agent_roles.get(r) == "mav"), "")
        end_reason = "blue_eliminated" if blue_alive == 0 and red_alive else "red_eliminated" if red_alive == 0 and blue_alive else "timeout" if step + 1 >= max_steps else "other"
        row = {
            "seed": seed, "action_mode": mode, "opponent_policy": opponent_mode,
            "opponent_act_called": True, "episode_length": step + 1,
            "trainer_effective_return": team_return, "fixed_initial_team_mean_return": fixed_return,
            "red_alive_final": red_alive, "blue_alive_final": blue_alive,
            "red_launches": launches["red"], "blue_launches": launches["blue"],
            "red_hits": hits["red"], "blue_hits": hits["blue"],
            "red_kills": len(env.blue_ids) - blue_alive, "blue_kills": len(env.red_ids) - red_alive,
            "termination_reason": end_reason, "mav_death_reason": death_reasons.get(mav_id, ""),
            "uav_death_reasons": "|".join(str(death_reasons.get(r, "")) for r in env.red_ids if env.agent_roles.get(r) == "attack_uav" and death_reasons.get(r)),
            "mav_low_altitude_death_count": int("low" in str(death_reasons.get(mav_id, "")).lower()),
            "uav_horizontal_oob_count": sum("out" in str(death_reasons.get(r, "")).lower() for r in env.red_ids[1:]),
            "reward_identity_max_error": identity_max, "uav_active_rows": uav_rows,
            "mav_active_rows": mav_rows, "action_min": min(action_values, default=0),
            "action_max": max(action_values, default=0), "first_obs_agent_count": len(first_obs or {}),
        }
        for key, value in uav_sums.items(): row[f"uav_{key}_sum"] = value; row[f"uav_{key}_mean"] = value / max(uav_rows, 1)
        for key, value in mav_sums.items(): row[f"mav_{key}_sum"] = value; row[f"mav_{key}_mean"] = value / max(mav_rows, 1)
        return row
    finally:
        env.close()


def _reward_summary(rows):
    output = []
    for mode in RED_MODES:
        selected = [r for r in rows if r["action_mode"] == mode]
        result = {"action_mode": mode, "episodes": len(selected)}
        numeric = [k for k, v in selected[0].items() if isinstance(v, (int, float, bool))] if selected else []
        for key in numeric: result[f"{key}_mean"] = float(np.mean([float(r[key]) for r in selected]))
        output.append(result)
    return output


def _objective_ordering(dense_per_step: float):
    base = [
        ("no_loss_timeout", 0.0), ("one_blue_kill_timeout", 20.0),
        ("half_blue_loss_timeout", 20.0), ("no_loss_full_win", 40.0),
        ("one_uav_loss_full_win", 22.5 + 20.0 / 3.0),
        ("mav_loss_full_win", 15.0 + 10.0 / 3.0),
        ("mav_loss_no_kill", -15.0 - 20.0 / 3.0),
        ("full_red_loss", -30.0 - 40.0 / 3.0),
    ]
    rows = []
    for length in (100, 250, 500, 750, 1000):
        for name, objective in base:
            rows.append({
                "audit_layer": "episode_length_sensitivity", "case": name,
                "episode_length": length, "dense_per_step": dense_per_step,
                "dense_return": dense_per_step * length, "event_terminal_return": objective,
                "total_return": dense_per_step * length + objective,
            })
    for name, objective in base:
        rows.append({"audit_layer": "same_dense_same_length", "case": name, "episode_length": 0, "dense_per_step": 0.0, "dense_return": 0.0, "event_terminal_return": objective, "total_return": objective})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--opponent-policy", default="brma_rule")
    parser.add_argument("--episodes-per-mode", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--output-dir", default="outputs/brma_tam_scale_aligned_v1_audit_v2")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    path_summaries, path_steps, cycles, discretization = _progress_audits(args.gamma)
    episodes = []
    for mode in RED_MODES:
        for ep in range(args.episodes_per_mode):
            episodes.append(_episode(args.config, mode, args.opponent_policy, args.seed_start + ep, args.max_steps))
    rewards = _reward_summary(episodes)
    safe_row = next((r for r in rewards if r["action_mode"] == "zero_absolute"), {})
    dense_reference = float(safe_row.get("trainer_effective_return_mean", 0.0)) / max(float(safe_row.get("episode_length_mean", 1.0)), 1.0)
    ordering = _objective_ordering(dense_reference)
    same = {r["case"]: r["total_return"] for r in ordering if r["audit_layer"] == "same_dense_same_length"}
    ordering_ok = same["no_loss_full_win"] > same["one_uav_loss_full_win"] > same["one_blue_kill_timeout"] > same["no_loss_timeout"] > same["mav_loss_no_kill"] > same["full_red_loss"]
    length_rows = [r for r in ordering if r["audit_layer"] == "episode_length_sensitivity"]
    safe_1000 = float(safe_row.get("trainer_effective_return_mean", 0.0))
    early_win = next(r["total_return"] for r in length_rows if r["case"] == "no_loss_full_win" and r["episode_length"] == 100)
    partial_1000 = next(r["total_return"] for r in length_rows if r["case"] == "one_blue_kill_timeout" and r["episode_length"] == 1000)
    early_full_red_loss = next(r["total_return"] for r in length_rows if r["case"] == "full_red_loss" and r["episode_length"] == 100)
    flight_cumulative = (
        float(safe_row.get("uav_scale_v1_flight_total_sum_mean", 0.0))
        + float(safe_row.get("mav_scale_v1_flight_total_sum_mean", 0.0))
    ) / 3.0
    mav_role_cumulative = float(safe_row.get("mav_scale_v1_mav_role_sum_mean", 0.0)) / 3.0
    objective_risk_reasons = []
    if safe_1000 > early_win: objective_risk_reasons.append("safe_timeout_above_early_no_loss_win")
    if partial_1000 > early_win: objective_risk_reasons.append("partial_kill_timeout_above_early_no_loss_win")
    if early_full_red_loss > safe_1000: objective_risk_reasons.append("early_full_red_loss_above_continued_safe_timeout")
    if abs(flight_cumulative) > 30.0: objective_risk_reasons.append("flight_cumulative_exceeds_terminal_scale")
    if abs(mav_role_cumulative) > 30.0: objective_risk_reasons.append("mav_role_cumulative_exceeds_terminal_scale")
    objective_risk = bool(objective_risk_reasons)
    summary = {
        "reward_contract_revision": 3, "progress_path_summaries": path_summaries,
        "progress_cycles": cycles, "progress_discretization": discretization,
        "episodes": len(episodes), "opponent_policy": args.opponent_policy,
        "opponent_act_all_called": all(r["opponent_act_called"] for r in episodes),
        "reward_identity_max_error": max(r["reward_identity_max_error"] for r in episodes),
        "objective_same_length_ordering_ok": ordering_ok,
        "objective_ordering_risk": objective_risk,
        "dense_reference_per_decision": dense_reference,
        "safe_timeout_1000": safe_1000, "early_no_loss_win_100": early_win,
        "partial_kill_timeout_1000": partial_1000,
        "early_full_red_loss_100": early_full_red_loss,
        "safe_timeout_flight_cumulative_team": flight_cumulative,
        "safe_timeout_mav_role_cumulative_team": mav_role_cumulative,
        "objective_ordering_risk_reasons": objective_risk_reasons,
        "nonfinite_count": sum(not math.isfinite(float(v)) for r in episodes for v in r.values() if isinstance(v, (int, float))),
    }
    _write_csv(out / "progress_path_audit.csv", path_steps)
    _write_csv(out / "progress_cycle_audit.csv", cycles)
    _write_csv(out / "progress_discretization_audit.csv", discretization)
    _write_csv(out / "environment_rollout_episode_summary.csv", episodes)
    _write_csv(out / "environment_rollout_reward_summary.csv", rewards)
    _write_csv(out / "environment_rollout_death_summary.csv", [{k: r[k] for k in ("seed","action_mode","mav_death_reason","uav_death_reasons","mav_low_altitude_death_count","uav_horizontal_oob_count")} for r in episodes])
    _write_csv(out / "environment_rollout_action_summary.csv", [{k: r[k] for k in ("seed","action_mode","opponent_policy","action_min","action_max")} for r in episodes])
    _write_csv(out / "objective_ordering_audit.csv", ordering)
    (out / "scale_aligned_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Scale-aligned strict audit", "",
        f"- Opponent act called: `{summary['opponent_act_all_called']}`",
        f"- Identity max error: `{summary['reward_identity_max_error']}`",
        f"- Objective same-length ordering: `{ordering_ok}`",
        f"- Objective ordering risk: `{objective_risk}`",
        f"- Risk reasons: `{objective_risk_reasons}`",
        f"- Dense reference per decision: `{dense_reference:.8f}`",
        f"- Actual safe timeout return: `{safe_1000:.8f}`",
        f"- Early full-red-loss sensitivity return: `{early_full_red_loss:.8f}`",
        f"- Safe-timeout flight cumulative team contribution: `{flight_cumulative:.8f}`",
        f"- Safe-timeout MAV-role cumulative team contribution: `{mav_role_cumulative:.8f}`",
    ]
    (out / "scale_aligned_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__": main()
