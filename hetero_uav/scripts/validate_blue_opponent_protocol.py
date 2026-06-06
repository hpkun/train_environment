"""Validate blue opponent protocol — compare rule_nearest vs greedy_fsm.

No training.  Runs fixed red policies + scripted blue opponents across
configs, collecting environment behaviour metrics to inform the default
opponent decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env import make_env

DEFAULT_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]

# -- helpers -----------------------------------------------------------


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _red_actions(env, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
    if mode == "bounded_random":
        return {
            rid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
            for rid in env.red_ids
        }
    raise ValueError(f"unknown red_policy: {mode}")


def _alive_counts(env) -> tuple[int, int]:
    red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
    blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
    return red_alive, blue_alive


def _classify_episode(env, any_terminated: bool, any_truncated: bool, steps: int) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    mav_sim = env.red_planes.get("red_0")
    mav_alive = bool(mav_sim is not None and mav_sim.is_alive)
    timeout = bool(any_truncated or steps >= getattr(env, "max_steps", 0))

    if blue_alive == 0 and red_alive > 0:
        end_reason = "red_win_elimination"
        winner = "red"
    elif red_alive == 0 and blue_alive > 0:
        end_reason = "blue_win_elimination"
        winner = "blue"
    elif red_alive == 0 and blue_alive == 0:
        end_reason = "mutual_elimination_draw"
        winner = "draw"
    elif timeout:
        end_reason = "timeout"
        if red_alive > blue_alive:
            winner = "red_alive_advantage"
        elif red_alive < blue_alive:
            winner = "blue_alive_advantage"
        else:
            winner = "draw"
    elif any_terminated:
        end_reason = "partial_agent_done"
        winner = "draw"
    else:
        end_reason = "not_ended"
        winner = "none"

    return {
        "episode_end_reason": end_reason,
        "winner": winner,
        "red_alive_final": int(red_alive),
        "blue_alive_final": int(blue_alive),
        "mav_survived": bool(mav_alive),
        "steps_executed": int(steps),
    }


def _opponent_difficulty_label(red_win: float, blue_win: float, mav_surv: float,
                                nan: bool) -> str:
    if nan:
        return "nan_failure"
    if blue_win >= 0.9 and mav_surv <= 0.2:
        return "too_strong_candidate"
    if red_win >= 0.9:
        return "too_weak_candidate"
    return "stable_candidate"


# -- per-config validation --------------------------------------------


def validate_config(
    config_path: str,
    blue_opponent: str,
    red_policy: str,
    steps: int,
    episodes: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    policy = OpponentPolicy(mode=blue_opponent, seed=seed + 13)

    nan_detected = False
    actions_seen: list[np.ndarray] = []
    episode_lengths: list[int] = []
    end_reason_counts: dict[str, int] = {}
    winner_counts: dict[str, int] = {}
    red_alive_finals: list[int] = []
    blue_alive_finals: list[int] = []
    mav_survivals: list[int] = []
    state_counts: dict[str, int] = {}
    assigned_target_counts: dict[str, int] = {}
    used_env_refresh = False
    used_env_kinematics = False
    used_env_positions = False

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            nan_detected = nan_detected or _contains_nan(obs)
            ep_steps = 0
            any_terminated = False
            any_truncated = False

            for _step in range(steps):
                red_acts = _red_actions(env, red_policy, rng)
                blue_acts = policy.act(obs, env.blue_ids, env=env)
                actions = {**red_acts, **blue_acts}

                # Validate blue actions
                for bid, action in blue_acts.items():
                    arr = np.asarray(action, dtype=np.float32)
                    if arr.shape != (3,):
                        raise RuntimeError(
                            f"{bid} action shape {arr.shape} in {blue_opponent}"
                        )
                    if not np.isfinite(arr).all():
                        raise RuntimeError(
                            f"{bid} action NaN/Inf in {blue_opponent}"
                        )
                    if np.any(arr < -1.0) or np.any(arr > 1.0):
                        raise RuntimeError(
                            f"{bid} action out of [-1,1] in {blue_opponent}: "
                            f"min={arr.min():.3f} max={arr.max():.3f}"
                        )
                    actions_seen.append(arr)

                obs, rewards, terminated, truncated, info = env.step(actions)
                ep_steps += 1
                nan_detected = (
                    nan_detected
                    or _contains_nan(obs)
                    or _contains_nan(rewards)
                )

                any_terminated = any_terminated or any(terminated.values())
                any_truncated = any_truncated or any(truncated.values())

                if blue_opponent == "greedy_fsm":
                    for state in policy.last_states.values():
                        state_counts[state] = state_counts.get(state, 0) + 1
                    assigned = getattr(policy, "last_assigned_targets", {})
                    for slot in assigned.values():
                        key = str(slot)
                        assigned_target_counts[key] = (
                            assigned_target_counts.get(key, 0) + 1
                        )

                if any_terminated or any_truncated:
                    break

            # After episode
            used_env_refresh = used_env_refresh or bool(
                getattr(policy, "used_env_refresh_engaged_targets", False)
            )
            used_env_kinematics = used_env_kinematics or bool(
                getattr(policy, "used_env_own_kinematics", False)
            )
            used_env_positions = used_env_positions or bool(
                getattr(policy, "used_env_own_positions", False)
            )

            episode_lengths.append(ep_steps)
            result = _classify_episode(env, any_terminated, any_truncated, ep_steps)
            end_reason_counts[result["episode_end_reason"]] = (
                end_reason_counts.get(result["episode_end_reason"], 0) + 1
            )
            winner_counts[result["winner"]] = (
                winner_counts.get(result["winner"], 0) + 1
            )
            red_alive_finals.append(result["red_alive_final"])
            blue_alive_finals.append(result["blue_alive_final"])
            mav_survivals.append(1 if result["mav_survived"] else 0)

    finally:
        env.close()

    # -- compute aggregate metrics --
    n = episodes
    red_win = winner_counts.get("red", 0) / n
    blue_win = winner_counts.get("blue", 0) / n
    draw = (
        winner_counts.get("draw", 0)
        + winner_counts.get("none", 0)
    ) / n
    timeout = (
        end_reason_counts.get("timeout", 0)
        + end_reason_counts.get("not_ended", 0)
    ) / n
    mav_surv = sum(mav_survivals) / n
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    red_alive_mean = float(np.mean(red_alive_finals)) if red_alive_finals else 0.0
    blue_alive_mean = float(np.mean(blue_alive_finals)) if blue_alive_finals else 0.0

    action_min = float(np.min(np.stack(actions_seen))) if actions_seen else 0.0
    action_max = float(np.max(np.stack(actions_seen))) if actions_seen else 0.0
    action_mean_abs = float(np.mean(np.abs(np.stack(actions_seen)))) if actions_seen else 0.0
    action_out_of_range = bool(action_min < -1.0001 or action_max > 1.0001)

    label = _opponent_difficulty_label(red_win, blue_win, mav_surv, nan_detected)

    record = {
        "config": config_path,
        "blue_opponent_policy": blue_opponent,
        "red_policy": red_policy,
        "episodes": n,
        "steps_limit": steps,
        "nan_detected": bool(nan_detected),
        "action_min": action_min,
        "action_max": action_max,
        "action_mean_abs": action_mean_abs,
        "action_out_of_range": action_out_of_range,
        "red_alive_final_mean": red_alive_mean,
        "blue_alive_final_mean": blue_alive_mean,
        "mav_survival_rate": mav_surv,
        "red_win_rate": red_win,
        "blue_win_rate": blue_win,
        "draw_rate": draw,
        "timeout_rate": timeout,
        "episode_end_reason_counts": end_reason_counts,
        "winner_counts": winner_counts,
        "avg_episode_length": avg_len,
        "opponent_difficulty_label": label,
    }

    if blue_opponent == "greedy_fsm":
        record["state_counts"] = state_counts
        record["assigned_target_counts"] = assigned_target_counts
        record["used_env_refresh_engaged_targets"] = bool(used_env_refresh)
        record["used_env_own_kinematics"] = bool(used_env_kinematics)
        record["used_env_own_positions"] = bool(used_env_positions)

    return record


# -- markdown -----------------------------------------------------------


def _markdown(data: dict) -> str:
    lines = [
        "# Blue Opponent Protocol Validation",
        "",
        "Purpose: compare `rule_nearest` and `greedy_fsm` as scripted blue",
        "opponents under fixed red policies.  This is not a training run and",
        "not a method module.",
        "",
        "## Summary",
        "",
        f"- records: {len(data['records'])}",
        f"- nan_records: {data['summary']['nan_records']}",
        f"- action_out_of_range: {data['summary']['any_action_out_of_range']}",
        "",
        "## Opponent Difficulty Labels",
        "",
        f"- too_strong_candidate: {data['summary'].get('too_strong_candidates', [])}",
        f"- too_weak_candidate: {data['summary'].get('too_weak_candidates', [])}",
        f"- stable_candidate: {data['summary'].get('stable_candidates', [])}",
        "",
        "## Records",
    ]

    for rec in data["records"]:
        cfg_name = Path(rec["config"]).name
        lines.extend([
            "",
            f"### {cfg_name} | {rec['blue_opponent_policy']} | red={rec['red_policy']}",
            "",
            f"- difficulty: **{rec['opponent_difficulty_label']}**",
            f"- episodes: {rec['episodes']}",
            f"- steps_limit: {rec['steps_limit']}",
            f"- nan_detected: {rec['nan_detected']}",
            f"- action_min: {rec['action_min']:.4f}",
            f"- action_max: {rec['action_max']:.4f}",
            f"- action_mean_abs: {rec['action_mean_abs']:.4f}",
            f"- red_alive_final_mean: {rec['red_alive_final_mean']:.2f}",
            f"- blue_alive_final_mean: {rec['blue_alive_final_mean']:.2f}",
            f"- mav_survival_rate: {rec['mav_survival_rate']:.2f}",
            f"- red_win_rate: {rec['red_win_rate']:.2f}",
            f"- blue_win_rate: {rec['blue_win_rate']:.2f}",
            f"- draw_rate: {rec['draw_rate']:.2f}",
            f"- timeout_rate: {rec['timeout_rate']:.2f}",
            f"- avg_episode_length: {rec['avg_episode_length']:.1f}",
            f"- episode_end_reason_counts: {rec['episode_end_reason_counts']}",
            f"- winner_counts: {rec['winner_counts']}",
        ])
        if "state_counts" in rec:
            lines.append(f"- greedy_fsm state_counts: {rec['state_counts']}")
        if "assigned_target_counts" in rec:
            lines.append(
                f"- greedy_fsm assigned_target_counts: {rec['assigned_target_counts']}"
            )

    lines.extend([
        "",
        "## Decision Use",
        "",
        "- `rule_nearest`: easy / default baseline opponent candidate",
        "- `greedy_fsm`: hard / diagnostic opponent candidate",
        "- Final default opponent decision requires user confirmation.",
        "- This is not a training run; no performance conclusions.",
    ])
    return "\n".join(lines) + "\n"


# -- main ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
    )
    parser.add_argument(
        "--blue-opponents",
        nargs="*",
        default=["rule_nearest", "greedy_fsm"],
    )
    parser.add_argument(
        "--red-policies",
        nargs="*",
        default=["zero", "bounded_random"],
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/blue_opponent_protocol_validation.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/blue_opponent_protocol_validation.md",
    )
    args = parser.parse_args()

    records: list[dict] = []
    for config in args.configs:
        for blue_opp in args.blue_opponents:
            for red_pol in args.red_policies:
                print(
                    f"[validate] {Path(config).name:50s} "
                    f"blue={blue_opp:14s} red={red_pol}",
                    flush=True,
                )
                rec = validate_config(
                    config_path=config,
                    blue_opponent=blue_opp,
                    red_policy=red_pol,
                    steps=args.steps,
                    episodes=args.episodes,
                    seed=args.seed,
                )

                # Hard-fail on NaN or action out-of-range
                if rec["nan_detected"]:
                    raise RuntimeError(
                        f"NaN detected: {config} blue={blue_opp} red={red_pol}"
                    )
                if rec["action_out_of_range"]:
                    raise RuntimeError(
                        f"Action out of [-1,1]: {config} blue={blue_opp} red={red_pol} "
                        f"min={rec['action_min']:.3f} max={rec['action_max']:.3f}"
                    )

                records.append(rec)

    # -- summary --
    nan_count = sum(1 for r in records if r["nan_detected"])
    too_strong = [
        f"{Path(r['config']).name}|{r['blue_opponent_policy']}|{r['red_policy']}"
        for r in records
        if r["opponent_difficulty_label"] == "too_strong_candidate"
    ]
    too_weak = [
        f"{Path(r['config']).name}|{r['blue_opponent_policy']}|{r['red_policy']}"
        for r in records
        if r["opponent_difficulty_label"] == "too_weak_candidate"
    ]
    stable = [
        f"{Path(r['config']).name}|{r['blue_opponent_policy']}|{r['red_policy']}"
        for r in records
        if r["opponent_difficulty_label"] == "stable_candidate"
    ]

    summary = {
        "records": len(records),
        "nan_records": nan_count,
        "any_action_out_of_range": any(r["action_out_of_range"] for r in records),
        "too_strong_candidates": too_strong,
        "too_weak_candidates": too_weak,
        "stable_candidates": stable,
    }
    data = {"records": records, "summary": summary}

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")

    print(f"\noutput_json: {out_json}", flush=True)
    print(f"output_md:  {out_md}", flush=True)
    print(f"nan_records: {nan_count}", flush=True)
    print(f"too_strong_candidates: {len(too_strong)} {too_strong}", flush=True)
    print(f"too_weak_candidates: {len(too_weak)} {too_weak}", flush=True)
    print(f"stable_candidates: {len(stable)} {stable}", flush=True)

    if nan_count:
        raise SystemExit("NaN detected — validation failed")


if __name__ == "__main__":
    main()
