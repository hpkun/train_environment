"""Read-only numerical and short-rollout audit for scale-aligned reward v1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv  # noqa: E402


DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v1.yaml"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _terminal(initial_blue, alive_blue, initial_attack, alive_attack, mav_alive):
    blue_loss = (initial_blue - alive_blue) / initial_blue
    red_loss = (initial_attack * (not mav_alive) + initial_attack - alive_attack) / (2 * initial_attack)
    return float(np.clip(30 * (blue_loss - red_loss), -30, 30)), blue_loss, red_loss


def _state_cases() -> list[dict]:
    phi = HeteroUavCombatEnv._scale_v1_distance_potential
    rows = []
    for name, previous, current in (
        ("initial_3v2", None, 22300), ("far_22_3km_stationary", 22300, 22300),
        ("approach_22_3_to_15", 22300, 15000), ("approach_15_to_10", 15000, 10000),
        ("approach_10_to_5", 10000, 5000), ("retreat_5_to_15", 5000, 15000),
        ("target_unchanged", 12000, 12000), ("target_switch", None, 15000),
        ("target_dead_switch", None, 18000),
    ):
        delta = 0.0 if previous is None else phi(current) - phi(previous)
        rows.append({
            "case": name, "previous_distance_m": previous, "current_distance_m": current,
            "phi_distance": phi(current), "distance_progress_raw": 5 * delta,
            "distance_progress_clipped": float(np.clip(5 * delta, -.5, .5)),
            "progress_reset": int(previous is None),
        })
    event_cases = (
        ("single_uav_kill", 10), ("single_uav_death", -10), ("mav_death", -20),
    )
    rows.extend({"case": name, "event": value} for name, value in event_cases)
    terminal_cases = (
        ("3v2_no_loss_win", 2, 0, 2, 2, True),
        ("3v2_one_uav_loss_win", 2, 0, 2, 1, True),
        ("3v2_mav_loss_win", 2, 0, 2, 2, False),
        ("3v2_no_loss_timeout", 2, 2, 2, 2, True),
        ("3v2_mav_loss_no_kill", 2, 2, 2, 2, False),
        ("3v2_red_eliminated", 2, 2, 2, 0, False),
        ("5v4_no_loss_win", 4, 0, 4, 4, True),
        ("5v4_no_loss_timeout", 4, 4, 4, 4, True),
        ("5v4_red_eliminated", 4, 4, 4, 0, False),
    )
    for name, ib, ab, ia, aa, ma in terminal_cases:
        value, lb, lr = _terminal(ib, ab, ia, aa, ma)
        rows.append({"case": name, "terminal": value, "blue_loss_fraction": lb, "red_loss_fraction": lr})
    return rows


def _actions(env, mode: str, rng) -> dict:
    if mode == "random":
        return {aid: rng.uniform(-0.25, 0.25, 3).astype(np.float32) for aid in env.agent_ids}
    if mode == "fixed_straight":
        return {aid: np.asarray([0.0, 0.0, 0.0], np.float32) for aid in env.agent_ids}
    return {aid: np.zeros(3, np.float32) for aid in env.agent_ids}


def _short_rollout(config: str, mode: str, steps: int, seed: int) -> dict:
    env = make_env(config, max_steps=steps)
    rng = np.random.default_rng(seed)
    team_rewards = []; progress = []; mav_role = []; flight = []; event = []; terminal = []
    identity_max = 0.0; non_event_outside = 0; clip_count = 0; nan_count = 0
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            _, rewards, terminated, truncated, info = env.step(_actions(env, mode, rng))
            comps = info.get("reward_components", {})
            active = [rid for rid in env.red_ids if env._scale_v1_alive_before_step.get(rid, False)]
            team_rewards.append(float(np.mean([rewards[rid] for rid in active])) if active else 0.0)
            for rid in active:
                comp = comps.get(rid, {})
                identity_max = max(identity_max, abs(float(comp.get("scale_v1_identity_error", 0))))
                progress.append(float(comp.get("scale_v1_progress_clipped", 0)))
                mav_role.append(float(comp.get("scale_v1_mav_role", 0)))
                flight.append(float(comp.get("scale_v1_flight_total", 0)))
                ev = float(comp.get("scale_v1_uav_event_total", 0)) + float(comp.get("scale_v1_mav_event_total", 0))
                term = float(comp.get("scale_v1_terminal", 0))
                event.append(ev); terminal.append(term)
                dense = float(comp.get("total", 0)) - ev - term
                non_event_outside += int(abs(dense) > 1.0)
                clip_count += int(abs(float(comp.get("scale_v1_progress_raw", 0)) - float(comp.get("scale_v1_progress_clipped", 0))) > 1e-12)
                nan_count += int(not all(math.isfinite(float(v)) for v in (rewards[rid], dense, ev, term)))
            if all(terminated.values()) or all(truncated.values()):
                break
    finally:
        env.close()
    def ms(values):
        return (float(np.mean(values)) if values else 0.0, float(np.std(values)) if values else 0.0)
    team_mean, team_std = ms(team_rewards); pmean, pstd = ms(progress); mmean, mstd = ms(mav_role); fmean, fstd = ms(flight)
    return {
        "action_mode": mode, "decisions": len(team_rewards), "team_reward_mean": team_mean,
        "team_reward_std": team_std, "team_reward_min": min(team_rewards, default=0),
        "team_reward_max": max(team_rewards, default=0), "uav_progress_mean": pmean,
        "uav_progress_std": pstd, "mav_role_mean": mmean, "mav_role_std": mstd,
        "flight_mean": fmean, "flight_std": fstd, "event_sum": sum(event),
        "terminal_sum": sum(terminal), "return": sum(team_rewards),
        "return_per_decision": sum(team_rewards) / max(len(team_rewards), 1),
        "reward_identity_max_error": identity_max, "nan_inf_count": nan_count,
        "non_event_abs_gt_1_ratio": non_event_outside / max(len(progress) + len(mav_role), 1),
        "progress_clip_ratio": clip_count / max(len(progress), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/brma_tam_scale_aligned_v1_audit")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    states = _state_cases()
    rollouts = [_short_rollout(args.config, mode, args.steps, args.seed + i) for i, mode in enumerate(("zero", "random", "fixed_straight"))]
    approach = 5 * (HeteroUavCombatEnv._scale_v1_distance_potential(5000) - HeteroUavCombatEnv._scale_v1_distance_potential(22300))
    summary = {
        "reward_contract_revision": 3, "state_case_count": len(states),
        "rollouts": rollouts, "cumulative_distance_progress_22_3km_to_5km": approach,
        "cumulative_distance_progress_5km_to_22_3km": -approach,
        "static_far_1000_step_distance_progress": 0.0,
        "identity_max_error": max(row["reward_identity_max_error"] for row in rollouts),
        "nan_inf_count": sum(row["nan_inf_count"] for row in rollouts),
        "note": "Zero/random/fixed-policy reward audit; not learned-policy performance.",
    }
    _write_csv(out / "scale_aligned_state_cases.csv", states)
    _write_csv(out / "scale_aligned_short_rollouts.csv", rollouts)
    (out / "scale_aligned_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# Scale-aligned reward audit", "", f"- Identity max error: `{summary['identity_max_error']}`", f"- NaN/Inf count: `{summary['nan_inf_count']}`", f"- 22.3 km -> 5 km distance progress: `{approach:.6f}`", f"- Reverse distance progress: `{-approach:.6f}`", "", "## Short rollouts"]
    report.extend(f"- {r['action_mode']}: return={r['return']:.6f}, mean={r['team_reward_mean']:.6f}, progress_clip_ratio={r['progress_clip_ratio']:.6f}" for r in rollouts)
    (out / "scale_aligned_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
