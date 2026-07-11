"""Short, non-training interface/stability audit for blue rule opponents."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.opponent_policy import OpponentPolicy
from uav_env.make_env import make_env


def run_case(config: str, mode: str, seed: int, max_steps: int) -> tuple[list[dict], dict]:
    env = make_env(config, max_steps=max_steps)
    opponent = OpponentPolicy(mode=mode, seed=seed + 17)
    obs, info = env.reset(seed=seed)
    opponent.reset_memory()
    rows = []
    action_vectors = []
    previous_alive = {bid: True for bid in env.blue_ids}
    blue_oob = blue_low_altitude = launch_count = 0
    for step in range(max_steps):
        blue_actions = opponent.act(obs, env.blue_ids, env=env)
        actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
        actions.update(blue_actions)
        finite = all(np.isfinite(a).all() for a in blue_actions.values())
        in_range = all((np.asarray(a) >= -1.0).all() and (np.asarray(a) <= 1.0).all() for a in blue_actions.values())
        action_vectors.extend(np.asarray(a).tolist() for a in blue_actions.values())
        obs, rewards, terminated, truncated, info = env.step(actions)
        for bid in env.blue_ids:
            agent_info = info.get(bid, {}) if isinstance(info, dict) else {}
            launch_count += int(agent_info.get("missiles_fired_this_step", 0) or 0)
            alive_now = bool(agent_info.get("alive", env.blue_planes[bid].is_alive))
            if previous_alive[bid] and not alive_now:
                reason = str(agent_info.get("death_reason", "")).lower()
                blue_oob += int("out_of" in reason or "boundary" in reason)
                blue_low_altitude += int("altitude" in reason or "ground" in reason)
            previous_alive[bid] = alive_now
        rows.append({
            "opponent_policy": mode,
            "step": step,
            "action_finite": int(finite),
            "action_in_range": int(in_range),
            "blue_alive": sum(int(env.blue_planes[bid].is_alive) for bid in env.blue_ids),
            "selected_targets": json.dumps(opponent.last_assigned_targets, sort_keys=True),
            "selected_maneuvers": json.dumps(opponent.last_states, sort_keys=True),
        })
        terminated_all = all(terminated.values()) if isinstance(terminated, dict) else bool(terminated)
        truncated_all = all(truncated.values()) if isinstance(truncated, dict) else bool(truncated)
        if terminated_all or truncated_all:
            break
    tam = opponent.tam_greedy_rule
    summary = {
        "opponent_policy": mode,
        "episode_steps": len(rows),
        "all_actions_finite": all(row["action_finite"] for row in rows),
        "all_actions_in_range": all(row["action_in_range"] for row in rows),
        "blue_alive_final": rows[-1]["blue_alive"] if rows else 0,
        "action_std": float(np.std(np.asarray(action_vectors))) if action_vectors else 0.0,
        "target_switch_count": tam.target_switch_count if mode == "tam_greedy_rule" else "",
        "candidate_selected_counts": dict(tam.selected_counts) if mode == "tam_greedy_rule" else {},
        "missile_warning_break_count": tam.warning_break_count if mode == "tam_greedy_rule" else "",
        "boundary_safety_count": tam.boundary_safety_count if mode == "tam_greedy_rule" else "",
        "hard_deck_recovery_count": tam.hard_deck_recovery_count if mode == "tam_greedy_rule" else "",
        "blue_horizontal_oob_count": blue_oob,
        "blue_low_altitude_death_count": blue_low_altitude,
        "launch_count": launch_count,
    }
    env.close()
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opponent-policies", nargs="+", default=["tam_greedy_rule", "brma_rule"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/tam_greedy_rule_audit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for mode in args.opponent_policies:
        rows, summary = run_case(args.config, mode, args.seed, args.max_steps)
        all_rows.extend(rows)
        summaries.append(summary)
    with (output / "steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]) if all_rows else ["opponent_policy", "step"])
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["opponent_policy"])
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
