"""Audit coordinate symmetry and rule-v-rule balance in formal combat."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from hetero_3v2_v2_audit_common import (
    AUDIT_DIR, ROOT, V1_CONFIG, perturbation, write_csv, write_json,
)
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from uav_env.JSBSim.formal_v1.scenario import TABLE6_INITIAL_STATES, jsbsim_initial_state
from uav_env.make_env import make_env


def _swap_state(agent_id: str) -> dict:
    mapping = {
        "red_0": "red_0", "red_1": "blue_0", "red_2": "blue_1",
        "blue_0": "red_1", "blue_1": "red_2",
    }
    source = mapping[agent_id]
    state = jsbsim_initial_state(source)
    state["ic/psi-true-deg"] = (
        float(state["ic/psi-true-deg"]) + 180.0) % 360.0
    return state


def run(seed: int, scenario: str, max_steps: int) -> dict:
    env = make_env(str(ROOT / V1_CONFIG))
    env.reset(seed=seed, options={"audit_initial_perturbation": perturbation(seed)})
    if scenario == "symmetric_2v2":
        env.aircraft["red_0"].shotdown()
    elif scenario == "swapped_3v2":
        for aid, aircraft in env.aircraft.items():
            aircraft.reload(new_state=_swap_state(aid), new_origin=env.origin)
        for aid, aircraft in env.aircraft.items():
            aircraft.partners = [x for oid, x in env.aircraft.items()
                                 if oid != aid and oid.split("_")[0] == aid.split("_")[0]]
            aircraft.enemies = [x for oid, x in env.aircraft.items()
                                if oid.split("_")[0] != aid.split("_")[0]]
    policy = PaperGreedyOpponent()
    for step in range(max_steps):
        actions = policy.actions(env, "red")
        _, _, _, _, info = env.step(actions)
        if info["team_done"]:
            break
    events = env.event_log
    row = {
        "scenario": scenario, "seed": seed, "steps": step + 1,
        "outcome": info["outcome"], "red_alive_final": info["red_alive"],
        "blue_alive_final": info["blue_alive"], "mav_alive_final": int(info["mav_alive"]),
        "red_launches": sum(e["event"] == "launch" and str(e["shooter_id"]).startswith("red")
                            for e in events),
        "blue_launches": sum(e["event"] == "launch" and str(e["shooter_id"]).startswith("blue")
                             for e in events),
        "red_hits": sum(e["event"] == "hit" and str(e["shooter_id"]).startswith("red")
                        for e in events),
        "blue_hits": sum(e["event"] == "hit" and str(e["shooter_id"]).startswith("blue")
                         for e in events),
        "finite": int(np.isfinite(info["critic_state"]).all()),
    }
    env.close()
    return row


def _run_task(task):
    seed, scenario, max_steps = task
    return run(seed, scenario, max_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    scenarios = ("symmetric_2v2", "formal_3v2", "swapped_3v2")
    tasks = [(seed, scenario, args.max_steps)
             for scenario in scenarios for seed in range(args.seeds)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_task, tasks))
    else:
        rows = [_run_task(task) for task in tasks]
    output = AUDIT_DIR.__class__(args.output_dir)
    write_csv(output / "balance_episodes.csv", rows)
    summary = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        summary[scenario] = {
            "episodes": len(subset),
            "outcomes": dict(Counter(row["outcome"] for row in subset)),
            "red_launches_mean": float(np.mean([row["red_launches"] for row in subset])),
            "blue_launches_mean": float(np.mean([row["blue_launches"] for row in subset])),
            "red_hits_mean": float(np.mean([row["red_hits"] for row in subset])),
            "blue_hits_mean": float(np.mean([row["blue_hits"] for row in subset])),
            "red_alive_final_mean": float(np.mean(
                [row["red_alive_final"] for row in subset])),
            "blue_alive_final_mean": float(np.mean(
                [row["blue_alive_final"] for row in subset])),
            "mav_survival_rate": float(np.mean(
                [row["mav_alive_final"] for row in subset])),
            "all_finite": all(row["finite"] for row in subset),
        }
    write_json(output / "balance_summary.json", summary)
    print(output / "balance_summary.json")


if __name__ == "__main__":
    main()
