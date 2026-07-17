"""In-memory A/B fire-control closure audit; no training or output files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.env import UavCombatEnv
from rule_based_agent import _paper_absolute_action


def pursuit_actions(env, own_ids, enemies):
    actions = {}
    taken = set()
    for aid in own_ids:
        own = env._get_sim(aid)
        if not own.is_alive:
            actions[aid] = np.zeros(3, dtype=np.float32)
            continue
        rows = sorted(
            (float(np.linalg.norm(enemy.get_position() - own.get_position())),
             int(enemy_id.split("_")[1]), enemy_id, enemy)
            for enemy_id, enemy in enemies.items() if enemy.is_alive)
        available = [row for row in rows if row[2] not in taken]
        selected = available or rows
        if not selected:
            actions[aid] = _paper_absolute_action(
                0.0, float(own.get_rpy()[2]), 300.0)
            continue
        _distance, _index, enemy_id, enemy = selected[0]
        taken.add(enemy_id)
        relative = enemy.get_position() - own.get_position()
        pitch = float(np.arctan2(
            relative[2], max(np.hypot(relative[0], relative[1]), 1e-9)))
        heading = float(np.arctan2(relative[1], relative[0]))
        actions[aid] = _paper_absolute_action(pitch, heading, 300.0)
    return actions


def straight_actions(env, ids, headings):
    return {
        aid: (_paper_absolute_action(0.0, headings[aid], 300.0)
              if env._get_sim(aid).is_alive
              else np.zeros(3, dtype=np.float32))
        for aid in ids
    }


def run_case(seed, case):
    env = UavCombatEnv()
    try:
        obs, _ = env.reset(seed=seed)
        headings = {
            aid: float(env._get_sim(aid).get_rpy()[2]) for aid in env.agent_ids}
        totals = {team: {} for team in ("red", "blue")}
        info = {}
        for step in range(1400):
            if case == "red_chase":
                red = pursuit_actions(env, env.red_ids, env.blue_planes)
                blue = straight_actions(env, env.blue_ids, headings)
            else:
                red = straight_actions(env, env.red_ids, headings)
                blue = env.blue_policy_actions({aid: obs[aid] for aid in env.blue_ids})
            obs, _reward, terminated, truncated, info = env.step({**red, **blue})
            for team in ("red", "blue"):
                for key, value in info["__launch_diag__"][team].items():
                    totals[team][key] = totals[team].get(key, 0) + int(value)
            if all(terminated[aid] or truncated[aid] for aid in env.agent_ids):
                break
        shooter = "red" if case == "red_chase" else "blue"
        terms = info.get("__missile_term__", {}).get(shooter, {})
        return {
            "seed": seed, "case": case, "steps": step + 1,
            "geometry": totals[shooter].get("geometry_ok_pairs", 0),
            "lock_mature": totals[shooter].get("lock_mature_pairs", 0),
            "launches": totals[shooter].get("launches", 0),
            "hits": int(terms.get("hit", 0)),
            "missile_numerical_invalid": int(terms.get("numerical_invalid", 0)),
            "invalid_episode": bool(
                info.get("__episode__", {}).get("invalid_numerical_episode", False)),
        }
    finally:
        env.close()


def run_audit(seeds):
    rows = [run_case(seed, case) for seed in seeds
            for case in ("red_chase", "blue_chase")]
    red = [row for row in rows if row["case"] == "red_chase"]
    blue = [row for row in rows if row["case"] == "blue_chase"]
    direction_pass = lambda group: (
        all(row["geometry"] > 0 and row["lock_mature"] > 0
            and row["launches"] > 0 for row in group)
        and sum(row["hits"] for row in group) > 0)
    healthy = all(not row["invalid_episode"]
                  and row["missile_numerical_invalid"] == 0 for row in rows)
    return {
        "FireControlClosureRed": "PASS" if direction_pass(red) else "FAIL",
        "FireControlClosureBlue": "PASS" if direction_pass(blue) else "FAIL",
        "numerical_health": "PASS" if healthy else "FAIL",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
    args = parser.parse_args()
    print(json.dumps(run_audit(args.seeds), indent=2, sort_keys=True))
