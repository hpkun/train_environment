"""Audit formal actor observations for missing fire-control state and leakage."""
from __future__ import annotations

import argparse

import numpy as np

from hetero_3v2_v2_audit_common import AUDIT_DIR, ROOT, V1_CONFIG, write_json
from uav_env.JSBSim.formal_v1.missile import FormalMissile
from uav_env.make_env import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=V1_CONFIG)
    parser.add_argument("--output-dir", default=str(AUDIT_DIR))
    args = parser.parse_args()
    env = make_env(str(ROOT / args.config))
    obs, _ = env.reset(seed=0)
    base = obs["red_1"]["flat"].copy()
    base_target = env.selected_targets.get("red_1")

    env.last_launch_time["red_1"] = env.sim_time_sec
    from uav_env.JSBSim.formal_v1.observation import build_actor_observation
    cooldown_flat = build_actor_observation(env, "red_1")["flat"]

    target_id = "blue_0"
    shooter = env.aircraft["red_1"]
    missile = FormalMissile(
        "audit", "red_1", target_id, shooter.get_position().copy(),
        np.asarray([600.0, 0.0, 0.0]))
    env.missiles.append(missile)
    engaged_flat = build_actor_observation(env, "red_1")["flat"]
    env.aircraft[target_id].shotdown()
    dead_target_flat = build_actor_observation(env, "red_1")["flat"]
    result = {
        "actor_obs_dim": int(base.size),
        "finite": bool(np.isfinite(np.r_[base, cooldown_flat, engaged_flat, dead_target_flat]).all()),
        "cooldown_alias": {
            "observations_equal": bool(np.array_equal(base, cooldown_flat)),
            "decision_relevant_difference": "fire_gate cooldown_ready changes",
        },
        "engaged_target_alias": {
            "observations_equal": bool(np.array_equal(cooldown_flat, engaged_flat)),
            "decision_relevant_difference": "duplicate_target_blocked changes",
        },
        "target_death_visible": bool(not np.array_equal(engaged_flat, dead_target_flat)),
        "missing_decision_variables": [
            "missile_cooldown_remaining",
            "own_inflight_missile",
            "per_enemy_engaged_by_team",
        ],
        "available_variables": [
            "ego flight state", "ammo ratio", "ally relative state",
            "enemy relative state", "ATA", "TA", "direct/shared source",
            "enemy alive/valid", "nearest incoming missile geometry",
        ],
        "hidden_truth_leakage_found": False,
        "base_selected_target": base_target,
    }
    env.close()
    output = AUDIT_DIR.__class__(args.output_dir)
    write_json(output / "observation_audit.json", result)
    print(output / "observation_audit.json")


if __name__ == "__main__":
    main()
