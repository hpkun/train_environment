"""Diagnose whether the A-4 init_altitude_offset_m=2000 is actually applied.

Does not run MAPPO, does not run win-rate experiments, does not add MAV GCAS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


def _actions(env, policy: str, rng: np.random.Generator) -> dict:
    if policy == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    if policy == "bounded_random":
        return {
            aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
            for aid in env.agent_ids
        }
    if policy == "random":
        return {
            aid: env.action_space.spaces[aid].sample().astype(np.float32)
            for aid in env.agent_ids
        }
    raise ValueError(f"unknown policy: {policy}")


def _scan(env) -> dict:
    sims = list(env.blue_planes.values()) + list(env.red_planes.values())
    alts = {s.uid: float(s.get_geodetic()[2]) for s in sims}
    alive = [s.uid for s in sims if s.is_alive]
    crashed = [s.uid for s in sims if s.is_crash]
    nan_detected = False
    for s in sims:
        v = np.concatenate([
            s.get_position().astype(np.float64),
            s.get_velocity().astype(np.float64),
            np.asarray(s.get_rpy(), dtype=np.float64),
        ])
        nan_detected = nan_detected or bool(np.isnan(v).any())
    return {"altitudes": alts, "alive": alive, "crashed": crashed, "nan": nan_detected}


def main() -> None:
    rng = np.random.default_rng(0)

    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=200,
        red_agent_types=["mav", "attack_uav"],
        blue_agent_types=["attack_uav", "attack_uav"],
        enable_gcas_for_blue=True,
        suppress_jsbsim_output=True,
        aircraft_type_params={
            "mav": {"init_altitude_offset_m": 2000.0, "init_speed_offset_mps": 0.0},
            "attack_uav": {"init_altitude_offset_m": 0.0, "init_speed_offset_mps": 0.0},
        },
    )
    try:
        obs, info = env.reset(seed=0)

        # ---- initial altitudes ----
        print("=== initial altitudes ===")
        scan0 = _scan(env)
        for aid in env.agent_ids:
            alt_m = scan0["altitudes"].get(aid, 0.0)
            offset = info.get("agent_init_offsets", {}).get(aid, {})
            print(f"  {aid}: alt={alt_m:.1f}m  offset={offset}")

        # Verify red_0 is ~2000m higher than red_1
        red0_alt = scan0["altitudes"].get("red_0", 0.0)
        red1_alt = scan0["altitudes"].get("red_1", 0.0)
        delta = red0_alt - red1_alt
        print(f"  red_0 - red_1 altitude delta: {delta:.1f}m")
        assert delta > 500.0, f"red_0 should be notably higher than red_1, got {delta:.1f}m"

        # Verify red_0 model is A-4
        models = info.get("agent_models", {})
        assert models.get("red_0", "") == "A-4", f"red_0 model: {models.get('red_0')}"

        # Verify offsets in info
        offsets = info.get("agent_init_offsets", {})
        assert abs(offsets.get("red_0", {}).get("altitude_offset_m", 0.0) - 2000.0) < 1.0
        assert offsets.get("red_1", {}).get("altitude_offset_m", 999.0) == 0.0
        assert offsets.get("blue_0", {}).get("altitude_offset_m", 999.0) == 0.0

        # ---- zero policy ----
        print("=== zero policy 200 steps ===")
        min_alt, crashed = _run_policy(env, "zero", rng, 200)
        print(f"  red_0 min_alt={min_alt.get('red_0',0):.1f}m crashed={crashed.get('red_0',False)}")

        # ---- bounded_random policy ----
        print("=== bounded_random policy 200 steps ===")
        min_alt2, crashed2 = _run_policy(env, "bounded_random", rng, 200)
        print(f"  red_0 min_alt={min_alt2.get('red_0',0):.1f}m crashed={crashed2.get('red_0',False)}")

        # ---- random policy ----
        print("=== random policy 200 steps ===")
        min_alt3, crashed3 = _run_policy(env, "random", rng, 200)
        print(f"  red_0 min_alt={min_alt3.get('red_0',0):.1f}m crashed={crashed3.get('red_0',False)}")

        print("diagnose_hetero_a4_init_applied: DONE")
    finally:
        env.close()


def _run_policy(env, policy, rng, steps):
    obs, info = env.reset(seed=0)
    min_alts = {}
    crashed = {}
    for _ in range(steps):
        actions = _actions(env, policy, rng)
        obs, _rew, terminated, truncated, info = env.step(actions)
        scan = _scan(env)
        for aid, alt in scan["altitudes"].items():
            min_alts[aid] = min(min_alts.get(aid, float("inf")), alt)
        for aid in scan["crashed"]:
            crashed[aid] = True
        if all(terminated.values()) or all(truncated.values()):
            break
    return min_alts, crashed


if __name__ == "__main__":
    main()
