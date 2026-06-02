"""Diagnose HeteroObsAdapter across all available hetero composition configs.

Does NOT train, does NOT run MAPPO, does NOT modify environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter


CONFIGS = [
    "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]


def main() -> None:
    adapter = HeteroObsAdapter()
    print(f"flat_actor_obs_dim = {adapter.flat_actor_obs_dim}")
    print(f"critic_state_dim   = {adapter.critic_state_dim}")
    print()

    # ---- artificial dead entity test (no JSBSim) ----
    print("=== Artificial dead entity test ===")
    fake_obs = {
        "ego_state": np.ones(11, dtype=np.float32),
        "ego_role": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "missile_warning": np.array([0.0], dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
        "ally_states": np.array([
            [1.0, 0, 0, 0, 0, 0, 300, 0, 1, 0, 1],   # alive ally
            [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # dead ally
        ], dtype=np.float32),
        "ally_roles": np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32),
        "enemy_states": np.array([
            [1.0, 0, 0, 0, 0, 0, 300, 0, 1, 0, 1],   # alive enemy
            [0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # dead enemy
        ], dtype=np.float32),
        "enemy_roles": np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32),
    }

    out = adapter.adapt_agent(
        "red_0", fake_obs,
        red_ids=["red_0", "red_1"],
        blue_ids=["blue_0", "blue_1"])
    av = out["ally_valid_mask"]
    aa = out["ally_alive_mask"]
    ev = out["enemy_valid_mask"]
    ea = out["enemy_alive_mask"]
    # 2v2: 1 real ally → valid=[1,0,0,0], alive=[1,0,0,0]
    #      2 real enemies, 1 dead → valid=[1,1,0,0], alive=[1,0,0,0]
    print(f"  ally_valid:  {av.tolist()}  (expect [1, 0, 0, 0])")
    print(f"  ally_alive:  {aa.tolist()}  (expect [1, 0, 0, 0])")
    print(f"  enemy_valid: {ev.tolist()} (expect [1, 1, 0, 0])")
    print(f"  enemy_alive: {ea.tolist()} (expect [1, 0, 0, 0])")
    assert np.allclose(av, [1, 0, 0, 0]), f"ally_valid: {av}"
    assert np.allclose(aa, [1, 0, 0, 0]), f"ally_alive: {aa}"
    assert np.allclose(ev, [1, 1, 0, 0]), f"enemy_valid: {ev}"
    assert np.allclose(ea, [1, 0, 0, 0]), f"enemy_alive: {ea}"
    print("  artificial dead entity test PASSED")
    print()

    for cfg_path in CONFIGS:
        print(f"=== {cfg_path} ===")
        env = None
        try:
            env = make_env(cfg_path, env_type="jsbsim_hetero", max_steps=10)
            obs, info = env.reset(seed=0)
            red_ids = getattr(env, "red_ids", None)
            blue_ids = getattr(env, "blue_ids", None)

            result = adapter.adapt_all(obs, info=info,
                                       red_ids=red_ids, blue_ids=blue_ids)

            print(f"  red_ids: {red_ids}")
            print(f"  blue_ids: {blue_ids}")
            print(f"  actor_obs keys: {sorted(result['actor_obs'].keys())}")
            for rid, arr in result["actor_obs"].items():
                print(f"    {rid}: shape={arr.shape} nan={np.isnan(arr).any()}")
            print(f"  critic_state: shape={result['critic_state'].shape} "
                  f"nan={np.isnan(result['critic_state']).any()}")
            print(f"  red_valid_mask: {result['red_valid_mask'].tolist()}")

            # red_0 masks
            r0 = result["structured_actor_obs"].get("red_0")
            if r0:
                print(f"  red_0 ally_valid:   {r0['ally_valid_mask'].tolist()}")
                print(f"  red_0 ally_alive:   {r0['ally_alive_mask'].tolist()}")
                print(f"  red_0 enemy_valid:  {r0['enemy_valid_mask'].tolist()}")
                print(f"  red_0 enemy_alive:  {r0['enemy_alive_mask'].tolist()}")
                print(f"  red_0 ego_feature:  shape={r0['ego_feature'].shape}")
                print(f"  red_0 ally ents:    shape={r0['ally_entities'].shape}")
                print(f"  red_0 enemy ents:   shape={r0['enemy_entities'].shape}")

            # zero action step 3 times
            for step in range(3):
                actions = {aid: np.zeros(3, dtype=np.float32)
                           for aid in env.agent_ids}
                obs, _rew, terminated, truncated, info = env.step(actions)
                result2 = adapter.adapt_all(obs, info=info,
                                            red_ids=red_ids, blue_ids=blue_ids)
                for rid, arr in result2["actor_obs"].items():
                    if np.isnan(arr).any():
                        print(f"  [WARN] {rid} flat_actor_obs NaN at step {step}")
                if np.isnan(result2["critic_state"]).any():
                    print(f"  [WARN] critic_state NaN at step {step}")

            print(f"  step smoke passed (3 zero-action steps)")

        finally:
            if env is not None:
                env.close()
        print()


if __name__ == "__main__":
    main()
