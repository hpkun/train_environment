"""Diagnose paper-aligned 3v2 and 5v4 configs: shapes, types, stability."""
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
    "uav_env/JSBSim/configs/hetero_paper_3v2_mav_2uav_vs_2uav.yaml",
    "uav_env/JSBSim/configs/hetero_paper_5v4_mav_4uav_vs_4uav.yaml",
]


def main():
    adapter = HeteroObsAdapter()
    rng = np.random.default_rng(0)

    for cfg_path in CONFIGS:
        print(f"=== {Path(cfg_path).stem} ===")
        env = None
        try:
            env = make_env(cfg_path, env_type="jsbsim_hetero")
            obs, info = env.reset(seed=0)

            print(f"  red_ids: {env.red_ids}")
            print(f"  blue_ids: {env.blue_ids}")
            print(f"  agent_types: {info['agent_types']}")
            print(f"  agent_roles: {info['agent_roles']}")
            print(f"  agent_models: {info['agent_models']}")

            # Missiles and initial state
            for aid in env.red_ids + env.blue_ids:
                sim = env._get_sim(aid)
                miss = (env._num_missiles_for(aid)
                        if hasattr(env, '_num_missiles_for') else 0)
                alt = float(sim.get_geodetic()[2]) if sim else 0.0
                spd = float(np.linalg.norm(sim.get_velocity())) if sim else 0.0
                yaw = float(np.rad2deg(sim.get_rpy()[2])) if sim else 0.0
                print(f"  {aid:7s}: model={info['agent_models'].get(aid,'?')} "
                      f"type={info['agent_types'].get(aid,'?')} "
                      f"missiles={miss} alt={alt:.0f}m spd={spd:.0f}m/s "
                      f"yaw={yaw:.1f}deg")

            # Adapter
            result = adapter.adapt_all(
                obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            r0dim = result["actor_obs"].get(env.red_ids[0],
                                            np.zeros(140)).shape[0]
            csdim = result["critic_state"].shape[0]
            print(f"  adapter actor_dim={r0dim} critic_dim={csdim}")
            print(f"  red_valid_mask={result['red_valid_mask'].tolist()}")

            # Zero action 3 steps
            for _ in range(3):
                actions = {aid: np.zeros(3, dtype=np.float32)
                           for aid in env.agent_ids}
                obs, _r, _t, _tr, info = env.step(actions)
                r2 = adapter.adapt_all(obs, info=info,
                                       red_ids=env.red_ids,
                                       blue_ids=env.blue_ids)
                assert not np.isnan(r2["critic_state"]).any()
            print("  zero_action 3 steps: OK")

            # Bounded random 3 steps
            for _ in range(3):
                actions = {aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
                           for aid in env.agent_ids}
                obs, _r, _t, _tr, info = env.step(actions)
                r3 = adapter.adapt_all(obs, info=info,
                                       red_ids=env.red_ids,
                                       blue_ids=env.blue_ids)
                assert not np.isnan(r3["critic_state"]).any()
            print("  bounded_random 3 steps: OK")

        finally:
            if env is not None:
                env.close()
        print()


if __name__ == "__main__":
    main()
