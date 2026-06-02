"""Diagnose hetero-only type/role observation metadata.

Does not run MAPPO, does not alter reward/missile/PID/termination/action logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import (  # noqa: E402
    HeteroUavCombatEnv,
    TYPE_VOCAB,
)


def _print_agent_metadata(obs: dict, agent_id: str) -> None:
    agent_obs = obs[agent_id]
    print(f"{agent_id}:")
    print(f"  ego_type={agent_obs['ego_type'].tolist()}")
    print(f"  ego_role={agent_obs['ego_role'].tolist()}")
    print(f"  ally_types={agent_obs['ally_types'].tolist()}")
    print(f"  enemy_types={agent_obs['enemy_types'].tolist()}")
    print(f"  ally_roles={agent_obs['ally_roles'].tolist()}")
    print(f"  enemy_roles={agent_obs['enemy_roles'].tolist()}")


def _assert_metadata(obs: dict, env: HeteroUavCombatEnv) -> None:
    mav = np.eye(len(TYPE_VOCAB), dtype=np.float32)[TYPE_VOCAB.index("mav")]
    attack = np.eye(len(TYPE_VOCAB), dtype=np.float32)[TYPE_VOCAB.index("attack_uav")]

    np.testing.assert_array_equal(obs["red_0"]["ego_type"], mav)
    np.testing.assert_array_equal(obs["red_1"]["ego_type"], attack)
    np.testing.assert_array_equal(obs["blue_0"]["ego_type"], attack)
    np.testing.assert_array_equal(obs["blue_1"]["ego_type"], attack)
    np.testing.assert_array_equal(obs["red_0"]["ally_types"][0], attack)
    np.testing.assert_array_equal(obs["red_0"]["enemy_types"][0], attack)
    np.testing.assert_array_equal(obs["red_0"]["enemy_types"][1], attack)

    for aid in env.agent_ids:
        agent_obs = obs[aid]
        assert agent_obs["ego_type"].shape == (4,)
        assert agent_obs["ego_role"].shape == (4,)
        if aid.startswith("red"):
            assert agent_obs["ally_types"].shape == (env.max_num_red - 1, 4)
            assert agent_obs["ally_roles"].shape == (env.max_num_red - 1, 4)
            assert agent_obs["enemy_types"].shape == (env.max_num_blue, 4)
            assert agent_obs["enemy_roles"].shape == (env.max_num_blue, 4)
        else:
            assert agent_obs["ally_types"].shape == (env.max_num_blue - 1, 4)
            assert agent_obs["ally_roles"].shape == (env.max_num_blue - 1, 4)
            assert agent_obs["enemy_types"].shape == (env.max_num_red, 4)
            assert agent_obs["enemy_roles"].shape == (env.max_num_red, 4)
        for key in [
            "ego_type",
            "ego_role",
            "ally_types",
            "ally_roles",
            "enemy_types",
            "enemy_roles",
        ]:
            assert not np.isnan(agent_obs[key]).any(), f"NaN in {aid}/{key}"


def main() -> None:
    env = HeteroUavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=12,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        print("reset_success: true")
        print(f"agent_types: {info['agent_types']}")
        print(f"agent_roles: {info['agent_roles']}")
        print(f"agent_models: {info['agent_models']}")
        for aid in env.agent_ids:
            _print_agent_metadata(obs, aid)
        _assert_metadata(obs, env)

        initial_shapes = {
            aid: {key: obs[aid][key].shape for key in obs[aid] if key.endswith(("type", "role", "types", "roles"))}
            for aid in env.agent_ids
        }
        for step in range(3):
            actions = {
                aid: env.action_space.spaces[aid].sample().astype(np.float32)
                for aid in env.agent_ids
            }
            obs, _rewards, _terminated, _truncated, _info = env.step(actions)
            _assert_metadata(obs, env)
            for aid in env.agent_ids:
                for key, shape in initial_shapes[aid].items():
                    assert obs[aid][key].shape == shape
            print(f"step={step + 1} metadata_shapes_ok: true")
        print("diagnose_hetero_observation: DONE")
    finally:
        env.close()


if __name__ == "__main__":
    main()
