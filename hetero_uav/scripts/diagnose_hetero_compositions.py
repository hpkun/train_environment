"""Diagnose formal heterogeneous composition configs.

This script only checks config loading, reset/step smoke behavior, and
type/role observation metadata. It does not run MAPPO or win-rate experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402


CONFIGS = [
    "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
    "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
]
HETERO_FIELDS = [
    "ego_type",
    "ego_role",
    "ally_types",
    "ally_roles",
    "enemy_types",
    "enemy_roles",
]


def _scan_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            if np.asarray(value).dtype.kind in {"f", "c"} and np.isnan(value).any():
                return True
    return False


def _scan_crashed(env) -> list[str]:
    crashed = []
    for sim in list(env.blue_planes.values()) + list(env.red_planes.values()):
        if getattr(sim, "is_crash", False):
            crashed.append(sim.uid)
    return crashed


def _zero_actions(env) -> dict:
    return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}


def _bounded_random_actions(env, rng: np.random.Generator) -> dict:
    return {
        aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        for aid in env.agent_ids
    }


def _smoke_steps(env, policy: str, steps: int = 5) -> tuple[bool, list[str]]:
    rng = np.random.default_rng(0)
    obs, _info = env.reset(seed=0)
    for _ in range(steps):
        actions = _zero_actions(env) if policy == "zero" else _bounded_random_actions(env, rng)
        obs, _rew, terminated, truncated, _info = env.step(actions)
        if _scan_nan(obs):
            return False, _scan_crashed(env)
        if any(_scan_crashed(env)):
            return False, _scan_crashed(env)
        if all(terminated.values()) or all(truncated.values()):
            break
    return True, _scan_crashed(env)


def _print_obs_space_shapes(env, agent_id: str) -> None:
    spaces = env.observation_space.spaces[agent_id].spaces
    print(f"  {agent_id} hetero observation_space shapes:")
    for key in HETERO_FIELDS:
        print(f"    {key}: {spaces[key].shape}")


def _diagnose_config(config_path: str) -> None:
    print(f"=== {config_path} ===")
    env = make_env(config_path)
    try:
        obs, info = env.reset(seed=0)
        red0_obs = obs["red_0"]
        print(f"max_num_red: {env.max_num_red}")
        print(f"max_num_blue: {env.max_num_blue}")
        print(f"agent_ids: {env.agent_ids}")
        print(f"agent_types: {info['agent_types']}")
        print(f"agent_roles: {info['agent_roles']}")
        print(f"agent_models: {info['agent_models']}")
        print(f"agent_init_offsets: {info['agent_init_offsets']}")
        for aid in env.agent_ids:
            print(f"{aid} observation keys: {list(obs[aid].keys())}")
        print(f"red_0 ego_type: {red0_obs['ego_type'].tolist()}")
        print(f"red_0 ego_role: {red0_obs['ego_role'].tolist()}")
        print(f"red_0 ally_types shape: {red0_obs['ally_types'].shape}")
        print(f"red_0 enemy_types shape: {red0_obs['enemy_types'].shape}")
        _print_obs_space_shapes(env, "red_0")

        zero_ok, zero_crashed = _smoke_steps(env, "zero", steps=5)
        print(f"zero_policy_5_step_ok: {zero_ok}")
        print(f"zero_policy_crashed_agents: {zero_crashed}")

        bounded_ok, bounded_crashed = _smoke_steps(env, "bounded_random", steps=5)
        print(f"bounded_random_5_step_ok: {bounded_ok}")
        print(f"bounded_random_crashed_agents: {bounded_crashed}")
        assert zero_ok
        assert bounded_ok
    finally:
        env.close()


def main() -> None:
    for config_path in CONFIGS:
        _diagnose_config(config_path)
    print("diagnose_hetero_compositions: DONE")


if __name__ == "__main__":
    main()
