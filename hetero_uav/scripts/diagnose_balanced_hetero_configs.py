"""Diagnose balanced hetero configs without running formal experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.adapter_utils import make_obs_adapter
from uav_env import make_env


CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_4v4.yaml",
]


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _actions(env, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    if mode == "bounded_random":
        return {
            aid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
            for aid in env.agent_ids
        }
    raise ValueError(mode)


def _missile_counts(env) -> dict[str, int]:
    counts = {}
    for aid, sim in {**env.red_planes, **env.blue_planes}.items():
        counts[aid] = int(getattr(sim, "num_left_missiles", -1))
    return counts


def _adapter_version(obs_mode: str) -> str:
    return "v2" if obs_mode == "mav_shared_geo" else "v1"


def diagnose_config(config: str) -> None:
    env = make_env(config, env_type="jsbsim_hetero")
    rng = np.random.default_rng(0)
    try:
        obs, info = env.reset(seed=0)
        obs_mode = getattr(env, "observation_mode", "brma_sensor")
        adapter = make_obs_adapter(_adapter_version(obs_mode))
        adapted = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        if _contains_nan(obs) or _contains_nan(adapted):
            raise RuntimeError(f"NaN detected after reset for {config}")

        print(f"=== {Path(config).name} ===")
        print(f"observation_mode: {obs_mode}")
        print(f"red_ids: {env.red_ids}")
        print(f"blue_ids: {env.blue_ids}")
        print(f"red_count: {len(env.red_ids)}")
        print(f"blue_count: {len(env.blue_ids)}")
        print(f"agent_types: {info.get('agent_types', {})}")
        print(f"model_names: {info.get('agent_models', {})}")
        print(f"missile_counts: {_missile_counts(env)}")
        print(f"actor_obs_dim: {adapter.flat_actor_obs_dim}")
        print(f"critic_state_dim: {adapter.critic_state_dim}")
        print(f"red_valid_mask: {adapted['red_valid_mask'].astype(int).tolist()}")

        for mode in ("zero", "bounded_random"):
            for _ in range(3):
                obs, rewards, terminated, truncated, info = env.step(
                    _actions(env, mode, rng))
                adapted = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                if _contains_nan(obs) or _contains_nan(adapted):
                    raise RuntimeError(f"NaN detected in {mode} step for {config}")
            print(f"{mode}_steps: 3 ok")
        print()
    finally:
        env.close()


def main() -> None:
    for config in CONFIGS:
        diagnose_config(config)
    print("balanced hetero config diagnostics passed")


if __name__ == "__main__":
    main()
