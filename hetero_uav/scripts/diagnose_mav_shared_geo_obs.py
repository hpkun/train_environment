"""Diagnose mav_shared_geo raw observations and HeteroObsAdapterV2."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.adapters import HeteroObsAdapterV2  # noqa: E402


CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
V2_KEYS = [
    "ego_geo_state",
    "ally_geo_states",
    "ally_alive_mask",
    "enemy_geo_states",
    "enemy_alive_mask",
    "enemy_observed_mask",
    "enemy_track_source",
]


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def _actions(env, policy: str, rng: np.random.Generator) -> dict:
    if policy == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    return {
        aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
        for aid in env.agent_ids
    }


def _run_steps(env, obs, info, policy: str) -> bool:
    adapter = HeteroObsAdapterV2()
    rng = np.random.default_rng(0)
    for _ in range(3):
        result = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        assert result["critic_state"].shape == (480,)
        obs, _rew, terminated, truncated, info = env.step(_actions(env, policy, rng))
        if _obs_has_nan(obs):
            return False
        if all(terminated.values()) or all(truncated.values()):
            break
    return True


def _diagnose(config_path: str) -> None:
    print(f"=== {config_path} ===")
    env = make_env(config_path)
    adapter = HeteroObsAdapterV2()
    try:
        obs, info = env.reset(seed=0)
        print(f"observation_mode: {info.get('observation_mode')}")
        print(f"red_ids: {env.red_ids}")
        print(f"blue_ids: {env.blue_ids}")
        print(f"red_0_is_mav: {info['agent_roles'].get('red_0') == 'mav'}")
        for rid in env.red_ids:
            print(f"{rid} v2 obs keys: {[key for key in V2_KEYS if key in obs[rid]]}")
        print(f"red_0 enemy_observed_mask: {obs['red_0']['enemy_observed_mask'].tolist()}")
        if "red_1" in obs:
            print(f"red_1 enemy_alive_mask: {obs['red_1']['enemy_alive_mask'].tolist()}")
            print(f"red_1 enemy_observed_mask: {obs['red_1']['enemy_observed_mask'].tolist()}")
            print(f"red_1 enemy_track_source: {obs['red_1']['enemy_track_source'].tolist()}")
            has_shared = bool((obs["red_1"]["enemy_track_source"][:, 1] > 0.5).any())
            print(f"red_1_has_mav_shared_source: {has_shared}")
            alive_unobserved = bool((
                (obs["red_1"]["enemy_alive_mask"] > 0.5)
                & (obs["red_1"]["enemy_observed_mask"] < 0.5)
            ).any())
            print(f"red_1_has_alive_but_unobserved_enemy: {alive_unobserved}")
            if not has_shared:
                print("warning: reset state has no mav_shared source for red_1")
            if not alive_unobserved:
                print("warning: reset state has no alive-but-unobserved enemy for red_1")
        adapted = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        print(f"actor dim is 96: {adapter.flat_actor_obs_dim == 96}")
        print(f"critic dim is 480: {adapter.critic_state_dim == 480}")
        print(f"critic_state shape: {adapted['critic_state'].shape}")
        print(f"zero_3_step_no_nan: {_run_steps(env, obs, info, 'zero')}")
        obs, info = env.reset(seed=1)
        print(f"bounded_random_3_step_no_nan: {_run_steps(env, obs, info, 'bounded_random')}")
    finally:
        env.close()


def main() -> None:
    for config_path in CONFIGS:
        _diagnose(config_path)
    print("diagnose_mav_shared_geo_obs: DONE")


if __name__ == "__main__":
    main()
