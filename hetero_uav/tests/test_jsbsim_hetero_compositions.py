from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env import make_env


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


CONFIG_CASES = [
    (
        "uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml",
        ["mav", "attack_uav"],
        ["attack_uav", "attack_uav"],
    ),
    (
        "uav_env/JSBSim/configs/hetero_test_3v3_mav_2attack.yaml",
        ["mav", "attack_uav", "attack_uav"],
        ["attack_uav", "attack_uav", "attack_uav"],
    ),
    (
        "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml",
        ["mav", "attack_uav", "scout_uav"],
        ["attack_uav", "attack_uav", "attack_uav"],
    ),
    (
        "uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_interceptor.yaml",
        ["mav", "attack_uav", "interceptor_uav"],
        ["attack_uav", "attack_uav", "attack_uav"],
    ),
]
HETERO_FIELDS = {
    "ego_type",
    "ego_role",
    "ally_types",
    "ally_roles",
    "enemy_types",
    "enemy_roles",
}


def _load_yaml(path: str) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _assert_no_nan(obs: dict) -> None:
    for aid, agent_obs in obs.items():
        for key, value in agent_obs.items():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"}:
                assert not np.isnan(arr).any(), f"NaN in {aid}/{key}"


def _step_policy(env, policy: str, steps: int = 3) -> None:
    rng = np.random.default_rng(0)
    obs, _info = env.reset(seed=0)
    _assert_no_nan(obs)
    for _ in range(steps):
        if policy == "zero":
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        else:
            actions = {
                aid: rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
                for aid in env.agent_ids
            }
        obs, _rewards, terminated, truncated, _info = env.step(actions)
        _assert_no_nan(obs)
        if all(terminated.values()) or all(truncated.values()):
            break


@pytest.mark.parametrize("config_path,red_types,blue_types", CONFIG_CASES)
def test_hetero_composition_config_reset_and_step(config_path, red_types, blue_types):
    cfg = _load_yaml(config_path)
    assert cfg["env_type"] == "jsbsim_hetero"
    assert cfg["red_agent_types"] == red_types
    assert cfg["blue_agent_types"] == blue_types

    env = make_env(config_path)
    try:
        obs, info = env.reset(seed=0)

        assert [info["agent_types"][f"red_{i}"] for i in range(env.max_num_red)] == red_types
        assert [info["agent_types"][f"blue_{i}"] for i in range(env.max_num_blue)] == blue_types
        assert info["agent_models"]["red_0"] == "A-4"
        for aid, type_name in info["agent_types"].items():
            expected_model = cfg["aircraft_type_params"][type_name]["aircraft_model"]
            assert info["agent_models"][aid] == expected_model

        assert info["agent_init_offsets"]["red_0"]["altitude_offset_m"] == 2000.0
        assert env._num_missiles_for("red_0") == 2
        for aid, type_name in info["agent_types"].items():
            expected_missiles = cfg["aircraft_type_params"][type_name]["num_missiles"]
            assert env._num_missiles_for(aid) == expected_missiles

        if "scout_uav" in red_types:
            scout_id = f"red_{red_types.index('scout_uav')}"
            assert env._num_missiles_for(scout_id) == 0
        if "interceptor_uav" in red_types:
            interceptor_id = f"red_{red_types.index('interceptor_uav')}"
            assert env._num_missiles_for(interceptor_id) == 2

        assert HETERO_FIELDS.issubset(obs["red_0"])
        np.testing.assert_array_equal(obs["red_0"]["ego_type"], np.array([1, 0, 0, 0], dtype=np.float32))
        if env.max_num_red == 3:
            assert obs["red_0"]["ally_types"].shape == (2, 4)
            assert obs["red_0"]["ally_roles"].shape == (2, 4)
            assert obs["red_0"]["enemy_types"].shape == (3, 4)
            assert obs["red_0"]["enemy_roles"].shape == (3, 4)

        _step_policy(env, "zero", steps=3)
        _step_policy(env, "bounded_random", steps=3)
    finally:
        env.close()


def test_jsbsim_brma_observation_unchanged_by_compositions():
    from uav_env.JSBSim.envs.uav_combat_env import UavCombatEnv

    env = UavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=20,
        suppress_jsbsim_output=True,
    )
    try:
        obs, _info = env.reset(seed=0)
        for aid in env.agent_ids:
            assert HETERO_FIELDS.isdisjoint(env.observation_space.spaces[aid].spaces)
            assert HETERO_FIELDS.isdisjoint(obs[aid])
    finally:
        env.close()
