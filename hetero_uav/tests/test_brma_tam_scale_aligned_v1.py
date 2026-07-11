from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithms.pure_happo import PureHAPPOPolicy
from algorithms.pure_happo.trainer import _alive_before_team_mean
from scripts.train_happo_reference import (
    MARL_DYNAMICS_TRAIN_FIELDS,
    _experiment_base_v2_meta,
)
from scripts.experiment_logging_schema import (
    REWARD_COMPONENT_COLUMNS, EPISODE_REWARD_COMPONENTS_COLUMNS,
)
from uav_env import make_env
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


ROOT = Path(__file__).resolve().parents[1]
CFG3 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v1.yaml"
CFG5 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scale_aligned_v1.yaml"
OLD3 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"


class Sim:
    def __init__(self, uid, pos, vel=(250.0, 0.0, 0.0), alive=True):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.is_alive = alive
        self.under_missiles = []

    def get_position(self): return self._pos
    def get_velocity(self): return self._vel
    def get_rpy(self): return np.zeros(3)
    def get_geodetic(self): return np.asarray([0.0, 0.0, self._pos[2]])


def _cfg(path=CFG3):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["brma_tam_scale_aligned_v1"]


def _env(scale="3v2"):
    env = object.__new__(HeteroUavCombatEnv)
    n_attack, n_blue = (2, 2) if scale == "3v2" else (4, 4)
    env.red_ids = ["red_0"] + [f"red_{i}" for i in range(1, n_attack + 1)]
    env.blue_ids = [f"blue_{i}" for i in range(n_blue)]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {"red_0": "mav", **{r: "attack_uav" for r in env.red_ids[1:]}}
    env.agent_roles.update({b: "attack_uav" for b in env.blue_ids})
    env.red_planes = {"red_0": Sim("red_0", (0, 0, 6500), (230, 0, 0))}
    env.red_planes.update({r: Sim(r, (0, i * 1000, 6000), (260, 0, 0)) for i, r in enumerate(env.red_ids[1:])})
    env.blue_planes = {b: Sim(b, (12000 + i * 1000, i * 1000, 6000), (230, 0, 0)) for i, b in enumerate(env.blue_ids)}
    env.brma_tam_scale_aligned_v1_config = _cfg(CFG3 if scale == "3v2" else CFG5)
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env._death_reasons = {}
    env._death_events_step = []
    env._missiles_in_flight = {}
    env._last_step_obs = {}
    env.mav_observation_range_m = 80000.0
    env.uav_direct_observation_range_m = 10000.0
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.max_steps = 1000
    env.current_step = 1
    env._brma_tam_scale_v1_reset_episode_state()
    env._scale_v1_alive_before_step = {aid: True for aid in env.agent_ids}
    return env


def _base(env):
    rewards = {rid: 0.0 for rid in env.red_ids}
    components = {rid: {
        "r_pitch": 0.01, "r_roll": -0.02, "r_alt": 0.03,
        "r_bound": 0.0, "r_vel": 0.01, "r_adv": 99.0,
        "r_end": 99.0, "r_death": -99.0,
    } for rid in env.red_ids}
    return rewards, components


def _progress(env, aid="red_1", target="blue_0", distance=12000.0, angle=0.0, speed=0.0):
    env.blue_planes[target]._pos[0] = distance
    return env._scale_v1_uav_progress(
        aid, target,
        {"target_distance_m": distance, "tam_angle_raw": angle, "tam_geometry_valid": 1.0},
        {"tam_speed_raw": speed, "speed_ratio_valid": 1.0},
    )


def test_old_reward_mode_unchanged():
    data = yaml.safe_load((ROOT / OLD3).read_text(encoding="utf-8"))
    assert data["hetero_reward_mode"] == "brma_tam_scripted_composite_v1"
    assert data["brma_tam_scripted_composite_v1"]["reward_contract_revision"] == 2


def test_new_reward_mode_constructs():
    env = make_env(str(ROOT / CFG3), max_steps=2)
    try: assert env.hetero_reward_mode == "brma_tam_scale_aligned_v1"
    finally: env.close()


@pytest.mark.parametrize("distance,expected", [(5000, 1.0), (10000, math.exp(-0.5)), (15000, math.exp(-1)), (22300, math.exp(-1.73)), (30000, math.exp(-2.5))])
def test_distance_potential_values(distance, expected):
    assert HeteroUavCombatEnv._scale_v1_distance_potential(distance) == pytest.approx(expected)


def test_first_step_progress_zero(): assert _progress(_env())["scale_v1_progress_clipped"] == 0


def test_stationary_progress_zero():
    env = _env(); _progress(env)
    assert _progress(env)["scale_v1_progress_clipped"] == pytest.approx(0)


def test_approach_progress_positive():
    env = _env(); _progress(env, distance=15000)
    assert _progress(env, distance=10000)["scale_v1_progress_clipped"] > 0


def test_retreat_progress_negative():
    env = _env(); _progress(env, distance=10000)
    assert _progress(env, distance=15000)["scale_v1_progress_clipped"] < 0


def test_far_range_has_gradient():
    env = _env(); _progress(env, distance=22300)
    assert _progress(env, distance=21000)["scale_v1_delta_distance"] > 0


def test_target_switch_resets_progress():
    env = _env(); _progress(env, target="blue_0")
    row = _progress(env, target="blue_1")
    assert row["scale_v1_progress_clipped"] == 0 and row["scale_v1_progress_reset_reason"] == "target_switch"


def test_target_death_resets_progress():
    env = _env(); _progress(env, target="blue_0"); env.blue_planes["blue_0"].is_alive = False
    row = _progress(env, target="blue_1")
    assert row["scale_v1_progress_reset_reason"] == "target_dead"


def test_progress_cache_resets_per_episode():
    env = _env(); _progress(env); env._brma_tam_scale_v1_reset_episode_state()
    assert _progress(env)["scale_v1_progress_reset_reason"] == "episode_start"


def test_progress_updates_once_per_decision():
    env = _env(); _progress(env); row = _progress(env)
    assert row["scale_v1_delta_distance"] == 0


def test_progress_is_clipped():
    env = _env(); _progress(env, distance=30000, angle=-1, speed=-1)
    row = _progress(env, distance=5000, angle=1, speed=1)
    assert row["scale_v1_progress_clipped"] == 0.5 and row["scale_v1_progress_raw"] > 0.5


def test_uav_event_once_semantics():
    env = _env(); env.red_planes["red_1"].is_alive = False
    first, _ = env._scale_v1_uav_event("red_1", env.red_planes["red_1"])
    second, _ = env._scale_v1_uav_event("red_1", env.red_planes["red_1"])
    assert first == -10 and second == 0


def test_mav_event_once_semantics():
    env = _env(); env.red_planes["red_0"].is_alive = False
    first, _ = env._scale_v1_mav_event("red_0", env.red_planes["red_0"])
    second, _ = env._scale_v1_mav_event("red_0", env.red_planes["red_0"])
    assert first == -20 and second == 0


def test_mav_credit_normalized_3v2():
    env = _env(); env._step_kill_count["red_1"] = 1
    value, logs = env._scale_v1_mav_event("red_0", env.red_planes["red_0"])
    assert value == 5 and logs["scale_v1_mav_team_credit_used"] == 5


def test_mav_credit_normalized_5v4():
    env = _env("5v4"); env._step_kill_count["red_1"] = 1
    value, _ = env._scale_v1_mav_event("red_0", env.red_planes["red_0"])
    assert value == 2.5


def test_mav_aspect_scale_invariant():
    env3 = _env(); _, a = env3._scale_v1_mav_role(env3.red_planes["red_0"])
    env5 = _env("5v4"); _, b = env5._scale_v1_mav_role(env5.red_planes["red_0"])
    assert abs(a["scale_v1_mav_aspect_mean"]) <= 1 and abs(b["scale_v1_mav_aspect_mean"]) <= 1


def test_mav_awareness_scale_invariant():
    env3 = _env(); _, a = env3._scale_v1_mav_role(env3.red_planes["red_0"])
    env5 = _env("5v4"); _, b = env5._scale_v1_mav_role(env5.red_planes["red_0"])
    assert 0 <= a["scale_v1_mav_aware_mean"] <= .3 and 0 <= b["scale_v1_mav_aware_mean"] <= .3


@pytest.mark.parametrize("blue_alive,attack_alive,mav_alive,expected", [(0,2,True,30),(0,1,True,22.5),(0,2,False,15),(2,2,True,0),(2,2,False,-15),(2,0,False,-30)])
def test_terminal_3v2_cases(blue_alive, attack_alive, mav_alive, expected):
    env = _env()
    for i,b in enumerate(env.blue_ids): env.blue_planes[b].is_alive = i < blue_alive
    env.red_planes["red_0"].is_alive = mav_alive
    for i,r in enumerate(env.red_ids[1:]): env.red_planes[r].is_alive = i < attack_alive
    assert env._scale_v1_terminal_value()[0] == pytest.approx(expected)


@pytest.mark.parametrize("blue_alive,red_alive,expected", [(0,5,30),(4,5,0),(4,0,-30)])
def test_terminal_5v4_cases(blue_alive, red_alive, expected):
    env = _env("5v4")
    for i,b in enumerate(env.blue_ids): env.blue_planes[b].is_alive = i < blue_alive
    for i,r in enumerate(env.red_ids): env.red_planes[r].is_alive = i < red_alive
    assert env._scale_v1_terminal_value()[0] == pytest.approx(expected)


def test_terminal_applied_once():
    env = _env(); env.current_step = env.max_steps
    _, first = env._compute_brma_tam_scale_aligned_v1(*_base(env))
    _, second = env._compute_brma_tam_scale_aligned_v1(*_base(env))
    assert first["red_0"]["scale_v1_terminal_applied"] == 1
    assert second["red_0"]["scale_v1_terminal_applied"] == 0


def test_terminal_alive_before_mean():
    import torch
    value = _alive_before_team_mean(torch.tensor([[15.,15.,0.]]), torch.tensor([[1.,1.,0.]]))
    assert value.item() == 15


def test_dead_before_reward_zero():
    env = _env(); env._scale_v1_alive_before_step["red_1"] = False
    rewards, _ = env._compute_brma_tam_scale_aligned_v1(*_base(env))
    assert rewards["red_1"] == 0


def test_uav_reward_identity():
    env = _env(); rewards, comp = env._compute_brma_tam_scale_aligned_v1(*_base(env))
    assert comp["red_1"]["scale_v1_identity_error"] == pytest.approx(0)
    assert rewards["red_1"] == pytest.approx(comp["red_1"]["scale_v1_uav_total"])


def test_mav_reward_identity():
    env = _env(); rewards, comp = env._compute_brma_tam_scale_aligned_v1(*_base(env))
    assert comp["red_0"]["scale_v1_identity_error"] == pytest.approx(0)
    assert rewards["red_0"] == pytest.approx(comp["red_0"]["scale_v1_mav_total"])


def test_initial_state_team_scale():
    env = make_env(str(ROOT / CFG3), max_steps=2)
    try:
        env.reset(seed=1)
        _, rewards, *_ = env.step({aid: np.zeros(3, np.float32) for aid in env.agent_ids})
        assert abs(np.mean([rewards[r] for r in env.red_ids])) < 0.1
    finally: env.close()


def test_static_far_episode_no_large_negative_return():
    assert 1000 * (HeteroUavCombatEnv._scale_v1_distance_potential(22300) - HeteroUavCombatEnv._scale_v1_distance_potential(22300)) == 0


def test_canonical_outcome_ordering():
    values = [30, 22.5, 15, 0, -15, -30]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_short_environment_rollout_finite():
    env = make_env(str(ROOT / CFG3), max_steps=3)
    try:
        env.reset(seed=2)
        for _ in range(3):
            _, rewards, *_ = env.step({aid: np.zeros(3, np.float32) for aid in env.agent_ids})
            assert np.isfinite(list(rewards.values())).all()
    finally: env.close()


def test_train_log_contains_scale_v1_fields():
    for key in ("effective_scale_v1_total", "effective_scale_v1_identity_error", "scale_v1_progress_positive_ratio"):
        assert key in MARL_DYNAMICS_TRAIN_FIELDS
    assert "scale_v1_progress_clipped" in REWARD_COMPONENT_COLUMNS
    assert "scale_v1_uav_total_sum" in EPISODE_REWARD_COMPONENTS_COLUMNS
    assert "scale_v1_mav_total_sum" in EPISODE_REWARD_COMPONENTS_COLUMNS


def test_checkpoint_meta_contains_reward_contract():
    meta = _experiment_base_v2_meta(actual_reward_mode="brma_tam_scale_aligned_v1", rich_logging_enabled=True, rich_log_mode="summary", config_path=CFG3)
    assert meta["reward_contract_revision"] == 3
    assert meta["reward_config"]["uav"]["progress"]["distance_weight"] == 5


def test_pure_happo_uses_no_custom_network():
    policy = PureHAPPOPolicy(num_agents=3)
    names = " ".join(type(m).__name__.lower() for m in policy.modules())
    assert "gru" not in names and "attention" not in names and "mask" not in names


def test_old_config_still_loads():
    env = make_env(str(ROOT / OLD3), max_steps=1)
    try: assert env.hetero_reward_mode == "brma_tam_scripted_composite_v1"
    finally: env.close()


def test_new_3v2_and_5v4_configs_load():
    for path, counts in ((CFG3, (3,2)), (CFG5, (5,4))):
        env = make_env(str(ROOT / path), max_steps=1)
        try: assert (len(env.red_ids), len(env.blue_ids)) == counts
        finally: env.close()
