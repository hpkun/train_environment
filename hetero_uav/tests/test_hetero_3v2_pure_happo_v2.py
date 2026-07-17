from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval_hetero_3v2_pure_happo_v1 import _validate_checkpoint_meta
from scripts.train_hetero_3v2_pure_happo_v1 import _contract_meta
from uav_env.JSBSim.formal_v1.geometry import combat_geometry
from uav_env.JSBSim.formal_v1.missile import FormalMissile
from uav_env.JSBSim.formal_v1.targeting import fire_gate
from uav_env.JSBSim.formal_v2.contract import (
    CREDIT_MODE, ENV_TYPE, OBSERVATION_CONTRACT, REWARD_CONTRACT_VERSION,
)
from uav_env.JSBSim.formal_v2.observation import (
    FIRE_CONTROL_FIELDS, build_actor_observation,
)
from uav_env.JSBSim.formal_v2.reward import (
    EVENT_REWARDS, MAV_SAFETY_WEIGHTS, MAV_SUPPORT_WEIGHTS, UAV_WEIGHTS,
    compute_role_rewards, mav_safety_components, mav_support_components,
    uav_angle_reward, uav_dodge_components, uav_speed_reward,
)
from uav_env.make_env import make_env

V1_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml"
V2_CONFIG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v2.yaml"


class FakeAircraft:
    def __init__(
        self, uid, position, velocity=(250.0, 0.0, 0.0), *,
        alive=True, missiles=2,
    ):
        self.uid = uid
        self._position = np.asarray(position, dtype=float)
        self._velocity = np.asarray(velocity, dtype=float)
        self._rpy = np.zeros(3, dtype=float)
        self.is_alive = alive
        self.num_missiles = missiles
        self.num_left_missiles = missiles

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def get_rpy(self):
        return self._rpy


def fake_env():
    aircraft = {
        "red_0": FakeAircraft("red_0", (-4_000, 0, 6_500), missiles=0),
        "red_1": FakeAircraft("red_1", (0, 0, 6_000)),
        "red_2": FakeAircraft("red_2", (0, 1_000, 6_000)),
        "blue_0": FakeAircraft("blue_0", (8_000, 0, 6_000), (-250, 0, 0)),
        "blue_1": FakeAircraft("blue_1", (16_000, 2_000, 6_000), (-250, 0, 0)),
    }
    return SimpleNamespace(
        aircraft=aircraft,
        red_ids=["red_0", "red_1", "red_2"],
        blue_ids=["blue_0", "blue_1"],
        roles={
            "red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav",
            "blue_0": "attack_uav", "blue_1": "attack_uav",
        },
        missiles=[],
        previous_missile_speed={},
        newly_dead=set(),
        death_reasons={},
        sim_time_sec=25.0,
        last_launch_time={agent_id: 0.0 for agent_id in aircraft},
        attack_interval_sec=25.0,
        attack_range_m=14_000.0,
        launch_ata_rad=np.deg2rad(60.0),
        launch_ta_rad=np.deg2rad(90.0),
        mav_detection_range_m=80_000.0,
        uav_detection_range_m=10_000.0,
        v2_mav_team_credit_used=0.0,
    )


def v1_meta():
    return {
        "formal_contract": "hetero_3v2_pure_happo_v1",
        "credit_mode": "shared_alive_team_mean",
        "reward_contract": {"version": "formal_role_reward_v2"},
        "algorithm_contract": "pure_happo_sequential_v2",
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
        "actor_obs_dim": 68,
        "critic_state_dim": 204,
        "action_dim": 3,
        "num_agents": 3,
    }


def v2_meta():
    return {
        "formal_contract": ENV_TYPE,
        "observation_contract": OBSERVATION_CONTRACT,
        "credit_mode": "shared_alive_team_mean",
        "reward_contract": REWARD_CONTRACT_VERSION,
        "algorithm_contract": "pure_happo_sequential_v2",
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
        "actor_obs_dim": 73,
        "critic_state_dim": 219,
        "action_dim": 3,
        "num_agents": 3,
    }


def test_v1_and_v2_construct_with_isolated_dimensions():
    v1 = make_env(V1_CONFIG)
    v2 = make_env(V2_CONFIG)
    obs1, info1 = v1.reset(seed=0)
    obs2, info2 = v2.reset(seed=0)
    assert obs1["red_1"]["flat"].shape == (68,)
    assert info1["critic_state"].shape == (204,)
    assert obs2["red_1"]["flat"].shape == (73,)
    assert info2["critic_state"].shape == (219,)
    assert tuple(FIRE_CONTROL_FIELDS) == (
        "missile_cooldown_remaining_norm", "fire_ready", "own_inflight_any",
        "team_inflight_to_enemy_slot_0", "team_inflight_to_enemy_slot_1",
    )
    assert np.isfinite(np.r_[obs2["red_1"]["flat"], info2["critic_state"]]).all()
    v1.close()
    v2.close()


def test_v2_cooldown_ready_and_inflight_slots_remove_aliases():
    env = make_env(V2_CONFIG)
    env.reset(seed=1)
    ready = build_actor_observation(env, "red_1")["fire_control"]
    assert ready.tolist() == pytest.approx([0, 1, 0, 0, 0])

    env.last_launch_time["red_1"] = env.sim_time_sec
    cooling = build_actor_observation(env, "red_1")["fire_control"]
    assert cooling[0] == pytest.approx(1.0)
    assert cooling[1] == 0.0
    assert not np.array_equal(ready, cooling)

    missile = FormalMissile(
        "red_1_m0", "red_1", "blue_0",
        env.aircraft["red_1"].get_position().copy(),
        np.asarray([600.0, 0.0, 0.0]),
    )
    env.missiles.append(missile)
    occupied = build_actor_observation(env, "red_1")["fire_control"]
    teammate = build_actor_observation(env, "red_2")["fire_control"]
    mav = build_actor_observation(env, "red_0")["fire_control"]
    assert occupied[2:].tolist() == [1.0, 1.0, 0.0]
    assert teammate[2:].tolist() == [0.0, 1.0, 0.0]
    assert mav.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    missile.state = "miss"
    restored = build_actor_observation(env, "red_1")["fire_control"]
    assert restored[2:].tolist() == [0.0, 0.0, 0.0]
    env.close()


def test_enemy_slot_identity_is_stable_after_death():
    env = make_env(V2_CONFIG)
    obs, _ = env.reset(seed=2)
    before = obs["red_1"]["enemies"].copy()
    env.aircraft["blue_0"].shotdown()
    after = build_actor_observation(env, "red_1")["enemies"]
    assert np.any(before[0] != 0.0)
    assert np.all(after[0] == 0.0)
    assert np.any(after[1] != 0.0)
    env.close()


def test_uav_published_weights_speed_angle_and_target_identity():
    assert UAV_WEIGHTS == {
        "height": 10.0, "speed": 10.0, "angle": 15.0,
        "distance": 10.0, "dodge": 30.0,
    }
    assert uav_speed_reward(200.0, 99.0) == 1.0
    assert uav_speed_reward(200.0, 100.0) == pytest.approx(1.0)
    assert uav_speed_reward(200.0, 200.0) == 0.0
    assert uav_speed_reward(200.0, 300.0) == pytest.approx(-1.0)
    assert uav_speed_reward(200.0, 301.0) == -1.0
    assert uav_angle_reward(0.0, np.pi) == pytest.approx(1.0)
    assert uav_angle_reward(np.pi, 0.0) == pytest.approx(-1.0)

    env = fake_env()
    first, detail1 = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_0"}, [])
    second, detail2 = compute_role_rewards(
        env, {"red_1": "blue_1", "red_2": "blue_0"}, [])
    assert first["red_1"] != second["red_1"]
    assert (
        detail1["per_agent"]["red_1"]["uav_distance"]
        != detail2["per_agent"]["red_1"]["uav_distance"]
    )


def test_dodge_formula_uses_current_incoming_missile_and_history():
    env = fake_env()
    missile = FormalMissile(
        "blue_0_m0", "blue_0", "red_1",
        np.asarray((-1_000.0, 0.0, 6_000.0)),
        np.asarray((600.0, 0.0, 0.0)),
    )
    env.missiles = [missile]
    first = uav_dodge_components(env, "red_1")
    assert first["dodge_angle"] == pytest.approx(-1.0)
    assert first["dodge_speed"] == 0.0
    missile.velocity = np.asarray((500.0, 0.0, 0.0))
    second = uav_dodge_components(env, "red_1")
    assert second["dodge_speed"] == pytest.approx(0.1)
    env.missiles = []
    assert uav_dodge_components(env, "red_1")["dodge"] == 0.0


def test_mav_weights_shared_metric_and_death_contract():
    assert MAV_SAFETY_WEIGHTS == {
        "distance": 0.5, "threat": 0.3, "aspect": 0.2}
    assert MAV_SUPPORT_WEIGHTS == {"position": 0.6, "awareness": 0.4}
    env = fake_env()
    support = mav_support_components(env)
    expected = (
        0.6 * support["support_position"]
        + 0.4 * support["awareness"])
    assert support["support"] == pytest.approx(expected)
    original_total, original = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_1"}, [])
    env.uav_detection_range_m = 0.0
    changed_total, changed = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_1"}, [])
    assert (
        original["per_agent"]["red_0"]["mav_shared_information_metric"]
        != changed["per_agent"]["red_0"]["mav_shared_information_metric"]
    )
    assert original_total["red_0"] == pytest.approx(changed_total["red_0"])

    env.aircraft["red_0"].is_alive = False
    env.newly_dead = {"red_0"}
    reward, detail = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_1"}, [])
    mav = detail["per_agent"]["red_0"]
    assert mav["raw_event_reward"] == EVENT_REWARDS["mav_death"]
    assert reward["red_0"] == pytest.approx(-2.0)
    assert mav["dense"] == 0.0
    assert all(mav[key] == 0.0 for key in (
        "mav_safety", "mav_support", "mav_awareness"))
    env.newly_dead = set()
    later, _ = compute_role_rewards(
        env, {"red_1": "blue_0", "red_2": "blue_1"}, [])
    assert later["red_0"] == 0.0


def test_mav_safety_is_exact_published_weighted_combination():
    env = fake_env()
    safety = mav_safety_components(env)
    expected = (
        0.5 * safety["safety_distance"]
        + 0.3 * safety["safety_threat"]
        + 0.2 * safety["safety_aspect"]
    )
    assert safety["safety"] == pytest.approx(expected)


def test_mav_position_and_awareness_are_not_constant():
    env = fake_env()
    baseline = mav_support_components(env)
    env.aircraft["red_0"]._position = np.asarray((-25_000.0, 0.0, 6_500.0))
    moved = mav_support_components(env)
    assert baseline["support_position"] != moved["support_position"]
    env.aircraft["red_0"]._velocity = np.asarray((-250.0, 0.0, 0.0))
    turned = mav_support_components(env)
    assert moved["awareness"] != turned["awareness"]


def test_reward_and_fire_gate_share_selected_target_geometry():
    env = make_env(V2_CONFIG)
    env.reset(seed=7)
    shooter = env.aircraft["red_1"]
    target_id = "blue_1"
    target = env.aircraft[target_id]
    geometry = combat_geometry(shooter, target)
    _, gate = fire_gate(env, "red_1", target_id)
    _, details = compute_role_rewards(
        env, {"red_1": target_id, "red_2": "blue_0"}, [])
    reward = details["per_agent"]["red_1"]

    assert gate["ata_rad"] == pytest.approx(geometry["ata_rad"])
    assert gate["ta_rad"] == pytest.approx(geometry["ta_rad"])
    assert reward["uav_angle"] == pytest.approx(
        uav_angle_reward(gate["ata_rad"], gate["ta_rad"]))
    env.close()


def test_v2_meta_and_cross_contract_rejection():
    env = make_env(V2_CONFIG)
    meta = {
        **_contract_meta(env),
        "credit_mode": CREDIT_MODE,
        "algorithm_contract": "pure_happo_sequential_v2",
        "policy_distribution": "tanh_squashed_gaussian_raw_action",
        "critic_contract": "centralized_shared_scalar_v",
        "gae_contract": "separated_termination_truncation",
    }
    assert meta["formal_contract"] == ENV_TYPE
    assert meta["observation_contract"] == OBSERVATION_CONTRACT
    assert meta["reward_contract"] == REWARD_CONTRACT_VERSION
    assert meta["actor_obs_dim"] == 73
    assert meta["critic_state_dim"] == 219
    _validate_checkpoint_meta(meta, ENV_TYPE)
    with pytest.raises(ValueError, match="does not match"):
        _validate_checkpoint_meta(v1_meta(), ENV_TYPE)
    with pytest.raises(ValueError, match="does not match"):
        _validate_checkpoint_meta(v2_meta(), "hetero_3v2_pure_happo_v1")
    wrong = dict(meta, actor_obs_dim=68)
    with pytest.raises(ValueError, match="actor_obs_dim"):
        _validate_checkpoint_meta(wrong, ENV_TYPE)
    wrong = dict(meta, reward_contract="wrong")
    with pytest.raises(ValueError, match="reward_contract"):
        _validate_checkpoint_meta(wrong, ENV_TYPE)
    env.close()
