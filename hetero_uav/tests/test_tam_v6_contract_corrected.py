from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.trainer import (
    _compute_grouped_gae,
    _compute_role_local_gae,
    _normalize_role_local_advantages,
)
from scripts.train_happo_reference import _build_actor_update_mask, _build_policy, _pure_happo_meta
from scripts.audit_high_level_pid_action_response import _audit_action, _summarize
from scripts.run_tam_v5_fast_learnability import _red_missile_kill_key
from scripts.run_tam_v6_contract_corrected_learnability import _json_safe
from uav_env import make_env
from uav_env.JSBSim.env import UavCombatEnv, make_empty_launch_diag
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
from uav_env.JSBSim.envs.alignment.target_assessment import (
    select_paper_assessment_target,
    target_hold_sequence_stats,
)


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_paper_formula_v5.yaml"
V6 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_happo_contract_corrected_v6.yaml"
V6_5V4 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_happo_contract_corrected_v6.yaml"


def test_v6_config_is_isolated_from_frozen_v5():
    old = yaml.safe_load(V5.read_text(encoding="utf-8"))
    new = yaml.safe_load(V6.read_text(encoding="utf-8"))
    assert old["missile_evasion"]["teams"] == "red_only"
    assert old["red_target_selection_mode"] == "closest"
    assert new["missile_evasion"] == {"mode": "brma_scripted", "teams": "none"}
    assert new["red_target_selection_mode"] == "paper_assessment"
    assert new["hetero_reward_mode"] == "tam_happo_paper_formula_v5"
    assert new["incoming_missile_observation"]["contract"] == "mav_shared_geo_v3_incoming_missile"
    transfer = yaml.safe_load(V6_5V4.read_text(encoding="utf-8"))
    assert transfer["max_num_red"] == 5 and transfer["max_num_blue"] == 4
    assert transfer["missile_evasion"]["teams"] == "none"
    assert transfer["red_target_selection_mode"] == "paper_assessment"


def test_v6_observation_and_action_contract_are_explicit_and_finite():
    env = make_env(str(V6), suppress_jsbsim_output=True)
    try:
        obs, _ = env.reset(seed=7)
        for aid in env.agent_ids:
            assert obs[aid]["incoming_missile_state"].shape == (7,)
            assert obs[aid]["incoming_missile_valid_mask"].shape == (1,)
            assert np.isfinite(obs[aid]["incoming_missile_state"]).all()
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _obs, _reward, _terminated, _truncated, info = env.step(actions)
        for rid in env.red_ids:
            assert info[rid]["action_overridden"] is False
            assert info[rid]["action_override_reason"] == "none"
            assert len(info[rid]["requested_action"]) == 3
            assert info[rid]["policy_requested_action"] == [0.0, 0.0, 0.0]
            assert info[rid]["post_trim_action"] == [0.0, 0.0, 0.0]
            assert info[rid]["action_transformed"] is False
            assert info[rid]["action_transform_reason"] == "none"
            assert len(info[rid]["executed_control_target"]) == 3
    finally:
        env.close()


def test_v6_contract_rejects_nonzero_action_trim(tmp_path):
    cfg = yaml.safe_load(V6.read_text(encoding="utf-8"))
    cfg["action_trim_by_role"]["mav"]["pitch"] = 0.1
    path = tmp_path / "v6_nonzero_trim.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="requires zero action trim"):
        make_env(str(path), suppress_jsbsim_output=True)


def test_nonzero_trim_is_reported_as_action_transformation():
    env = make_env(str(V6), suppress_jsbsim_output=True)
    try:
        env.reset(seed=3)
        # Other experiments may opt into trim; emulate that after the v6 startup guard.
        env.action_trim_by_role["mav"] = np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _obs, _reward, _terminated, _truncated, info = env.step(actions)
        red = info["red_0"]
        assert red["policy_requested_action"] == [0.0, 0.0, 0.0]
        np.testing.assert_allclose(red["post_trim_action"], [0.1, 0.0, 0.0])
        assert red["action_transformed"] is True
        assert red["action_transform_reason"] == "action_trim"
        assert red["action_overridden"] is False
    finally:
        env.close()


def test_v6_high_level_pid_action_mapping_is_absolute_and_bounded():
    env = make_env(str(V6), suppress_jsbsim_output=True)
    try:
        env.reset(seed=11)
        target = env._parse_actions({
            "red_0": np.asarray([0.25, -0.5, 1.0], dtype=np.float32),
        })["red_0"]
        assert target[0] == pytest.approx(0.25 * np.pi / 2.0)
        assert target[1] == pytest.approx(-0.5 * np.pi)
        assert target[2] == pytest.approx(env.VELOCITY_MAX)
        low_speed = env._parse_actions({
            "red_0": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        })["red_0"]
        assert low_speed[2] == pytest.approx(env.VELOCITY_MIN)
        np.testing.assert_array_equal(_audit_action("heading", 1.0), [0.0, 1.0, 0.0])
    finally:
        env.close()


def test_v6_adapter_expands_schema_without_changing_v5_default():
    assert HeteroObsAdapterV2().flat_actor_obs_dim == 140
    adapter = HeteroObsAdapterV2(include_incoming_missile=True)
    assert adapter.flat_actor_obs_dim == 148
    assert adapter.critic_state_dim == 740


def test_v6_incoming_missile_normalization_is_bounded_in_both_adapters():
    obs = {
        "ego_geo_state": np.zeros(7, dtype=np.float32),
        "ego_role": np.asarray([1, 0, 0, 0], dtype=np.float32),
        "missile_warning": np.asarray([1], dtype=np.float32),
        "incoming_missile_state": np.asarray(
            [5000, -50000, 100000, 4 * np.pi, 4 * np.pi, 5000, 600],
            dtype=np.float32),
        "incoming_missile_valid_mask": np.asarray([1], dtype=np.float32),
    }
    flat = HeteroObsAdapterV2(include_incoming_missile=True)._build_ego(obs)
    entity = HeteroEntitySetAdapter(include_incoming_missile=True)._self_token(obs)
    assert np.max(np.abs(flat[-8:-1])) <= 1.0
    assert np.max(np.abs(entity[21:28])) <= 1.0


def test_actor_update_mask_is_distinct_from_alive_mask():
    active = np.ones(3, dtype=np.float32)
    info = {
        "red_0": {"action_overridden": False},
        "red_1": {"action_overridden": True, "action_override_reason": "missile_evasion"},
        "red_2": {"action_overridden": False},
    }
    np.testing.assert_array_equal(
        _build_actor_update_mask(active, info, ["red_0", "red_1", "red_2"]),
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(active, np.ones(3, dtype=np.float32))


def _local_buffer() -> HAPPORolloutBuffer:
    buffer = HAPPORolloutBuffer(8, 3, 6, 12, 3, [0, 1, 1], value_dim=3)
    rng = np.random.default_rng(4)
    for step in range(8):
        active = np.ones(3, dtype=np.float32)
        actor_mask = active.copy(); actor_mask[1] = 0.0
        rewards = np.asarray([1.0, 10.0 if step == 0 else 0.0, -1.0], dtype=np.float32)
        buffer.store(
            rng.normal(size=(3, 6)).astype(np.float32),
            rng.normal(size=12).astype(np.float32),
            np.tanh(rng.normal(size=(3, 3))).astype(np.float32),
            np.zeros(3, dtype=np.float32), rewards, np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32), active, actor_update_masks=actor_mask,
            next_value=np.zeros(3, dtype=np.float32),
        )
    return buffer


def test_role_local_vector_critic_shape_masked_update_and_reward_isolation():
    torch.manual_seed(2)
    policy = PureHAPPOPolicy(6, 12, 3, 3, credit_mode="role_local_vector_critic")
    assert policy.value(torch.zeros(2, 12)).shape == (2, 3)
    stats = PureHAPPOTrainer(policy, ppo_epochs=1, critic_epochs=1, seed=2).update(_local_buffer())
    assert np.isfinite(stats["critic_loss"])
    assert stats["valid_sample_count_per_agent"] == [8, 0, 8]
    assert stats["credit_mode"] == "role_local_vector_critic"


def test_role_local_transition_bootstrap_and_reward_are_agent_local():
    rewards = torch.tensor([[1.0, 10.0], [1.0, 10.0], [1.0, 10.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards)
    env_ids = torch.zeros(3, dtype=torch.long)
    base, _ = _compute_role_local_gae(
        rewards, values, next_values, dones, env_ids, gamma=1.0, lam=1.0)

    changed_next = next_values.clone()
    changed_next[1, 0] = 5.0
    changed, _ = _compute_role_local_gae(
        rewards, values, changed_next, dones, env_ids, gamma=1.0, lam=1.0)
    assert not torch.equal(changed[:2, 0], base[:2, 0])
    assert changed[2, 0] == base[2, 0]
    torch.testing.assert_close(changed[:, 1], base[:, 1])

    changed_reward = rewards.clone()
    changed_reward[0, 0] += 7.0
    reward_adv, _ = _compute_role_local_gae(
        changed_reward, values, next_values, dones, env_ids, gamma=1.0, lam=1.0)
    torch.testing.assert_close(reward_adv[:, 1], base[:, 1])
    assert reward_adv[0, 0] == base[0, 0] + 7.0


def test_role_local_agent_done_stops_only_that_agents_gae_chain():
    rewards = torch.ones((3, 2))
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards)
    dones[1, 0] = 1.0
    advantages, _ = _compute_role_local_gae(
        rewards, values, next_values, dones, torch.zeros(3, dtype=torch.long),
        gamma=1.0, lam=1.0)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([2.0, 1.0, 1.0]))
    torch.testing.assert_close(advantages[:, 1], torch.tensor([3.0, 2.0, 1.0]))


def test_shared_scalar_gae_keeps_per_transition_bootstrap_contract():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    next_values = torch.tensor([2.0, 3.0, 4.0])
    dones = torch.tensor([0.0, 0.0, 1.0])
    advantages, returns = _compute_grouped_gae(
        rewards, values, next_values, dones, torch.zeros(3, dtype=torch.long),
        gamma=1.0, lam=0.0)
    torch.testing.assert_close(advantages, torch.tensor([3.0, 4.0, 1.0]))
    torch.testing.assert_close(returns, advantages)


def test_role_local_advantage_normalization_excludes_invalid_actor_samples():
    advantages = torch.tensor([[1.0, 10.0], [3.0, 20.0], [1000.0, 30.0]])
    actor_masks = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    normalized = _normalize_role_local_advantages(advantages, actor_masks)
    torch.testing.assert_close(normalized[:, 0], torch.tensor([-1.0, 1.0, 0.0]))
    torch.testing.assert_close(normalized[:, 1], torch.tensor([0.0, -1.0, 1.0]))
    assert torch.isfinite(normalized).all()


def test_credit_mode_checkpoint_contract_rejects_mismatch(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"credit_mode": "role_local_vector_critic"}), encoding="utf-8")
    with pytest.raises(ValueError, match="credit_mode"):
        _build_policy("pure_happo", 6, 12, torch.device("cpu"),
                      init_checkpoint_meta=meta, num_agents=3,
                      credit_mode="shared_alive_team_mean")
    policy = PureHAPPOPolicy(6, 12, 3, 3)
    contract = _pure_happo_meta(policy)
    assert contract["algorithm"] == "pure_happo_baseline"
    assert contract["has_gru"] is False
    assert contract["paper_action_space_exact"] is False


def test_v6_observation_checkpoint_contract_rejects_legacy_dimensions(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({
        "policy_arch": "pure_happo", "credit_mode": "shared_alive_team_mean",
        "actor_obs_dim": 140, "critic_state_dim": 700, "action_dim": 3,
        "num_agents": 3, "observation_contract": "mav_shared_geo_v2",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="incoming-missile observation checkpoints"):
        _build_policy("pure_happo", 148, 740, torch.device("cpu"),
                      init_checkpoint_meta=meta, num_agents=3, action_dim=3)


def test_v6_pure_happo_meta_names_heterogeneity_boundaries():
    policy = PureHAPPOPolicy(148, 740, 3, 3)
    args = type("Args", (), {
        "actor_lr": 5e-4, "critic_lr": 5e-4, "clip_param": 0.2,
        "entropy_coef": 0.01, "value_coef": 0.5, "max_grad_norm": 10.0,
        "ppo_epochs": 5, "critic_epochs": 5, "gamma": 0.99,
        "gae_lambda": 0.95, "rollout_length": 256, "num_envs": 1,
        "max_steps": 1000, "seed": 0, "opponent_policy": "tam_greedy_rule",
        "observation_contract": "mav_shared_geo_v3_incoming_missile",
        "incoming_missile_normalization": {}, "aircraft_dynamics_homogeneous": True,
        "roles_heterogeneous": True, "mav_has_missiles": False,
        "mav_aircraft_model": "f16", "attack_uav_aircraft_model": "f16",
        "dynamics_heterogeneity": False, "role_heterogeneity": True,
        "sensor_heterogeneity": True, "payload_heterogeneity": True,
        "reward_heterogeneity": True, "evasion_scripted": False,
    })()
    meta = _pure_happo_meta(policy, args)
    assert meta["mav_aircraft_model"] == "f16"
    assert meta["attack_uav_aircraft_model"] == "f16"
    assert meta["dynamics_heterogeneity"] is False
    assert all(meta[key] is True for key in (
        "role_heterogeneity", "sensor_heterogeneity",
        "payload_heterogeneity", "reward_heterogeneity"))
    assert meta["evasion_scripted"] is False
    assert meta["critic_output_mode"] == "scalar_team_value"
    assert meta["paper_standard_shared_team_advantage"] is True
    assert meta["credit_ablation"] is False
    assert meta["launch_gate_statistical_unit"] == "attack_uav_agent_decision"

    local = PureHAPPOPolicy(148, 740, 3, 3, credit_mode="role_local_vector_critic")
    local_meta = _pure_happo_meta(local, args)
    assert local_meta["global_v_critic"] is False
    assert local_meta["critic_output_mode"] == "per_agent_vector_value"
    assert local_meta["paper_standard_shared_team_advantage"] is False
    assert local_meta["credit_ablation"] is True
    assert local_meta["credit_contract_claim"] == "diagnostic_ablation_not_default_happo"


def test_v6_payload_contract_is_heterogeneous_even_before_combat():
    env = make_env(str(V6), suppress_jsbsim_output=True)
    try:
        assert [env._num_missiles_for(rid) for rid in env.red_ids] == [0, 2, 2]
    finally:
        env.close()


class _StateObject:
    def __init__(self, uid, position, velocity, *, alive=True, target=None):
        self.uid = uid
        self._position = np.asarray(position, dtype=np.float64)
        self._velocity = np.asarray(velocity, dtype=np.float64)
        self.is_alive = alive
        self.target_aircraft = target
        self._target_id = getattr(target, "uid", "")
        self.under_missiles = []

    def get_position(self):
        return self._position.copy()

    def get_velocity(self):
        return self._velocity.copy()


def test_incoming_missile_contract_selects_minimum_positive_tgo():
    aircraft = _StateObject("red_1", [0, 0, 6000], [200, 0, 0])
    receding = _StateObject("m0", [1000, 0, 6100], [300, 0, 0], target=aircraft)
    slow = _StateObject("m1", [-2000, 0, 6200], [400, 0, 0], target=aircraft)
    urgent = _StateObject("m2", [-1000, 0, 6500], [600, 0, 0], target=aircraft)
    aircraft.under_missiles = [receding, slow, urgent]
    env = object.__new__(UavCombatEnv)
    state, valid, diag = UavCombatEnv._incoming_missile_state(env, aircraft.uid, aircraft)
    assert valid == 1.0
    assert diag["incoming_missile_id"] == "m2"
    assert state.shape == (7,)
    assert np.isfinite(state).all()
    assert state[1] == pytest.approx(500.0)
    assert state[2] == pytest.approx(np.linalg.norm([1000, 0, 500]))
    assert state[5] > 0.0 and state[6] > 0.0


def test_incoming_missile_contract_zeroes_invalid_or_absent_threat():
    aircraft = _StateObject("red_1", [0, 0, 6000], [200, 0, 0])
    aircraft.under_missiles = [
        _StateObject("receding", [1000, 0, 6000], [300, 0, 0], target=aircraft)
    ]
    env = object.__new__(UavCombatEnv)
    state, valid, diag = UavCombatEnv._incoming_missile_state(env, aircraft.uid, aircraft)
    np.testing.assert_array_equal(state, np.zeros(7, dtype=np.float32))
    assert valid == 0.0 and diag == {}


class _AssessmentEnv:
    def __init__(self):
        self.current_step = 4
        self._engagement_target_state = {}
        self._engagement_target_step_diag = {}
        self._engagement_reallocation_counts = {}
        self._engaged_targets = set()
        self._lock_target = {}
        self._missiles_in_flight = {}
        self.blue_ids = ["blue_0", "blue_1"]
        self.blue_planes = {
            "blue_0": _StateObject("blue_0", [4000, 0, 6000], [200, 0, 0]),
            "blue_1": _StateObject("blue_1", [12000, 0, 6000], [200, 0, 0]),
        }

    def _has_launch_track(self, _aid, _bid):
        return True, "direct"

    def _brma_tam_3d_geometry(self, _attacker, target):
        # blue_1 deliberately receives the better assessment despite distance.
        if target.uid == "blue_1":
            return {"tam_ata_rad": 0.0, "tam_aa_rad": 0.0, "target_distance_m": 12000.0}
        return {"tam_ata_rad": np.pi, "tam_aa_rad": np.pi, "target_distance_m": 4000.0}


def test_paper_assessment_uses_shared_score_and_records_hold_diagnostics():
    env = _AssessmentEnv()
    attacker = _StateObject("red_1", [0, 0, 6000], [200, 0, 0])
    cfg = {"target_assessment": {"engagement_range_m": 14000.0, "hold_steps": 0}}
    target_id, _target, values = select_paper_assessment_target(
        env, attacker, cfg, require_observed=True)
    assert target_id == "blue_1"
    assert values["closest_target_id"] == "blue_0"
    target_id2, _target2, values2 = select_paper_assessment_target(
        env, attacker, cfg, require_observed=True)
    assert target_id2 == "blue_1"
    assert values2["target_held"] == 1.0
    assert env._engagement_target_step_diag["red_1"]["held"] is True


def test_paper_assessment_ranks_only_fire_control_candidates_and_reallocates():
    env = _AssessmentEnv()
    attacker = _StateObject("red_1", [0, 0, 6000], [200, 0, 0])
    cfg = {"target_assessment": {"engagement_range_m": 14000.0, "hold_steps": 10}}
    target_id, _target, _values = select_paper_assessment_target(
        env, attacker, cfg, require_observed=True, candidate_target_ids={"blue_1"})
    assert target_id == "blue_1"
    env.current_step += 1
    target_id, _target, values = select_paper_assessment_target(
        env, attacker, cfg, require_observed=True, candidate_target_ids={"blue_0"})
    assert target_id == "blue_0"
    assert values["target_reallocated"] == 1.0
    assert values["target_reallocation_reason"] == "not_fire_control_candidate"
    assert env._engagement_reallocation_counts["red_1"] == 1


def test_paper_assessment_deconflicts_two_attackers():
    env = _AssessmentEnv()
    cfg = {"target_assessment": {"engagement_range_m": 14000.0, "hold_steps": 0}}
    first = _StateObject("red_1", [0, 0, 6000], [200, 0, 0])
    second = _StateObject("red_2", [0, 100, 6000], [200, 0, 0])
    first_id, _target, _values = select_paper_assessment_target(env, first, cfg)
    assert first_id == "blue_1"
    env._engaged_targets.add(first_id)
    second_id, _target, _values = select_paper_assessment_target(env, second, cfg)
    assert second_id == "blue_0"


def _fire_control_env(*, include_alternate: bool = True, agent_ids=None):
    env = object.__new__(UavCombatEnv)
    env.agent_ids = list(agent_ids or ["red_1", "red_2"])
    env.red_planes = {
        aid: SimpleNamespace(
            uid=aid, color="Red", is_alive=True, num_left_missiles=2,
            get_position=lambda: np.asarray([0.0, 0.0, 6000.0]),
            get_velocity=lambda: np.asarray([250.0, 0.0, 0.0]),
        )
        for aid in env.agent_ids
    }
    env.blue_planes = {
        "blue_0": _StateObject("blue_0", [5000, 0, 6000], [200, 0, 0]),
    }
    if include_alternate:
        env.blue_planes["blue_1"] = _StateObject(
            "blue_1", [6000, 0, 6000], [200, 0, 0])
    env.blue_ids = list(env.blue_planes)
    env.agent_roles = {aid: "attack_uav" for aid in env.agent_ids}
    env.red_target_selection_mode = "paper_assessment"
    env.tam_happo_paper_formula_v5_config = {
        "target_assessment": {"engagement_range_m": 14000.0, "hold_steps": 0}
    }
    env.current_step = 10
    env._engagement_target_state = {
        aid: {"target_id": "blue_0", "selected_step": 9, "switch_count": 0}
        for aid in env.agent_ids
    }
    env._engagement_target_step_diag = {}
    env._engagement_reallocation_counts = {}
    env._engaged_targets = set()
    env._missiles_in_flight = {}
    env._lock_target = {aid: "blue_0" for aid in env.agent_ids}
    env._lock_timer = {aid: 1 for aid in env.agent_ids}
    env._missile_cooldown = {aid: 0 for aid in env.agent_ids}
    env.missile_lock_delay_frames = 2
    env._agents_deny_kill = set()
    env.use_boresight_launch_gate = False
    env._launch_diag_step = make_empty_launch_diag()
    env._fire_candidate_target_step = {}
    env._launch_gate_accum = {}
    env._get_sim = lambda aid: env.red_planes.get(aid)
    env._has_launch_track = lambda _aid, _bid: (True, "direct")
    env._missile_candidate_metrics = lambda _shooter, target: {
        "range_m": 5000.0 if target.uid == "blue_0" else 6000.0,
        "range_ok": True, "ao_ok": True, "ta_ok": True,
        "boresight_ok_3d": True, "launch_geometry_ok_3d": True,
    }
    env._score_mav_aware_target = lambda _shooter, target, metrics: {
        "score": 1.0 if target.uid == "blue_0" else 0.0,
        "range_m": metrics["range_m"],
    }
    env._brma_tam_3d_geometry = lambda _shooter, target: {
        "tam_ata_rad": 0.0 if target.uid == "blue_0" else np.pi,
        "tam_aa_rad": 0.0 if target.uid == "blue_0" else np.pi,
        "target_distance_m": 5000.0 if target.uid == "blue_0" else 6000.0,
    }
    env._build_launch_quality_record = lambda *_args, **_kwargs: {}
    launches = []

    def launch(parent, target, _quality):
        launches.append((parent.uid, target.uid))
        parent.num_left_missiles -= 1
        env._missiles_in_flight[f"m{len(launches)}"] = SimpleNamespace(
            parent_aircraft=parent, target_aircraft=target,
            _parent_id=parent.uid, _target_id=target.uid,
        )

    env._launch_missile = launch
    return env, launches


def test_fire_control_reallocates_second_shooter_after_same_frame_launch():
    env, launches = _fire_control_env(include_alternate=True)
    UavCombatEnv._check_missile_launch(env)
    assert launches == [("red_1", "blue_0")]
    assert env._lock_target["red_2"] == "blue_1"
    assert env._lock_timer["red_2"] == 1


def test_fire_control_clears_second_shooter_lock_without_alternate_target():
    env, launches = _fire_control_env(include_alternate=False)
    UavCombatEnv._check_missile_launch(env)
    assert launches == [("red_1", "blue_0")]
    assert env._lock_target["red_2"] is None
    assert env._lock_timer["red_2"] == 0


def test_fire_control_preserves_own_lock_when_only_own_missile_occupies_target():
    env, launches = _fire_control_env(include_alternate=False, agent_ids=["red_2"])
    env._engaged_targets.add("blue_0")
    env._missiles_in_flight["own"] = SimpleNamespace(
        parent_aircraft=env.red_planes["red_2"],
        target_aircraft=env.blue_planes["blue_0"],
        _parent_id="red_2", _target_id="blue_0",
    )
    env._missile_cooldown["red_2"] = 2
    UavCombatEnv._check_missile_launch(env)
    assert launches == []
    assert env._lock_target["red_2"] == "blue_0"
    assert env._lock_timer["red_2"] == 2


def test_target_hold_metric_is_mean_contiguous_segment_length():
    valid, segments, mean = target_hold_sequence_stats(
        ["A", "A", "A", "B", "B", None, "C", "C"])
    assert valid == 7
    assert segments == 3
    assert mean == pytest.approx(7.0 / 3.0)


def test_red_kill_attribution_uses_blue_death_event_not_reward_credit():
    event = {
        "side": "blue", "agent_id": "blue_0", "killed_by_missile": True,
        "missile_owner": "red_1",
    }
    assert _red_missile_kill_key(event, 1002) == (1002, "blue_0")
    assert _red_missile_kill_key({**event, "missile_owner": "blue_1"}, 1002) is None
    assert _red_missile_kill_key({**event, "killed_by_missile": False}, 1002) is None

    # A preceding MAV death is irrelevant; the later attack-UAV kill remains countable.
    mav_death = {"side": "red", "agent_id": "red_0", "killed_by_missile": True,
                 "missile_owner": "blue_0"}
    keys = {
        key for key in (
            _red_missile_kill_key(mav_death, 1002),
            _red_missile_kill_key(event, 1002),
            _red_missile_kill_key(event, 1002),
        ) if key is not None
    }
    assert keys == {(1002, "blue_0")}


def test_v6_runner_json_sanitizes_nonstandard_nonfinite_values():
    payload = _json_safe({"missing": float("nan"), "positive": float("inf"), "ok": 1.0})
    assert payload == {"missing": None, "positive": None, "ok": 1.0}
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_pid_audit_uses_unit_safe_response_deltas_and_honest_saturation_names():
    rows = []
    for axis in ("pitch", "heading", "speed"):
        for value in (-1.0, 0.0, 1.0):
            for step in (1, 5, 10, 25):
                rows.append({
                    "axis": axis, "action_value": value, "step": step,
                    "roll_rad": value * 0.01, "pitch_rad": value * 0.02,
                    "yaw_rad": value * 0.03, "speed_mps": 200.0 + value,
                    "position_n_m": 0.0, "position_e_m": 0.0, "position_u_m": 6000.0,
                    "altitude_m": 6000.0 + value, "vertical_speed_mps": value,
                    "overload_g": 1.0, "crash": 0, "out_of_bounds": 0,
                    "fcs_throttle_cmd_norm": None, "fcs_aileron_cmd_norm": None,
                    "fcs_elevator_cmd_norm": None, "fcs_rudder_cmd_norm": None,
                })
    summary = _summarize(rows, (-1.0, 0.0, 1.0))
    axis_summary = summary["axes"]["pitch"]
    assert "saturation_rate" not in axis_summary
    assert "cross_axis_coupling_ratio_at_step_25" not in axis_summary
    assert axis_summary["requested_action_boundary_fraction"] == pytest.approx(2.0 / 3.0)
    assert axis_summary["actual_controller_saturation_fraction"] is None
    assert "speed_delta_mps" in axis_summary["endpoint_response_deltas_at_step_25"]["+1.00"]
