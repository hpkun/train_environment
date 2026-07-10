from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.happo.happo_buffer import HAPPORolloutBuffer
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTrainer
from algorithms.pure_happo.policy import LegacyClampPureHAPPOPolicy
from algorithms.pure_happo.trainer import _alive_before_team_mean
from scripts.rich_logging import RichExperimentLogger
from scripts.train_happo_reference import (
    MARL_DYNAMICS_TRAIN_FIELDS,
    PURE_HAPPO_DEFAULTS,
    _apply_policy_specific_defaults,
    _build_policy,
    _pure_happo_meta,
    _resolve_pure_happo_eval_configs,
)
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


ROOT = Path(__file__).resolve().parents[1]
CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)


class FakeSim:
    def __init__(self, uid, pos, vel, alive=True):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.is_alive = bool(alive)
        self.under_missiles = []

    def get_position(self): return self._pos
    def get_velocity(self): return self._vel
    def get_rpy(self): return np.zeros(3, dtype=np.float64)
    def get_geodetic(self): return np.asarray([0.0, 0.0, self._pos[2]])


@pytest.fixture
def test_output_dir(request):
    path = ROOT / "outputs" / "test_tmp" / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _reward_cfg():
    data = yaml.safe_load((ROOT / CFG_3V2).read_text(encoding="utf-8"))
    return data["brma_tam_scripted_composite_v1"]


def _bare_reward_env():
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = ["blue_0", "blue_1"]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {
        "red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav",
        "blue_0": "attack_uav", "blue_1": "attack_uav",
    }
    env.red_planes = {
        "red_0": FakeSim("red_0", (0, 0, 6500), (250, 0, 0)),
        "red_1": FakeSim("red_1", (0, 0, 6000), (300, 0, 0)),
        "red_2": FakeSim("red_2", (0, 3000, 6000), (300, 0, 0)),
    }
    env.blue_planes = {
        "blue_0": FakeSim("blue_0", (4000, 0, 6000), (230, 0, 0)),
        "blue_1": FakeSim("blue_1", (12000, 0, 7000), (230, 0, 0)),
    }
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env._death_reasons = {}
    env._death_events_step = []
    env._evasion_step_records = []
    env._launch_quality_step_records = []
    env._missiles_in_flight = {}
    env._lock_target = {aid: None for aid in env.agent_ids}
    env._lock_timer = {aid: 0 for aid in env.agent_ids}
    env.brma_tam_scripted_composite_v1_config = _reward_cfg()
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 14000.0
    env.mav_observation_range_m = 80000.0
    env.uav_direct_observation_range_m = 10000.0
    env._brma_tam_scripted_reset_episode_state()
    env._brma_tam_alive_before_step = {aid: True for aid in env.agent_ids}
    return env


def _base_reward_inputs():
    rewards = {rid: 0.0 for rid in ("red_0", "red_1", "red_2")}
    components = {
        rid: {"r_pitch": 0.1, "r_roll": -0.2, "r_vel": 0.3, "total": 999.0}
        for rid in rewards
    }
    return rewards, components


def _fake_buffer(policy, steps=16):
    rng = np.random.default_rng(77)
    buffer = HAPPORolloutBuffer(steps, 3, 96, 480, 3, [0, 1, 1])
    for step in range(steps):
        obs = rng.normal(0, 0.2, (3, 96)).astype(np.float32)
        critic = rng.normal(0, 0.2, 480).astype(np.float32)
        with torch.no_grad():
            out = policy.act(torch.as_tensor(obs), critic_state=torch.as_tensor(critic))
        rewards = np.asarray([0.2, -0.1, 0.05], dtype=np.float32) + step * 0.001
        buffer.store(
            obs, critic, out["action"].numpy(), out["log_prob"].numpy(), rewards,
            np.zeros(3, np.float32), float(out["value"].item()),
            np.ones(3, np.float32), next_value=float(out["value"].item()), env_id=0,
        )
    return buffer


def _meta_args():
    values = dict(PURE_HAPPO_DEFAULTS)
    values.update({
        "rollout_length": 256, "num_envs": 1, "max_steps": 1000,
        "seed": 1, "opponent_policy": "brma_rule",
    })
    return argparse.Namespace(**values)


def test_pure_happo_is_canonical_tanh_policy():
    policy = _build_policy("pure_happo", 96, 480, torch.device("cpu"), num_agents=3)
    assert type(policy) is PureHAPPOPolicy
    assert not isinstance(policy, LegacyClampPureHAPPOPolicy) or type(policy) is not LegacyClampPureHAPPOPolicy
    out = policy.act(torch.zeros(3, 96), critic_state=torch.zeros(480))
    replay, *_ = policy.evaluate_actions(
        torch.zeros(1, 3, 96), torch.zeros(1, 480), out["action"].unsqueeze(0)
    )
    assert torch.max(torch.abs(replay.squeeze(0) - out["log_prob"])) < 1e-5


def test_pure_happo_has_independent_actors_shared_critic():
    policy = PureHAPPOPolicy(num_agents=3)
    ids = [{id(param) for param in actor.parameters()} for actor in policy.actors]
    assert all(not (ids[i] & ids[j]) for i in range(3) for j in range(i + 1, 3))
    assert len(policy.actors) == 3
    assert policy.critic is not None


def test_pure_happo_has_no_custom_modules():
    names = {module.__class__.__name__.lower() for module in PureHAPPOPolicy(num_agents=3).modules()}
    assert not any(any(token in name for token in ("attention", "gru", "mask", "entity")) for name in names)


def test_reward_identity_uav():
    env = _bare_reward_env()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(*_base_reward_inputs())
    for rid in ("red_1", "red_2"):
        comp = components[rid]
        expected = sum(comp[key] for key in (
            "brma_pitch", "brma_roll", "brma_vel", "tam_speed_weighted",
            "tam_angle_weighted", "tam_distance_weighted", "uav_event_total",
        ))
        assert rewards[rid] == pytest.approx(expected, abs=1e-6)


def test_reward_identity_mav():
    env = _bare_reward_env()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(*_base_reward_inputs())
    comp = components["red_0"]
    expected = sum(comp[key] for key in (
        "brma_pitch", "brma_roll", "brma_vel", "mav_dist_weighted",
        "mav_threat_weighted", "mav_aspect_weighted", "mav_pos_weighted",
        "mav_aware_weighted", "mav_event_total",
    ))
    assert rewards["red_0"] == pytest.approx(expected, abs=1e-6)


def test_reward_piecewise_boundaries():
    h = HeteroUavCombatEnv
    assert h._brma_tam_distance_raw(5000)["tam_distance_raw"] == 1.0
    assert h._brma_tam_distance_raw(10000)["tam_distance_raw"] == -1.0
    assert h._brma_tam_mav_dist_raw(8000, 8000, 15000) == pytest.approx(-0.5)
    assert h._brma_tam_mav_dist_raw(15000, 8000, 15000) == pytest.approx(0.2)
    assert h._brma_tam_mav_pos_raw(8000, 8000, 25000) == pytest.approx(1.0)
    assert h._brma_tam_mav_pos_raw(25000, 8000, 25000) == pytest.approx(-0.5)


def test_reward_event_once_semantics():
    env = _bare_reward_env()
    uav = env.red_planes["red_1"]
    env._step_kill_count["red_1"] = 1
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == 200.0
    env._step_kill_count["red_1"] = 0
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == 0.0
    uav.is_alive = False
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == -200.0
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == 0.0

    env = _bare_reward_env()
    uav = env.red_planes["red_1"]
    uav._pos[0] = 50000
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == -100.0
    assert env._brma_tam_uav_event("red_1", uav, _reward_cfg())[0] == 0.0


def test_active_agent_mean_after_mav_death():
    rewards = torch.tensor([[3.0, 6.0, 9.0], [100.0, 6.0, 10.0], [1.0, 2.0, 12.0], [5.0, 6.0, 7.0]])
    active = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(_alive_before_team_mean(rewards, active), torch.tensor([6.0, 8.0, 12.0, 0.0]))


def test_pure_happo_update_metrics_are_finite():
    policy = PureHAPPOPolicy(num_agents=3)
    metrics = PureHAPPOTrainer(policy, ppo_epochs=2, critic_epochs=2, seed=1).update(_fake_buffer(policy))
    scalar = [float(value) for value in metrics.values() if isinstance(value, (int, float, np.number))]
    assert scalar and all(math.isfinite(value) for value in scalar)
    assert metrics["critic_update_norm"] > 0.0
    assert all(value > 0.0 for value in metrics["policy_update_norm_per_agent"])
    assert metrics["gradient_nonfinite_count"] == 0


def test_numeric_metrics_persist_to_train_log():
    required = {
        "actor_loss_mean", "critic_loss_unscaled", "value_explained_variance_old",
        "value_explained_variance_new", "return_min", "return_max",
        "actor_grad_norm_mav", "critic_update_norm", "reward_nan_count",
        "gradient_nonfinite_count", "reward_identity_max_error",
    }
    assert required <= set(MARL_DYNAMICS_TRAIN_FIELDS)
    source = (ROOT / "scripts/train_happo_reference.py").read_text(encoding="utf-8")
    assert "*MARL_DYNAMICS_TRAIN_FIELDS" in source


def test_meta_disables_custom_network():
    meta = _pure_happo_meta(PureHAPPOPolicy(num_agents=3), _meta_args())
    assert meta["policy_arch"] == "pure_happo"
    assert meta["attention"] is False
    assert meta["recurrent"] is False
    assert meta["random_scale_mask"] is False
    assert meta["biased_mask"] is False
    assert meta["separate_actors"] is True
    assert meta["centralized_critic"] is True
    assert meta["sequential_update"] is True
    assert meta["critic_epochs"] == 5
    assert meta["initial_action_log_std"] == pytest.approx(-1.204)

    policy = PureHAPPOPolicy(num_agents=3)
    with torch.no_grad():
        policy.action_log_stds[0].add_(0.5)
    assert _pure_happo_meta(policy, _meta_args())["initial_action_log_std"] == pytest.approx(-1.204)


def test_summary_mode_does_not_write_heavy_step_logs(test_output_dir):
    logger = RichExperimentLogger(
        test_output_dir, run_id="test", method_name="pure_happo", scenario_name="3v2",
        device="cpu", num_envs=1, rollout_length_per_env=256,
        transitions_per_rollout=256, mode="summary",
    )
    logger.close()
    names = {path.name for path in test_output_dir.iterdir()}
    assert "train_metrics.csv" in names
    assert "episode_reward_components.csv" in names
    assert not names & {
        "aircraft_timeseries.csv", "missile_timeseries.csv", "missile_events.csv",
        "reward_components.csv", "reward_target_diagnostics.csv", "attention_metrics.csv",
    }


def test_pure_happo_eval_rejects_5v4():
    policy = PureHAPPOPolicy(num_agents=3)
    args = argparse.Namespace(config=CFG_3V2, eval_configs=[CFG_5V4])
    with pytest.raises(ValueError, match="training 3V2 config"):
        _resolve_pure_happo_eval_configs(args, policy)
    args.eval_configs = None
    assert _resolve_pure_happo_eval_configs(args, policy) == [CFG_3V2]


def test_one_update_reward_component_identity():
    env = _bare_reward_env()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(*_base_reward_inputs())
    errors = []
    for rid in env.red_ids:
        role = env.agent_roles[rid]
        keys = (
            ("brma_pitch", "brma_roll", "brma_vel", "mav_dist_weighted",
             "mav_threat_weighted", "mav_aspect_weighted", "mav_pos_weighted",
             "mav_aware_weighted", "mav_event_total")
            if role == "mav" else
            ("brma_pitch", "brma_roll", "brma_vel", "tam_speed_weighted",
             "tam_angle_weighted", "tam_distance_weighted", "uav_event_total")
        )
        errors.append(abs(rewards[rid] - sum(components[rid][key] for key in keys)))
    assert max(errors) <= 1e-6
