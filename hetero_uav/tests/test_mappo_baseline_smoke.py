"""Smoke tests for MAPPO baseline pipeline. No HAPPO, no attention, no GRU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CONFIG = 'uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml'
CONFIG_3V3 = 'uav_env/JSBSim/configs/hetero_test_3v3_mav_attack_scout.yaml'


def test_import_succeeds():
    from algorithms.mappo.policy import MAPPOActorCritic
    m = MAPPOActorCritic()
    assert m.actor_obs_dim == 140


def test_actor_forward():
    from algorithms.mappo.policy import MAPPOActorCritic
    m = MAPPOActorCritic()
    actor_in = torch.randn(2, 140)
    critic_in = torch.randn(2, 700)
    dist, value, action, log_prob, entropy = m(actor_in, critic_in)
    assert action.shape == (2, 3)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.isfinite(action).all()


def test_opponent_policy_zero_shape():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    obs = {"blue_0": {}, "blue_1": {}}
    actions = OpponentPolicy("zero").act(obs, ["blue_0", "blue_1"])
    assert set(actions) == {"blue_0", "blue_1"}
    for act in actions.values():
        assert act.shape == (3,)
        assert np.allclose(act, 0.0)


def test_opponent_policy_random_range():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    obs = {"blue_0": {}, "blue_1": {}}
    actions = OpponentPolicy("random", seed=0).act(obs, ["blue_0", "blue_1"])
    for act in actions.values():
        assert act.shape == (3,)
        assert np.all(act >= -1.0)
        assert np.all(act <= 1.0)


def test_opponent_policy_rule_nearest_range():
    from algorithms.mappo.opponent_policy import OpponentPolicy
    obs = {
        "blue_0": {
            "enemy_states": np.array([
                [0.5, -0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0.3, -0.2, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float32)
        }
    }
    actions = OpponentPolicy("rule_nearest").act(obs, ["blue_0"])
    act = actions["blue_0"]
    assert act.shape == (3,)
    assert np.all(act >= -1.0)
    assert np.all(act <= 1.0)


def test_critic_shape():
    from algorithms.mappo.policy import MAPPOActorCritic
    m = MAPPOActorCritic()
    critic_in = torch.randn(1, 700)
    value = m.critic(critic_in)
    assert value.shape == (1, 1)


def test_adapter_plus_env():
    from uav_env.JSBSim.adapters.hetero_obs_adapter import HeteroObsAdapter
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=5,
        suppress_jsbsim_output=True,
        red_agent_types=['mav', 'attack_uav'],
        blue_agent_types=['attack_uav', 'attack_uav'],
    )
    try:
        obs, info = env.reset(seed=0)
        adapter = HeteroObsAdapter()
        result = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
        assert result['critic_state'].shape == (700,)
        r0 = result['actor_obs']['red_0']
        assert r0.shape == (140,)
    finally:
        env.close()


def _run_train_smoke(opponent_policy: str):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'train_mappo_baseline.py'),
         '--config', CONFIG, '--iterations', '1', '--rollout-length', '8',
         '--debug', '--device', 'cpu',
         '--opponent-policy', opponent_policy],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr[:800]}'
    assert 'Saved' in result.stdout
    assert f'opponent_policy={opponent_policy}' in result.stdout


def test_train_smoke_zero_runs():
    _run_train_smoke('zero')


def test_train_smoke_rule_nearest_runs():
    _run_train_smoke('rule_nearest')


def test_model_saved():
    model_path = ROOT / 'outputs' / 'mappo_baseline' / 'latest' / 'model.pt'
    assert model_path.exists(), str(model_path)


def test_eval_smoke_runs():
    model_path = str(ROOT / 'outputs' / 'mappo_baseline' / 'latest' / 'model.pt')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'eval_mappo_baseline.py'),
         '--model', model_path, '--config', CONFIG,
         '--episodes', '1', '--device', 'cpu',
         '--opponent-policy', 'rule_nearest'],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr[:800]}'
    assert 'avg_return' in result.stdout
    assert 'opponent_policy: rule_nearest' in result.stdout
    assert 'nan_detected: False' in result.stdout


def test_zero_shot_eval_smoke_runs():
    model_path = str(ROOT / 'outputs' / 'mappo_baseline' / 'latest' / 'model.pt')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'eval_mappo_zero_shot.py'),
         '--model', model_path,
         '--episodes', '1', '--device', 'cpu',
         '--opponent-policy', 'rule_nearest',
         '--configs', CONFIG, CONFIG_3V3],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180,
    )
    assert result.returncode == 0, f'stderr: {result.stderr[:800]}'
    assert result.stdout.count('avg_return') >= 2
    assert 'actor_obs_dim check: True' in result.stdout
    assert 'critic_state_dim check: True' in result.stdout
    assert 'nan_detected: False' in result.stdout


def test_no_nan_in_policy_init():
    from algorithms.mappo.policy import MAPPOActorCritic
    m = MAPPOActorCritic()
    out = m.actor(torch.randn(1, 140))
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
