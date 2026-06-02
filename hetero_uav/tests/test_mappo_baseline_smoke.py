"""Smoke tests for MAPPO baseline pipeline. No HAPPO, no attention, no GRU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

CONFIG = 'uav_env/JSBSim/configs/hetero_train_2v2_mav_attack.yaml'


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


def test_train_smoke_runs():
    """Train 1 iteration, 8-step rollout, debug mode."""
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'train_mappo_baseline.py'),
         '--config', CONFIG, '--iterations', '1', '--rollout-length', '8',
         '--debug', '--device', 'cpu'],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr[:800]}'
    assert 'Saved' in result.stdout


def test_model_saved():
    model_path = ROOT / 'outputs' / 'mappo_baseline' / 'latest' / 'model.pt'
    assert model_path.exists(), str(model_path)


def test_eval_smoke_runs():
    model_path = str(ROOT / 'outputs' / 'mappo_baseline' / 'latest' / 'model.pt')
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'eval_mappo_baseline.py'),
         '--model', model_path, '--config', CONFIG,
         '--episodes', '1', '--device', 'cpu'],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert result.returncode == 0, f'stderr: {result.stderr[:800]}'
    assert 'avg_return' in result.stdout


def test_no_nan_in_policy_init():
    from algorithms.mappo.policy import MAPPOActorCritic
    m = MAPPOActorCritic()
    out = m.actor(torch.randn(1, 140))
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
