from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_parallel_runner_help_exposes_true_parallel_flags():
    result = subprocess.run(
        [sys.executable, "scripts/train_happo_reference_parallel.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0
    assert "--num-envs" in result.stdout
    assert "--reset-timeout-sec" in result.stdout
    assert "--step-timeout-sec" in result.stdout
    assert "--worker-startup-delay-sec" in result.stdout
    assert "--policy-arch" in result.stdout
    assert "pure_happo" in result.stdout


def test_parallel_runner_has_pure_happo_trainer_branch():
    """Source-level: parallel runner imports and branches for PureHAPPOTrainer."""
    from algorithms.pure_happo import PureHAPPOTrainer
    assert PureHAPPOTrainer.__name__ == "PureHAPPOTrainer"
    import scripts.train_happo_reference_parallel as mod
    import inspect
    src = inspect.getsource(mod)
    assert "PureHAPPOTrainer" in src
    assert "args.policy_arch == \"pure_happo\"" in src
    assert "trainer.update(buffer)" in src


def test_build_policy_supports_pure_happo():
    """Static: _build_policy('pure_happo', ..., num_agents=5) constructs 5 actors."""
    from scripts.train_happo_reference import _build_policy
    import torch
    policy = _build_policy("pure_happo", 96, 480, torch.device("cpu"), num_agents=5)
    assert policy.num_agents == 5
    assert len(policy.actors) == 5
    assert len(policy.action_log_stds) == 5


def test_parallel_runner_imports_worker_classes():
    from scripts.train_happo_reference_parallel import ParallelEnv, RemoteEnvProxy

    assert ParallelEnv.__name__ == "ParallelEnv"
    proxy = RemoteEnvProxy(
        {
            "red_ids": ["red_0"],
            "blue_ids": ["blue_0"],
            "agent_ids": ["red_0", "blue_0"],
            "max_steps": 10,
        },
        {"engaged_targets": ["red_0"], "blue_own_positions": {"blue_0": [0, 0, 0]}},
    )
    assert proxy.refresh_engaged_targets() == {"red_0"}
    assert "blue_0" in proxy.get_blue_own_positions()
