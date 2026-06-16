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
