from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_happo_reference import HeartbeatLogger, _transitions_per_rollout


def test_train_happo_help_exposes_num_envs_and_heartbeat_flags():
    result = subprocess.run(
        [sys.executable, "scripts/train_happo_reference.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0
    assert "--num-envs" in result.stdout
    assert "--heartbeat-log" in result.stdout
    assert "--heartbeat-every-steps" in result.stdout
    assert "--debug-rollout-heartbeat" in result.stdout
    assert "--eval-at-start" in result.stdout


def test_transitions_per_rollout_uses_configured_num_envs():
    assert _transitions_per_rollout(256, 1) == 256
    assert _transitions_per_rollout(256, 4) == 1024


def test_heartbeat_logger_writes_env_step_markers(tmp_path: Path):
    path = tmp_path / "heartbeat.log"
    logger = HeartbeatLogger(path, every_steps=1, enabled=True)
    logger.write(
        "before_env_step",
        iteration=1,
        rollout_local_step=7,
        env_idx=0,
        total_steps=42,
        episode_length=3,
        alive_agents={"red": 3, "blue": 2},
        done=False,
        truncated=False,
    )
    logger.write(
        "after_env_step",
        iteration=1,
        rollout_local_step=7,
        env_idx=0,
        total_steps=43,
        episode_length=4,
        alive_agents={"red": 3, "blue": 2},
        done=False,
        truncated=False,
    )
    logger.close()

    text = path.read_text(encoding="utf-8")
    assert "before_env_step" in text
    assert "after_env_step" in text
    assert "total_env_steps_actual=43" in text
