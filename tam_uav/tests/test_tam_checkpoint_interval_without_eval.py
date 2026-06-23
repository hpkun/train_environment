"""Test checkpoint interval saving without eval subprocess."""
from __future__ import annotations
import json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_checkpoint_interval_without_eval_subprocess():
    """num-envs=4, checkpoint-interval=512, no-subprocess-eval smoke"""
    import os
    out_dir = ROOT / ".tmp/test_ckpt_interval_noeval"
    if out_dir.exists():
        for f in out_dir.rglob("*"):
            if f.is_file():
                f.unlink()
        for d in sorted(out_dir.rglob("*"), reverse=True):
            if d.is_dir():
                d.rmdir()

    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts/train_tam_happo_direct.py"),
        "--config", "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "--output-dir", str(out_dir),
        "--total-env-steps", "1024",
        "--rollout-length", "128",
        "--num-envs", "4",
        "--max-steps", "1000",
        "--device", "cpu",
        "--policy-arch", "tam_categorical_recurrent",
        "--opponent-policy", "tam_direct_fsm",
        "--reward-mode", "tam_paper_reward_v1",
        "--tam-paper-mode",
        "--happo-update-granularity", "agent",
        "--advantage-mode", "per_agent_reward",
        "--actor-lr", "0.0005",
        "--critic-lr", "0.0005",
        "--entropy-coef", "0.01",
        "--clip-param", "0.2",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--max-grad-norm", "10.0",
        "--ppo-epochs", "2",
        "--eval-during-training",
        "--no-subprocess-eval",
        "--checkpoint-interval-steps", "512",
        "--keep-checkpoints", "3",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Training failed:\n{result.stderr[:1000]}"

    # Verify checkpoint was saved
    ckpt_dir = out_dir / "checkpoints" / "step_00000512"
    assert ckpt_dir.exists(), f"Expected {ckpt_dir}"
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "meta.json").exists()

    # Verify latest
    assert (out_dir / "latest" / "model.pt").exists()

    # Verify meta
    meta = json.loads((out_dir / "latest" / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_envs"] == 4
    assert meta["rollout_length"] == 128
    assert meta["transitions_per_rollout"] == 512  # 4*128
    assert meta["checkpoint_interval_steps"] == 512
    assert meta["subprocess_eval_enabled"] == False
    assert meta["multi_env_rollout_mode"] == "serial_env_batching"

    # Verify eval subprocess was NOT called (eval_log should be empty or not exist)
    eval_log = out_dir / "eval_log.csv"
    if eval_log.exists():
        content = eval_log.read_text()
        # Only header row (from eval-at-start) or empty
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert len(lines) <= 1, f"eval_log should be empty or header-only, got {len(lines)} lines"


def test_keep_checkpoints_limit():
    """keep-checkpoints=2 should prune step dirs."""
    import os
    out_dir = ROOT / ".tmp/test_keep_ckpt_limit"
    if out_dir.exists():
        for f in out_dir.rglob("*"):
            if f.is_file(): f.unlink()
        for d in sorted(out_dir.rglob("*"), reverse=True):
            if d.is_dir(): d.rmdir()

    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts/train_tam_happo_direct.py"),
        "--config", "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml",
        "--output-dir", str(out_dir),
        "--total-env-steps", "1280",
        "--rollout-length", "128",
        "--num-envs", "4",
        "--max-steps", "1000",
        "--device", "cpu",
        "--policy-arch", "tam_categorical_recurrent",
        "--opponent-policy", "tam_direct_fsm",
        "--reward-mode", "tam_paper_reward_v1",
        "--tam-paper-mode",
        "--happo-update-granularity", "agent",
        "--advantage-mode", "per_agent_reward",
        "--actor-lr", "0.0005", "--critic-lr", "0.0005",
        "--entropy-coef", "0.01", "--clip-param", "0.2",
        "--gamma", "0.99", "--gae-lambda", "0.95",
        "--max-grad-norm", "10.0", "--ppo-epochs", "2",
        "--eval-during-training",
        "--no-subprocess-eval",
        "--checkpoint-interval-steps", "256",
        "--keep-checkpoints", "2",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Training failed:\n{result.stderr[:1000]}"

    step_dirs = sorted([d for d in (out_dir / "checkpoints").glob("step_*") if d.is_dir()],
                       key=lambda d: d.name)
    assert len(step_dirs) <= 2, f"Should have <= 2 step dirs, got {len(step_dirs)}: {[d.name for d in step_dirs]}"
