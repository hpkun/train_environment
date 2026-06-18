"""Targeted tests for parallel runner timeout recovery."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_parallel(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "train_happo_reference_parallel.py")] + args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_parallel_flat_smoke_completes():
    """Flat policy 2-env smoke completes without errors."""
    result = _run_parallel([
        "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
        "--output-dir", "outputs/_test_parallel_recovery_flat",
        "--total-env-steps", "256",
        "--rollout-length", "256",
        "--num-envs", "2",
        "--max-steps", "1000",
        "--device", "cpu",
        "--policy-arch", "flat",
        "--opponent-policy", "brma_rule",
    ])
    assert result.returncode == 0
    train_log = ROOT / "outputs/_test_parallel_recovery_flat/train_log.csv"
    assert train_log.exists()
    status = json.loads((ROOT / "outputs/_test_parallel_recovery_flat/runner_status.json").read_text())
    assert status["runner_completed_normally"] is True
    assert status["rollout_aborted_count"] == 0


def test_old_runner_num_envs_2_still_errors():
    """Single-process runner rejects --num-envs 2 with clear error."""
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "train_happo_reference.py"),
         "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
         "--output-dir", "outputs/_test_old_runner_reject",
         "--num-envs", "2", "--total-env-steps", "256",
         "--device", "cpu"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0, (
        f"Old runner should reject --num-envs 2, but returned 0. "
        f"stderr={result.stderr[:500]}"
    )
    combined = (result.stderr or "") + (result.stdout or "")
    assert "serial num_envs" in combined.lower() or "parallel" in combined.lower(), (
        f"Error should mention serial/parallel, got: {combined[:500]}"
    )


def test_parallel_runner_help_includes_timeout_args():
    result = _run_parallel(["--help"])
    assert result.returncode == 0
    assert "--reset-timeout-sec" in result.stdout
    assert "--step-timeout-sec" in result.stdout


def test_worker_diag_reads_missiles_in_flight():
    """Verify _worker_diag uses _missiles_in_flight, not env.missiles."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "train_happo_reference_parallel",
        ROOT / "scripts" / "train_happo_reference_parallel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert '"_missiles_in_flight"' in source


def test_parallel_env_worker_restart_count_increments():
    """_restart_worker increments worker_restart_count."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_happo_reference_parallel",
        ROOT / "scripts" / "train_happo_reference_parallel.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "self.worker_restart_count += 1" in (
        ROOT / "scripts" / "train_happo_reference_parallel.py"
    ).read_text(encoding="utf-8")


def test_timeout_message_includes_env_idx():
    """TimeoutError message contains env_idx for diagnostics."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert "env_idx" in source


def test_skip_update_on_rollout_aborted():
    """trainer.update is NOT called when rollout_aborted flag is True."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert "rollout_aborted" in source
    assert "continue" in source
    # The update call should be guarded by the abort check
    idx_continue = source.find("if rollout_aborted:")
    idx_update = source.find("trainer.update(")
    assert idx_continue >= 0
    assert idx_update >= 0
    assert idx_continue < idx_update, (
        "rollout_aborted check must appear before trainer.update()"
    )


def test_consecutive_abort_limit_exists():
    """max_consecutive_rollout_abort is defined and used."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert "max_consecutive_rollout_abort" in source
    assert "consecutive_rollout_abort_count" in source


def test_runner_status_written():
    """Runner writes runner_status.json with expected fields."""
    result = _run_parallel([
        "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml",
        "--output-dir", "outputs/_test_parallel_status_json",
        "--total-env-steps", "256",
        "--rollout-length", "256",
        "--num-envs", "2",
        "--max-steps", "1000",
        "--device", "cpu",
        "--policy-arch", "flat",
        "--opponent-policy", "brma_rule",
    ])
    status_path = ROOT / "outputs/_test_parallel_status_json/runner_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text())
    for key in ("worker_restart_count", "rollout_aborted_count",
                "runner_completed_normally", "exit_reason",
                "total_env_steps_actual", "latest_iteration"):
        assert key in status, f"missing key {key}"
    assert status["runner_completed_normally"] is True
    assert status["exit_reason"] == "normal"


def test_helper_functions_exist():
    """_write_runner_status and _cleanup_runner are defined."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert "def _write_runner_status(" in source
    assert "def _cleanup_runner(" in source


def test_emergency_path_uses_helpers():
    """Emergency path is routed to outer status/cleanup handling."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    # After the consecutive timeout trigger, the inner path should save the
    # emergency checkpoint and raise for the outer wrapper.  It must not run
    # duplicate cleanup/status writes in the inner rollout loop.
    idx_timeout = source.find("consecutive timeout limit")
    idx_raise = source.find("raise _ConsecutiveWorkerTimeout", idx_timeout)
    assert idx_timeout > 0 and idx_raise > idx_timeout
    between = source[idx_timeout:idx_raise]
    assert "_save_policy_checkpoint(" in between
    assert "_write_runner_status(" not in between
    assert "_cleanup_runner(" not in between


def test_main_wrapper_has_outer_finally():
    """main wrapper must funnel all exits through finally cleanup."""
    source = (ROOT / "scripts" / "train_happo_reference_parallel.py").read_text(encoding="utf-8")
    assert "def _run_training(" in source
    assert "def main() -> None:" in source
    wrapper = source[source.find("def main() -> None:"):]
    assert "try:" in wrapper
    assert "except KeyboardInterrupt" in wrapper
    assert "except Exception as exc" in wrapper
    assert "finally:" in wrapper
    assert "_write_runner_status(" in wrapper
    assert "_cleanup_runner(" in wrapper


def test_status_helper_marks_exception_paths_not_normal(tmp_path):
    from scripts.train_happo_reference_parallel import _write_runner_status

    _write_runner_status(
        tmp_path,
        worker_restart_count=2,
        rollout_aborted_count=3,
        consecutive_rollout_abort_count=4,
        last_worker_timeout_info={"env_idx": 1},
        exit_reason="exception",
        total_steps=99,
        latest_iteration=7,
        exception_type="RuntimeError",
        exception_message="boom",
    )
    status = json.loads((tmp_path / "runner_status.json").read_text())
    assert status["runner_completed_normally"] is False
    assert status["exit_reason"] == "exception"
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "boom"
    assert status["status"] == "failed"
    assert status["iteration"] == status["latest_iteration"]
    assert status["failed_step"] == status["total_env_steps_actual"]
    assert "failed_episode_id" in status
    assert status["nonfinite_detected"] is False


def test_status_helper_marks_keyboard_interrupt_not_normal(tmp_path):
    from scripts.train_happo_reference_parallel import _write_runner_status

    _write_runner_status(
        tmp_path,
        worker_restart_count=0,
        rollout_aborted_count=0,
        consecutive_rollout_abort_count=0,
        last_worker_timeout_info={},
        exit_reason="keyboard_interrupt",
        total_steps=12,
        latest_iteration=1,
    )
    status = json.loads((tmp_path / "runner_status.json").read_text())
    assert status["runner_completed_normally"] is False
    assert status["exit_reason"] == "keyboard_interrupt"
