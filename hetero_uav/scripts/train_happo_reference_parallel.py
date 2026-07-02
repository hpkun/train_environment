"""Train HAPPO reference with true multiprocessing rollout workers.

This runner keeps the policy, trainer, reward, environment dynamics, and action
space unchanged.  It only replaces the rollout collection layer with
process-isolated JSBSim environment workers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _WorkerTimeout(Exception):
    """Raised when a worker does not respond within the timeout window."""
    def __init__(self, env_idx: int, command: str, timeout_sec: float):
        self.env_idx = int(env_idx)
        self.command = str(command)
        self.timeout_sec = float(timeout_sec)
        super().__init__(
            f"[env_idx={env_idx}] {command} timed out after {timeout_sec:.1f}s"
        )


class _ConsecutiveWorkerTimeout(Exception):
    """Raised after repeated worker timeouts force emergency shutdown."""


_RUNNER_RESOURCES = {
    "vec_env": None,
    "rich_logger": None,
    "env_rich_loggers": [],
    "policy": None,
}

_RUNNER_PROGRESS = {
    "out_dir": None,
    "total_steps": 0,
    "latest_iteration": 0,
    "worker_restart_count": 0,
    "rollout_aborted_count": 0,
    "consecutive_rollout_abort_count": 0,
    "last_worker_timeout_info": {},
}


from algorithms.happo import HAPPORolloutBuffer, HAPPOReferenceTrainer
from algorithms.happo.rollout_safety import (
    zero_inactive_actions,
    zero_inactive_hidden,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from algorithms.pure_happo import PureHAPPOTrainer
from eval_checkpoint_selection import (
    best_metric_name,
    build_eval_checkpoint_meta,
)
from scripts.rich_logging import RichExperimentLogger, write_not_available_attention
from scripts.train_happo_reference import (
    DEFAULT_CONFIG,
    DEFAULT_EVAL_CONFIGS,
    _build_policy,
    _entity_policy_meta,
    _eval_checkpoint_extra,
    _load_uav_imitation_dataset,
    _reject_unsafe_random_scale_mask,
    _reject_unsafe_random_scale_mask_checkpoint,
    _rel,
    _run_eval,
    _sample_uav_imitation_batch,
    _save_policy_checkpoint,
    _score_eval,
    _team_done,
    _transitions_per_rollout,
    _prune_eval_checkpoints,
    _write_failure_artifacts,
    _pure_happo_meta,
)


def _worker_diag(env) -> dict:
    red_alive = sum(
        1 for rid in env.red_ids
        if env.red_planes.get(rid) is not None and env.red_planes[rid].is_alive
    )
    blue_alive = sum(
        1 for bid in env.blue_ids
        if env.blue_planes.get(bid) is not None and env.blue_planes[bid].is_alive
    )
    try:
        engaged_targets = env.refresh_engaged_targets()
    except Exception:
        engaged_targets = set()
    try:
        blue_positions = env.get_blue_own_positions()
    except Exception:
        blue_positions = {}
    try:
        blue_kinematics = env.get_blue_own_kinematics()
    except Exception:
        blue_kinematics = {}
    return {
        "red_alive": int(red_alive),
        "blue_alive": int(blue_alive),
        "mav_alive": bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive),
        "missile_count": int(len(getattr(env, "_missiles_in_flight", {}))),
        "sim_time": float(getattr(env, "current_time", 0.0)),
        "engaged_targets": list(engaged_targets or []),
        "blue_own_positions": blue_positions or {},
        "blue_own_kinematics": blue_kinematics or {},
    }


def _worker_meta(env) -> dict:
    role_ids = []
    for rid in env.red_ids:
        plane = env.red_planes.get(rid)
        role = getattr(plane, "role", "") or getattr(plane, "aircraft_type", "")
        role_ids.append(0 if "mav" in str(role).lower() or rid == "red_0" else 1)
    return {
        "red_ids": list(env.red_ids),
        "blue_ids": list(env.blue_ids),
        "agent_ids": list(env.agent_ids),
        "max_steps": int(getattr(env, "max_steps", 0)),
        "role_ids": role_ids,
    }


def _env_worker(remote, parent_remote, env_kwargs: dict) -> None:
    parent_remote.close()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env = None
    try:
        from uav_env import make_env

        env = make_env(
            env_kwargs["config"],
            env_type="jsbsim_hetero",
            hetero_reward_mode=env_kwargs["reward_mode"],
            max_steps=env_kwargs["max_steps"],
        )
        remote.send(("ready", _worker_meta(env), _worker_diag(env)))
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send(("ok", obs, info, _worker_diag(env)))
            elif cmd == "step":
                obs, rewards, terminated, truncated, info = env.step(data)
                remote.send(("ok", obs, rewards, terminated, truncated, info, _worker_diag(env)))
            elif cmd == "close":
                remote.send(("ok",))
                break
            else:
                remote.send(("error", f"unknown worker command: {cmd}"))
    except Exception as exc:
        try:
            remote.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except Exception:
            pass


class RemoteEnvProxy:
    def __init__(self, meta: dict, diag: dict | None = None):
        self.red_ids = list(meta["red_ids"])
        self.blue_ids = list(meta["blue_ids"])
        self.agent_ids = list(meta["agent_ids"])
        self.max_steps = int(meta.get("max_steps", 0))
        self.diag = diag or {}

    def update_diag(self, diag: dict) -> None:
        self.diag = diag or {}

    def refresh_engaged_targets(self):
        return set(self.diag.get("engaged_targets", []))

    def get_blue_own_positions(self):
        return self.diag.get("blue_own_positions", {}) or {}

    def get_blue_own_kinematics(self):
        return self.diag.get("blue_own_kinematics", {}) or {}


class ParallelEnv:
    def __init__(self, num_envs: int, env_kwargs: dict,
                 reset_timeout: float, step_timeout: float,
                 startup_delay: float = 0.5):
        self.num_envs = int(num_envs)
        self.env_kwargs = dict(env_kwargs)
        self.reset_timeout = float(reset_timeout)
        self.step_timeout = float(step_timeout)
        self.startup_delay = float(startup_delay)
        self.ctx = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        self.metas = []
        self.diags = []
        self.worker_restart_count = 0
        self.last_timeout_info: dict = {}
        for idx in range(self.num_envs):
            self._start_worker(idx)
            if idx < self.num_envs - 1 and self.startup_delay > 0:
                time.sleep(self.startup_delay)

    def _start_worker(self, idx: int) -> None:
        parent_remote, worker_remote = self.ctx.Pipe()
        proc = self.ctx.Process(
            target=_env_worker,
            args=(worker_remote, parent_remote, self.env_kwargs),
            daemon=True,
        )
        proc.start()
        worker_remote.close()
        if not parent_remote.poll(self.reset_timeout):
            proc.terminate()
            raise TimeoutError(f"worker {idx} did not initialize within {self.reset_timeout:.1f}s")
        status = parent_remote.recv()
        if status[0] != "ready":
            proc.terminate()
            raise RuntimeError(f"worker {idx} failed to initialize: {status}")
        if idx < len(self.remotes):
            self.remotes[idx] = parent_remote
            self.processes[idx] = proc
            self.metas[idx] = status[1]
            self.diags[idx] = status[2]
        else:
            self.remotes.append(parent_remote)
            self.processes.append(proc)
            self.metas.append(status[1])
            self.diags.append(status[2])

    def _restart_worker(self, idx: int) -> None:
        self.worker_restart_count += 1
        try:
            self.remotes[idx].close()
        except Exception:
            pass
        proc = self.processes[idx]
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        self._start_worker(idx)

    def _recv(self, idx: int, timeout: float, command: str):
        remote = self.remotes[idx]
        if remote.poll(timeout):
            msg = remote.recv()
            if msg[0] == "error":
                raise RuntimeError(f"worker {idx} {command} failed: {msg[1]}")
            return msg
        self._restart_worker(idx)
        exc = _WorkerTimeout(idx, command, timeout)
        self.last_timeout_info = {
            "env_idx": idx,
            "command": command,
            "timeout_sec": timeout,
        }
        raise exc

    def reset_all(self, seed: int):
        states = []
        for idx, remote in enumerate(self.remotes):
            remote.send(("reset", seed + idx))
        for idx in range(self.num_envs):
            msg = self._recv(idx, self.reset_timeout, "reset")
            _status, obs, info, diag = msg
            self.diags[idx] = diag
            states.append((obs, info, diag))
        return states

    def reset_one(self, idx: int, seed: int):
        self.remotes[idx].send(("reset", seed))
        msg = self._recv(idx, self.reset_timeout, "reset")
        _status, obs, info, diag = msg
        self.diags[idx] = diag
        return obs, info, diag

    def step_all(self, action_dicts: list[dict]):
        for remote, actions in zip(self.remotes, action_dicts):
            remote.send(("step", actions))
        results = []
        for idx in range(self.num_envs):
            msg = self._recv(idx, self.step_timeout, "step")
            _status, obs, rewards, terminated, truncated, info, diag = msg
            self.diags[idx] = diag
            results.append((obs, rewards, terminated, truncated, info, diag))
        return results

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                if remote.poll(2):
                    remote.recv()
            except Exception:
                pass
            try:
                remote.close()
            except Exception:
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def _episode_outcome_from_diag(diag: dict, truncated: dict, length: int, max_steps: int) -> dict:
    red_alive = int(diag.get("red_alive", 0))
    blue_alive = int(diag.get("blue_alive", 0))
    timeout = bool((truncated and all(truncated.values())) or length >= max_steps)
    if blue_alive == 0 and red_alive > 0:
        winner, reason, timeout_adv = "red", "blue_eliminated", ""
    elif red_alive == 0 and blue_alive > 0:
        winner, reason, timeout_adv = "blue", "red_eliminated", ""
    elif red_alive == 0 and blue_alive == 0:
        winner, reason, timeout_adv = "draw", "mutual_elimination", ""
    elif timeout:
        reason = "timeout"
        winner = "draw"
        if red_alive > blue_alive:
            timeout_adv = "red_timeout_alive_advantage"
        elif blue_alive > red_alive:
            timeout_adv = "blue_timeout_alive_advantage"
        else:
            timeout_adv = "timeout_draw"
    else:
        winner, reason, timeout_adv = "none", "ongoing", ""
    return {"winner": winner, "end_reason": reason,
            "timeout_alive_advantage": timeout_adv,
            "red_alive": red_alive, "blue_alive": blue_alive}


def _build_red_alive_mask_from_info(info: dict, proxy: RemoteEnvProxy) -> np.ndarray:
    mask = np.zeros(len(proxy.red_ids), dtype=np.float32)
    fallback_alive = int(proxy.diag.get("red_alive", 0))
    for i, rid in enumerate(proxy.red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            # Fallback is conservative when per-agent info is absent.  It is
            # only used for diagnostics/smoke paths; normal env info carries
            # per-agent alive fields.
            alive = i < fallback_alive
        mask[i] = 1.0 if alive else 0.0
    return mask


def _alive_agents_from_diag(diag: dict) -> dict:
    return {"red": int(diag.get("red_alive", 0)), "blue": int(diag.get("blue_alive", 0))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="outputs/happo_reference_parallel")
    parser.add_argument("--total-env-steps", type=int, default=1024)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-arch", default="flat",
                        choices=["flat", "entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent", "pure_happo", "pure_happo_tanh"])
    parser.add_argument("--brma-random-scale-mask", action="store_true")
    parser.add_argument("--brma-biased-mask", action="store_true")
    parser.add_argument("--brma-random-mask-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule", "brma_rule_safe_pursuit", "tam_greedy_easy", "brma_rule_safe_pursuit_easy"])
    parser.add_argument("--reward-mode", default=None,
                        help="Override YAML hetero_reward_mode. If omitted, uses YAML value.")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eval-during-training", action="store_true")
    parser.add_argument("--eval-interval-steps", type=int, default=25000)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--train-eval-episodes", type=int, default=1)
    parser.add_argument("--eval-configs", nargs="*", default=None)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--eval-checkpoint-metric", default="combined",
                        choices=["combined", "3v2", "5v4", "7v6"])
    parser.add_argument("--keep-eval-checkpoints", type=int, default=20)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--uav-imitation-dataset", default=None)
    parser.add_argument("--uav-imitation-coef", type=float, default=0.0)
    parser.add_argument("--uav-imitation-until-steps", type=int, default=0)
    parser.add_argument("--uav-imitation-batch-size", type=int, default=1024)
    parser.add_argument("--enable-rich-logging", action="store_true")
    parser.add_argument("--rich-log-dir", default=None)
    parser.add_argument("--reset-timeout-sec", type=float, default=300.0)
    parser.add_argument("--step-timeout-sec", type=float, default=120.0)
    parser.add_argument("--worker-startup-delay-sec", type=float, default=0.5)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0,
                        help="Save periodic checkpoint every N steps (0=disabled)")
    parser.add_argument("--keep-checkpoints", type=int, default=8,
                        help="Max periodic checkpoints to keep")
    return parser.parse_args()


def _write_runner_status(out_dir: Path, *, worker_restart_count: int,
                         rollout_aborted_count: int,
                         consecutive_rollout_abort_count: int,
                         last_worker_timeout_info: dict,
                         exit_reason: str, total_steps: int,
                         latest_iteration: int,
                         exception_type: str = "",
                         exception_message: str = "",
                         failure_checkpoint_saved: bool = False) -> None:
    """Write runner_status.json with structured exit information."""
    payload = {
        "status": "normal" if exit_reason == "normal" else "failed",
        "worker_restart_count": worker_restart_count,
        "rollout_aborted_count": rollout_aborted_count,
        "consecutive_rollout_abort_count": consecutive_rollout_abort_count,
        "last_worker_timeout_info": last_worker_timeout_info,
        "runner_completed_normally": exit_reason == "normal",
        "exit_reason": exit_reason,
        "total_env_steps_actual": total_steps,
        "latest_iteration": latest_iteration,
        "iteration": latest_iteration,
        "failed_step": total_steps if exit_reason != "normal" else None,
        "failed_episode_id": "",
        "output_dir": str(out_dir),
        "nan_detected": _exception_is_nonfinite_message(exception_message),
        "nonfinite_detected": _exception_is_nonfinite_message(exception_message),
        "failure_checkpoint_saved": bool(failure_checkpoint_saved),
    }
    if exception_type:
        payload["exception_type"] = exception_type
    if exception_message:
        payload["exception_message"] = exception_message
    try:
        (out_dir / "runner_status.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _exception_is_nonfinite_message(message: str) -> bool:
    text = str(message).lower().replace("-", "")
    return "nonfinite" in text or "nan" in text or "inf" in text


def _cleanup_runner(*, vec_env, rich_logger,
                    env_rich_loggers) -> None:
    """Close all runner resources; each step is independently fault-tolerant."""
    for logger in env_rich_loggers:
        try:
            logger.close()
        except Exception:
            pass
    if rich_logger is not None:
        try:
            rich_logger.close()
        except Exception:
            pass
    if vec_env is not None:
        try:
            vec_env.close()
        except Exception:
            pass


def _run_training(args: argparse.Namespace) -> None:
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter

    entity_mode = args.policy_arch == "hetero_entity_recurrent"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / args.output_dir
    (out_dir / "latest").mkdir(parents=True, exist_ok=True)
    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Experiment repro info
    try:
        import shutil
        shutil.copy(args.config, out_dir / "config_snapshot.yaml")
    except Exception:
        pass
    try:
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    except Exception:
        pass
    try:
        (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    except Exception:
        pass
    try:
        import subprocess
        git_info = {"commit": "", "dirty": ""}
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0: git_info["commit"] = r.stdout.strip()
        r = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0: git_info["dirty"] = "yes" if r.stdout.strip() else "clean"
        (out_dir / "git_info.json").write_text(json.dumps(git_info))
    except Exception:
        pass
    _RUNNER_PROGRESS.update({
        "out_dir": out_dir,
        "total_steps": 0,
        "latest_iteration": 0,
        "worker_restart_count": 0,
        "rollout_aborted_count": 0,
        "consecutive_rollout_abort_count": 0,
        "last_worker_timeout_info": {},
    })

    # Resolve reward_mode: CLI takes priority, but validate against YAML
    import yaml as _yaml
    _yaml_cfg = {}
    _yaml_reward = None
    try:
        with open(args.config, encoding="utf-8") as _f:
            _yaml_cfg = _yaml.safe_load(_f) or {}
        _yaml_reward = _yaml_cfg.get("hetero_reward_mode")
    except Exception:
        pass
    cli_reward_mode = args.reward_mode  # may be None
    if cli_reward_mode is None:
        if _yaml_reward is None:
            raise ValueError(
                "No --reward-mode provided and config YAML has no hetero_reward_mode. "
                "Please specify --reward-mode or fix the config."
            )
        resolved_reward_mode = _yaml_reward
    else:
        if _yaml_reward is not None and cli_reward_mode != _yaml_reward:
            raise ValueError(
                f"CLI --reward-mode={cli_reward_mode!r} conflicts with "
                f"config YAML hetero_reward_mode={_yaml_reward!r}. "
                f"Use matching values or omit --reward-mode to use the YAML value."
            )
        resolved_reward_mode = cli_reward_mode
    print(f"[reward-mode] resolved: {resolved_reward_mode}", flush=True)

    vec_env = ParallelEnv(
        args.num_envs,
        {
            "config": args.config,
            "reward_mode": resolved_reward_mode,
            "max_steps": args.max_steps,
        },
        reset_timeout=args.reset_timeout_sec,
        step_timeout=args.step_timeout_sec,
        startup_delay=args.worker_startup_delay_sec,
    )
    _RUNNER_RESOURCES["vec_env"] = vec_env
    proxies = [RemoteEnvProxy(meta, diag) for meta, diag in zip(vec_env.metas, vec_env.diags)]
    env = proxies[0]
    if entity_mode:
        adapter = HeteroEntitySetAdapter()
        actor_dim = 0
        critic_dim = 0
    else:
        adapter = HeteroObsAdapterV2()
        actor_dim = adapter.flat_actor_obs_dim
        critic_dim = adapter.critic_state_dim
    roles = vec_env.metas[0]["role_ids"]
    if any(meta["red_ids"] != env.red_ids or meta["blue_ids"] != env.blue_ids for meta in vec_env.metas):
        raise RuntimeError("parallel workers returned inconsistent agent ids")

    init_meta_path = None
    if args.init_checkpoint:
        init_path_for_meta = _rel(args.init_checkpoint)
        init_meta_path = init_path_for_meta.parent / "meta.json"
        if args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"} and not init_meta_path.exists():
            raise ValueError(
                f"{args.policy_arch} init checkpoint requires meta.json with policy_arch={args.policy_arch}"
            )
        # Validate init checkpoint consistency
        if init_meta_path.exists():
            init_meta = json.loads(init_meta_path.read_text(encoding="utf-8"))
            meta_pa = init_meta.get("policy_arch", "")
            meta_rm = init_meta.get("reward_mode", "")
            if meta_pa and meta_pa != args.policy_arch:
                raise ValueError(
                    f"init checkpoint policy_arch mismatch: "
                    f"checkpoint={meta_pa!r}, CLI={args.policy_arch!r}. "
                    f"Checkpoint: {init_meta_path}"
                )
            if meta_rm and meta_rm != resolved_reward_mode:
                raise ValueError(
                    f"init checkpoint reward_mode mismatch: "
                    f"checkpoint={meta_rm!r}, CLI={resolved_reward_mode!r}. "
                    f"Checkpoint: {init_meta_path}"
                )
        _reject_unsafe_random_scale_mask_checkpoint(args.policy_arch, init_meta_path)
    policy = _build_policy(
        args.policy_arch, actor_dim, critic_dim, device,
        init_checkpoint_meta=init_meta_path,
        brma_random_scale_mask=args.brma_random_scale_mask,
        brma_biased_mask=args.brma_biased_mask,
        brma_random_mask_prob=args.brma_random_mask_prob,
        num_agents=len(env.red_ids),
    )
    _RUNNER_RESOURCES["policy"] = policy
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        policy.load(init_path, map_location=device)
        print(f"Loaded init_checkpoint: {init_path}", flush=True)

    if args.policy_arch in {"pure_happo", "pure_happo_tanh"}:
        trainer = PureHAPPOTrainer(
            policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            seed=args.seed,
        )
    else:
        trainer = HAPPOReferenceTrainer(
            policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
        )
    uav_imitation_data = None
    if args.uav_imitation_dataset and args.uav_imitation_coef > 0.0:
        if args.policy_arch in {"pure_happo", "pure_happo_tanh"}:
            print("WARNING: pure_happo/pure_happo_tanh does not support UAV imitation. "
                  "Ignoring --uav-imitation-* flags.", flush=True)
        else:
            uav_imitation_data = _load_uav_imitation_dataset(args.uav_imitation_dataset)

    opponents = [OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17 + i)
                 for i in range(args.num_envs)]
    env_states = vec_env.reset_all(args.seed)
    obs_list = [state[0] for state in env_states]
    info_list = [state[1] for state in env_states]
    for proxy, state in zip(proxies, env_states):
        proxy.update_diag(state[2])

    transitions_per_rollout = _transitions_per_rollout(args.rollout_length, args.num_envs)

    rich_logger = None
    env_rich_loggers = []
    if args.enable_rich_logging:
        rich_dir = _rel(args.rich_log_dir) if args.rich_log_dir else out_dir / "rich_logs"
        rich_logger = RichExperimentLogger(
            rich_dir,
            run_id=out_dir.name,
            method_name="happo_reference_v0_parallel",
            scenario_name=Path(args.config).stem,
            device=str(args.device),
            num_envs=args.num_envs,
            rollout_length_per_env=args.rollout_length,
            transitions_per_rollout=transitions_per_rollout,
        )
        write_not_available_attention(rich_dir, "happo_reference_v0_parallel", Path(args.config).stem)
        _RUNNER_RESOURCES["rich_logger"] = rich_logger
        for idx in range(args.num_envs):
            env_dir = rich_dir / f"env_{idx:02d}"
            env_rich_loggers.append(RichExperimentLogger(
                env_dir,
                run_id=f"{out_dir.name}_env_{idx:02d}",
                method_name="happo_reference_v0_parallel",
                scenario_name=Path(args.config).stem,
                device=str(args.device),
                num_envs=1,
                rollout_length_per_env=args.rollout_length,
                transitions_per_rollout=args.rollout_length,
            ))
        _RUNNER_RESOURCES["env_rich_loggers"] = env_rich_loggers

    iterations = int(math.ceil(args.total_env_steps / transitions_per_rollout))
    total_steps = 0
    episodes = 0
    current_ep_return = [np.zeros(len(env.red_ids), dtype=np.float32) for _ in range(args.num_envs)]
    current_ep_len = [0 for _ in range(args.num_envs)]
    current_ep_id = [0 for _ in range(args.num_envs)]
    recent = deque(maxlen=100)
    prev_hit_totals = [{"red": 0, "blue": 0} for _ in range(args.num_envs)]
    # Episode-level aggregation (reward components, death reasons, missile stats)
    current_ep_reward_comp = [{} for _ in range(args.num_envs)]
    current_ep_red_death = [defaultdict(int) for _ in range(args.num_envs)]
    current_ep_blue_death = [defaultdict(int) for _ in range(args.num_envs)]
    current_ep_missile = [{"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}
                          for _ in range(args.num_envs)]
    nan_detected = False
    best_score = -float("inf")
    eval_best_scores = {"best_3v2": -float("inf"), "best_5v4": -float("inf"), "best_7v6": -float("inf"), "best_combined": -float("inf")}
    last_eval = -999999 if args.eval_at_start else 0
    next_checkpoint_step = args.checkpoint_interval_steps if args.checkpoint_interval_steps > 0 else None
    worker_restart_count = 0
    rollout_aborted_count = 0
    consecutive_rollout_abort_count = 0
    max_consecutive_rollout_abort = max(3, args.num_envs * 2)
    last_worker_timeout_info: dict = {}

    rnn_hidden = None
    if getattr(policy, "rnn_hidden_size", 0):
        rnn_hidden = np.zeros((args.num_envs, len(env.red_ids), policy.rnn_hidden_size), dtype=np.float32)

    train_log = out_dir / "train_log.csv"
    with train_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "total_steps", "avg_return", "red_win", "blue_win",
            "draw", "timeout", "mav_survival", "red_alive_final",
            "blue_alive_final", "red_missiles_fired", "blue_missiles_fired",
            "red_missile_hits", "blue_missile_hits", "actor_loss_mav", "actor_loss_uav",
            "critic_loss", "entropy_mav", "entropy_uav",
            "mav_action_saturation_rate", "uav_action_saturation_rate",
            "uav_imitation_loss", "entropy_mav_valid_count",
            "entropy_uav_valid_count", "mav_active_sample_count",
            "uav_active_sample_count", "action_log_std_mav_min",
            "action_log_std_mav_max", "action_log_std_mav_mean",
            "action_log_std_uav_min", "action_log_std_uav_max",
            "action_log_std_uav_mean", "approx_kl_mav", "approx_kl_uav",
            "mask_keep_ratio", "mask_entropy", "masked_entity_count",
            "worker_restart_count", "rollout_aborted_count",
            "episodes_completed", "recent_episode_count",
            "avg_episode_length",
            "red_elimination_win", "blue_elimination_win",
            "red_timeout_alive_advantage", "blue_timeout_alive_advantage",
            "mav_return_mean", "uav_return_mean",
            "red_death_missile_hit", "red_death_Crash_LowAlt",
            "red_death_Crash_OverG", "red_death_Crash_Extreme",
            "red_death_Crash_NonFiniteState", "red_death_unknown",
            "mav_death_count", "red_uav_death_count",
            "blue_death_missile_hit",
            "mav_safety_sum", "mav_support_sum", "mav_event_sum",
            "mav_death_reward_sum",
            "uav_attack_window_sum", "uav_fire_reward_sum",
            "uav_hit_reward_sum", "uav_low_speed_fire_reward_sum",
            "uav_death_reward_sum",
            "tam_v7_mav_flight_sum", "tam_v7_mav_safety_sum",
            "tam_v7_mav_support_sum", "tam_v7_mav_event_sum",
            "tam_v7_mav_terminal_sum", "tam_v7_mav_total_sum",
            "tam_v7_uav_flight_sum", "tam_v7_uav_situation_sum",
            "tam_v7_uav_event_sum", "tam_v7_uav_terminal_sum",
            "tam_v7_uav_total_sum", "tam_v7_total_sum",
            "tam_v7_uav_altitude_sum", "tam_v7_uav_speed_sum",
            "tam_v7_uav_boundary_sum", "tam_v7_mav_altitude_sum",
            "tam_v7_mav_speed_sum", "tam_v7_mav_boundary_sum",
            "red_episode_missiles_fired_mean", "red_episode_missile_hits_mean",
            "blue_episode_missiles_fired_mean", "blue_episode_missile_hits_mean",
            "nan_detected",
        ])
        eval_writer = None
        eval_f = None
        if args.eval_during_training:
            eval_f = (out_dir / "eval_log.csv").open("w", newline="", encoding="utf-8")
            eval_writer = csv.writer(eval_f)
            eval_writer.writerow([
                "total_steps", "iteration", "config", "red_win_rate",
                "blue_win_rate", "draw_rate", "timeout_rate",
                "red_elimination_win_rate", "red_timeout_alive_advantage_rate",
                "red_kill_fraction", "net_kill_fraction",
                "mav_survival_rate", "blue_dead_mean",
                "red_missile_hits_mean", "blue_missile_hits_mean",
            ])

        for iteration in range(1, iterations + 1):
            _RUNNER_PROGRESS["latest_iteration"] = iteration
            rollout_transitions = min(transitions_per_rollout, args.total_env_steps - total_steps)
            if rollout_transitions <= 0:
                break
            buffer_kwargs = {}
            if entity_mode:
                buffer_kwargs = {
                    "actor_token_count": len(env.red_ids) + len(env.blue_ids),
                    "critic_token_count": len(env.red_ids) + len(env.blue_ids),
                    "entity_dim": adapter.entity_dim,
                }
            buffer = HAPPORolloutBuffer(
                rollout_transitions, len(env.red_ids), actor_dim, critic_dim, 3,
                roles, rnn_hidden_size=getattr(policy, "rnn_hidden_size", 0),
                **buffer_kwargs,
            )
            red_fired = blue_fired = red_hits = blue_hits = 0
            rollout_aborted = False

            while len(buffer) < rollout_transitions and total_steps < args.total_env_steps:
                batch_records = []
                action_dicts = []
                for env_idx, proxy in enumerate(proxies):
                    if len(buffer) + len(batch_records) >= rollout_transitions or total_steps + len(batch_records) >= args.total_env_steps:
                        break
                    obs = obs_list[env_idx]
                    info = info_list[env_idx]
                    rollout_local_step = len(buffer) + len(batch_records)
                    adapted = adapter.adapt_all(
                        obs, info=info, red_ids=proxy.red_ids, blue_ids=proxy.blue_ids)
                    if entity_mode:
                        actor_obs = None
                        critic = None
                        actor_tokens = adapted["actor_entity_tokens"].copy()
                        actor_keep = adapted["actor_keep_mask"].copy()
                        critic_tokens = adapted["critic_entity_tokens"].copy()
                        critic_keep = adapted["critic_keep_mask"].copy()
                        critic_counts_t = adapted.get("critic_counts",
                                                      np.zeros(4, dtype=np.float32))
                    else:
                        actor_obs = np.stack([
                            adapted["actor_obs"].get(rid, np.zeros(actor_dim, dtype=np.float32))
                            for rid in proxy.red_ids
                        ])
                        critic = adapted["critic_state"]
                    active = _build_red_alive_mask_from_info(info, proxy)
                    active_rows = active > 0.5

                    # ---- Zero inactive hidden BEFORE act (align with single-proc) ----
                    if rnn_hidden is not None:
                        rnn_hidden[env_idx] = zero_inactive_hidden(rnn_hidden[env_idx], active)
                    # Save pre-action hidden AFTER inactive rows are zeroed
                    rnn_hidden_pre = None
                    act_kwargs = {}
                    if rnn_hidden is not None:
                        rnn_hidden_pre = rnn_hidden[env_idx].copy()
                        act_kwargs["rnn_hidden"] = torch.as_tensor(rnn_hidden[env_idx], device=device)

                    # ---- Sanitize entity inputs for inactive rows (entity_mode) ----
                    if entity_mode:
                        actor_tokens[~active_rows] = 0.0
                        actor_keep[~active_rows] = 0.0
                        actor_keep[~active_rows, 0] = 1.0

                    with torch.no_grad():
                        if entity_mode:
                            out = policy.act(
                                torch.as_tensor(actor_tokens, device=device),
                                torch.as_tensor(actor_keep, device=device),
                                torch.as_tensor(roles, device=device),
                                torch.as_tensor(critic_tokens, device=device),
                                torch.as_tensor(critic_keep, device=device),
                                deterministic=False,
                                critic_counts=torch.as_tensor(critic_counts_t, device=device),
                                **act_kwargs,
                            )
                        else:
                            out = policy.act(
                                torch.as_tensor(actor_obs, device=device),
                                roles=roles,
                                critic_state=torch.as_tensor(critic, device=device),
                                deterministic=False,
                                **act_kwargs,
                            )

                    # ---- Zero inactive actions (align with single-proc) ----
                    actions_raw = out["action"].cpu().numpy()
                    actions = zero_inactive_actions(actions_raw, active)
                    log_probs = out["log_prob"].cpu().numpy()
                    value = float(out["value"].item())

                    # ---- Zero inactive returned hidden ----
                    if rnn_hidden is not None and "rnn_hidden" in out:
                        rnn_post = out["rnn_hidden"].cpu().numpy()
                        rnn_hidden[env_idx] = zero_inactive_hidden(rnn_post, active)

                    # ---- Finite check: only active rows ----
                    if active_rows.any():
                        if not np.isfinite(actions[active_rows]).all():
                            nan_detected = True
                            break
                        if not np.isfinite(value):
                            nan_detected = True
                            break
                    action_dict = {rid: actions[i].astype(np.float32)
                                   for i, rid in enumerate(proxy.red_ids)}
                    action_dict.update(opponents[env_idx].act(obs, proxy.blue_ids, env=proxy))
                    action_dicts.append(action_dict)
                    batch_records.append({
                        "env_idx": env_idx,
                        "actor_obs": actor_obs,
                        "critic": critic,
                        "actions": actions,
                        "log_probs": log_probs,
                        "value": value,
                        "active": active,
                        "rnn_hidden_pre": rnn_hidden_pre,
                        "rollout_local_step": rollout_local_step,
                        "actor_tokens": actor_tokens if entity_mode else None,
                        "actor_keep": actor_keep if entity_mode else None,
                        "critic_tokens": critic_tokens if entity_mode else None,
                        "critic_keep": critic_keep if entity_mode else None,
                        "critic_counts": critic_counts_t if entity_mode else None,
                    })
                if nan_detected or not batch_records:
                    break

                try:
                    step_results = vec_env.step_all(action_dicts)
                except _WorkerTimeout as exc:
                    rollout_aborted = True
                    rollout_aborted_count += 1
                    consecutive_rollout_abort_count += 1
                    last_worker_timeout_info = {
                        "env_idx": exc.env_idx,
                        "command": exc.command,
                        "timeout_sec": exc.timeout_sec,
                        "iteration": iteration,
                        "total_steps": total_steps,
                    }
                    _RUNNER_PROGRESS.update({
                        "total_steps": total_steps,
                        "latest_iteration": iteration,
                        "worker_restart_count": getattr(vec_env, "worker_restart_count", 0),
                        "rollout_aborted_count": rollout_aborted_count,
                        "consecutive_rollout_abort_count": consecutive_rollout_abort_count,
                        "last_worker_timeout_info": last_worker_timeout_info,
                    })
                    print(
                        f"[happo-parallel] iter={iteration:04d} worker timeout during step; "
                        f"discarding rollout buffer, resetting all envs, continuing "
                        f"(consecutive aborts: {consecutive_rollout_abort_count})",
                        flush=True,
                    )
                    if consecutive_rollout_abort_count >= max_consecutive_rollout_abort:
                        print(
                            f"[happo-parallel] consecutive timeout limit "
                            f"({max_consecutive_rollout_abort}) reached; saving emergency "
                            "checkpoint and exiting",
                            flush=True,
                        )
                        _save_policy_checkpoint(policy, out_dir / "emergency", {
                            "algorithm": "happo_reference_v0",
                            "runner": "multiprocessing_parallel",
                            "policy_arch": args.policy_arch,
                            "total_steps": total_steps,
                            "iteration": iteration,
                            "reason": "consecutive_worker_timeout",
                            "consecutive_rollout_abort_count": consecutive_rollout_abort_count,
                            "last_worker_timeout_info": last_worker_timeout_info,
                        })
                        raise _ConsecutiveWorkerTimeout(
                            "consecutive worker timeout limit reached"
                        )
                    env_states = vec_env.reset_all(args.seed + total_steps)
                    obs_list[:] = [s[0] for s in env_states]
                    info_list[:] = [s[1] for s in env_states]
                    for proxy, state in zip(proxies, env_states):
                        proxy.update_diag(state[2])
                    if rnn_hidden is not None:
                        rnn_hidden[:] = 0.0
                    prev_hit_totals = [{"red": 0, "blue": 0} for _ in range(args.num_envs)]
                    for i in range(args.num_envs):
                        current_ep_return[i][:] = 0.0
                        current_ep_len[i] = 0
                    break  # restart while loop with fresh buffer
                for record, result in zip(batch_records, step_results):
                    env_idx = record["env_idx"]
                    proxy = proxies[env_idx]
                    next_obs, rewards, terminated, truncated, next_info, diag = result
                    proxy.update_diag(diag)
                    reward_np = np.array([float(rewards.get(rid, 0.0)) for rid in proxy.red_ids], dtype=np.float32)
                    done = _team_done(terminated, truncated)
                    done_np = np.full((len(proxy.red_ids),), float(done), dtype=np.float32)
                    if done:
                        next_value = 0.0
                    else:
                        next_adapted = adapter.adapt_all(
                            next_obs, info=next_info, red_ids=proxy.red_ids, blue_ids=proxy.blue_ids)
                        with torch.no_grad():
                            if entity_mode:
                                next_value = float(policy.value(
                                    next_adapted["critic_entity_tokens"],
                                    next_adapted["critic_keep_mask"],
                                    critic_counts=next_adapted.get("critic_counts"),
                                ).item())
                            else:
                                next_value = float(policy.value(
                                    torch.as_tensor(next_adapted["critic_state"], device=device).unsqueeze(0)
                                ).item())
                    store_kwargs = {}
                    if record["rnn_hidden_pre"] is not None:
                        store_kwargs["rnn_hidden"] = record["rnn_hidden_pre"]
                    if entity_mode:
                        store_kwargs.update({
                            "actor_entity_tokens": record["actor_tokens"],
                            "actor_keep_mask": record["actor_keep"],
                            "critic_entity_tokens": record["critic_tokens"],
                            "critic_keep_mask": record["critic_keep"],
                            "critic_counts": record["critic_counts"],
                        })
                    buffer.store(
                        record["actor_obs"], record["critic"], record["actions"],
                        record["log_probs"], reward_np, done_np, record["value"],
                        record["active"], next_value=next_value, env_id=env_idx,
                        **store_kwargs,
                    )
                    current_ep_return[env_idx] += reward_np
                    current_ep_len[env_idx] += 1
                    total_steps += 1
                    _RUNNER_PROGRESS["total_steps"] = total_steps
                    if env_rich_loggers:
                        env_rich_loggers[env_idx].write_missile_events(
                            next_info,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=diag.get("sim_time", 0.0),
                        )
                        env_rich_loggers[env_idx].write_reward_components(
                            next_info,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=diag.get("sim_time", 0.0),
                        )
                    for aid in proxy.agent_ids:
                        fired = int(next_info.get(aid, {}).get("missiles_fired_this_step", 0))
                        if aid.startswith("red_"):
                            red_fired += fired
                        else:
                            blue_fired += fired
                    mt = next_info.get("__missile_term__", {})
                    if isinstance(mt, dict):
                        red_hit_total = int(mt.get("red", {}).get("hit", 0))
                        blue_hit_total = int(mt.get("blue", {}).get("hit", 0))
                        # Compute delta ONCE, use for both rollout-level and episode-level
                        # (prev_hit_totals must NOT be updated between the two uses,
                        #  otherwise episode-level delta is always 0 after the first use).
                        delta_red = max(red_hit_total - prev_hit_totals[env_idx]["red"], 0)
                        delta_blue = max(blue_hit_total - prev_hit_totals[env_idx]["blue"], 0)
                        red_hits += delta_red
                        blue_hits += delta_blue
                        current_ep_missile[env_idx]["red_hits"] += delta_red
                        current_ep_missile[env_idx]["blue_hits"] += delta_blue
                        prev_hit_totals[env_idx]["red"] = red_hit_total
                        prev_hit_totals[env_idx]["blue"] = blue_hit_total
                    # Aggregate per-step reward components and death events
                    rc = next_info.get("reward_components", {}) if isinstance(next_info, dict) else {}
                    for aid in env.red_ids:
                        comp = rc.get(aid, {}) if isinstance(rc, dict) else {}
                        if isinstance(comp, dict):
                            for k, v in comp.items():
                                try:
                                    delta = float(v)
                                except (ValueError, TypeError):
                                    continue  # skip non-numeric fields (e.g. uav_reward_target_id)
                                current_ep_reward_comp[env_idx][k] = (
                                    current_ep_reward_comp[env_idx].get(k, 0.0) + delta)
                    for de in next_info.get("death_events", []) if isinstance(next_info, dict) else []:
                        if not isinstance(de, dict):
                            continue
                        reason = str(de.get("death_reason", "unknown"))
                        if str(de.get("agent_id", "")).startswith("red_"):
                            current_ep_red_death[env_idx][reason] += 1
                        else:
                            current_ep_blue_death[env_idx][reason] += 1
                    # Missile fired stats (hit deltas computed above with __missile_term__)
                    em = current_ep_missile[env_idx]
                    for aid, ai in (next_info or {}).items():
                        if isinstance(ai, dict):
                            f = int(ai.get("missiles_fired_this_step", 0))
                            if aid.startswith("red_"): em["red_fired"] += f
                            else: em["blue_fired"] += f

                    if done:
                        outcome = _episode_outcome_from_diag(
                            diag, truncated, current_ep_len[env_idx], proxy.max_steps)
                        mav_return = float(current_ep_return[env_idx][0])
                        uav_return = float(current_ep_return[env_idx][1:].mean()) if len(env.red_ids) > 1 else 0.0
                        recent.append({
                            "return": float(current_ep_return[env_idx].mean()),
                            "winner": outcome["winner"],
                            "end_reason": outcome["end_reason"],
                            "mav": bool(diag.get("mav_alive", False)),
                            "red_alive": int(diag.get("red_alive", 0)),
                            "blue_alive": int(diag.get("blue_alive", 0)),
                            "ep_len": current_ep_len[env_idx],
                            "return_mav": mav_return,
                            "return_uav": uav_return,
                            "reward_comp": dict(current_ep_reward_comp[env_idx]),
                            "red_death": dict(current_ep_red_death[env_idx]),
                            "blue_death": dict(current_ep_blue_death[env_idx]),
                            "missile": dict(current_ep_missile[env_idx]),
                        })
                        episodes += 1
                        current_ep_return[env_idx][:] = 0.0
                        current_ep_len[env_idx] = 0
                        current_ep_reward_comp[env_idx] = {}
                        current_ep_red_death[env_idx] = defaultdict(int)
                        current_ep_blue_death[env_idx] = defaultdict(int)
                        current_ep_missile[env_idx] = {"red_fired": 0, "blue_fired": 0,
                                                       "red_hits": 0, "blue_hits": 0}
                        next_obs, next_info, reset_diag = vec_env.reset_one(
                            env_idx, args.seed + total_steps + env_idx)
                        proxy.update_diag(reset_diag)
                        current_ep_id[env_idx] += 1
                        if rnn_hidden is not None:
                            rnn_hidden[env_idx][:] = 0.0
                        prev_hit_totals[env_idx] = {"red": 0, "blue": 0}
                    obs_list[env_idx] = next_obs
                    info_list[env_idx] = next_info

            if nan_detected:
                break
            if rollout_aborted:
                continue  # skip PPO update, restart for loop with new buffer

            # Successful rollout — reset consecutive abort counter
            consecutive_rollout_abort_count = 0
            _RUNNER_PROGRESS.update({
                "total_steps": total_steps,
                "latest_iteration": iteration,
                "worker_restart_count": getattr(vec_env, "worker_restart_count", 0),
                "rollout_aborted_count": rollout_aborted_count,
                "consecutive_rollout_abort_count": consecutive_rollout_abort_count,
                "last_worker_timeout_info": last_worker_timeout_info,
            })

            if args.policy_arch in {"pure_happo", "pure_happo_tanh"}:
                stats = trainer.update(buffer)
            else:
                imitation_batch = None
                imitation_active = (
                    uav_imitation_data is not None
                    and args.uav_imitation_coef > 0.0
                    and (args.uav_imitation_until_steps <= 0
                         or total_steps <= args.uav_imitation_until_steps)
                )
                if imitation_active:
                    imitation_batch = _sample_uav_imitation_batch(
                        uav_imitation_data, args.uav_imitation_batch_size, device)
                stats = trainer.update(
                    buffer,
                    uav_imitation_batch=imitation_batch,
                    uav_imitation_coef=args.uav_imitation_coef if imitation_active else 0.0,
                )
            rec = list(recent)
            n = max(len(rec), 1)
            avg_return = float(np.mean([r["return"] for r in rec])) if rec else 0.0
            red_win = sum(1 for r in rec if r["winner"] == "red") / n
            blue_win = sum(1 for r in rec if r["winner"] == "blue") / n
            draw = sum(1 for r in rec if r["winner"] == "draw") / n
            timeout = sum(1 for r in rec if r["end_reason"] == "timeout") / n
            mav_surv = sum(1 for r in rec if r["mav"]) / n
            red_alive = float(np.mean([r["red_alive"] for r in rec])) if rec else 0.0
            blue_alive = float(np.mean([r["blue_alive"] for r in rec])) if rec else 0.0
            # Extended metrics
            red_elim = sum(1 for r in rec if r["end_reason"] == "blue_eliminated") / n
            blue_elim = sum(1 for r in rec if r["end_reason"] == "red_eliminated") / n
            red_tout_adv = sum(1 for r in rec if r.get("timeout_alive_advantage") == "red_timeout_alive_advantage") / n
            blue_tout_adv = sum(1 for r in rec if r.get("timeout_alive_advantage") == "blue_timeout_alive_advantage") / n
            mav_return = float(np.mean([r.get("return_mav", 0) for r in rec])) if rec else 0.0
            uav_return = float(np.mean([r.get("return_uav", 0) for r in rec])) if rec else 0.0
            ep_len_mean = float(np.mean([r.get("ep_len", 0) for r in rec])) if rec else 0.0
            # Death reasons
            red_d = defaultdict(int)
            blue_d = defaultdict(int)
            for r in rec:
                for k, v in r.get("red_death", {}).items():
                    red_d[k] += v
                for k, v in r.get("blue_death", {}).items():
                    blue_d[k] += v
            mav_death = sum(r.get("red_death", {}).get("Crash_LowAlt", 0)
                           + r.get("red_death", {}).get("Crash_OverG", 0)
                           + r.get("red_death", {}).get("Crash_Extreme", 0)
                           + r.get("red_death", {}).get("Crash_NonFiniteState", 0)
                           + r.get("red_death", {}).get("missile_hit", 0)
                           + r.get("red_death", {}).get("unknown", 0) for r in rec)
            # Reward component sums
            rc_sum = defaultdict(float)
            for r in rec:
                for k, v in r.get("reward_comp", {}).items():
                    rc_sum[k] += v
            # Missile stats
            red_mf = float(np.mean([r.get("missile", {}).get("red_fired", 0) for r in rec])) if rec else 0.0
            red_mh = float(np.mean([r.get("missile", {}).get("red_hits", 0) for r in rec])) if rec else 0.0
            blue_mf = float(np.mean([r.get("missile", {}).get("blue_fired", 0) for r in rec])) if rec else 0.0
            blue_mh = float(np.mean([r.get("missile", {}).get("blue_hits", 0) for r in rec])) if rec else 0.0

            writer.writerow([
                iteration, total_steps, f"{avg_return:.4f}", f"{red_win:.4f}",
                f"{blue_win:.4f}", f"{draw:.4f}", f"{timeout:.4f}",
                f"{mav_surv:.4f}", f"{red_alive:.2f}", f"{blue_alive:.2f}",
                red_fired, blue_fired, red_hits, blue_hits, f"{stats['actor_loss_mav']:.6f}",
                f"{stats['actor_loss_uav']:.6f}", f"{stats['critic_loss']:.6f}",
                f"{stats['entropy_mav']:.6f}", f"{stats['entropy_uav']:.6f}",
                f"{stats['mav_action_saturation_rate']:.6f}",
                f"{stats['uav_action_saturation_rate']:.6f}",
                f"{stats.get('uav_imitation_loss', 0.0):.6f}",
                f"{stats.get('entropy_mav_valid_count', 0.0):.1f}",
                f"{stats.get('entropy_uav_valid_count', 0.0):.1f}",
                f"{stats.get('mav_active_sample_count', 0.0):.1f}",
                f"{stats.get('uav_active_sample_count', 0.0):.1f}",
                f"{stats.get('action_log_std_mav_min', 0.0):.6f}",
                f"{stats.get('action_log_std_mav_max', 0.0):.6f}",
                f"{stats.get('action_log_std_mav_mean', 0.0):.6f}",
                f"{stats.get('action_log_std_uav_min', 0.0):.6f}",
                f"{stats.get('action_log_std_uav_max', 0.0):.6f}",
                f"{stats.get('action_log_std_uav_mean', 0.0):.6f}",
                f"{stats.get('approx_kl_mav', 0.0):.6f}",
                f"{stats.get('approx_kl_uav', 0.0):.6f}",
                f"{stats.get('mask_keep_ratio', 1.0):.6f}",
                f"{stats.get('mask_entropy', 0.0):.6f}",
                f"{stats.get('masked_entity_count', 0.0):.2f}",
                getattr(vec_env, "worker_restart_count", 0), rollout_aborted_count,
                episodes, n,
                f"{ep_len_mean:.1f}",
                f"{red_elim:.4f}", f"{blue_elim:.4f}",
                f"{red_tout_adv:.4f}", f"{blue_tout_adv:.4f}",
                f"{mav_return:.4f}", f"{uav_return:.4f}",
                red_d.get("missile_hit", 0), red_d.get("Crash_LowAlt", 0),
                red_d.get("Crash_OverG", 0), red_d.get("Crash_Extreme", 0),
                red_d.get("Crash_NonFiniteState", 0), red_d.get("unknown", 0),
                mav_death, sum(red_d.values()) - mav_death,
                blue_d.get("missile_hit", 0),
                f"{rc_sum.get('mav_safety', 0):.4f}", f"{rc_sum.get('mav_support', 0):.4f}",
                f"{rc_sum.get('mav_event', 0):.4f}", f"{rc_sum.get('mav_death', 0):.4f}",
                f"{rc_sum.get('uav_attack_window', 0):.4f}", f"{rc_sum.get('uav_fire', 0):.4f}",
                f"{rc_sum.get('uav_hit', 0):.4f}", f"{rc_sum.get('uav_low_speed_fire', 0):.4f}",
                f"{rc_sum.get('uav_death', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_flight', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_safety', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_support', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_event', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_terminal', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_total', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_flight', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_situation', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_event', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_terminal', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_total', 0):.4f}",
                f"{rc_sum.get('tam_v7_total', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_altitude', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_speed', 0):.4f}",
                f"{rc_sum.get('tam_v7_uav_boundary', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_altitude', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_speed', 0):.4f}",
                f"{rc_sum.get('tam_v7_mav_boundary', 0):.4f}",
                f"{red_mf:.2f}", f"{red_mh:.2f}",
                f"{blue_mf:.2f}", f"{blue_mh:.2f}",
                int(nan_detected),
            ])
            if rich_logger is not None:
                red_dead = max(0.0, float(len(env.red_ids)) - red_alive)
                blue_dead = max(0.0, float(len(env.blue_ids)) - blue_alive)
                rich_logger.write_train_metrics({
                    "train_steps": iteration,
                    "total_env_steps_actual": total_steps,
                    "avg_episode_return": avg_return,
                    "avg_team_reward": avg_return,
                    "red_win_rate": red_win,
                    "blue_win_rate": blue_win,
                    "draw_rate": draw,
                    "timeout_rate": timeout,
                    "mav_survival_rate": mav_surv,
                    "red_alive_final_mean": red_alive,
                    "blue_alive_final_mean": blue_alive,
                    "red_missiles_fired_mean": red_fired / max(args.num_envs, 1),
                    "blue_missiles_fired_mean": blue_fired / max(args.num_envs, 1),
                    "red_missile_hits_mean": red_hits / max(args.num_envs, 1),
                    "actor_loss": (stats["actor_loss_mav"] + stats["actor_loss_uav"]) / 2.0,
                    "critic_loss": stats["critic_loss"],
                    "entropy": (stats["entropy_mav"] + stats["entropy_uav"]) / 2.0,
                    "mav_action_saturation_rate": stats["mav_action_saturation_rate"],
                    "uav_action_saturation_rate": stats["uav_action_saturation_rate"],
                    "approx_kl_mav": stats.get("approx_kl_mav", 0.0),
                    "approx_kl_uav": stats.get("approx_kl_uav", 0.0),
                    "mask_keep_ratio": stats.get("mask_keep_ratio", 1.0),
                    "mask_entropy": stats.get("mask_entropy", 0.0),
                    "masked_entity_count": stats.get("masked_entity_count", 0.0),
                    "nan_detected": int(nan_detected),
                })
            try:
                f.flush()
            except Exception:
                pass
            # Compact terminal summary (detailed data in train_log.csv)
            print(
                f"[happo-parallel] it={iteration:04d} step={total_steps}/{args.total_env_steps} "
                f"ret={avg_return:+.1f} win R/B/D={red_win:.2f}/{blue_win:.2f}/{draw:.2f} "
                f"timeout={timeout:.2f} mav={mav_surv:.2f}",
                flush=True,
            )

            # Periodic checkpoint (crossing trigger, not modulo)
            while next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
                ckpt_dir = out_dir / "checkpoints" / f"step_{next_checkpoint_step:07d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save(ckpt_dir / "model.pt")
                try:
                    with open(args.config, encoding="utf-8") as _f:
                        _cfg = __import__("yaml").safe_load(_f) or {}
                    _rtsm = _cfg.get("red_target_selection_mode", "")
                except Exception:
                    _rtsm = ""
                meta = {
                    "policy_arch": args.policy_arch, "reward_mode": resolved_reward_mode,
                    "config": args.config, "opponent_policy": args.opponent_policy,
                    "init_checkpoint": args.init_checkpoint,
                    "total_env_steps_actual": total_steps, "episodes": episodes,
                    "checkpoint_step": next_checkpoint_step,
                    "red_target_selection_mode": _rtsm,
                    "resolved_config_path": str(Path(args.config).resolve()),
                    "checkpoint_interval_steps": args.checkpoint_interval_steps,
                    "keep_checkpoints": args.keep_checkpoints,
                    "observation_adapter": "HeteroEntitySetAdapter" if entity_mode else "HeteroObsAdapterV2",
                    "entity_dim": getattr(policy, "entity_dim", None),
                    "num_envs": args.num_envs,
                    "rollout_length_per_env": args.rollout_length,
                    "transitions_per_rollout": transitions_per_rollout,
                    **_entity_policy_meta(policy),
                    **_pure_happo_meta(policy, args),
                }
                (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                # Also update latest
                policy.save(out_dir / "latest" / "model.pt")
                (out_dir / "latest" / "meta.json").write_text(json.dumps(meta, indent=2))
                next_checkpoint_step += args.checkpoint_interval_steps
            # Prune old periodic checkpoints (not latest, best, eval, emergency)
            existing = sorted(
                [d for d in (out_dir / "checkpoints").iterdir() if d.is_dir() and d.name.startswith("step_")],
                key=lambda d: d.name)
            for old in existing[:-args.keep_checkpoints]:
                import shutil
                shutil.rmtree(old, ignore_errors=True)

            if total_steps - last_eval >= args.eval_interval_steps and args.eval_during_training:
                last_eval = total_steps
                tmp_model = out_dir / "_tmp_eval.pt"
                policy.save(tmp_model)
                tmp_meta = {
                    "algorithm": "happo_reference_v0",
                    "runner": "multiprocessing_parallel",
                    "policy_arch": args.policy_arch,
                    "actor_obs_dim": actor_dim,
                    "critic_state_dim": critic_dim,
                    "entity_dim": getattr(policy, "entity_dim", 21),
                    "attention": args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "brma_entity_encoder": args.policy_arch in {"brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "recurrent": args.policy_arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "rnn_hidden_size": getattr(policy, "rnn_hidden_size", None),
                    "random_scale_mask": bool(getattr(policy, "random_scale_mask", False)),
                    "biased_mask": bool(getattr(policy, "biased_mask", False)),
                    "random_mask_prob": float(getattr(policy, "random_mask_prob", 0.0)),
                    **_entity_policy_meta(policy),
                    **_pure_happo_meta(policy, args),
                }
                (out_dir / "_tmp_eval_meta.json").write_text(json.dumps(tmp_meta, indent=2), encoding="utf-8")
                (out_dir / "meta.json").unlink(missing_ok=True)
                (tmp_model.parent / "meta.json").write_text(
                    (out_dir / "_tmp_eval_meta.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                records = _run_eval(str(tmp_model), args, str((out_dir / "_tmp_eval.json").relative_to(ROOT)))
                if records and eval_writer is not None:
                    for r in records:
                        eval_writer.writerow([
                            total_steps, iteration, r["config"], r["red_win_rate"],
                            r["blue_win_rate"], r["draw_rate"], r["timeout_rate"],
                            r.get("red_elimination_win_rate", 0.0),
                            r.get("red_timeout_alive_advantage_rate", 0.0),
                            r.get("red_kill_fraction", 0.0),
                            r.get("net_kill_fraction", 0.0),
                            r["mav_survival_rate"], r["blue_dead_mean"],
                            r["red_missile_hits_mean"], r.get("blue_missile_hits_mean", 0.0),
                        ])
                    eval_f.flush()
                    if args.save_eval_checkpoints:
                        meta = build_eval_checkpoint_meta(
                            step=total_steps,
                            iteration=iteration,
                            policy_arch=args.policy_arch,
                            records=records,
                            extra={
                                **_eval_checkpoint_extra(
                                    args, policy, actor_dim, critic_dim,
                                    transitions_per_rollout),
                                "runner": "multiprocessing_parallel",
                                "eval_checkpoint_metric": args.eval_checkpoint_metric,
                            },
                        )
                        eval_ckpt_dir = out_dir / "eval_checkpoints" / f"step_{total_steps:06d}"
                        _save_policy_checkpoint(policy, eval_ckpt_dir, meta)
                        _prune_eval_checkpoints(out_dir / "eval_checkpoints", args.keep_eval_checkpoints)
                        for best_name in ("best_3v2", "best_5v4", "best_7v6", "best_combined"):
                            metric_name = best_metric_name(best_name)
                            metric_score = float(meta["scores"].get(metric_name, 0.0))
                            if metric_score > eval_best_scores[best_name]:
                                eval_best_scores[best_name] = metric_score
                                best_meta = dict(meta)
                                best_meta["best_kind"] = best_name
                                best_meta["best_score"] = metric_score
                                best_meta["best_score_metric"] = metric_name
                                _save_policy_checkpoint(policy, out_dir / best_name, best_meta)
                    score = _score_eval(records)
                    if score > best_score:
                        best_score = score
                        _save_policy_checkpoint(policy, out_dir / "best", {
                            **tmp_meta,
                            "best_score": best_score,
                            "num_envs": args.num_envs,
                            "rollout_length_per_env": args.rollout_length,
                            "transitions_per_rollout": transitions_per_rollout,
                        })
                tmp_model.unlink(missing_ok=True)
                (out_dir / "_tmp_eval_meta.json").unlink(missing_ok=True)
                (out_dir / "meta.json").unlink(missing_ok=True)
                (out_dir / "_tmp_eval.json").unlink(missing_ok=True)

        if eval_f is not None:
            eval_f.close()

    latest_model = out_dir / "latest" / "model.pt"
    policy.save(latest_model)
    meta = {
        "algorithm": "happo_reference_v0",
        "runner": "multiprocessing_parallel",
        "policy_arch": args.policy_arch,
        "config": args.config,
        "reward_mode": resolved_reward_mode,
        "opponent_policy": args.opponent_policy,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": critic_dim,
        "entity_dim": getattr(policy, "entity_dim", None),
        "separate_actors": True,
        "centralized_critic": True,
        "sequential_update": True,
        "attention": args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "brma_entity_encoder": args.policy_arch in {"brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "recurrent": args.policy_arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "rnn_hidden_size": getattr(policy, "rnn_hidden_size", None),
        "random_scale_mask": bool(getattr(policy, "random_scale_mask", False)),
        "biased_mask": bool(getattr(policy, "biased_mask", False)),
        "random_mask_prob": float(getattr(policy, "random_mask_prob", 0.0)),
        "num_envs": args.num_envs,
        "rollout_length_per_env": args.rollout_length,
        "transitions_per_rollout": transitions_per_rollout,
        "init_checkpoint": args.init_checkpoint,
        "uav_imitation_dataset": args.uav_imitation_dataset,
        "uav_imitation_coef": args.uav_imitation_coef,
        "uav_imitation_until_steps": args.uav_imitation_until_steps,
        "uav_imitation_batch_size": args.uav_imitation_batch_size,
        "total_env_steps_actual": total_steps,
        "episodes": episodes,
        "nan_detected": nan_detected,
        "observation_adapter": "HeteroEntitySetAdapter" if entity_mode else "HeteroObsAdapterV2",
        "config_snapshot_path": str((out_dir / "config_snapshot.yaml").resolve()),
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "keep_checkpoints": args.keep_checkpoints,
        **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
    }
    try:
        with open(args.config, encoding="utf-8") as _f:
            _cfg = __import__("yaml").safe_load(_f) or {}
        meta["red_target_selection_mode"] = _cfg.get("red_target_selection_mode", "")
    except Exception:
        meta["red_target_selection_mode"] = ""
    (out_dir / "latest" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "main_experiment_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if rich_logger is not None:
        rich_logger.write_training_efficiency(total_steps, nan_detected=nan_detected)
    _RUNNER_PROGRESS.update({
        "total_steps": total_steps,
        "latest_iteration": iteration,
        "worker_restart_count": getattr(vec_env, "worker_restart_count", 0),
        "rollout_aborted_count": rollout_aborted_count,
        "consecutive_rollout_abort_count": consecutive_rollout_abort_count,
        "last_worker_timeout_info": last_worker_timeout_info,
    })
    print(f"Saved {latest_model}", flush=True)


def main() -> None:
    args = _parse_args()
    _reject_unsafe_random_scale_mask(args)
    out_dir = ROOT / args.output_dir
    _RUNNER_RESOURCES.update({
        "vec_env": None,
        "rich_logger": None,
        "env_rich_loggers": [],
        "policy": None,
    })
    _RUNNER_PROGRESS["out_dir"] = out_dir
    exit_reason = "normal"
    exception_type = ""
    exception_message = ""
    exit_code = 0
    try:
        _run_training(args)
    except _ConsecutiveWorkerTimeout as exc:
        exit_reason = "consecutive_worker_timeout"
        exception_type = type(exc).__name__
        exception_message = str(exc)
        exit_code = 1
    except KeyboardInterrupt as exc:
        exit_reason = "keyboard_interrupt"
        exception_type = type(exc).__name__
        exception_message = str(exc)
        exit_code = 130
    except Exception as exc:
        exit_reason = "exception"
        exception_type = type(exc).__name__
        exception_message = str(exc)
        exit_code = 1
    finally:
        status_out_dir = _RUNNER_PROGRESS.get("out_dir") or out_dir
        vec_env = _RUNNER_RESOURCES.get("vec_env")
        worker_restart_count = int(_RUNNER_PROGRESS.get("worker_restart_count", 0))
        if vec_env is not None:
            worker_restart_count = int(getattr(vec_env, "worker_restart_count", worker_restart_count))
        failure_checkpoint_saved = False
        if exit_reason != "normal" and _RUNNER_RESOURCES.get("policy") is not None:
            failure_state = {
                "output_dir": Path(status_out_dir),
                "total_steps": int(_RUNNER_PROGRESS.get("total_steps", 0)),
                "iteration": int(_RUNNER_PROGRESS.get("latest_iteration", 0)),
                "episode_id": "",
                "meta": {
                    "algorithm": "happo_reference_v0",
                    "policy_arch": args.policy_arch,
                    "config": args.config,
                    "num_envs": args.num_envs,
                },
            }
            failure_exc = RuntimeError(exception_message or exit_reason)
            try:
                _write_failure_artifacts(
                    _RUNNER_RESOURCES["policy"], failure_state, failure_exc
                )
            except Exception:
                pass
            failure_checkpoint_saved = (
                Path(status_out_dir) / "latest_failure" / "model.pt"
            ).exists()
        try:
            Path(status_out_dir).mkdir(parents=True, exist_ok=True)
            _write_runner_status(
                Path(status_out_dir),
                worker_restart_count=worker_restart_count,
                rollout_aborted_count=int(_RUNNER_PROGRESS.get("rollout_aborted_count", 0)),
                consecutive_rollout_abort_count=int(
                    _RUNNER_PROGRESS.get("consecutive_rollout_abort_count", 0)
                ),
                last_worker_timeout_info=dict(
                    _RUNNER_PROGRESS.get("last_worker_timeout_info", {})
                ),
                exit_reason=exit_reason,
                total_steps=int(_RUNNER_PROGRESS.get("total_steps", 0)),
                latest_iteration=int(_RUNNER_PROGRESS.get("latest_iteration", 0)),
                exception_type=exception_type,
                exception_message=exception_message,
                failure_checkpoint_saved=failure_checkpoint_saved,
            )
        finally:
            _cleanup_runner(
                vec_env=_RUNNER_RESOURCES.get("vec_env"),
                rich_logger=_RUNNER_RESOURCES.get("rich_logger"),
                env_rich_loggers=_RUNNER_RESOURCES.get("env_rich_loggers") or [],
            )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
