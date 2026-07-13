"""Train minimal HAPPO reference v0 on the heterogeneous JSBSim env."""
from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import math
import multiprocessing as _mp
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _format_step_count(n: int) -> str:
    n = int(n)
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}m" if n % 1_000_000 else f"{int(v)}m"
    if n >= 1000:
        v = n / 1000
        return f"{v:.1f}k" if n % 1000 else f"{int(v)}k"
    return str(n)


def _format_happo_stdout_line(
    *,
    iteration: int,
    total_steps: int,
    target_steps: int,
    rec_count: int,
    avg_return: float,
    red_win: float,
    blue_win: float,
    draw: float,
    timeout: float,
    red_alive: float,
    blue_alive: float,
    mav_survival: float,
) -> str:
    step_text = (
        f"{_format_step_count(total_steps)}/"
        f"{_format_step_count(target_steps)}"
    )
    if rec_count <= 0:
        return (
            f"[happo] it={iteration:04d} step={step_text} ret=-- "
            f"win:R/B/D/T=-- alive:R/B/M=--"
        )
    return (
        f"[happo] it={iteration:04d} step={step_text} ret={avg_return:+.2f} "
        f"win:R/B/D/T={red_win:.2f}/{blue_win:.2f}/{draw:.2f}/{timeout:.2f} "
        f"alive:R/B/M={red_alive:.2f}/{blue_alive:.2f}/{mav_survival:.2f}"
    )


def _finalize_happo_v1_episode_reward_components(
    component_sums: dict,
    episode_length: int,
) -> dict:
    """Convert v1 per-step sums into episode-level means/rates."""
    out = {}
    for key, value in (component_sums or {}).items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue

    length = max(int(episode_length), 1)
    for key in ("mav_observed_ratio", "mav_shared_track_ratio"):
        raw = out.get(key, out.get(key + "_sum", 0.0))
        value = float(raw) / float(length)
        out[key] = value
        if key + "_sum" in out:
            out[key + "_sum"] = value

    before_launches = out.get(
        "red_launch_before_mav_death",
        out.get("red_launch_before_mav_death_sum", 0.0),
    )
    before_alive_steps = out.get(
        "red_uav_alive_steps_before_mav_death",
        out.get("red_uav_alive_steps_before_mav_death_sum", 0.0),
    )
    after_launches = out.get(
        "red_launch_after_mav_death",
        out.get("red_launch_after_mav_death_sum", 0.0),
    )
    after_alive_steps = out.get(
        "red_uav_alive_steps_after_mav_death",
        out.get("red_uav_alive_steps_after_mav_death_sum", 0.0),
    )
    before_rate = (
        float(before_launches) / float(before_alive_steps)
        if float(before_alive_steps) > 0.0 else 0.0
    )
    after_rate = (
        float(after_launches) / float(after_alive_steps)
        if float(after_alive_steps) > 0.0 else 0.0
    )
    out["red_launch_rate_before_mav_death"] = before_rate
    out["red_launch_rate_after_mav_death"] = after_rate
    if "red_launch_rate_before_mav_death_sum" in out:
        out["red_launch_rate_before_mav_death_sum"] = before_rate
    if "red_launch_rate_after_mav_death_sum" in out:
        out["red_launch_rate_after_mav_death_sum"] = after_rate
    return out


def _recent_component_mean(records: list[dict], key: str) -> float:
    values = []
    for rec in records:
        comp = rec.get("reward_comp", {})
        if key in comp:
            values.append(float(comp[key]))
    return float(np.mean(values)) if values else 0.0


def _make_env_kwargs_for_reward_mode(reward_mode: str | None) -> dict:
    if reward_mode is None:
        return {}
    return {"hetero_reward_mode": reward_mode}


def _config_declares_env_type(config_path: str | Path | None) -> bool:
    if not config_path:
        return False
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return "env_type" in cfg


def _env_type_override_kwargs(config_path: str | Path | None) -> dict:
    return {} if _config_declares_env_type(config_path) else {"env_type": "jsbsim_hetero"}


def _infer_env_action_dim(env) -> int:
    """Infer per-agent continuous action dimension from env.action_space."""

    env_action_dim = getattr(env, "action_dim", None)
    if env_action_dim is not None:
        return int(env_action_dim)
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return 3
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, dict):
        for rid in getattr(env, "red_ids", []):
            space = spaces.get(rid)
            shape = getattr(space, "shape", None)
            if shape:
                return int(np.prod(shape))
    shape = getattr(action_space, "shape", None)
    if shape:
        return int(np.prod(shape))
    return 3


def _finite_metric(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _format_metric(value) -> str:
    return f"{_finite_metric(value):.6f}"


def _array_distribution_stats(array: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(array)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_abs_max": 0.0,
            f"{prefix}_nan_count": 0.0,
        }
    nan_count = int((~np.isfinite(arr)).sum())
    finite = arr[np.isfinite(arr)].astype(np.float64)
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_abs_max": 0.0,
            f"{prefix}_nan_count": float(nan_count),
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std()),
        f"{prefix}_abs_max": float(np.max(np.abs(finite))),
        f"{prefix}_nan_count": float(nan_count),
    }


def _rollout_distribution_stats(buffer) -> dict:
    n = len(buffer)
    return {
        **_array_distribution_stats(buffer.actor_obs[:n], "actor_obs"),
        **_array_distribution_stats(buffer.critic_state[:n], "critic_state"),
    }


def _aggregate_scale_v2_step(red_ids, roles, active_before, reward_components) -> dict:
    """Return alive-before team means for one v2 decision step."""
    sums = defaultdict(float)
    active_count = 0.0
    for index, rid in enumerate(red_ids):
        weight = float(active_before[index]) if index < len(active_before) else 0.0
        if weight <= 0.0:
            continue
        comp = reward_components.get(rid, {}) if isinstance(reward_components, dict) else {}
        if not isinstance(comp, dict):
            continue
        active_count += weight
        role = roles.get(rid, "")
        prefix = "mav" if role == "mav" else "uav"
        sums[f"effective_scale_v2_{prefix}_flight_raw"] += weight * float(comp.get("scale_v2_flight_raw_total", 0.0) or 0.0)
        sums[f"effective_scale_v2_{prefix}_flight_scaled"] += weight * float(comp.get("scale_v2_flight_scaled_total", 0.0) or 0.0)
        if role == "mav":
            sums["effective_scale_v2_mav_role"] += weight * float(comp.get("scale_v2_mav_role", 0.0) or 0.0)
            sums["effective_scale_v2_mav_event"] += weight * float(comp.get("scale_v2_event", 0.0) or 0.0)
        else:
            sums["effective_scale_v2_uav_progress"] += weight * float(comp.get("scale_v2_progress", 0.0) or 0.0)
            sums["effective_scale_v2_uav_event"] += weight * float(comp.get("scale_v2_event", 0.0) or 0.0)
        sums["effective_scale_v2_terminal"] += weight * float(comp.get("scale_v2_terminal", 0.0) or 0.0)
        sums["effective_scale_v2_total"] += weight * float(comp.get("total", 0.0) or 0.0)
    denom = max(active_count, 1.0)
    keys = (
        "effective_scale_v2_uav_flight_raw", "effective_scale_v2_uav_flight_scaled",
        "effective_scale_v2_uav_progress", "effective_scale_v2_uav_event",
        "effective_scale_v2_mav_flight_raw", "effective_scale_v2_mav_flight_scaled",
        "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
        "effective_scale_v2_terminal", "effective_scale_v2_total",
    )
    result = {key: sums[key] / denom for key in keys}
    result["effective_scale_v2_component_sum"] = sum(result[key] for key in (
        "effective_scale_v2_uav_flight_scaled", "effective_scale_v2_uav_progress",
        "effective_scale_v2_uav_event", "effective_scale_v2_mav_flight_scaled",
        "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
        "effective_scale_v2_terminal",
    ))
    result["effective_scale_v2_identity_error"] = (
        result["effective_scale_v2_total"] - result["effective_scale_v2_component_sum"]
    )
    result["scale_v2_active_red_count"] = active_count
    return result


def _experiment_base_v2_meta(
    *,
    actual_reward_mode: str,
    rich_logging_enabled: bool,
    rich_log_mode: str,
    config_path: str | None = None,
) -> dict:
    full_rich = bool(rich_logging_enabled and rich_log_mode == "full")
    meta = {
        "reward_mode": actual_reward_mode,
        "actual_reward_mode": actual_reward_mode,
        "rich_logging_enabled": bool(rich_logging_enabled),
        "rich_log_mode": rich_log_mode,
        "per_step_reward_components": full_rich,
        "aircraft_timeseries": full_rich,
    }
    if actual_reward_mode == "brma_tam_scripted_composite_v1":
        meta["reward_contract_revision"] = 2
    if actual_reward_mode == "brma_tam_scale_aligned_v1":
        path = Path(config_path or "")
        if path and not path.is_absolute():
            path = ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        reward_cfg = dict((data or {}).get("brma_tam_scale_aligned_v1", {}) or {})
        meta.update({
            "reward_contract_revision": 3,
            "reward_config": reward_cfg,
            "progress_formula_version": "potential_difference_exp_distance_tam_angle_speed_v1",
            "terminal_formula_version": "loss_fraction_mav_weighted_v1",
            "mav_normalization": "aspect_and_awareness_divide_by_alive_blue_count",
            "decision_frequency_hz": float(data.get("sim_freq", 60)) / max(float(data.get("agent_interaction_steps", 12)), 1.0),
            "agent_interaction_steps": int(data.get("agent_interaction_steps", 12)),
            "max_steps": int(data.get("max_steps", 1000)),
            "initial_red_count": int(data.get("max_num_red", len(data.get("red_agent_types", [])))),
            "initial_blue_count": int(data.get("max_num_blue", len(data.get("blue_agent_types", [])))),
        })
    if actual_reward_mode == "brma_tam_scale_aligned_v2":
        path = Path(config_path or "")
        if path and not path.is_absolute():
            path = ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        reward_cfg = dict((data or {}).get("brma_tam_scale_aligned_v2", {}) or {})
        missile_protocol = dict((data or {}).get("missile_protocol", {}) or {})
        meta.update({
            "reward_contract_revision": 4,
            "flight_scale": float(reward_cfg.get("flight_scale", 0.1)),
            "reward_config": reward_cfg,
            "progress_formula_version": "potential_difference_exp_distance_tam_angle_speed_v1",
            "terminal_formula_version": "loss_fraction_mav_weighted_v1",
            "mav_normalization": "aspect_and_awareness_divide_by_alive_blue_count",
            "decision_frequency_hz": float(data.get("sim_freq", 60)) / max(float(data.get("agent_interaction_steps", 12)), 1.0),
            "agent_interaction_steps": int(data.get("agent_interaction_steps", 12)),
            "max_steps": int(data.get("max_steps", 1000)),
            "initial_red_count": int(data.get("max_num_red", len(data.get("red_agent_types", [])))),
            "initial_blue_count": int(data.get("max_num_blue", len(data.get("blue_agent_types", [])))),
            **missile_protocol,
        })
    return meta


# ---------------------------------------------------------------------------
#  Multiprocess env helpers — each env in its own process, pipe-based proxy
# ---------------------------------------------------------------------------
def _mp_env_worker(remote, parent_remote, config: str,
                   reward_mode: str | None, max_steps: int) -> None:
    """Child-process entry point: construct env, loop on pipe commands."""
    parent_remote.close()
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env = None
    try:
        from uav_env import make_env
        env = make_env(
            config,
            max_steps=max_steps,
            **_env_type_override_kwargs(config),
            **_make_env_kwargs_for_reward_mode(reward_mode),
        )
        meta = {
            "red_ids": list(env.red_ids),
            "blue_ids": list(env.blue_ids),
            "agent_ids": list(env.agent_ids),
            "max_steps": int(env.max_steps),
            "agent_roles": dict(env.agent_roles),
            "agent_types": dict(env.agent_types),
            "agent_models": dict(env.agent_models),
            "actual_reward_mode": getattr(env, "hetero_reward_mode", reward_mode),
            "action_dim": _infer_env_action_dim(env),
        }
        remote.send(("ready", meta))
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rewards, terminated, truncated, info = env.step(data)
                remote.send(("ok", obs, rewards, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send(("ok", obs, info))
            elif cmd == "call":
                method, args, kwargs = data
                result = getattr(env, method)(*args, **kwargs)
                remote.send(("ok", result))
            elif cmd == "close":
                remote.send(("ok",))
                break
    except Exception as exc:
        try:
            remote.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        try: remote.close()
        except Exception: pass


class _MpEnvProxy:
    """Thin proxy that looks like a real env to the rollout loop.

    Internal JSBSim attributes accessed by heartbeat / debug paths return
    safe fallback values so diagnostics don't block on pipe round-trips.
    """

    def __init__(self, meta: dict, remote):
        self.red_ids = list(meta["red_ids"])
        self.blue_ids = list(meta["blue_ids"])
        self.agent_ids = list(meta["agent_ids"])
        self.max_steps = int(meta["max_steps"])
        self.agent_roles = dict(meta.get("agent_roles", {}))
        self.agent_types = dict(meta.get("agent_types", {}))
        self.agent_models = dict(meta.get("agent_models", {}))
        self.hetero_reward_mode = meta.get("actual_reward_mode", "")
        self.action_dim = int(meta.get("action_dim", 3))
        self._remote = remote
        self.current_step = 0
        self.env_dt = 0.0
        # Alive caches — updated from step result info
        self._cached_red_alive = 0
        self._cached_blue_alive = 0

    # red_planes / blue_planes are accessed by _alive_counts().
    # Return simple objects with .is_alive so _alive_counts works.
    class _AliveSim:
        def __init__(self, alive: bool): self.is_alive = alive

    @property
    def red_planes(self):
        return {rid: self._AliveSim(i < self._cached_red_alive)
                for i, rid in enumerate(self.red_ids)}

    @property
    def blue_planes(self):
        return {bid: self._AliveSim(i < self._cached_blue_alive)
                for i, bid in enumerate(self.blue_ids)}

    def step(self, actions: dict):
        self._remote.send(("step", actions))
        msg = self._remote.recv()
        if msg[0] != "ok":
            raise RuntimeError(f"mp env step error: {msg}")
        _obs, _rewards, _terminated, _truncated, info = msg[1], msg[2], msg[3], msg[4], msg[5]
        self._cached_red_alive = sum(
            1 for rid in self.red_ids
            if info.get(rid, {}).get("alive", False))
        self._cached_blue_alive = sum(
            1 for bid in self.blue_ids
            if info.get(bid, {}).get("alive", False))
        return _obs, _rewards, _terminated, _truncated, info

    def reset(self, *, seed: int):
        self._remote.send(("reset", seed))
        msg = self._remote.recv()
        if msg[0] != "ok":
            raise RuntimeError(f"mp env reset error: {msg}")
        obs, info = msg[1], msg[2]
        self._cached_red_alive = len(self.red_ids)
        self._cached_blue_alive = len(self.blue_ids)
        return obs, info

    def refresh_engaged_targets(self):
        self._remote.send(("call", ("refresh_engaged_targets", (), {})))
        msg = self._remote.recv()
        return msg[1] if msg[0] == "ok" else set()

    def get_blue_own_positions(self):
        self._remote.send(("call", ("get_blue_own_positions", (), {})))
        msg = self._remote.recv()
        return msg[1] if msg[0] == "ok" else {}

    def get_blue_own_kinematics(self):
        self._remote.send(("call", ("get_blue_own_kinematics", (), {})))
        msg = self._remote.recv()
        return msg[1] if msg[0] == "ok" else ({}, {})

    def close(self):
        try:
            self._remote.send(("close", None))
            if self._remote.poll(2):
                self._remote.recv()
        except Exception:
            pass
        try:
            self._remote.close()
        except Exception:
            pass
        proc = getattr(self, "_proc", None)
        if proc is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def _create_mp_envs(num_envs: int, config: str, reward_mode: str,
                    max_steps: int):
    """Spawn ``num_envs`` worker processes, return list of ``_MpEnvProxy``."""
    ctx = _mp.get_context("spawn")
    proxies = []
    for i in range(num_envs):
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_mp_env_worker,
                           args=(child, parent, config, reward_mode, max_steps),
                           daemon=True)
        proc.start()
        child.close()
        if not parent.poll(300):
            proc.terminate()
            raise TimeoutError(f"mp env worker {i} startup timed out")
        msg = parent.recv()
        if msg[0] != "ready":
            proc.terminate()
            raise RuntimeError(f"mp env worker {i} init error: {msg}")
        proxy = _MpEnvProxy(msg[1], parent)
        proxy._proc = proc
        proxies.append(proxy)
        if i < num_envs - 1:
            time.sleep(0.5)
    return proxies

from algorithms.happo import (
    BRMAEntityHAPPOReferencePolicy,
    BRMARecurrentMaskedHAPPOReferencePolicy,
    BRMARecurrentHAPPOReferencePolicy,
    EntityHAPPOReferencePolicy,
    HAPPOReferencePolicy,
    HAPPORolloutBuffer,
    HAPPOReferenceTrainer,
)
from algorithms.pure_happo import PureHAPPOPolicy, PureHAPPOTanhPolicy, PureHAPPOTrainer
from algorithms.happo.rollout_safety import (
    sanitize_policy_inputs,
    zero_inactive_actions,
    zero_inactive_hidden,
)
from algorithms.mappo.opponent_policy import OpponentPolicy
from algorithms.mappo.tam_greedy_rule import (
    CANDIDATE_MANEUVERS as TAM_CANDIDATE_MANEUVERS,
    PROTOCOL_VERSION as TAM_RULE_PROTOCOL_VERSION,
    RULE_CLAIM as TAM_RULE_CLAIM,
    SCORE_WEIGHTS as TAM_SCORE_WEIGHTS,
)
from eval_checkpoint_selection import (
    best_metric_name,
    build_eval_checkpoint_meta,
    compute_eval_scores,
)
from scripts.experiment_logging_schema import (
    BRMA_TAM_SCRIPTED_COMPONENT_COLUMNS,
    BRMA_TAM_SCALE_V1_COMPONENT_COLUMNS,
    _SCALE_V1_EPISODE_LAST_FIELDS,
)
from uav_env.JSBSim.envs.role_situation_v3 import (
    V3_EFFECTIVE_FIELDS,
    V3_EPISODE_FIELDS,
    V3_REWARD_COMPONENT_FIELDS,
    accumulate_v3_episode_step,
    collect_v3_effective_samples,
)
from uav_env.JSBSim.envs.paper_calibrated_v4 import V4_COMPONENT_FIELDS
from scripts.rich_logging import RichExperimentLogger, write_not_available_attention


def _opponent_protocol_meta(opponent_policy: str) -> dict:
    if opponent_policy != "tam_greedy_rule":
        return {}
    return {
        "tam_rule_protocol_version": TAM_RULE_PROTOCOL_VERSION,
        "tam_candidate_maneuvers": list(TAM_CANDIDATE_MANEUVERS),
        "tam_score_weights": dict(TAM_SCORE_WEIGHTS),
        "tam_rule_claim": TAM_RULE_CLAIM,
    }


DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml"
DEFAULT_EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0_f22_pid.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f22_pid.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_7v6_f22_pid.yaml",
]

PURE_HAPPO_DEFAULTS = {
    "actor_lr": 5e-4,
    "critic_lr": 5e-4,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 10.0,
    "ppo_epochs": 5,
    "critic_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "train_eval_episodes": 10,
}

MARL_DYNAMICS_TRAIN_FIELDS = [
    "actor_loss_mean", "entropy_mean",
    "clip_fraction_mav", "clip_fraction_uav",
    "approx_kl_abs_mav", "approx_kl_abs_uav",
    "ratio_mean_mav", "ratio_mean_uav",
    "ratio_std_mav", "ratio_std_uav",
    "ratio_p95_mav", "ratio_p95_uav",
    "ratio_p99_mav", "ratio_p99_uav",
    "actor_grad_norm_mav", "actor_grad_norm_uav", "critic_grad_norm",
    "policy_update_norm_mav", "policy_update_norm_uav", "critic_update_norm",
    "critic_epochs", "critic_loss_unscaled", "critic_loss_scaled",
    "critic_loss_mean_over_epochs", "critic_loss_first_epoch", "critic_loss_last_epoch",
    "critic_grad_norm_mean_over_epochs", "critic_grad_norm_max_over_epochs",
    "value_explained_variance",
    "value_explained_variance_old", "value_explained_variance_new",
    "value_pred_mean", "value_pred_std",
    "value_pred_old_mean", "value_pred_old_std", "value_pred_new_mean", "value_pred_new_std",
    "return_mean", "return_std", "return_min", "return_max",
    "advantage_raw_mean", "advantage_raw_std", "advantage_raw_min", "advantage_raw_max",
    "advantage_norm_mean", "advantage_norm_std", "advantage_norm_min", "advantage_norm_max",
    "mav_action_mean_pitch", "mav_action_mean_heading", "mav_action_mean_speed",
    "uav_action_mean_pitch", "uav_action_mean_heading", "uav_action_mean_speed",
    "mav_action_std_pitch", "mav_action_std_heading", "mav_action_std_speed",
    "uav_action_std_pitch", "uav_action_std_heading", "uav_action_std_speed",
    "mav_action_mean_abs_pitch", "mav_action_mean_abs_heading", "mav_action_mean_abs_speed",
    "uav_action_mean_abs_pitch", "uav_action_mean_abs_heading", "uav_action_mean_abs_speed",
    "mav_action_mean_abs_pitch_active", "mav_action_mean_abs_heading_active", "mav_action_mean_abs_speed_active",
    "uav_action_mean_abs_pitch_active", "uav_action_mean_abs_heading_active", "uav_action_mean_abs_speed_active",
    "mav_action_saturation_pitch", "mav_action_saturation_heading", "mav_action_saturation_speed",
    "uav_action_saturation_pitch", "uav_action_saturation_heading", "uav_action_saturation_speed",
    "mav_action_mean_pitch_active", "mav_action_mean_heading_active", "mav_action_mean_speed_active",
    "uav_action_mean_pitch_active", "uav_action_mean_heading_active", "uav_action_mean_speed_active",
    "mav_action_std_pitch_active", "mav_action_std_heading_active", "mav_action_std_speed_active",
    "uav_action_std_pitch_active", "uav_action_std_heading_active", "uav_action_std_speed_active",
    "mav_action_saturation_pitch_active", "mav_action_saturation_heading_active", "mav_action_saturation_speed_active",
    "uav_action_saturation_pitch_active", "uav_action_saturation_heading_active", "uav_action_saturation_speed_active",
    "rollout_transitions", "ppo_epochs", "actor_lr", "critic_lr", "clip_param",
    "entropy_coef", "value_coef", "gamma", "gae_lambda", "max_grad_norm",
    "actor_obs_mean", "actor_obs_std", "actor_obs_abs_max", "actor_obs_nan_count",
    "critic_state_mean", "critic_state_std", "critic_state_abs_max", "critic_state_nan_count",
    "reward_nan_count", "action_nan_count", "value_nan_count",
    "log_prob_nan_count", "gradient_nonfinite_count",
    "avg_return_per_decision", "avg_episode_length",
    "attack_uav_reward_target_distance_mean", "reward_target_valid_ratio",
    "reward_distance_le_5km_ratio", "reward_distance_5_to_10km_ratio",
    "reward_distance_ge_10km_ratio", "uav_first_horizontal_oob_count",
    "mav_low_altitude_death_count", "reward_identity_max_error",
    "reward_identity_max_error_mav", "reward_identity_max_error_uav",
    "reward_identity_failure_count",
    "effective_scale_v1_uav_flight", "effective_scale_v1_uav_progress",
    "effective_scale_v1_uav_event", "effective_scale_v1_mav_flight",
    "effective_scale_v1_mav_role", "effective_scale_v1_mav_event",
    "effective_scale_v1_terminal", "effective_scale_v1_total",
    "effective_scale_v1_component_sum", "effective_scale_v1_identity_error",
    "reward_distance_10_to_20km_ratio", "reward_distance_ge_20km_ratio",
    "scale_v1_progress_positive_ratio", "scale_v1_progress_negative_ratio",
    "scale_v1_progress_zero_ratio", "scale_v1_progress_clip_ratio",
    "scale_v1_target_switch_count", "scale_v1_mav_aspect_raw_sum_mean",
    "scale_v1_mav_aspect_mean", "scale_v1_mav_awareness_raw_sum_mean",
    "scale_v1_mav_awareness_mean", "scale_v1_terminal_mean",
    "scale_v1_kill_count", "scale_v1_death_count", "scale_v1_oob_count",
    "effective_scale_v2_uav_flight_raw", "effective_scale_v2_uav_flight_scaled",
    "effective_scale_v2_uav_progress", "effective_scale_v2_uav_event",
    "effective_scale_v2_mav_flight_raw", "effective_scale_v2_mav_flight_scaled",
    "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
    "effective_scale_v2_terminal", "effective_scale_v2_total",
    "effective_scale_v2_component_sum", "effective_scale_v2_identity_error",
    "scale_v2_progress_positive_ratio", "scale_v2_progress_negative_ratio",
    "scale_v2_progress_zero_ratio", "scale_v2_progress_clip_ratio",
    "scale_v2_target_switch_count", "scale_v2_terminal_mean",
    "scale_v2_kill_count", "scale_v2_death_count", "scale_v2_oob_count",
    "scale_v2_flight_raw_mean", "scale_v2_flight_scaled_mean",
]

UPDATE_DIAGNOSTIC_ARRAY_FIELDS = [
    "actor_loss_per_agent", "entropy_per_agent", "approx_kl_per_agent",
    "approx_kl_abs_per_agent", "clip_fraction_per_agent",
    "ratio_mean_per_agent", "ratio_std_per_agent", "ratio_p95_per_agent",
    "ratio_p99_per_agent", "actor_grad_norm_per_agent",
    "policy_update_norm_per_agent", "valid_sample_count_per_agent",
    "active_sample_ratio_per_agent", "last_update_order",
    "m_mean_after_each_agent", "m_std_after_each_agent",
    "m_abs_mean_after_each_agent", "m_abs_max_after_each_agent",
    "critic_loss_per_epoch", "critic_grad_norm_per_epoch",
]


def _entity_policy_meta(policy) -> dict:
    if policy.__class__.__name__ != "HeteroEntityRecurrentPolicy":
        return {}
    return {
        "feature_schema_version": "hetero_entity_set_v2",
        "adapter_mode": "hetero_entity_set",
        "actor_obs_format": "entity_tokens_keep_mask",
        "critic_obs_format": "global_entity_tokens_keep_mask",
        "entity_dim": policy.entity_dim,
        "role_dim": 4,
        "role_vocab": ["mav", "attack_uav", "scout_uav", "interceptor_uav"],
        "action_dim": policy.action_dim,
        "rnn_hidden_size": policy.rnn_hidden_size,
        "hidden_dim": policy.hidden_dim,
        "num_attention_heads": policy.num_attention_heads,
        "policy_arch": "hetero_entity_recurrent",
        "actor_arch": "entity_attention_grucell_role_heads",
        "critic_arch": "global_entity_attention_value_v2",
        "scale_support_mode": "variable_token_count",
        "padding_mode": "keep_mask",
        "policy_class": policy.__class__.__name__,
        "critic_class": policy.critic.__class__.__name__,
        "observation_adapter": "HeteroEntitySetAdapter",
        "hetero_entity_recurrent_full_geometry": False,
        "full_geometry_path": "unsupported_hetero_entity_set_adapter",
        "full_geometry_features_used": False,
    }


def _full_geometry_meta(policy_arch: str, entity_dim, args=None) -> dict:
    """Compute full-geometry wiring meta fields."""
    enemy_flat_dim = 18
    entity_dim_required = 30
    full_geo_extra = enemy_flat_dim - 7  # 11

    meta = {
        "observation_mode": "mav_shared_geo",
        "mav_shared_full_geometry": True,
        "enemy_flat_dim": enemy_flat_dim,
        "enemy_entity_dim": enemy_flat_dim,
        "entity_dim_required_for_full_geometry": entity_dim_required,
        "full_geometry_extra_dim": full_geo_extra,
    }

    if policy_arch in ("pure_happo", "pure_happo_tanh", "flat"):
        meta.update({
            "full_geometry_features_used": True,
            "full_geometry_path": "flat_actor_obs",
            "full_geometry_checkpoint_compatible": True,
            "hetero_entity_recurrent_full_geometry": False,
            "entity_dim": "n/a (flat policy)",
        })
    elif policy_arch in ("entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked"):
        ed = int(entity_dim) if isinstance(entity_dim, (int, float)) else 30
        uses_fg = ed >= entity_dim_required
        meta.update({
            "entity_dim": ed,
            "full_geometry_features_used": uses_fg,
            "full_geometry_path": "flat_to_entity_token_19_30",
            "full_geometry_checkpoint_compatible": True if uses_fg else False,
            "hetero_entity_recurrent_full_geometry": False,
        })
        if not uses_fg:
            meta["full_geometry_checkpoint_reason"] = (
                f"old checkpoint entity_dim={ed} truncates full-geometry "
                f"(need >= {entity_dim_required})"
            )
    elif policy_arch == "hetero_entity_recurrent":
        meta.update({
            "entity_dim": int(entity_dim) if isinstance(entity_dim, (int, float)) else 21,
            "full_geometry_features_used": False,
            "full_geometry_path": "unsupported_hetero_entity_set_adapter",
            "full_geometry_checkpoint_compatible": False,
            "hetero_entity_recurrent_full_geometry": False,
        })
    return meta


def _pure_happo_meta(policy, args=None) -> dict:
    """Extra meta fields for paper-aligned pure HAPPO baseline."""
    cls_name = getattr(getattr(policy, "__class__", None), "__name__", "")
    if cls_name not in {"PureHAPPOPolicy", "PureHAPPOTanhPolicy",
                        "LegacyClampPureHAPPOPolicy"}:
        return {}
    meta = {
        "num_agents": int(policy.num_agents),
        "paper_aligned_happo": True,
        "parameter_sharing": False,
        "per_agent_independent_actors": True,
        "global_v_critic": True,
        "sequential_correction_factor": True,
        "happo_update_unit": "agent",
        "policy_arch": "pure_happo",
        "bounded_action_distribution": "tanh_squashed_gaussian",
        "action_distribution_type": "tanh_squashed_gaussian",
        "logprob_correction": "tanh_jacobian",
        "attention": False,
        "recurrent": False,
        "random_scale_mask": False,
        "biased_mask": False,
        "entity_encoder": False,
        "mask_generator": False,
        "separate_actors": True,
        "centralized_critic": True,
        "sequential_update": True,
        "advantage_normalization": True,
        "optimizer": "Adam",
        "actor_hidden_sizes": [256, 128],
        "critic_hidden_sizes": [256, 128],
        "initial_action_log_std": float(getattr(
            policy,
            "initial_action_log_std",
            policy.action_log_stds[0].detach().mean().item(),
        )),
    }
    if args is not None:
        for key in (
            "actor_lr", "critic_lr", "clip_param", "entropy_coef", "value_coef",
            "max_grad_norm", "ppo_epochs", "critic_epochs", "gamma", "gae_lambda",
            "rollout_length", "num_envs", "max_steps", "seed", "opponent_policy",
        ):
            meta[key] = getattr(args, key)
        meta["transitions_per_rollout"] = int(args.rollout_length) * int(args.num_envs)
    return meta


def _apply_policy_specific_defaults(args) -> None:
    pure = str(args.policy_arch) == "pure_happo"
    legacy = {
        "actor_lr": 2e-4, "critic_lr": 5e-4, "clip_param": 0.2,
        "entropy_coef": 0.02, "value_coef": 0.5, "max_grad_norm": 10.0,
        "ppo_epochs": 2, "critic_epochs": 1, "gamma": 0.99,
        "gae_lambda": 0.95, "train_eval_episodes": 1,
    }
    selected = PURE_HAPPO_DEFAULTS if pure else legacy
    for key, value in selected.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def _resolve_pure_happo_eval_configs(args, policy) -> list[str]:
    configs = list(args.eval_configs) if args.eval_configs else [str(args.config)]
    train_path = Path(args.config).resolve()
    if len(configs) != 1 or Path(configs[0]).resolve() != train_path:
        raise ValueError(
            "pure_happo fixed-agent evaluation requires exactly the training 3V2 config; "
            "5V4/other scale evaluation is not supported"
        )
    cfg_path = ROOT / configs[0] if not Path(configs[0]).is_absolute() else Path(configs[0])
    with cfg_path.open(encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}
    eval_num_red = int(config_data.get("max_num_red", -1))
    if eval_num_red != int(policy.num_agents):
        raise ValueError(
            f"pure_happo policy has {policy.num_agents} fixed actors but eval config has {eval_num_red} red agents"
        )
    return configs


_SINGLE_RUNNER_STATE = {
    "policy": None,
    "output_dir": None,
    "total_steps": 0,
    "iteration": 0,
    "episode_id": 0,
    "meta": {},
    "envs": [],
    "heartbeat": None,
    "watchdog": None,
    "rich_logger": None,
}

UNSAFE_RANDOM_SCALE_MASK_ERROR = (
    "brma_random_scale_mask is disabled for main training. The current "
    "random_scale_mask path resamples entity masks independently during "
    "rollout policy.act and PPO update evaluate_actions, which breaks "
    "old/new log_prob alignment. Use the no-random-mask "
    "brma_recurrent_masked path for current main experiments; re-enable this "
    "only after implementing rollout mask replay or the full BRMA biased "
    "mask objective."
)


def _reject_unsafe_random_scale_mask(args) -> None:
    if getattr(args, "brma_random_scale_mask", False):
        raise SystemExit(UNSAFE_RANDOM_SCALE_MASK_ERROR)


UNSAFE_RANDOM_SCALE_MASK_CHECKPOINT_ERROR = (
    "unsafe random_scale_mask checkpoint is disabled for main training. This "
    "init checkpoint was saved with random_scale_mask=true; the current main "
    "training path rejects it because rollout and PPO update masks were not "
    "replayed. Use it only for diagnostic eval, or re-enable training after "
    "implementing rollout mask replay or the full BRMA biased mask objective."
)


def _reject_unsafe_random_scale_mask_checkpoint(policy_arch: str, init_checkpoint_meta: str | Path | None) -> None:
    if policy_arch != "brma_recurrent_masked" or init_checkpoint_meta is None:
        return
    meta_path = Path(init_checkpoint_meta)
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if bool(meta.get("random_scale_mask", False)):
        raise SystemExit(UNSAFE_RANDOM_SCALE_MASK_CHECKPOINT_ERROR)


def _transitions_per_rollout(rollout_length: int, num_envs: int) -> int:
    return int(rollout_length) * int(num_envs)


class HeartbeatLogger:
    def __init__(self, path: str | Path | None, every_steps: int = 50,
                 enabled: bool = False, debug_all: bool = False,
                 static_fields: dict | None = None) -> None:
        self.enabled = bool(enabled)
        self.every_steps = max(1, int(every_steps))
        self.debug_all = bool(debug_all)
        self.path = Path(path) if path is not None else None
        self.static_fields = static_fields or {}
        self.last_write_time = time.time()
        self.last_entries: deque[str] = deque(maxlen=20)
        self.last_event: dict = {}
        self._lock = threading.Lock()
        self._file = None
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: str, *, iteration: int, rollout_local_step: int,
              env_idx: int | str, total_steps: int, episode_length: int | str = "",
              alive_agents=None, done=None, terminated=None, truncated=None,
              env_episode_id: int | str = "", missile_count: int | str = "",
              sim_time: float | str = "") -> None:
        if not self.enabled or self._file is None:
            return
        if not self.debug_all and int(total_steps) % self.every_steps != 0:
            return
        alive_text = ""
        if isinstance(alive_agents, dict):
            alive_text = ",".join(f"{k}:{v}" for k, v in sorted(alive_agents.items()))
        fields = {
            "wall_time": f"{time.time():.3f}",
            "iteration": iteration,
            "rollout_local_step": rollout_local_step,
            "env_idx": env_idx,
            "event": event,
            "total_env_steps_actual": total_steps,
            "env_episode_step": episode_length,
            "env_episode_id": env_episode_id,
            "alive_agents": alive_text,
            "red_alive_count": alive_agents.get("red", "") if isinstance(alive_agents, dict) else "",
            "blue_alive_count": alive_agents.get("blue", "") if isinstance(alive_agents, dict) else "",
            "missile_count": missile_count,
            "sim_time": sim_time,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            **self.static_fields,
        }
        row = " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"
        with self._lock:
            self._file.write(row)
            self._file.flush()
            self.last_write_time = time.time()
            self.last_entries.append(row.rstrip("\n"))
            self.last_event = fields.copy()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class HeartbeatStallWatchdog:
    def __init__(self, logger: HeartbeatLogger, output_dir: Path,
                 timeout_sec: float, exit_on_stall: bool = False) -> None:
        self.logger = logger
        self.output_dir = output_dir
        self.timeout_sec = float(timeout_sec)
        self.exit_on_stall = bool(exit_on_stall)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="heartbeat-stall-watchdog", daemon=True)
        self.triggered = False

    def start(self) -> None:
        if self.timeout_sec > 0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(min(5.0, max(0.5, self.timeout_sec / 10.0))):
            elapsed = time.time() - self.logger.last_write_time
            if elapsed < self.timeout_sec:
                continue
            self.triggered = True
            self._write_report(elapsed)
            if self.exit_on_stall:
                os._exit(88)
            return

    def _write_report(self, elapsed: float) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.logger._lock:
            last_entries = list(self.logger.last_entries)
            last_event = dict(self.logger.last_event)
        report = {
            "stall_timeout_sec": self.timeout_sec,
            "elapsed_since_last_heartbeat_sec": elapsed,
            "last_event": last_event,
            "last_heartbeat_entries": last_entries,
            "exit_on_heartbeat_stall": self.exit_on_stall,
        }
        (self.output_dir / "heartbeat_stall_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        md = [
            "# Heartbeat Stall Report",
            "",
            f"- timeout_sec: {self.timeout_sec}",
            f"- elapsed_since_last_heartbeat_sec: {elapsed:.1f}",
            f"- last_event: `{last_event.get('event', 'unknown')}`",
            f"- env_idx: `{last_event.get('env_idx', '')}`",
            f"- rollout_local_step: `{last_event.get('rollout_local_step', '')}`",
            "",
            "## Last Heartbeats",
            "",
            "```text",
            *last_entries,
            "```",
        ]
        (self.output_dir / "heartbeat_stall_report.md").write_text("\n".join(md), encoding="utf-8")
        with (self.output_dir / "heartbeat_stall_stack.txt").open("w", encoding="utf-8") as f:
            faulthandler.dump_traceback(file=f, all_threads=True)


def _build_red_alive_mask(info: dict, env, red_ids: list[str]) -> np.ndarray:
    mask = np.zeros(len(red_ids), dtype=np.float32)
    for i, rid in enumerate(red_ids):
        agent_info = info.get(rid, {}) if isinstance(info, dict) else {}
        if isinstance(agent_info, dict) and "alive" in agent_info:
            alive = bool(agent_info["alive"])
        else:
            sim = env.red_planes.get(rid)
            alive = bool(sim is not None and sim.is_alive)
        mask[i] = 1.0 if alive else 0.0
    return mask


def _team_done(terminated: dict, truncated: dict) -> bool:
    return bool(all(terminated.values()) or all(truncated.values()))


def _alive_counts(env) -> tuple[int, int]:
    return (
        sum(1 for sim in env.red_planes.values() if sim.is_alive),
        sum(1 for sim in env.blue_planes.values() if sim.is_alive),
    )


def _mav_alive(env) -> bool:
    sim = env.red_planes.get("red_0")
    return bool(sim is not None and sim.is_alive)


def _missile_count(env) -> int:
    inflight = getattr(env, "_missiles_in_flight", None)
    if inflight is not None:
        return len(inflight)
    missiles = getattr(env, "missiles", None)
    if isinstance(missiles, list):
        return len(missiles)
    if isinstance(missiles, dict):
        return sum(len(v) if isinstance(v, list) else 1 for v in missiles.values())
    return 0


def _sim_time(env) -> float:
    return float(getattr(env, "current_step", 0)) * float(getattr(env, "env_dt", 0.0))


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _rel(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_checkpoint_meta(model_path: str | Path | None) -> dict:
    if model_path is None:
        return {}
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _build_policy(policy_arch: str, actor_dim: int, critic_dim: int,
                  device: torch.device, init_checkpoint_meta: str | Path | None = None,
                  num_agents: int = 3,
                  action_dim: int = 3,
                  brma_random_scale_mask: bool = False,
                  brma_biased_mask: bool = False,
                  brma_random_mask_prob: float = 0.25,
                  max_allies: int | None = None,
                  max_enemies: int | None = None):
    if policy_arch == "hetero_entity_recurrent":
        from algorithms.happo.hetero_entity_recurrent_policy import (
            HeteroEntityRecurrentPolicy,
            validate_entity_policy_meta,
        )
        meta = {}
        if init_checkpoint_meta is not None and Path(init_checkpoint_meta).exists():
            meta = json.loads(Path(init_checkpoint_meta).read_text(encoding="utf-8"))
            if meta.get("policy_arch") != policy_arch:
                raise ValueError("hetero_entity_recurrent requires a matching checkpoint meta")
            validate_entity_policy_meta(meta)
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta.get("entity_dim", 21)),
            action_dim=action_dim,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
    if policy_arch in ("pure_happo", "pure_happo_tanh"):
        return PureHAPPOPolicy(
            actor_obs_dim=actor_dim, critic_state_dim=critic_dim,
            action_dim=action_dim, num_agents=num_agents,
        ).to(device)
    if policy_arch == "flat":
        return HAPPOReferencePolicy(actor_dim, critic_dim, action_dim=action_dim).to(device)
    if policy_arch == "entity_attention":
        meta = {}
        if init_checkpoint_meta is not None:
            meta_path = Path(init_checkpoint_meta)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("policy_arch", "flat") != "entity_attention":
                raise ValueError(
                    "entity_attention cannot load a flat checkpoint; use an "
                    "entity_attention checkpoint or omit --init-checkpoint"
                )
        entity_dim = int(meta.get("entity_dim", 30))
        _ma = int(max_allies) if max_allies is not None else 4
        _me = int(max_enemies) if max_enemies is not None else 4
        return EntityHAPPOReferencePolicy(
            entity_dim=entity_dim, critic_state_dim=critic_dim,
            action_dim=action_dim,
            max_allies=_ma, max_enemies=_me).to(device)
    if policy_arch == "brma_entity":
        meta = {}
        if init_checkpoint_meta is not None:
            meta_path = Path(init_checkpoint_meta)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("policy_arch", "flat") != "brma_entity":
                raise ValueError(
                    "brma_entity cannot load a flat or entity_attention checkpoint; "
                    "use a brma_entity checkpoint or omit --init-checkpoint"
                )
        entity_dim = int(meta.get("entity_dim", 30))
        _ma = int(max_allies) if max_allies is not None else 4
        _me = int(max_enemies) if max_enemies is not None else 4
        return BRMAEntityHAPPOReferencePolicy(
            entity_dim=entity_dim,
            critic_state_dim=critic_dim,
            action_dim=action_dim,
            max_allies=_ma, max_enemies=_me,
        ).to(device)
    if policy_arch == "brma_recurrent":
        meta = {}
        if init_checkpoint_meta is not None:
            meta_path = Path(init_checkpoint_meta)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("policy_arch", "flat") not in ("brma_recurrent", "brma_entity"):
                raise ValueError(
                    "brma_recurrent cannot load a flat or entity_attention checkpoint; "
                    "use a brma_recurrent or brma_entity checkpoint or omit --init-checkpoint"
                )
        entity_dim = int(meta.get("entity_dim", 30))
        rnn_hidden_size = int(meta.get("rnn_hidden_size", 128))
        _ma = int(max_allies) if max_allies is not None else 4
        _me = int(max_enemies) if max_enemies is not None else 4
        return BRMARecurrentHAPPOReferencePolicy(
            entity_dim=entity_dim,
            critic_state_dim=critic_dim,
            action_dim=action_dim,
            rnn_hidden_size=rnn_hidden_size,
            max_allies=_ma, max_enemies=_me,
        ).to(device)
    if policy_arch == "brma_recurrent_masked":
        meta = {}
        if init_checkpoint_meta is not None:
            meta_path = Path(init_checkpoint_meta)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("policy_arch", "flat") != "brma_recurrent_masked":
                raise ValueError(
                    "brma_recurrent_masked cannot load flat, entity_attention, brma_entity, "
                    "or brma_recurrent checkpoints; use a masked checkpoint or omit --init-checkpoint"
                )
        entity_dim = int(meta.get("entity_dim", 30))
        rnn_hidden_size = int(meta.get("rnn_hidden_size", 128))
        _ma = int(max_allies) if max_allies is not None else 4
        _me = int(max_enemies) if max_enemies is not None else 4
        return BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=entity_dim,
            critic_state_dim=critic_dim,
            action_dim=action_dim,
            rnn_hidden_size=rnn_hidden_size,
            random_scale_mask=bool(meta.get("random_scale_mask", brma_random_scale_mask)),
            random_mask_prob=float(meta.get("random_mask_prob", brma_random_mask_prob)),
            biased_mask=bool(meta.get("biased_mask", brma_biased_mask)),
            max_allies=_ma, max_enemies=_me,
        ).to(device)
    raise ValueError(f"unsupported --policy-arch: {policy_arch}")


def _load_uav_imitation_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(_rel(path), allow_pickle=True)
    obs = np.asarray(data["actor_obs"], dtype=np.float32)
    action = np.asarray(data["oracle_action"], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != 96:
        raise ValueError(f"uav imitation actor_obs must have shape [N,96], got {obs.shape}")
    if action.ndim != 2 or action.shape[1] != 3:
        raise ValueError(f"uav imitation oracle_action must have shape [N,3], got {action.shape}")
    return {"actor_obs": obs, "oracle_action": np.clip(action, -1.0, 1.0)}


def _sample_uav_imitation_batch(data: dict[str, np.ndarray], batch_size: int,
                                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data["actor_obs"].shape[0])
    if n <= 0:
        raise ValueError("uav imitation dataset is empty")
    idx = np.random.randint(0, n, size=max(1, int(batch_size)))
    obs = torch.as_tensor(data["actor_obs"][idx], dtype=torch.float32, device=device)
    action = torch.as_tensor(data["oracle_action"][idx], dtype=torch.float32, device=device)
    return obs, action


def _episode_outcome(env, truncated: dict, length: int) -> dict:
    red_alive, blue_alive = _alive_counts(env)
    timeout = bool(all(truncated.values()) or length >= getattr(env, "max_steps", 0))
    if blue_alive == 0 and red_alive > 0:
        winner, reason = "red", "blue_eliminated"
    elif red_alive == 0 and blue_alive > 0:
        winner, reason = "blue", "red_eliminated"
    elif red_alive == 0 and blue_alive == 0:
        winner, reason = "draw", "mutual_elimination"
    elif timeout:
        reason = "timeout"
        if red_alive > blue_alive:
            winner = "red"
        elif blue_alive > red_alive:
            winner = "blue"
        else:
            winner = "draw"
    else:
        winner, reason = "none", "ongoing"
    return {"winner": winner, "end_reason": reason}


def _run_eval(model_path: str, args, summary_json: str, train_num_agents: int = None,
              eval_configs_override=None) -> list[dict] | None:
    configs = eval_configs_override if eval_configs_override is not None else (args.eval_configs or DEFAULT_EVAL_CONFIGS)
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "eval_happo_reference.py"),
        "--model", model_path,
        "--episodes", str(args.train_eval_episodes),
        "--device", str(args.device),
        "--opponent-policy", args.opponent_policy,
        "--max-steps-override", str(args.max_steps),
        "--seed", str(args.seed),
        "--summary-json", summary_json,
        "--configs", *configs,
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                            encoding="utf-8", errors="replace", timeout=1200)
    if result.returncode != 0:
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        return None
    try:
        return json.loads((ROOT / summary_json).read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_eval(records: list[dict], metric: str = "combined") -> float:
    scores = compute_eval_scores(records)
    key = "score_combined" if metric == "combined" else f"score_{metric}"
    if key not in scores:
        raise ValueError(f"unsupported eval checkpoint metric: {metric}")
    return scores[key]


def _eval_checkpoint_extra(args, policy, actor_dim: int, critic_dim: int,
                           transitions_per_rollout: int) -> dict:
    actual_reward_mode = getattr(args, "actual_reward_mode", args.reward_mode)
    return {
        "algorithm": "happo_reference_v0",
        **_experiment_base_v2_meta(
            actual_reward_mode=str(actual_reward_mode),
            rich_logging_enabled=bool(getattr(args, "enable_rich_logging", False)),
            rich_log_mode=str(getattr(args, "rich_log_mode", "summary")),
            config_path=str(getattr(args, "config", "")),
        ),
        "opponent_policy": args.opponent_policy,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": critic_dim,
        "action_dim": int(getattr(policy, "action_dim", 3)),
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
        **_full_geometry_meta(args.policy_arch, getattr(policy, "entity_dim", None), args),
        **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
    }


def _save_policy_checkpoint(policy, directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save(directory / "model.pt")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _exception_is_nonfinite(exc: BaseException | None) -> bool:
    text = str(exc or "").lower().replace("-", "")
    return "nonfinite" in text or "nan" in text or "inf" in text


def _write_runner_status(
    out_dir: Path,
    *,
    status: str,
    total_steps: int,
    iteration: int,
    exception: BaseException | None = None,
    failed_episode_id: int | str = "",
    failure_checkpoint_saved: bool = False,
    extra: dict | None = None,
) -> dict:
    nonfinite = _exception_is_nonfinite(exception)
    payload = {
        "status": status,
        "runner_completed_normally": status == "normal",
        "total_env_steps_actual": int(total_steps),
        "iteration": int(iteration),
        "exception_type": type(exception).__name__ if exception is not None else "",
        "exception_message": str(exception) if exception is not None else "",
        "failed_step": int(total_steps) if status != "normal" else None,
        "failed_episode_id": failed_episode_id if status != "normal" else None,
        "output_dir": str(out_dir),
        "nan_detected": bool(nonfinite),
        "nonfinite_detected": bool(nonfinite),
        "failure_checkpoint_saved": bool(failure_checkpoint_saved),
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runner_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _write_failure_artifacts(policy, state: dict, exc: BaseException) -> None:
    out_dir = Path(state["output_dir"])
    checkpoint_saved = False
    if policy is not None and all(torch.isfinite(p).all() for p in policy.parameters()):
        try:
            failure_dir = out_dir / "latest_failure"
            failure_meta = {
                **dict(state.get("meta", {})),
                "status": "failed",
                "total_env_steps_actual": int(state.get("total_steps", 0)),
                "iteration": int(state.get("iteration", 0)),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            _save_policy_checkpoint(policy, failure_dir, failure_meta)
            checkpoint_saved = True
        except Exception:
            checkpoint_saved = False
    _write_runner_status(
        out_dir,
        status="failed",
        total_steps=int(state.get("total_steps", 0)),
        iteration=int(state.get("iteration", 0)),
        exception=exc,
        failed_episode_id=state.get("episode_id", ""),
        failure_checkpoint_saved=checkpoint_saved,
    )


def _launch_diagnostic_config_for_scenario(scenario: str) -> str:
    mapping = {
        "3v2": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
        "5v4": "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
    }
    if scenario not in mapping:
        raise ValueError(f"unsupported launch diagnostic scenario: {scenario}")
    return mapping[scenario]


def _run_launch_diagnostics_after_training(args, out_dir: Path) -> str:
    if not getattr(args, "run_launch_diagnostics_after_training", False):
        return "not_requested"
    scenarios = list(getattr(args, "launch_diagnostic_scenarios", []) or ["3v2", "5v4"])
    base_out = _rel(args.launch_diagnostic_output_dir) if args.launch_diagnostic_output_dir else out_dir / "launch_diagnostics"
    diagnostic_configs = {
        scenario: _launch_diagnostic_config_for_scenario(scenario)
        for scenario in scenarios
    }
    try:
        for scenario in scenarios:
            diag_out = base_out / scenario
            try:
                output_dir_arg = str(out_dir.relative_to(ROOT))
            except ValueError:
                output_dir_arg = str(out_dir)
            try:
                diag_out_arg = str(diag_out.relative_to(ROOT))
            except ValueError:
                diag_out_arg = str(diag_out)
            cmd = [
                sys.executable,
                "scripts/eval_policy_launch_diagnostics.py",
                "--output-dir", output_dir_arg,
                "--checkpoint-name", "latest",
                "--scenario", str(scenario),
                "--episodes", str(int(args.launch_diagnostic_episodes)),
                "--device", str(args.device),
                "--opponent-policy", str(args.opponent_policy),
                "--max-steps", str(int(args.max_steps)),
                "--diagnostic-output-dir", diag_out_arg,
            ]
            if scenario in diagnostic_configs:
                cmd += ["--config", diagnostic_configs[scenario]]
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=1200,
            )
            if result.returncode != 0:
                print(result.stdout, flush=True)
                print(result.stderr, flush=True)
                return "failed"
        return "success"
    except Exception as exc:
        print(f"[launch-diagnostics] failed: {exc}", flush=True)
        return "failed"


def _cleanup_single_runner() -> None:
    for env in _SINGLE_RUNNER_STATE.get("envs", []):
        try:
            env.close()
        except Exception:
            pass
    for key, method in (("rich_logger", "close"), ("watchdog", "stop"), ("heartbeat", "close")):
        resource = _SINGLE_RUNNER_STATE.get(key)
        if resource is not None:
            try:
                getattr(resource, method)()
            except Exception:
                pass


def _prune_eval_checkpoints(eval_dir: Path, keep: int) -> None:
    if keep <= 0 or not eval_dir.exists():
        return
    checkpoints = sorted(
        (p for p in eval_dir.glob("step_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for path in checkpoints[:-keep]:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def _run_training_main() -> None:
    _SINGLE_RUNNER_STATE.update({
        "policy": None, "output_dir": None, "total_steps": 0,
        "iteration": 0, "episode_id": 0, "meta": {}, "envs": [],
        "heartbeat": None, "watchdog": None, "rich_logger": None,
    })
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="outputs/happo_reference")
    parser.add_argument("--total-env-steps", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-arch", default="flat",
                        choices=["flat", "entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent", "pure_happo"])
    parser.add_argument("--brma-random-scale-mask", action="store_true",
                        help="Accepted for compatibility but rejected: unsafe without rollout mask replay.")
    parser.add_argument("--brma-biased-mask", action="store_true",
                        help="Enable learned BRMA biased mask generator for brma_recurrent_masked.")
    parser.add_argument("--brma-random-mask-prob", type=float, default=0.25,
                        help="Non-self entity drop probability for BRMA random scale mask.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=sorted(OpponentPolicy.MODES))
    parser.add_argument(
        "--reward-mode",
        default=None,
        help="Override YAML hetero_reward_mode. If omitted, uses the YAML value.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--critic-epochs", type=int, default=None)
    parser.add_argument("--clip-param", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--eval-during-training", action="store_true")
    parser.add_argument("--eval-interval-steps", type=int, default=25000)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--train-eval-episodes", type=int, default=None)
    parser.add_argument("--eval-configs", nargs="*", default=None)
    parser.add_argument("--save-eval-checkpoints", action="store_true",
                        help="Save every eval checkpoint and maintain best_3v2/best_5v4/best_7v6/best_combined.")
    parser.add_argument("--eval-checkpoint-metric", default="combined",
                        choices=["combined", "3v2", "5v4", "7v6"],
                        help="Reference metric for eval checkpoint summaries; all best dirs are still maintained.")
    parser.add_argument("--keep-eval-checkpoints", type=int, default=20,
                        help="Maximum number of per-eval checkpoint directories to keep when enabled.")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0,
                        help="Save a periodic checkpoint every N env steps (0=disabled)")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                        help="Max periodic checkpoints to keep (oldest removed first)")
    parser.add_argument("--uav-imitation-dataset", default=None)
    parser.add_argument("--uav-imitation-coef", type=float, default=0.0)
    parser.add_argument("--uav-imitation-until-steps", type=int, default=0)
    parser.add_argument("--uav-imitation-batch-size", type=int, default=1024)
    parser.add_argument("--enable-rich-logging", action="store_true")
    parser.add_argument(
        "--rich-log-mode",
        choices=["full", "summary"],
        default="summary",
        help="summary skips per-step reward_components and aircraft_timeseries; full keeps legacy rich logs.",
    )
    parser.add_argument("--rich-log-dir", default=None)
    parser.add_argument("--heartbeat-log", default=None)
    parser.add_argument("--heartbeat-every-steps", type=int, default=50)
    parser.add_argument("--debug-rollout-heartbeat", action="store_true")
    parser.add_argument("--heartbeat-stall-timeout-sec", type=float, default=0.0)
    parser.add_argument("--exit-on-heartbeat-stall", action="store_true")
    parser.add_argument("--timeseries-episodes-limit", type=int, default=3)
    parser.add_argument("--timeseries-step-stride", type=int, default=5)
    parser.add_argument("--run-launch-diagnostics-after-training", action="store_true")
    parser.add_argument("--launch-diagnostic-episodes", type=int, default=5)
    parser.add_argument("--launch-diagnostic-scenarios", nargs="*", default=["3v2", "5v4"])
    parser.add_argument("--launch-diagnostic-output-dir", default=None)
    args = parser.parse_args()
    _apply_policy_specific_defaults(args)
    _reject_unsafe_random_scale_mask(args)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.critic_epochs < 1:
        raise ValueError("--critic-epochs must be >= 1")
    if args.policy_arch == "pure_happo":
        if args.init_checkpoint:
            raise ValueError("pure_happo formal training must start from scratch; --init-checkpoint is not allowed")
        if args.uav_imitation_dataset or args.uav_imitation_coef > 0.0:
            raise ValueError("pure_happo formal training does not allow imitation or BC inputs")
        if args.brma_biased_mask or args.brma_random_scale_mask:
            raise ValueError("pure_happo does not allow BRMA random/biased mask modules")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / args.output_dir
    _SINGLE_RUNNER_STATE["output_dir"] = out_dir
    (out_dir / "latest").mkdir(parents=True, exist_ok=True)
    (out_dir / "best").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.num_envs > 1:
        envs = _create_mp_envs(args.num_envs, args.config, args.reward_mode,
                               args.max_steps)
    else:
        envs = [
            make_env(
                args.config,
                max_steps=args.max_steps,
                **_env_type_override_kwargs(args.config),
                **_make_env_kwargs_for_reward_mode(args.reward_mode),
            )
            for _ in range(args.num_envs)
        ]
    env = envs[0]
    action_dim = _infer_env_action_dim(env)
    actual_reward_mode = str(getattr(env, "hetero_reward_mode", args.reward_mode or ""))
    args.actual_reward_mode = actual_reward_mode
    base_v2_meta = _experiment_base_v2_meta(
        actual_reward_mode=actual_reward_mode,
        rich_logging_enabled=args.enable_rich_logging,
        rich_log_mode=args.rich_log_mode,
        config_path=args.config,
    )
    base_v2_meta.update(_opponent_protocol_meta(args.opponent_policy))
    print(
        "[experiment] "
        f"config={args.config} policy_arch={args.policy_arch} "
        f"opponent_policy={args.opponent_policy} "
        f"actual_reward_mode={actual_reward_mode} "
        f"rich_log_mode={args.rich_log_mode} num_envs={args.num_envs} "
        f"rollout_length={args.rollout_length} max_steps={args.max_steps}",
        flush=True,
    )
    if actual_reward_mode == "brma_tam_scale_aligned_v1":
        print(
            "[reward_contract] revision=3 "
            f"config={json.dumps(base_v2_meta.get('reward_config', {}), sort_keys=True)} "
            f"progress={base_v2_meta.get('progress_formula_version')} "
            f"terminal={base_v2_meta.get('terminal_formula_version')} "
            f"mav_normalization={base_v2_meta.get('mav_normalization')}",
            flush=True,
        )
    if args.policy_arch == "pure_happo":
        print(
            "[pure_happo] "
            f"actor_lr={args.actor_lr} critic_lr={args.critic_lr} "
            f"clip={args.clip_param} entropy={args.entropy_coef} value_coef={args.value_coef} "
            f"max_grad_norm={args.max_grad_norm} ppo_epochs={args.ppo_epochs} "
            f"critic_epochs={args.critic_epochs} gamma={args.gamma} "
            f"gae_lambda={args.gae_lambda} seed={args.seed}",
            flush=True,
        )
    _SINGLE_RUNNER_STATE["envs"] = envs
    entity_mode = args.policy_arch == "hetero_entity_recurrent"
    adapter = HeteroEntitySetAdapter() if entity_mode else HeteroObsAdapterV2()
    actor_dim = 0 if entity_mode else adapter.flat_actor_obs_dim
    critic_dim = 0 if entity_mode else adapter.critic_state_dim
    init_meta_path = None
    if args.init_checkpoint:
        init_path_for_meta = _rel(args.init_checkpoint)
        init_meta_path = init_path_for_meta.parent / "meta.json"
        if args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"} and not init_meta_path.exists():
            raise ValueError(
                f"{args.policy_arch} init checkpoint requires meta.json with policy_arch={args.policy_arch}"
            )
        _reject_unsafe_random_scale_mask_checkpoint(args.policy_arch, init_meta_path)
    policy = _build_policy(args.policy_arch, actor_dim, critic_dim, device,
                           init_checkpoint_meta=init_meta_path,
                           brma_random_scale_mask=args.brma_random_scale_mask,
                           brma_biased_mask=args.brma_biased_mask,
                           brma_random_mask_prob=args.brma_random_mask_prob,
                           num_agents=len(env.red_ids),
                           action_dim=action_dim,
                           max_allies=getattr(adapter, "max_allies", None),
                           max_enemies=getattr(adapter, "max_enemies", None))
    _SINGLE_RUNNER_STATE["policy"] = policy
    _SINGLE_RUNNER_STATE["meta"] = {
        "algorithm": "happo_reference_v0",
        "policy_arch": args.policy_arch,
        "config": args.config,
        **base_v2_meta,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": critic_dim,
        "action_dim": action_dim,
        **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
    }
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        policy.load(init_path, map_location=device)
        print(f"Loaded init_checkpoint: {init_path}", flush=True)
    if args.policy_arch in {"pure_happo"}:
        trainer = PureHAPPOTrainer(
            policy, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            clip_param=args.clip_param, entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm, ppo_epochs=args.ppo_epochs,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            seed=args.seed, critic_epochs=args.critic_epochs,
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
        if actor_dim != 96:
            raise ValueError(
                "legacy 96-dim imitation dataset is incompatible with canonical "
                "mav_shared_geo full-geometry actor obs; imitation is disabled "
                "for current main experiments"
            )
        uav_imitation_data = _load_uav_imitation_dataset(args.uav_imitation_dataset)
        print(
            f"Loaded uav_imitation_dataset: {_rel(args.uav_imitation_dataset)} "
            f"samples={uav_imitation_data['actor_obs'].shape[0]} "
            f"coef={args.uav_imitation_coef}",
            flush=True,
        )
    opponents = [
        OpponentPolicy(mode=args.opponent_policy, seed=args.seed + 17 + i)
        for i in range(args.num_envs)
    ]
    env_states = [e.reset(seed=args.seed + i) for i, e in enumerate(envs)]
    obs_list = [state[0] for state in env_states]
    info_list = [state[1] for state in env_states]
    roles = _role_ids(env)
    transitions_per_rollout = _transitions_per_rollout(args.rollout_length, args.num_envs)
    heartbeat_path = _rel(args.heartbeat_log) if args.heartbeat_log else out_dir / "heartbeat.log"
    heartbeat = HeartbeatLogger(
        heartbeat_path,
        every_steps=args.heartbeat_every_steps,
        enabled=bool(args.heartbeat_log or args.debug_rollout_heartbeat),
        debug_all=args.debug_rollout_heartbeat,
        static_fields={"max_steps": args.max_steps, "num_envs": args.num_envs},
    )
    _SINGLE_RUNNER_STATE["heartbeat"] = heartbeat
    watchdog = HeartbeatStallWatchdog(
        heartbeat,
        out_dir,
        timeout_sec=args.heartbeat_stall_timeout_sec,
        exit_on_stall=args.exit_on_heartbeat_stall,
    )
    watchdog.start()
    _SINGLE_RUNNER_STATE["watchdog"] = watchdog
    rich_logger = None
    if args.enable_rich_logging:
        rich_dir = _rel(args.rich_log_dir) if args.rich_log_dir else out_dir
        rich_logger = RichExperimentLogger(
            rich_dir,
            run_id=out_dir.name,
            method_name="happo_reference_v0",
            scenario_name=Path(args.config).stem,
            device=str(args.device),
            num_envs=args.num_envs,
            rollout_length_per_env=args.rollout_length,
            transitions_per_rollout=transitions_per_rollout,
            mode=args.rich_log_mode,
            timeseries_episodes_limit=args.timeseries_episodes_limit,
            timeseries_step_stride=args.timeseries_step_stride,
        )
        _SINGLE_RUNNER_STATE["rich_logger"] = rich_logger
        if args.rich_log_mode == "full":
            write_not_available_attention(rich_dir, "happo_reference_v0", Path(args.config).stem)
    iterations = int(math.ceil(args.total_env_steps / transitions_per_rollout))
    total_steps = 0
    episodes = 0
    current_ep_return = [np.zeros(len(env.red_ids), dtype=np.float32) for _ in range(args.num_envs)]
    current_ep_len = [0 for _ in range(args.num_envs)]
    current_ep_id = [0 for _ in range(args.num_envs)]
    current_ep_reward_comp = [{} for _ in range(args.num_envs)]
    current_ep_reward_comp_by_agent = [
        {rid: {} for rid in env.red_ids}
        for _ in range(args.num_envs)
    ]
    current_ep_launch_stats = [
        {"red_launch_count": 0, "red_hit_count": 0, "blue_launch_count": 0, "blue_hit_count": 0}
        for _ in range(args.num_envs)
    ]
    current_ep_hit_totals = [
        {"red": 0, "blue": 0}
        for _ in range(args.num_envs)
    ]
    prev_hit_totals = [{"red": 0, "blue": 0} for _ in range(args.num_envs)]
    recent = deque(maxlen=100)
    best_score = -float("inf")
    eval_best_scores = {
        "best_3v2": -float("inf"),
        "best_5v4": -float("inf"),
        "best_7v6": -float("inf"),
        "best_combined": -float("inf"),
    }
    nan_detected = False
    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)
    rnn_hidden = None
    if _rnn_hidden_size > 0:
        rnn_hidden = [
            np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)
            for _ in range(args.num_envs)
        ]

    train_log = out_dir / "train_log.csv"
    with train_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "total_steps", "actual_reward_mode", "avg_return", "red_win", "blue_win",
            "draw", "timeout", "mav_survival", "red_alive_final",
            "blue_alive_final", "red_missiles_fired", "blue_missiles_fired",
            "red_missile_hits", "blue_missile_hits", "actor_loss_mav", "actor_loss_uav",
            "critic_loss", "entropy_mav", "entropy_uav",
            "mav_action_saturation_rate", "uav_action_saturation_rate",
            "uav_imitation_loss",
            "entropy_mav_valid_count", "entropy_uav_valid_count",
            "mav_active_sample_count", "uav_active_sample_count",
            "action_log_std_mav_min", "action_log_std_mav_max",
            "action_log_std_mav_mean", "action_log_std_uav_min",
            "action_log_std_uav_max", "action_log_std_uav_mean",
            "approx_kl_mav", "approx_kl_uav",
            "mask_keep_ratio", "mask_entropy", "masked_entity_count",
            "tam_v7_mav_flight_sum", "tam_v7_mav_safety_sum",
            "tam_v7_mav_support_sum", "tam_v7_mav_event_sum",
            "tam_v7_mav_terminal_sum", "tam_v7_mav_total_sum",
            "tam_v7_uav_flight_sum", "tam_v7_uav_situation_sum",
            "tam_v7_uav_event_sum", "tam_v7_uav_terminal_sum",
            "tam_v7_uav_total_sum", "tam_v7_total_sum",
            "tam_v7_uav_altitude_sum", "tam_v7_uav_speed_sum",
            "tam_v7_uav_boundary_sum", "tam_v7_mav_altitude_sum",
            "tam_v7_mav_speed_sum", "tam_v7_mav_boundary_sum",
            "paper_v1_uav_flight_sum", "paper_v1_uav_adv_sum",
            "paper_v1_uav_end_sum", "paper_v1_uav_total_sum",
            "paper_v1_mav_flight_sum", "paper_v1_mav_safety_sum",
            "paper_v1_mav_support_sum", "paper_v1_mav_event_raw_sum",
            "paper_v1_mav_scaled_tam_sum", "paper_v1_mav_total_sum",
            "v1_mav_safety_sum", "v1_mav_support_sum",
            "v1_mav_event_sum", "v1_mav_total_sum",
            "mav_observed_ratio", "mav_shared_track_ratio",
            "red_launch_with_mav_shared_track", "red_hit_with_mav_shared_track",
            "team_kill_while_mav_alive", "team_kill_after_mav_death",
            "red_launch_rate_before_mav_death", "red_launch_rate_after_mav_death",
            "mav_removed_r_adv_sum", "mav_removed_r_end_sum",
            "effective_team_brma_flight", "effective_team_uav_speed",
            "effective_team_uav_angle", "effective_team_uav_distance",
            "effective_team_uav_event", "effective_team_mav_safety",
            "effective_team_mav_support", "effective_team_mav_event",
            "effective_team_total", "active_red_count",
            "effective_component_sum", "effective_total_minus_component_sum",
            "initial_red_count", "effective_initial_team_mean_total",
            *MARL_DYNAMICS_TRAIN_FIELDS,
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
        last_eval = -999999 if args.eval_at_start else 0
        for iteration in range(1, iterations + 1):
            _SINGLE_RUNNER_STATE["iteration"] = iteration
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
            buffer = HAPPORolloutBuffer(rollout_transitions, len(env.red_ids), actor_dim,
                                        critic_dim, action_dim, roles,
                                        rnn_hidden_size=getattr(policy, 'rnn_hidden_size', 0),
                                        **buffer_kwargs)
            red_fired = blue_fired = red_hits = blue_hits = 0
            rollout_reward_sums: defaultdict[str, float] = defaultdict(float)
            rollout_reward_counts: defaultdict[str, float] = defaultdict(float)
            rollout_reward_steps = 0
            rollout_target_distance_sum = 0.0
            rollout_target_valid_count = 0
            rollout_target_zone_counts = [0, 0, 0, 0]
            rollout_scale_progress_counts = [0, 0, 0, 0]
            rollout_scale_target_switch_count = 0.0
            rollout_scale_mav_aspect_sum = 0.0
            rollout_scale_mav_aspect_mean_sum = 0.0
            rollout_scale_mav_aware_sum = 0.0
            rollout_scale_mav_aware_mean_sum = 0.0
            rollout_scale_mav_rows = 0
            rollout_scale_terminal_sum = 0.0
            rollout_scale_event_counts = [0.0, 0.0, 0.0]
            rollout_scale_v2_progress_counts = [0, 0, 0, 0]
            rollout_scale_v2_target_switch_count = 0.0
            rollout_scale_v2_terminal_sum = 0.0
            rollout_scale_v2_event_counts = [0.0, 0.0, 0.0]
            rollout_scale_v2_flight_raw_sum = 0.0
            rollout_scale_v2_flight_scaled_sum = 0.0
            rollout_scale_v2_active_rows = 0
            rollout_uav_oob_count = 0
            rollout_mav_low_altitude_death_count = 0
            rollout_identity_max = {"mav": 0.0, "uav": 0.0}
            rollout_identity_failures = 0
            while len(buffer) < rollout_transitions and total_steps < args.total_env_steps:
                for env_idx, rollout_env in enumerate(envs):
                    if len(buffer) >= rollout_transitions or total_steps >= args.total_env_steps:
                        break
                    obs = obs_list[env_idx]
                    info = info_list[env_idx]
                    rollout_local_step = len(buffer)
                    heartbeat.write(
                        "before_policy_act",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                    )
                    adapted = adapter.adapt_all(
                        obs, info=info, red_ids=rollout_env.red_ids, blue_ids=rollout_env.blue_ids)
                    active = _build_red_alive_mask(info, rollout_env, rollout_env.red_ids)
                    san_ctx = {
                        "iteration": iteration, "env_idx": env_idx,
                        "total_steps": total_steps, "episode_id": current_ep_id[env_idx],
                    }
                    _sh = rnn_hidden[env_idx] if rnn_hidden is not None else None
                    if entity_mode:
                        actor_obs = None
                        critic = None
                        actor_tokens = adapted["actor_entity_tokens"].copy()
                        actor_keep = adapted["actor_keep_mask"].copy()
                        critic_tokens = adapted["critic_entity_tokens"].copy()
                        critic_keep = adapted["critic_keep_mask"].copy()
                        critic_counts = adapted.get("critic_counts",
                                                    np.zeros(4, dtype=np.float32))
                        actor_tokens[active <= 0.5] = 0.0
                        actor_keep[active <= 0.5] = 0.0
                        actor_keep[active <= 0.5, 0] = 1.0
                        if not np.isfinite(actor_tokens[active > 0.5]).all():
                            raise ValueError(f"Non-finite active entity tokens: {san_ctx}")
                        if not np.isfinite(critic_tokens[critic_keep > 0.5]).all():
                            raise ValueError(f"Non-finite active critic entity tokens: {san_ctx}")
                        if rnn_hidden is not None:
                            rnn_hidden[env_idx] = zero_inactive_hidden(rnn_hidden[env_idx], active)
                    else:
                        actor_obs = np.stack([
                            adapted["actor_obs"].get(rid, np.zeros(actor_dim, dtype=np.float32))
                            for rid in rollout_env.red_ids
                        ])
                        critic = adapted["critic_state"]
                        san = sanitize_policy_inputs(
                            actor_obs, active, critic_state=critic, rnn_hidden=_sh, context=san_ctx)
                        actor_obs = san["actor_obs"]
                        critic = san["critic_state"] if san["critic_state"] is not None else critic
                        if rnn_hidden is not None and san["rnn_hidden"] is not None:
                            rnn_hidden[env_idx] = san["rnn_hidden"]
                    rnn_hidden_pre = None
                    if rnn_hidden is not None:
                        rnn_hidden_pre = rnn_hidden[env_idx].copy()
                    act_kwargs = {}
                    if rnn_hidden is not None:
                        act_kwargs["rnn_hidden"] = torch.as_tensor(
                            rnn_hidden[env_idx], device=device)
                    with torch.no_grad():
                        if entity_mode:
                            out = policy.act(
                                torch.as_tensor(actor_tokens, device=device),
                                torch.as_tensor(actor_keep, device=device),
                                torch.as_tensor(roles, device=device),
                                torch.as_tensor(critic_tokens, device=device),
                                torch.as_tensor(critic_keep, device=device),
                                deterministic=False,
                                critic_counts=torch.as_tensor(critic_counts, device=device),
                                **act_kwargs)
                        else:
                            out = policy.act(
                                torch.as_tensor(actor_obs, device=device), roles=roles,
                                critic_state=torch.as_tensor(critic, device=device),
                                deterministic=False, **act_kwargs)
                    # ---- Zero out actions for inactive agents ----
                    actions_np = zero_inactive_actions(
                        out["action"].cpu().numpy(), active)
                    out["action"] = torch.as_tensor(actions_np, device=device)
                    # ---- Zero out returned rnn_hidden for inactive agents ----
                    if rnn_hidden is not None and "rnn_hidden" in out:
                        rnn_hidden[env_idx] = zero_inactive_hidden(
                            out["rnn_hidden"].cpu().numpy(), active)
                    heartbeat.write(
                        "after_policy_act",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                    )
                    actions = out["action"].cpu().numpy()
                    log_probs = out["log_prob"].cpu().numpy()
                    value = float(out["value"].item())
                    # Check finite for active agents only (inactive were zeroed)
                    active_mask_np = active > 0.5
                    if active_mask_np.any():
                        if not np.isfinite(actions[active_mask_np]).all():
                            raise ValueError(
                                f"Non-finite action for active agent: "
                                f"iter={iteration} env={env_idx} step={total_steps}"
                            )
                        if not np.isfinite(value):
                            raise ValueError(
                                f"Non-finite value: "
                                f"iter={iteration} env={env_idx} step={total_steps} value={value}"
                            )
                        if not np.isfinite(log_probs[active_mask_np]).all():
                            raise ValueError(
                                f"Non-finite log_prob for active agent: "
                                f"iter={iteration} env={env_idx} step={total_steps}"
                            )
                    # Check finite for returned rnn_hidden (active agents)
                    if rnn_hidden is not None and "rnn_hidden" in out and active_mask_np.any():
                        rh = out["rnn_hidden"].cpu().numpy()
                        if not np.isfinite(rh[active_mask_np]).all():
                            raise ValueError(
                                f"Non-finite returned rnn_hidden for active agent: "
                                f"iter={iteration} env={env_idx} step={total_steps}"
                            )
                    action_dict = {rid: actions[i].astype(np.float32)
                                   for i, rid in enumerate(rollout_env.red_ids)}
                    heartbeat.write(
                        "before_opponent_act",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                    )
                    action_dict.update(opponents[env_idx].act(obs, rollout_env.blue_ids, env=rollout_env))
                    heartbeat.write(
                        "after_opponent_act",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                    )
                    heartbeat.write(
                        "before_env_step",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                    )
                    next_obs, rewards, terminated, truncated, next_info = rollout_env.step(action_dict)
                    heartbeat.write(
                        "after_env_step",
                        iteration=iteration,
                        rollout_local_step=rollout_local_step,
                        env_idx=env_idx,
                        total_steps=total_steps,
                        episode_length=current_ep_len[env_idx],
                        env_episode_id=current_ep_id[env_idx],
                        alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                        missile_count=_missile_count(rollout_env),
                        sim_time=_sim_time(rollout_env),
                        done=_team_done(terminated, truncated),
                        terminated=bool(all(terminated.values())) if terminated else False,
                        truncated=bool(all(truncated.values())) if truncated else False,
                    )
                    reward_np = np.array([float(rewards.get(rid, 0.0)) for rid in rollout_env.red_ids], dtype=np.float32)
                    done = _team_done(terminated, truncated)
                    done_np = np.full((len(rollout_env.red_ids),), float(done), dtype=np.float32)
                    if done:
                        next_value = 0.0
                    else:
                        next_adapted = adapter.adapt_all(
                            next_obs, info=next_info, red_ids=rollout_env.red_ids, blue_ids=rollout_env.blue_ids)
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
                    if rnn_hidden_pre is not None:
                        store_kwargs["rnn_hidden"] = rnn_hidden_pre
                    if entity_mode:
                        store_kwargs.update({
                            "actor_entity_tokens": actor_tokens,
                            "actor_keep_mask": actor_keep,
                            "critic_entity_tokens": critic_tokens,
                            "critic_keep_mask": critic_keep,
                            "critic_counts": critic_counts,
                        })
                    buffer.store(
                        actor_obs, critic, actions, log_probs, reward_np, done_np,
                        value, active, next_value=next_value, env_id=env_idx,
                        **store_kwargs)
                    current_ep_return[env_idx] += reward_np
                    current_ep_len[env_idx] += 1
                    total_steps += 1
                    _SINGLE_RUNNER_STATE["total_steps"] = total_steps
                    _SINGLE_RUNNER_STATE["episode_id"] = current_ep_id[env_idx]
                    rc = next_info.get("reward_components", {}) if isinstance(next_info, dict) else {}
                    if actual_reward_mode == "brma_tam_role_situation_v3":
                        v3_sums, v3_counts = collect_v3_effective_samples(
                            rc, rollout_env.agent_roles, active, rollout_env.red_ids, step=total_steps,
                        )
                        for key, value in v3_sums.items():
                            rollout_reward_sums[key] += value
                            rollout_reward_counts[key] += v3_counts[key]
                    for aid in rollout_env.red_ids:
                        comp = rc.get(aid, {}) if isinstance(rc, dict) else {}
                        if not isinstance(comp, dict):
                            continue
                        for key, value in comp.items():
                            try:
                                delta = float(value)
                            except (TypeError, ValueError):
                                continue
                            current_ep_reward_comp[env_idx][key] = (
                                current_ep_reward_comp[env_idx].get(key, 0.0) + delta
                            )
                    # Per-agent episode accumulators (for episode_reward_components.csv)
                    for rid in rollout_env.red_ids:
                        comp = rc.get(rid, {}) if isinstance(rc, dict) else {}
                        if not isinstance(comp, dict):
                            continue
                        agent_acc = current_ep_reward_comp_by_agent[env_idx].setdefault(rid, {})
                        if actual_reward_mode == "brma_tam_role_situation_v3":
                            ridx = rollout_env.red_ids.index(rid)
                            accumulate_v3_episode_step(
                                agent_acc,
                                comp,
                                agent_id=rid,
                                role=rollout_env.agent_roles.get(rid, ""),
                                alive_before=float(active[ridx]) > 0.0,
                                step=total_steps,
                            )
                        for key, value in comp.items():
                            if key in V3_REWARD_COMPONENT_FIELDS:
                                continue
                            if key == "v4_j_combat":
                                agent_acc["v4_final_j_combat"] = float(value)
                                continue
                            if key == "v4_identity_error":
                                agent_acc["v4_max_abs_identity_error"] = max(
                                    float(agent_acc.get("v4_max_abs_identity_error", 0.0)), abs(float(value))
                                )
                                continue
                            if not (
                                key.startswith("tam_v7_")
                                or key.startswith("tam_table1_")
                                or key.startswith("v1_mav_")
                                or key.startswith("paper_v1_")
                                or key in BRMA_TAM_SCRIPTED_COMPONENT_COLUMNS
                                or key in V4_COMPONENT_FIELDS
                                or key in BRMA_TAM_SCALE_V1_COMPONENT_COLUMNS
                                or key in {
                                    "mav_observed_ratio",
                                    "mav_shared_track_ratio",
                                    "red_launch_with_mav_shared_track",
                                    "red_hit_with_mav_shared_track",
                                    "team_kill_while_mav_alive",
                                    "team_kill_after_mav_death",
                                    "red_launch_before_mav_death",
                                    "red_launch_after_mav_death",
                                    "red_uav_alive_steps_before_mav_death",
                                    "red_uav_alive_steps_after_mav_death",
                                    "red_launch_rate_before_mav_death",
                                    "red_launch_rate_after_mav_death",
                                    "mav_reward_safety_sum",
                                    "mav_reward_support_sum",
                                    "mav_reward_event_sum",
                                    "mav_reward_total_sum",
                                    "mav_removed_r_adv_sum",
                                    "mav_removed_r_end_sum",
                                }
                            ):
                                continue
                            if key in {"scale_v1_progress_reset_reason", "scale_v1_reward_target_id"}:
                                agent_acc[key + "_last"] = str(value)
                                continue
                            try:
                                delta = float(value)
                            except (TypeError, ValueError):
                                continue
                            if key == "max_altitude_m":
                                agent_acc[key] = max(agent_acc.get(key, delta), delta)
                            elif key == "above_altitude_max_episode_flag":
                                agent_acc[key] = max(agent_acc.get(key, 0.0), delta)
                            elif key == "tam_dodge_missing_reason":
                                continue
                            elif key == "tam_v7_mav_team_credit_used":
                                agent_acc["tam_v7_mav_team_credit_used_max"] = max(
                                    agent_acc.get("tam_v7_mav_team_credit_used_max", delta), delta)
                            elif key in ("tam_v7_blue_loss_frac", "tam_v7_red_loss_weighted"):
                                agent_acc[key + "_last"] = delta
                            elif key in _SCALE_V1_EPISODE_LAST_FIELDS:
                                agent_acc[key + "_last"] = delta
                            elif key in {
                                "mav_reward_safety_sum",
                                "mav_reward_support_sum",
                                "mav_reward_event_sum",
                                "mav_reward_total_sum",
                                "mav_removed_r_adv_sum",
                                "mav_removed_r_end_sum",
                            }:
                                agent_acc[key] = agent_acc.get(key, 0.0) + delta
                            else:
                                sum_key = key + "_sum"
                                agent_acc[sum_key] = agent_acc.get(sum_key, 0.0) + delta
                    if rc and isinstance(rc, dict):
                        for rid in rollout_env.red_ids:
                            comp = rc.get(rid, {}) if isinstance(rc, dict) else {}
                            if not isinstance(comp, dict):
                                continue
                            role = rollout_env.agent_roles.get(rid, "")
                            scale_v1 = actual_reward_mode == "brma_tam_scale_aligned_v1"
                            scale_v2 = actual_reward_mode == "brma_tam_scale_aligned_v2"
                            if actual_reward_mode in {"brma_tam_role_situation_v3", "brma_tam_paper_calibrated_v4"}:
                                continue
                            if scale_v1 and role == "mav":
                                identity_keys = (
                                    "scale_v1_flight_total", "scale_v1_mav_role",
                                    "scale_v1_mav_event_total", "scale_v1_terminal",
                                )
                                identity_role = "mav"
                            elif scale_v1:
                                identity_keys = (
                                    "scale_v1_flight_total", "scale_v1_progress_clipped",
                                    "scale_v1_uav_event_total", "scale_v1_terminal",
                                )
                                identity_role = "uav"
                            elif scale_v2 and role == "mav":
                                identity_keys = (
                                    "scale_v2_flight_scaled_total", "scale_v2_mav_role",
                                    "scale_v2_event", "scale_v2_terminal",
                                )
                                identity_role = "mav"
                            elif scale_v2:
                                identity_keys = (
                                    "scale_v2_flight_scaled_total", "scale_v2_progress",
                                    "scale_v2_event", "scale_v2_terminal",
                                )
                                identity_role = "uav"
                            elif role == "mav":
                                identity_keys = (
                                    "brma_pitch", "brma_roll", "brma_vel",
                                    "mav_dist_weighted", "mav_threat_weighted",
                                    "mav_aspect_weighted", "mav_pos_weighted",
                                    "mav_aware_weighted", "mav_event_total",
                                )
                                identity_role = "mav"
                            else:
                                identity_keys = (
                                    "brma_pitch", "brma_roll", "brma_vel",
                                    "tam_speed_weighted", "tam_angle_weighted",
                                    "tam_distance_weighted", "uav_event_total",
                                )
                                identity_role = "uav"
                            identity_sum = sum(float(comp.get(key, 0.0) or 0.0) for key in identity_keys)
                            identity_error = abs(float(comp.get("total", 0.0) or 0.0) - identity_sum)
                            rollout_identity_max[identity_role] = max(
                                rollout_identity_max[identity_role], identity_error
                            )
                            if identity_error > 1e-6:
                                rollout_identity_failures += 1
                                raise ValueError(
                                    f"reward identity failure agent={rid} step={total_steps} "
                                    f"error={identity_error:.9g}"
                                )
                            geometry_valid = float(comp.get(
                                "scale_v1_geometry_valid" if scale_v1 else "tam_geometry_valid", 0.0
                            ) or 0.0)
                            if role != "mav" and geometry_valid > 0.5:
                                distance = float(comp.get(
                                    "scale_v1_reward_target_distance_m" if scale_v1 else "reward_target_distance_m", 0.0
                                ) or 0.0)
                                rollout_target_distance_sum += distance
                                rollout_target_valid_count += 1
                                if distance <= 5000.0:
                                    rollout_target_zone_counts[0] += 1
                                elif distance < 10000.0:
                                    rollout_target_zone_counts[1] += 1
                                elif distance < 20000.0:
                                    rollout_target_zone_counts[2] += 1
                                else:
                                    rollout_target_zone_counts[3] += 1
                            if scale_v1 and role != "mav":
                                progress = float(comp.get("scale_v1_progress_clipped", 0.0) or 0.0)
                                raw_progress = float(comp.get("scale_v1_progress_raw", 0.0) or 0.0)
                                if progress > 1e-12:
                                    rollout_scale_progress_counts[0] += 1
                                elif progress < -1e-12:
                                    rollout_scale_progress_counts[1] += 1
                                else:
                                    rollout_scale_progress_counts[2] += 1
                                if abs(progress - raw_progress) > 1e-12:
                                    rollout_scale_progress_counts[3] += 1
                                rollout_scale_target_switch_count += float(comp.get("scale_v1_progress_reset_reason") == "target_switch")
                            if scale_v1 and role == "mav":
                                rollout_scale_mav_aspect_sum += float(comp.get("scale_v1_mav_aspect_raw_sum", 0.0) or 0.0)
                                rollout_scale_mav_aspect_mean_sum += float(comp.get("scale_v1_mav_aspect_mean", 0.0) or 0.0)
                                rollout_scale_mav_aware_sum += float(comp.get("scale_v1_mav_aware_raw_sum", 0.0) or 0.0)
                                rollout_scale_mav_aware_mean_sum += float(comp.get("scale_v1_mav_aware_mean", 0.0) or 0.0)
                                rollout_scale_mav_rows += 1
                            if scale_v1:
                                rollout_scale_terminal_sum += float(comp.get("scale_v1_terminal", 0.0) or 0.0)
                                rollout_scale_event_counts[0] += max(float(comp.get("scale_v1_uav_event_kill", 0.0) or 0.0), 0.0) / 10.0
                                rollout_scale_event_counts[1] += float(
                                    float(comp.get("scale_v1_uav_event_death", 0.0) or 0.0) < 0.0
                                    or float(comp.get("scale_v1_mav_event_death", 0.0) or 0.0) < 0.0
                                )
                                rollout_scale_event_counts[2] += float(float(comp.get("scale_v1_uav_event_oob", 0.0) or 0.0) < 0.0)
                            if scale_v2:
                                _v2_total = float(comp.get("scale_v2_total", 0.0) or 0.0)
                                _v2_flight_raw = float(comp.get("scale_v2_flight_raw_total", 0.0) or 0.0)
                                _v2_event = float(comp.get("scale_v2_event", 0.0) or 0.0)
                                _v2_terminal = float(comp.get("scale_v2_terminal", 0.0) or 0.0)
                                _v2_dead_before = (
                                    abs(_v2_total) < 1e-12
                                    and abs(_v2_flight_raw) < 1e-12
                                    and abs(_v2_event) < 1e-12
                                    and abs(_v2_terminal) < 1e-12
                                )
                                if _v2_dead_before:
                                    continue
                                rollout_scale_v2_flight_raw_sum += _v2_flight_raw
                                rollout_scale_v2_flight_scaled_sum += float(comp.get("scale_v2_flight_scaled_total", 0.0) or 0.0)
                                rollout_scale_v2_active_rows += 1
                                rollout_scale_v2_terminal_sum += _v2_terminal
                                rollout_scale_v2_event_counts[0] += max(
                                    float(comp.get("scale_v1_uav_event_kill", 0.0) or 0.0), 0.0
                                ) / 10.0
                                rollout_scale_v2_event_counts[1] += float(
                                    float(comp.get("scale_v1_uav_event_death", 0.0) or 0.0) < 0.0
                                    or float(comp.get("scale_v1_mav_event_death", 0.0) or 0.0) < 0.0
                                )
                                rollout_scale_v2_event_counts[2] += float(
                                    float(comp.get("scale_v1_uav_event_oob", 0.0) or 0.0) < 0.0
                                )
                                if role != "mav":
                                    progress = float(comp.get("scale_v2_progress", 0.0) or 0.0)
                                    raw_progress = float(comp.get("scale_v1_progress_raw", progress) or 0.0)
                                    if progress > 1e-12:
                                        rollout_scale_v2_progress_counts[0] += 1
                                    elif progress < -1e-12:
                                        rollout_scale_v2_progress_counts[1] += 1
                                    else:
                                        rollout_scale_v2_progress_counts[2] += 1
                                    if abs(progress - raw_progress) > 1e-12:
                                        rollout_scale_v2_progress_counts[3] += 1
                                    rollout_scale_v2_target_switch_count += float(
                                        comp.get("scale_v1_progress_reset_reason") == "target_switch"
                                    )
                            if role != "mav" and float(comp.get("uav_event_first_horizontal_out_of_zone", 0.0) or 0.0) != 0.0:
                                rollout_uav_oob_count += 1
                            if role == "mav" and float(comp.get("mav_event_death", 0.0) or 0.0) < 0.0:
                                altitude = float(comp.get("max_altitude_m", 0.0) or 0.0)
                                if altitude < float(getattr(rollout_env, "BATTLEFIELD_ALTITUDE_MIN", 2500.0)):
                                    rollout_mav_low_altitude_death_count += 1
                        active_count = 0.0
                        eff = {
                            "effective_team_brma_flight": 0.0,
                            "effective_team_uav_speed": 0.0,
                            "effective_team_uav_angle": 0.0,
                            "effective_team_uav_distance": 0.0,
                            "effective_team_uav_event": 0.0,
                            "effective_team_mav_safety": 0.0,
                            "effective_team_mav_support": 0.0,
                            "effective_team_mav_event": 0.0,
                            "effective_team_total": 0.0,
                            "effective_scale_v1_uav_flight": 0.0,
                            "effective_scale_v1_uav_progress": 0.0,
                            "effective_scale_v1_uav_event": 0.0,
                            "effective_scale_v1_mav_flight": 0.0,
                            "effective_scale_v1_mav_role": 0.0,
                            "effective_scale_v1_mav_event": 0.0,
                            "effective_scale_v1_terminal": 0.0,
                            "effective_scale_v1_total": 0.0,
                            "effective_scale_v2_uav_flight_raw": 0.0,
                            "effective_scale_v2_uav_flight_scaled": 0.0,
                            "effective_scale_v2_uav_progress": 0.0,
                            "effective_scale_v2_uav_event": 0.0,
                            "effective_scale_v2_mav_flight_raw": 0.0,
                            "effective_scale_v2_mav_flight_scaled": 0.0,
                            "effective_scale_v2_mav_role": 0.0,
                            "effective_scale_v2_mav_event": 0.0,
                            "effective_scale_v2_terminal": 0.0,
                            "effective_scale_v2_total": 0.0,
                        }
                        for ridx, rid in enumerate(rollout_env.red_ids):
                            try:
                                mask_value = float(active[ridx])
                            except (TypeError, ValueError, IndexError):
                                mask_value = 0.0
                            if mask_value <= 0.0:
                                continue
                            comp = rc.get(rid, {}) if isinstance(rc, dict) else {}
                            if not isinstance(comp, dict):
                                continue
                            active_count += mask_value
                            eff["effective_team_brma_flight"] += mask_value * sum(
                                float(comp.get(k, 0.0) or 0.0)
                                for k in ("brma_pitch", "brma_roll", "brma_vel")
                            )
                            eff["effective_team_uav_speed"] += mask_value * float(comp.get("tam_speed_weighted", 0.0) or 0.0)
                            eff["effective_team_uav_angle"] += mask_value * float(comp.get("tam_angle_weighted", 0.0) or 0.0)
                            eff["effective_team_uav_distance"] += mask_value * float(comp.get("tam_distance_weighted", 0.0) or 0.0)
                            eff["effective_team_uav_event"] += mask_value * float(comp.get("uav_event_total", 0.0) or 0.0)
                            eff["effective_team_mav_safety"] += mask_value * (
                                float(comp.get("mav_dist_weighted", 0.0) or 0.0)
                                + float(comp.get("mav_threat_weighted", 0.0) or 0.0)
                                + float(comp.get("mav_aspect_weighted", 0.0) or 0.0)
                            )
                            eff["effective_team_mav_support"] += mask_value * (
                                float(comp.get("mav_pos_weighted", 0.0) or 0.0)
                                + float(comp.get("mav_aware_weighted", 0.0) or 0.0)
                            )
                            eff["effective_team_mav_event"] += mask_value * float(comp.get("mav_event_total", 0.0) or 0.0)
                            eff["effective_team_total"] += mask_value * float(comp.get("total", 0.0) or 0.0)
                            if scale_v1:
                                if rollout_env.agent_roles.get(rid, "") == "mav":
                                    eff["effective_scale_v1_mav_flight"] += mask_value * float(comp.get("scale_v1_flight_total", 0.0) or 0.0)
                                    eff["effective_scale_v1_mav_role"] += mask_value * float(comp.get("scale_v1_mav_role", 0.0) or 0.0)
                                    eff["effective_scale_v1_mav_event"] += mask_value * float(comp.get("scale_v1_mav_event_total", 0.0) or 0.0)
                                else:
                                    eff["effective_scale_v1_uav_flight"] += mask_value * float(comp.get("scale_v1_flight_total", 0.0) or 0.0)
                                    eff["effective_scale_v1_uav_progress"] += mask_value * float(comp.get("scale_v1_progress_clipped", 0.0) or 0.0)
                                    eff["effective_scale_v1_uav_event"] += mask_value * float(comp.get("scale_v1_uav_event_total", 0.0) or 0.0)
                                eff["effective_scale_v1_terminal"] += mask_value * float(comp.get("scale_v1_terminal", 0.0) or 0.0)
                                eff["effective_scale_v1_total"] += mask_value * float(comp.get("total", 0.0) or 0.0)
                            if scale_v2:
                                f_raw = float(comp.get("scale_v2_flight_raw_total", 0.0) or 0.0)
                                f_scaled = float(comp.get("scale_v2_flight_scaled_total", 0.0) or 0.0)
                                if rollout_env.agent_roles.get(rid, "") == "mav":
                                    eff["effective_scale_v2_mav_flight_raw"] += mask_value * f_raw
                                    eff["effective_scale_v2_mav_flight_scaled"] += mask_value * f_scaled
                                    eff["effective_scale_v2_mav_role"] += mask_value * float(comp.get("scale_v2_mav_role", 0.0) or 0.0)
                                    eff["effective_scale_v2_mav_event"] += mask_value * float(comp.get("scale_v2_event", 0.0) or 0.0)
                                else:
                                    eff["effective_scale_v2_uav_flight_raw"] += mask_value * f_raw
                                    eff["effective_scale_v2_uav_flight_scaled"] += mask_value * f_scaled
                                    eff["effective_scale_v2_uav_progress"] += mask_value * float(comp.get("scale_v2_progress", 0.0) or 0.0)
                                    eff["effective_scale_v2_uav_event"] += mask_value * float(comp.get("scale_v2_event", 0.0) or 0.0)
                                eff["effective_scale_v2_terminal"] += mask_value * float(comp.get("scale_v2_terminal", 0.0) or 0.0)
                                eff["effective_scale_v2_total"] += mask_value * float(comp.get("total", 0.0) or 0.0)
                        denom = max(active_count, 1.0)
                        normalized = {}
                        for key, value in eff.items():
                            normalized[key] = float(value) / denom
                            current_ep_reward_comp[env_idx][key] = (
                                current_ep_reward_comp[env_idx].get(key, 0.0) + normalized[key]
                            )
                        if scale_v2:
                            normalized.update(_aggregate_scale_v2_step(
                                rollout_env.red_ids, rollout_env.agent_roles, active, rc
                            ))
                        component_sum = sum(
                            normalized[key] for key in (
                                "effective_team_brma_flight",
                                "effective_team_uav_speed",
                                "effective_team_uav_angle",
                                "effective_team_uav_distance",
                                "effective_team_uav_event",
                                "effective_team_mav_safety",
                                "effective_team_mav_support",
                                "effective_team_mav_event",
                            )
                        )
                        scale_component_sum = sum(normalized[key] for key in (
                            "effective_scale_v1_uav_flight", "effective_scale_v1_uav_progress",
                            "effective_scale_v1_uav_event", "effective_scale_v1_mav_flight",
                            "effective_scale_v1_mav_role", "effective_scale_v1_mav_event",
                            "effective_scale_v1_terminal",
                        ))
                        normalized["effective_scale_v1_component_sum"] = scale_component_sum
                        normalized["effective_scale_v1_identity_error"] = (
                            normalized["effective_scale_v1_total"] - scale_component_sum
                        )
                        scale_v2_component_sum = sum(normalized[key] for key in (
                            "effective_scale_v2_uav_flight_scaled", "effective_scale_v2_uav_progress",
                            "effective_scale_v2_uav_event", "effective_scale_v2_mav_flight_scaled",
                            "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
                            "effective_scale_v2_terminal",
                        )) if scale_v2 else 0.0
                        if scale_v2:
                            normalized["effective_scale_v2_component_sum"] = scale_v2_component_sum
                            normalized["effective_scale_v2_identity_error"] = (
                                normalized["effective_scale_v2_total"] - scale_v2_component_sum
                            )
                        current_ep_reward_comp[env_idx]["effective_scale_v1_component_sum"] = (
                            current_ep_reward_comp[env_idx].get("effective_scale_v1_component_sum", 0.0)
                            + scale_component_sum
                        )
                        current_ep_reward_comp[env_idx]["effective_scale_v1_identity_error"] = (
                            current_ep_reward_comp[env_idx].get("effective_scale_v1_identity_error", 0.0)
                            + normalized["effective_scale_v1_identity_error"]
                        )
                        current_ep_reward_comp[env_idx]["effective_component_sum"] = (
                            current_ep_reward_comp[env_idx].get("effective_component_sum", 0.0)
                            + component_sum
                        )
                        current_ep_reward_comp[env_idx]["effective_total_minus_component_sum"] = (
                            current_ep_reward_comp[env_idx].get("effective_total_minus_component_sum", 0.0)
                            + normalized["effective_team_total"] - component_sum
                        )
                        current_ep_reward_comp[env_idx]["effective_initial_team_mean_total"] = (
                            current_ep_reward_comp[env_idx].get("effective_initial_team_mean_total", 0.0)
                            + float(eff["effective_team_total"]) / max(float(len(rollout_env.red_ids)), 1.0)
                        )
                        current_ep_reward_comp[env_idx]["active_red_count"] = (
                            current_ep_reward_comp[env_idx].get("active_red_count", 0.0) + active_count
                        )
                        for key, value in normalized.items():
                            rollout_reward_sums[key] += value
                        rollout_reward_sums["effective_component_sum"] += component_sum
                        rollout_reward_sums["effective_total_minus_component_sum"] += (
                            normalized["effective_team_total"] - component_sum
                        )
                        rollout_reward_sums["active_red_count"] += active_count
                        rollout_reward_sums["effective_initial_team_mean_total"] += (
                            float(eff["effective_team_total"]) / max(float(len(rollout_env.red_ids)), 1.0)
                        )
                        rollout_reward_steps += 1
                    # Episode-local launch stats
                    for aid in rollout_env.agent_ids:
                        fired = int(next_info.get(aid, {}).get("missiles_fired_this_step", 0))
                        if aid.startswith("red_"):
                            current_ep_launch_stats[env_idx]["red_launch_count"] += fired
                        else:
                            current_ep_launch_stats[env_idx]["blue_launch_count"] += fired
                    mt = next_info.get("__missile_term__", {})
                    if isinstance(mt, dict):
                        red_hit_ep = int(mt.get("red", {}).get("hit", 0))
                        blue_hit_ep = int(mt.get("blue", {}).get("hit", 0))
                        red_delta = max(red_hit_ep - current_ep_hit_totals[env_idx]["red"], 0)
                        blue_delta = max(blue_hit_ep - current_ep_hit_totals[env_idx]["blue"], 0)
                        current_ep_launch_stats[env_idx]["red_hit_count"] += red_delta
                        current_ep_launch_stats[env_idx]["blue_hit_count"] += blue_delta
                        current_ep_hit_totals[env_idx]["red"] = red_hit_ep
                        current_ep_hit_totals[env_idx]["blue"] = blue_hit_ep
                    if rich_logger is not None:
                        rich_logger.write_missile_events(
                            next_info,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=_sim_time(rollout_env),
                        )
                        rich_logger.write_aircraft_timeseries(
                            rollout_env,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=_sim_time(rollout_env),
                        )
                        rich_logger.write_reward_components(
                            next_info,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=_sim_time(rollout_env),
                        )
                        rich_logger.write_reward_target_diagnostics(
                            next_info,
                            scenario=Path(args.config).stem,
                            episode_id=current_ep_id[env_idx],
                            step=total_steps,
                            sim_time=_sim_time(rollout_env),
                        )
                    for aid in rollout_env.agent_ids:
                        fired = int(next_info.get(aid, {}).get("missiles_fired_this_step", 0))
                        if aid.startswith("red_"):
                            red_fired += fired
                        else:
                            blue_fired += fired
                    mt = next_info.get("__missile_term__", {})
                    if isinstance(mt, dict):
                        red_hit_total = int(mt.get("red", {}).get("hit", 0))
                        blue_hit_total = int(mt.get("blue", {}).get("hit", 0))
                        red_hits += max(red_hit_total - prev_hit_totals[env_idx]["red"], 0)
                        blue_hits += max(blue_hit_total - prev_hit_totals[env_idx]["blue"], 0)
                        prev_hit_totals[env_idx]["red"] = red_hit_total
                        prev_hit_totals[env_idx]["blue"] = blue_hit_total
                    if done:
                        current_ep_reward_comp[env_idx]["initial_red_count"] = float(len(rollout_env.red_ids))
                        outcome = _episode_outcome(rollout_env, truncated, current_ep_len[env_idx])
                        ra, ba = _alive_counts(rollout_env)
                        episode_reward_comp = _finalize_happo_v1_episode_reward_components(
                            current_ep_reward_comp[env_idx],
                            current_ep_len[env_idx],
                        )
                        recent.append({
                            "return": float(current_ep_return[env_idx].mean()),
                            "episode_length": int(current_ep_len[env_idx]),
                            "winner": outcome["winner"],
                            "end_reason": outcome["end_reason"],
                            "mav": _mav_alive(rollout_env),
                            "red_alive": ra,
                            "blue_alive": ba,
                            "reward_comp": episode_reward_comp,
                        })
                        episodes += 1
                        # ---- Terminal episode audit CSV (BEFORE reset) ----
                        _death_events = next_info.get("death_events", []) if isinstance(next_info, dict) else []
                        _red_deaths = [e for e in _death_events if str(e.get("agent_id","")).startswith("red_")]
                        _blue_deaths = [e for e in _death_events if str(e.get("agent_id","")).startswith("blue_")]
                        _ra_env, _ba_env = ra, ba
                        _ra_info = sum(1 for rid in rollout_env.red_ids if next_info.get(rid, {}).get("alive", False))
                        _ba_info = sum(1 for bid in rollout_env.blue_ids if next_info.get(bid, {}).get("alive", False))
                        _mav_info = bool(next_info.get("red_0", {}).get("alive", False))
                        if _ra_env != _ra_info or _ba_env != _ba_info:
                            print(f"  [WARN] alive mismatch env vs info: red env={_ra_env} info={_ra_info} blue env={_ba_env} info={_ba_info}")
                        with open(out_dir / "terminal_episode_audit.csv", "a", newline="") as _ta_f:
                            _ta_w = csv.writer(_ta_f)
                            if _ta_f.tell() == 0:
                                _ta_w.writerow(["iteration","total_steps","env_idx","episode_id","episode_length","winner","end_reason","all_terminated","all_truncated","red_alive_env","blue_alive_env","mav_alive_env","red_alive_info","blue_alive_info","mav_alive_info","alive_mismatch","red_death_reasons","blue_death_reasons","red_launch_count","blue_launch_count","red_hit_count","blue_hit_count"])
                            _ta_w.writerow([iteration, total_steps, env_idx, episodes, current_ep_len[env_idx], outcome["winner"], outcome["end_reason"], int(all(terminated.values())), int(all(truncated.values())), _ra_env, _ba_env, int(_mav_alive(rollout_env)), _ra_info, _ba_info, int(_mav_info), int(_ra_env != _ra_info or _ba_env != _ba_info), str([e.get("death_reason","?") for e in _red_deaths]), str([e.get("death_reason","?") for e in _blue_deaths]), red_fired, blue_fired, red_hits, blue_hits])
                        # Write per-agent episode reward components
                        if rich_logger is not None:
                            for rid in rollout_env.red_ids:
                                agent_idx = rollout_env.red_ids.index(rid)
                                ep_ret_val = float(current_ep_return[env_idx][agent_idx])
                                agent_component_sums = _finalize_happo_v1_episode_reward_components(
                                    current_ep_reward_comp_by_agent[env_idx].get(rid, {}),
                                    current_ep_len[env_idx],
                                )
                                rich_logger.write_episode_reward_components(
                                    scenario=Path(args.config).stem,
                                    episode_id=current_ep_id[env_idx],
                                    agent_id=rid,
                                    role=rollout_env.agent_roles.get(rid, ""),
                                    team="red",
                                    episode_length=current_ep_len[env_idx],
                                    episode_return=ep_ret_val,
                                    component_sums=agent_component_sums,
                                    launch_stats=dict(current_ep_launch_stats[env_idx]),
                                    final_state={
                                        "mav_alive_final": int(_mav_alive(rollout_env)),
                                        "red_alive_final": ra,
                                        "blue_alive_final": ba,
                                    },
                                    outcome=outcome["winner"],
                                    end_reason=outcome["end_reason"],
                                )
                        current_ep_return[env_idx][:] = 0.0
                        current_ep_len[env_idx] = 0
                        current_ep_reward_comp[env_idx] = {}
                        current_ep_reward_comp_by_agent[env_idx] = {rid: {} for rid in rollout_env.red_ids}
                        current_ep_launch_stats[env_idx] = {
                            "red_launch_count": 0, "red_hit_count": 0,
                            "blue_launch_count": 0, "blue_hit_count": 0,
                        }
                        current_ep_hit_totals[env_idx] = {"red": 0, "blue": 0}
                        heartbeat.write(
                            "before_reset",
                            iteration=iteration,
                            rollout_local_step=rollout_local_step,
                            env_idx=env_idx,
                            total_steps=total_steps,
                            episode_length=0,
                            env_episode_id=current_ep_id[env_idx],
                            alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                            missile_count=_missile_count(rollout_env),
                            sim_time=_sim_time(rollout_env),
                            done=done,
                            terminated=bool(all(terminated.values())) if terminated else False,
                            truncated=bool(all(truncated.values())) if truncated else False,
                        )
                        next_obs, next_info = rollout_env.reset(seed=args.seed + total_steps + env_idx)
                        opponents[env_idx].reset_memory()
                        current_ep_id[env_idx] += 1
                        _SINGLE_RUNNER_STATE["episode_id"] = current_ep_id[env_idx]
                        if rnn_hidden is not None:
                            rnn_hidden[env_idx][:] = 0.0
                        heartbeat.write(
                            "after_reset",
                            iteration=iteration,
                            rollout_local_step=rollout_local_step,
                            env_idx=env_idx,
                            total_steps=total_steps,
                            episode_length=0,
                            env_episode_id=current_ep_id[env_idx],
                            alive_agents=dict(zip(("red", "blue"), _alive_counts(rollout_env))),
                            missile_count=_missile_count(rollout_env),
                            sim_time=_sim_time(rollout_env),
                            done=done,
                            terminated=bool(all(terminated.values())) if terminated else False,
                            truncated=bool(all(truncated.values())) if truncated else False,
                        )
                        prev_hit_totals[env_idx] = {"red": 0, "blue": 0}
                    obs_list[env_idx] = next_obs
                    info_list[env_idx] = next_info
                if nan_detected:
                    break
            if nan_detected:
                break
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
            if args.policy_arch in {"pure_happo"}:
                stats = trainer.update(buffer)
            else:
                stats = trainer.update(
                    buffer,
                    uav_imitation_batch=imitation_batch,
                    uav_imitation_coef=args.uav_imitation_coef if imitation_active else 0.0,
                )
            stats.update(_rollout_distribution_stats(buffer))
            m_means = np.asarray(stats.get("m_mean_after_each_agent", []), dtype=np.float64)
            m_stds = np.asarray(stats.get("m_std_after_each_agent", []), dtype=np.float64)
            m_abs_means = np.asarray(stats.get("m_abs_mean_after_each_agent", []), dtype=np.float64)
            m_abs_maxes = np.asarray(stats.get("m_abs_max_after_each_agent", []), dtype=np.float64)
            stats.update({
                "advantage_raw_abs_max": max(
                    abs(float(stats.get("advantage_raw_min", 0.0))),
                    abs(float(stats.get("advantage_raw_max", 0.0))),
                ),
                "correction_M_mean": float(m_means[-1]) if m_means.size else 0.0,
                "correction_M_std": float(m_stds[-1]) if m_stds.size else 0.0,
                "correction_M_mean_abs": float(m_abs_means[-1]) if m_abs_means.size else 0.0,
                "correction_M_max_abs": float(m_abs_maxes[-1]) if m_abs_maxes.size else 0.0,
            })
            reward_step_denom = max(rollout_reward_steps, 1)
            for key in (
                "effective_team_brma_flight", "effective_team_uav_speed",
                "effective_team_uav_angle", "effective_team_uav_distance",
                "effective_team_uav_event", "effective_team_mav_safety",
                "effective_team_mav_support", "effective_team_mav_event",
                "effective_team_total", "effective_component_sum",
                "effective_total_minus_component_sum", "active_red_count",
                "effective_initial_team_mean_total",
                "effective_scale_v1_uav_flight", "effective_scale_v1_uav_progress",
                "effective_scale_v1_uav_event", "effective_scale_v1_mav_flight",
                "effective_scale_v1_mav_role", "effective_scale_v1_mav_event",
                "effective_scale_v1_terminal", "effective_scale_v1_total",
                "effective_scale_v1_component_sum", "effective_scale_v1_identity_error",
                "effective_scale_v2_uav_flight_raw", "effective_scale_v2_uav_flight_scaled",
                "effective_scale_v2_uav_progress", "effective_scale_v2_uav_event",
                "effective_scale_v2_mav_flight_raw", "effective_scale_v2_mav_flight_scaled",
                "effective_scale_v2_mav_role", "effective_scale_v2_mav_event",
                "effective_scale_v2_terminal", "effective_scale_v2_total",
                "effective_scale_v2_component_sum", "effective_scale_v2_identity_error",
            ):
                stats[key] = rollout_reward_sums[key] / reward_step_denom
            for key in V3_EFFECTIVE_FIELDS:
                count = rollout_reward_counts[key]
                stats[key] = rollout_reward_sums[key] / count if count > 0.0 else 0.0
            stats.update({
                "initial_red_count": float(len(env.red_ids)),
                "attack_uav_reward_target_distance_mean": (
                    rollout_target_distance_sum / max(rollout_target_valid_count, 1)
                ),
                "reward_target_valid_ratio": rollout_target_valid_count / max(rollout_reward_steps * max(len(env.red_ids) - 1, 1), 1),
                "reward_distance_le_5km_ratio": rollout_target_zone_counts[0] / max(rollout_target_valid_count, 1),
                "reward_distance_5_to_10km_ratio": rollout_target_zone_counts[1] / max(rollout_target_valid_count, 1),
                "reward_distance_ge_10km_ratio": (rollout_target_zone_counts[2] + rollout_target_zone_counts[3]) / max(rollout_target_valid_count, 1),
                "reward_distance_10_to_20km_ratio": rollout_target_zone_counts[2] / max(rollout_target_valid_count, 1),
                "reward_distance_ge_20km_ratio": rollout_target_zone_counts[3] / max(rollout_target_valid_count, 1),
                "uav_first_horizontal_oob_count": float(rollout_uav_oob_count),
                "mav_low_altitude_death_count": float(rollout_mav_low_altitude_death_count),
                "reward_identity_max_error_mav": rollout_identity_max["mav"],
                "reward_identity_max_error_uav": rollout_identity_max["uav"],
                "reward_identity_max_error": max(rollout_identity_max.values()),
                "reward_identity_failure_count": float(rollout_identity_failures),
                "scale_v1_progress_positive_ratio": rollout_scale_progress_counts[0] / max(sum(rollout_scale_progress_counts[:3]), 1),
                "scale_v1_progress_negative_ratio": rollout_scale_progress_counts[1] / max(sum(rollout_scale_progress_counts[:3]), 1),
                "scale_v1_progress_zero_ratio": rollout_scale_progress_counts[2] / max(sum(rollout_scale_progress_counts[:3]), 1),
                "scale_v1_progress_clip_ratio": rollout_scale_progress_counts[3] / max(sum(rollout_scale_progress_counts[:3]), 1),
                "scale_v1_target_switch_count": rollout_scale_target_switch_count,
                "scale_v1_mav_aspect_raw_sum_mean": rollout_scale_mav_aspect_sum / max(rollout_scale_mav_rows, 1),
                "scale_v1_mav_aspect_mean": rollout_scale_mav_aspect_mean_sum / max(rollout_scale_mav_rows, 1),
                "scale_v1_mav_awareness_raw_sum_mean": rollout_scale_mav_aware_sum / max(rollout_scale_mav_rows, 1),
                "scale_v1_mav_awareness_mean": rollout_scale_mav_aware_mean_sum / max(rollout_scale_mav_rows, 1),
                "scale_v1_terminal_mean": rollout_scale_terminal_sum / max(rollout_reward_steps, 1),
                "scale_v1_kill_count": rollout_scale_event_counts[0],
                "scale_v1_death_count": rollout_scale_event_counts[1],
                "scale_v1_oob_count": rollout_scale_event_counts[2],
                "scale_v2_progress_positive_ratio": rollout_scale_v2_progress_counts[0] / max(sum(rollout_scale_v2_progress_counts[:3]), 1),
                "scale_v2_progress_negative_ratio": rollout_scale_v2_progress_counts[1] / max(sum(rollout_scale_v2_progress_counts[:3]), 1),
                "scale_v2_progress_zero_ratio": rollout_scale_v2_progress_counts[2] / max(sum(rollout_scale_v2_progress_counts[:3]), 1),
                "scale_v2_progress_clip_ratio": rollout_scale_v2_progress_counts[3] / max(sum(rollout_scale_v2_progress_counts[:3]), 1),
                "scale_v2_target_switch_count": rollout_scale_v2_target_switch_count,
                "scale_v2_terminal_mean": rollout_scale_v2_terminal_sum / max(rollout_scale_v2_active_rows, 1),
                "scale_v2_kill_count": rollout_scale_v2_event_counts[0],
                "scale_v2_death_count": rollout_scale_v2_event_counts[1],
                "scale_v2_oob_count": rollout_scale_v2_event_counts[2],
                "scale_v2_flight_raw_mean": rollout_scale_v2_flight_raw_sum / max(rollout_scale_v2_active_rows, 1),
                "scale_v2_flight_scaled_mean": rollout_scale_v2_flight_scaled_sum / max(rollout_scale_v2_active_rows, 1),
            })
            stats.update({
                "rollout_transitions": rollout_transitions,
                "ppo_epochs": args.ppo_epochs,
                "critic_epochs": args.critic_epochs,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "clip_param": args.clip_param,
                "entropy_coef": args.entropy_coef,
                "value_coef": args.value_coef,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "max_grad_norm": args.max_grad_norm,
            })
            update_diag = {
                "iteration": iteration,
                "total_steps": total_steps,
                "policy_arch": args.policy_arch,
            }
            for key in UPDATE_DIAGNOSTIC_ARRAY_FIELDS:
                if key in stats:
                    value = stats.get(key)
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        value = float(value) if isinstance(value, np.floating) else int(value)
                    update_diag[key] = value
            with (out_dir / "update_diagnostics.jsonl").open("a", encoding="utf-8") as _udf:
                _udf.write(json.dumps(update_diag, ensure_ascii=False, default=str) + "\n")
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
            rc_sum = {}
            for r in rec:
                for key, value in r.get("reward_comp", {}).items():
                    rc_sum[key] = rc_sum.get(key, 0.0) + float(value)
            rc_mean = {
                key: _recent_component_mean(rec, key)
                for key in (
                    "mav_observed_ratio",
                    "mav_shared_track_ratio",
                    "red_launch_rate_before_mav_death",
                    "red_launch_rate_after_mav_death",
                )
            }
            stats["avg_episode_length"] = (
                float(np.mean([r.get("episode_length", 0) for r in rec])) if rec else 0.0
            )
            stats["avg_return_per_decision"] = (
                float(np.mean([
                    float(r["return"]) / max(int(r.get("episode_length", 0)), 1)
                    for r in rec
                ])) if rec else 0.0
            )
            writer.writerow([
                iteration, total_steps, actual_reward_mode, f"{avg_return:.4f}", f"{red_win:.4f}",
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
                f"{rc_sum.get('paper_v1_uav_flight', 0):.4f}",
                f"{rc_sum.get('paper_v1_uav_adv', 0):.4f}",
                f"{rc_sum.get('paper_v1_uav_end', 0):.4f}",
                f"{rc_sum.get('paper_v1_uav_total', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_flight', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_safety', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_support', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_event_raw', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_scaled_tam', 0):.4f}",
                f"{rc_sum.get('paper_v1_mav_total', 0):.4f}",
                f"{rc_sum.get('v1_mav_safety', 0):.4f}",
                f"{rc_sum.get('v1_mav_support', 0):.4f}",
                f"{rc_sum.get('v1_mav_event', 0):.4f}",
                f"{rc_sum.get('v1_mav_total', 0):.4f}",
                f"{rc_mean.get('mav_observed_ratio', 0):.4f}",
                f"{rc_mean.get('mav_shared_track_ratio', 0):.4f}",
                f"{rc_sum.get('red_launch_with_mav_shared_track', 0):.4f}",
                f"{rc_sum.get('red_hit_with_mav_shared_track', 0):.4f}",
                f"{rc_sum.get('team_kill_while_mav_alive', 0):.4f}",
                f"{rc_sum.get('team_kill_after_mav_death', 0):.4f}",
                f"{rc_mean.get('red_launch_rate_before_mav_death', 0):.4f}",
                f"{rc_mean.get('red_launch_rate_after_mav_death', 0):.4f}",
                f"{rc_sum.get('v1_mav_removed_r_adv', 0):.4f}",
                f"{rc_sum.get('v1_mav_removed_r_end', 0):.4f}",
                f"{stats.get('effective_team_brma_flight', 0):.4f}",
                f"{stats.get('effective_team_uav_speed', 0):.4f}",
                f"{stats.get('effective_team_uav_angle', 0):.4f}",
                f"{stats.get('effective_team_uav_distance', 0):.4f}",
                f"{stats.get('effective_team_uav_event', 0):.4f}",
                f"{stats.get('effective_team_mav_safety', 0):.4f}",
                f"{stats.get('effective_team_mav_support', 0):.4f}",
                f"{stats.get('effective_team_mav_event', 0):.4f}",
                f"{stats.get('effective_team_total', 0):.4f}",
                f"{stats.get('active_red_count', 0):.4f}",
                f"{stats.get('effective_component_sum', 0):.4f}",
                f"{stats.get('effective_total_minus_component_sum', 0):.4f}",
                f"{stats.get('initial_red_count', 0):.4f}",
                f"{stats.get('effective_initial_team_mean_total', 0):.4f}",
                *[_format_metric(stats.get(field, 0.0)) for field in MARL_DYNAMICS_TRAIN_FIELDS],
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
                    "avg_mav_reward": "",
                    "avg_uav_reward": "",
                    "red_win_rate": red_win,
                    "blue_win_rate": blue_win,
                    "draw_rate": draw,
                    "timeout_rate": timeout,
                    "red_elimination_win_rate": sum(1 for r in rec if r["end_reason"] == "blue_eliminated") / n,
                    "red_timeout_alive_advantage_rate": sum(
                        1 for r in rec if r["winner"] == "red" and r["end_reason"] == "timeout"
                    ) / n,
                    "mav_survival_rate": mav_surv,
                    "red_alive_final_mean": red_alive,
                    "blue_alive_final_mean": blue_alive,
                    "red_missiles_fired_mean": red_fired / max(args.num_envs, 1),
                    "blue_missiles_fired_mean": blue_fired / max(args.num_envs, 1),
                    "red_missile_hits_mean": red_hits / max(args.num_envs, 1),
                    "blue_missile_hits_mean": "",
                    "red_dead_mean": red_dead,
                    "blue_dead_mean": blue_dead,
                    "kill_death_ratio": blue_dead / max(red_dead, 1e-6),
                    "relative_win_ratio": red_win / max(blue_win, 1e-6),
                    "actor_loss": (stats["actor_loss_mav"] + stats["actor_loss_uav"]) / 2.0,
                    "critic_loss": stats["critic_loss"],
                    "entropy": (stats["entropy_mav"] + stats["entropy_uav"]) / 2.0,
                    "policy_gradient_norm": "",
                    "value_gradient_norm": "",
                    "action_saturation_rate": max(
                        stats["mav_action_saturation_rate"],
                        stats["uav_action_saturation_rate"],
                    ),
                    "mav_action_saturation_rate": stats["mav_action_saturation_rate"],
                    "uav_action_saturation_rate": stats["uav_action_saturation_rate"],
                    "approx_kl_mav": stats.get("approx_kl_mav", 0.0),
                    "approx_kl_uav": stats.get("approx_kl_uav", 0.0),
                    "mask_keep_ratio": stats.get("mask_keep_ratio", 1.0),
                    "mask_entropy": stats.get("mask_entropy", 0.0),
                    "masked_entity_count": stats.get("masked_entity_count", 0.0),
                    "effective_scale_v1_total": stats.get("effective_scale_v1_total", 0.0),
                    "effective_scale_v1_identity_error": stats.get("effective_scale_v1_identity_error", 0.0),
                    "scale_v1_progress_positive_ratio": stats.get("scale_v1_progress_positive_ratio", 0.0),
                    "scale_v1_progress_clip_ratio": stats.get("scale_v1_progress_clip_ratio", 0.0),
                    "effective_scale_v2_total": stats.get("effective_scale_v2_total", 0.0),
                    "effective_scale_v2_uav_flight_raw": stats.get("effective_scale_v2_uav_flight_raw", 0.0),
                    "effective_scale_v2_uav_flight_scaled": stats.get("effective_scale_v2_uav_flight_scaled", 0.0),
                    "effective_scale_v2_uav_progress": stats.get("effective_scale_v2_uav_progress", 0.0),
                    "effective_scale_v2_uav_event": stats.get("effective_scale_v2_uav_event", 0.0),
                    "effective_scale_v2_mav_flight_raw": stats.get("effective_scale_v2_mav_flight_raw", 0.0),
                    "effective_scale_v2_mav_flight_scaled": stats.get("effective_scale_v2_mav_flight_scaled", 0.0),
                    "effective_scale_v2_mav_role": stats.get("effective_scale_v2_mav_role", 0.0),
                    "effective_scale_v2_mav_event": stats.get("effective_scale_v2_mav_event", 0.0),
                    "effective_scale_v2_terminal": stats.get("effective_scale_v2_terminal", 0.0),
                    "effective_scale_v2_component_sum": stats.get("effective_scale_v2_component_sum", 0.0),
                    "effective_scale_v2_identity_error": stats.get("effective_scale_v2_identity_error", 0.0),
                    "scale_v2_progress_positive_ratio": stats.get("scale_v2_progress_positive_ratio", 0.0),
                    "scale_v2_progress_negative_ratio": stats.get("scale_v2_progress_negative_ratio", 0.0),
                    "scale_v2_progress_zero_ratio": stats.get("scale_v2_progress_zero_ratio", 0.0),
                    "scale_v2_progress_clip_ratio": stats.get("scale_v2_progress_clip_ratio", 0.0),
                    "scale_v2_target_switch_count": stats.get("scale_v2_target_switch_count", 0.0),
                    "scale_v2_terminal_mean": stats.get("scale_v2_terminal_mean", 0.0),
                    "scale_v2_kill_count": stats.get("scale_v2_kill_count", 0.0),
                    "scale_v2_death_count": stats.get("scale_v2_death_count", 0.0),
                    "scale_v2_oob_count": stats.get("scale_v2_oob_count", 0.0),
                    "scale_v2_flight_raw_mean": stats.get("scale_v2_flight_raw_mean", 0.0),
                    "scale_v2_flight_scaled_mean": stats.get("scale_v2_flight_scaled_mean", 0.0),
                    "final_approx_kl_mav": stats.get("final_approx_kl_mav", 0.0),
                    "final_approx_kl_uav": stats.get("final_approx_kl_uav", 0.0),
                    "final_approx_kl_abs_mav": stats.get("final_approx_kl_abs_mav", 0.0),
                    "final_approx_kl_abs_uav": stats.get("final_approx_kl_abs_uav", 0.0),
                    "final_clip_fraction_mav": stats.get("final_clip_fraction_mav", 0.0),
                    "final_clip_fraction_uav": stats.get("final_clip_fraction_uav", 0.0),
                    "final_ratio_mean_mav": stats.get("final_ratio_mean_mav", 0.0),
                    "final_ratio_mean_uav": stats.get("final_ratio_mean_uav", 0.0),
                    "final_ratio_std_mav": stats.get("final_ratio_std_mav", 0.0),
                    "final_ratio_std_uav": stats.get("final_ratio_std_uav", 0.0),
                    "final_ratio_p95_mav": stats.get("final_ratio_p95_mav", 0.0),
                    "final_ratio_p95_uav": stats.get("final_ratio_p95_uav", 0.0),
                    "final_ratio_p99_mav": stats.get("final_ratio_p99_mav", 0.0),
                    "final_ratio_p99_uav": stats.get("final_ratio_p99_uav", 0.0),
                    "final_actor_parameter_delta_mav": stats.get("final_actor_parameter_delta_mav", 0.0),
                    "final_actor_parameter_delta_uav": stats.get("final_actor_parameter_delta_uav", 0.0),
                    **{
                        key: stats.get(key, 0.0)
                        for key in (
                            "actor_loss_mav", "actor_loss_uav",
                            "critic_loss_scaled", "critic_loss_unscaled",
                            "value_explained_variance_old", "value_explained_variance_new",
                            "return_mean", "return_std", "value_pred_old_mean",
                            "value_pred_old_std", "value_pred_new_mean", "value_pred_new_std",
                            "advantage_raw_mean", "advantage_raw_std", "advantage_raw_abs_max",
                            "actor_grad_norm_mav", "actor_grad_norm_uav", "critic_grad_norm",
                            "policy_update_norm_mav", "policy_update_norm_uav", "critic_update_norm",
                            "entropy_mav", "entropy_uav", "action_log_std_mav_mean",
                            "action_log_std_uav_mean", "correction_M_mean", "correction_M_std",
                            "correction_M_mean_abs", "correction_M_max_abs", "reward_nan_count",
                            "action_nan_count", "value_nan_count", "log_prob_nan_count",
                            "gradient_nonfinite_count",
                        )
                    },
                    **{key: stats.get(key, 0.0) for key in V3_EFFECTIVE_FIELDS},
                    "nan_detected": int(nan_detected),
                })
            if not f.closed:
                f.flush()
            heartbeat.write(
                "after_logging",
                iteration=iteration,
                rollout_local_step=len(buffer),
                env_idx="all",
                total_steps=total_steps,
                episode_length="",
                alive_agents=dict(zip(("red", "blue"), _alive_counts(env))),
            )
            print(
                _format_happo_stdout_line(
                    iteration=iteration,
                    total_steps=total_steps,
                    target_steps=args.total_env_steps,
                    rec_count=len(rec),
                    avg_return=avg_return,
                    red_win=red_win,
                    blue_win=blue_win,
                    draw=draw,
                    timeout=timeout,
                    red_alive=red_alive,
                    blue_alive=blue_alive,
                    mav_survival=mav_surv,
                ),
                flush=True,
            )
            # ---- Periodic checkpoint ----
            if args.checkpoint_interval_steps > 0:
                milestone = (total_steps // args.checkpoint_interval_steps) * args.checkpoint_interval_steps
                prev_milestone = ((total_steps - transitions_per_rollout) // args.checkpoint_interval_steps) * args.checkpoint_interval_steps
                if milestone > prev_milestone or total_steps >= args.total_env_steps:
                    ckpt_dir = out_dir / "checkpoints" / f"step_{total_steps:09d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    policy.save(ckpt_dir / "model.pt")
                    (ckpt_dir / "meta.json").write_text(json.dumps({
                        "algorithm": "happo_reference_v0",
                        "policy_arch": args.policy_arch,
                        **base_v2_meta,
                        "actor_obs_dim": actor_dim,
                        "critic_state_dim": critic_dim,
                        "action_dim": action_dim,
                        "entity_dim": getattr(policy, "entity_dim", None),
                        "num_agents": len(env.red_ids),
                        "total_env_steps": total_steps,
                        "iteration": iteration,
                    }, indent=2), encoding="utf-8")
                    # Rotate old checkpoints
                    existing = sorted(
                        (out_dir / "checkpoints").glob("step_*"),
                        key=lambda p: p.name,
                    )
                    while len(existing) > args.keep_checkpoints:
                        import shutil
                        shutil.rmtree(existing.pop(0), ignore_errors=True)
            if total_steps - last_eval >= args.eval_interval_steps and args.eval_during_training:
                last_eval = total_steps
                tmp_model = out_dir / "_tmp_eval.pt"
                _ckpt_finite = all(
                    torch.isfinite(p).all() for _n, p in policy.named_parameters()
                )
                if not _ckpt_finite:
                    raise RuntimeError(
                        "Refusing to save non-finite checkpoint at "
                        f"iter={iteration} steps={total_steps}"
                    )
                policy.save(tmp_model)
                (out_dir / "_tmp_eval_meta.json").write_text(json.dumps({
                    "algorithm": "happo_reference_v0",
                    "policy_arch": args.policy_arch,
                    **base_v2_meta,
                    "actor_obs_dim": actor_dim,
                    "critic_state_dim": critic_dim,
                    "action_dim": action_dim,
                    "entity_dim": getattr(policy, "entity_dim", None),
                    "attention": args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "brma_entity_encoder": args.policy_arch in {"brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "recurrent": args.policy_arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
                    "rnn_hidden_size": getattr(policy, "rnn_hidden_size", None),
                    "random_scale_mask": bool(getattr(policy, "random_scale_mask", False)),
                    "biased_mask": bool(getattr(policy, "biased_mask", False)),
                    "random_mask_prob": float(getattr(policy, "random_mask_prob", 0.0)),
                    **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
                }, indent=2), encoding="utf-8")
                (out_dir / "meta.json").unlink(missing_ok=True)
                (tmp_model.parent / "meta.json").write_text(
                    (out_dir / "_tmp_eval_meta.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                tmp_json = str((out_dir / "_tmp_eval.json").relative_to(ROOT))
                heartbeat.write(
                    "before_eval",
                    iteration=iteration,
                    rollout_local_step=len(buffer),
                    env_idx="all",
                    total_steps=total_steps,
                    episode_length="",
                    alive_agents=dict(zip(("red", "blue"), _alive_counts(env))),
                )
                eval_configs_for_this_run = args.eval_configs or DEFAULT_EVAL_CONFIGS
                if args.policy_arch in {"pure_happo"}:
                    eval_configs_for_this_run = _resolve_pure_happo_eval_configs(args, policy)
                    records = _run_eval(str(tmp_model), args, tmp_json,
                                       eval_configs_override=eval_configs_for_this_run)
                else:
                    records = _run_eval(str(tmp_model), args, tmp_json)
                heartbeat.write(
                    "after_eval",
                    iteration=iteration,
                    rollout_local_step=len(buffer),
                    env_idx="all",
                    total_steps=total_steps,
                    episode_length="",
                    alive_agents=dict(zip(("red", "blue"), _alive_counts(env))),
                )
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
                                    args,
                                    policy,
                                    actor_dim,
                                    critic_dim,
                                    transitions_per_rollout,
                                ),
                                "eval_checkpoint_metric": args.eval_checkpoint_metric,
                            },
                        )
                        eval_ckpt_dir = out_dir / "eval_checkpoints" / f"step_{total_steps:06d}"
                        _save_policy_checkpoint(policy, eval_ckpt_dir, meta)
                        _prune_eval_checkpoints(out_dir / "eval_checkpoints", args.keep_eval_checkpoints)
                        for best_name in (
                            "best_3v2", "best_5v4", "best_7v6", "best_combined"):
                            metric_name = best_metric_name(best_name)
                            metric_score = float(meta["scores"].get(metric_name, 0.0))
                            if metric_score > eval_best_scores[best_name]:
                                eval_best_scores[best_name] = metric_score
                                best_meta = dict(meta)
                                best_meta["best_kind"] = best_name
                                best_meta["best_score"] = metric_score
                                best_meta["best_score_metric"] = metric_name
                                _save_policy_checkpoint(policy, out_dir / best_name, best_meta)
                    score = _score_eval(records, args.eval_checkpoint_metric)
                    if score > best_score:
                        best_score = score
                        policy.save(out_dir / "best" / "model.pt")
                        (out_dir / "best" / "meta.json").write_text(json.dumps({
                            "algorithm": "happo_reference_v0",
                            "policy_arch": args.policy_arch,
                            **base_v2_meta,
                            "opponent_policy": args.opponent_policy,
                            "best_score": best_score,
                            "actor_obs_dim": actor_dim,
                            "critic_state_dim": critic_dim,
                            "action_dim": action_dim,
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
                            **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
                        }, indent=2), encoding="utf-8")
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
        "policy_arch": args.policy_arch,
        "config": args.config,
        **base_v2_meta,
        "opponent_policy": args.opponent_policy,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": critic_dim,
        "action_dim": action_dim,
        "entity_dim": getattr(policy, "entity_dim", None),
        "separate_actors": True,
        "centralized_critic": True,
        "sequential_update": True,
        "sequential_update_detail": "simplified HAPPO-style v0 role-wise PPO",
        "attention": args.policy_arch in {"entity_attention", "brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "brma_entity_encoder": args.policy_arch in {"brma_entity", "brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "recurrent": args.policy_arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "rnn_hidden_size": getattr(policy, "rnn_hidden_size", None),
        "random_scale_mask": bool(getattr(policy, "random_scale_mask", False)),
        "biased_mask": bool(getattr(policy, "biased_mask", False)),
        "random_mask_prob": float(getattr(policy, "random_mask_prob", 0.0)),
        "missile_scripted": True,
        "evasion_scripted": True,
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
        **_entity_policy_meta(policy),
        **_pure_happo_meta(policy, args),
    }
    (out_dir / "latest" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    launch_diagnostics_status = _run_launch_diagnostics_after_training(args, out_dir)
    meta["launch_diagnostics_status"] = launch_diagnostics_status
    (out_dir / "latest" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "main_experiment_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_runner_status(
        out_dir,
        status="normal",
        total_steps=total_steps,
        iteration=iteration,
        extra={**base_v2_meta, "launch_diagnostics_status": launch_diagnostics_status},
    )
    if rich_logger is not None:
        rich_logger.write_training_efficiency(total_steps, nan_detected=nan_detected)
        rich_logger.close()
    watchdog.stop()
    heartbeat.close()
    for rollout_env in envs:
        rollout_env.close()
    print(f"Saved {latest_model}", flush=True)


def main() -> None:
    try:
        _run_training_main()
    except (KeyboardInterrupt, Exception) as exc:
        out_dir = _SINGLE_RUNNER_STATE.get("output_dir")
        if out_dir is not None:
            _write_failure_artifacts(
                _SINGLE_RUNNER_STATE.get("policy"), _SINGLE_RUNNER_STATE, exc
            )
        raise
    finally:
        _cleanup_single_runner()


if __name__ == "__main__":
    main()
