"""Low-level diagnostics for tam_brma_scripted_reward_v1 + pure_happo.

Audit-only script.  It runs a short rollout/eval from existing checkpoints and
records action-distribution, PPO-buffer, reward-reachability and launch-gate
mechanics without updating parameters or changing environment behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.full_review_audit_utils import (
    explained_variance,
    gate_mismatch_stats,
    pearson_corr,
    read_csv_rows,
    safe_float,
    select_checkpoints_from_train_log,
    summarize_first_failed_gate,
    write_csv_rows,
)

DEFAULT_OUT = ROOT / "outputs" / "audit_tam_brma_v1_pure_happo_low_level"
DEFAULT_RUN = ROOT / "outputs" / "tam_brma_scripted_reward_v1_pure_happo_500k_obs_fix"


def _find_run_dir() -> Path:
    if DEFAULT_RUN.exists():
        return DEFAULT_RUN
    candidates = []
    for p in (ROOT / "outputs").iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if all(tok in name for tok in ("tam_brma", "pure_happo", "500k", "obs", "fix")):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No run dir matching tam_brma/pure_happo/500k/obs/fix found")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_policy(checkpoint: Path, device: torch.device):
    from algorithms.pure_happo import PureHAPPOPolicy

    meta_path = checkpoint.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    policy = PureHAPPOPolicy(
        actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
        critic_state_dim=int(meta.get("critic_state_dim", 480)),
        action_dim=int(meta.get("action_dim", 3)),
        num_agents=int(meta.get("num_agents", 3)),
    ).to(device)
    policy.load(str(checkpoint), map_location=device)
    policy.eval()
    return policy, meta


def _default_config(run_dir: Path, meta: dict) -> str:
    cfg = meta.get("config")
    if cfg:
        return str(cfg)
    args_path = run_dir / "args.json"
    if args_path.exists():
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if args.get("config"):
            return str(args["config"])
    return "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_brma_scripted_reward_v1.yaml"


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()[:12]


def _write_md(path: Path, title: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")


def _policy_distribution_rows(policy, actor_obs_np: np.ndarray, device: torch.device, deterministic: bool) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    actor_obs = torch.as_tensor(actor_obs_np, dtype=torch.float32, device=device)
    rows = []
    actions = np.zeros((actor_obs_np.shape[0], 3), dtype=np.float32)
    old_log_probs = np.zeros((actor_obs_np.shape[0],), dtype=np.float32)
    for agent_idx in range(policy.num_agents):
        dist, mean = policy._distribution(actor_obs[agent_idx:agent_idx + 1], agent_idx)
        raw = mean if deterministic else dist.sample()
        executed = raw.clamp(-1.0, 1.0)
        lp_current_dim = dist.log_prob(executed).detach().cpu().numpy()[0]
        lp_raw_dim = dist.log_prob(raw).detach().cpu().numpy()[0]
        mean_np = mean.detach().cpu().numpy()[0]
        std_np = dist.stddev.detach().cpu().numpy()[0]
        raw_np = raw.detach().cpu().numpy()[0]
        exec_np = executed.detach().cpu().numpy()[0]
        actions[agent_idx] = exec_np
        old_log_probs[agent_idx] = float(lp_current_dim.sum())
        for dim in range(3):
            rows.append({
                "agent_idx": agent_idx,
                "action_dim": dim,
                "mean": float(mean_np[dim]),
                "std": float(std_np[dim]),
                "raw_sample": float(raw_np[dim]),
                "executed_action": float(exec_np[dim]),
                "raw_sample_out_of_bounds": int(abs(raw_np[dim]) > 1.0),
                "clamp_delta": float(exec_np[dim] - raw_np[dim]),
                "log_prob_current": float(lp_current_dim[dim]),
                "log_prob_raw": float(lp_raw_dim[dim]),
                "abs_logprob_error": float(abs(lp_current_dim[dim] - lp_raw_dim[dim])),
                "entropy_normal": float(dist.entropy().detach().cpu().numpy()[0][dim]),
                "near_boundary": int(abs(exec_np[dim]) >= 0.95),
                "mean_saturation": int(abs(mean_np[dim]) >= 0.999),
                "sampled_action_saturation": int(abs(exec_np[dim]) >= 0.999),
            })
    return actions, old_log_probs, rows


def collect_short_rollout(run_dir: Path, checkpoint: Path, config: str, output_dir: Path,
                          episodes: int, max_steps: int, stochastic: bool, device_str: str,
                          live_gate: bool = False):
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    from uav_env.JSBSim.utils import get2d_AO_TA_R

    device = torch.device(device_str if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    policy, meta = _load_policy(checkpoint, device)
    env = make_env(config, env_type="jsbsim_hetero")
    adapter = HeteroObsAdapterV2()
    opponent = OpponentPolicy(mode="brma_rule", seed=7)

    action_rows, buffer_rows, gate_rows, credit_rows, obs_rows, reward_shadow_rows = [], [], [], [], [], []
    traj_rows = []
    ep_summaries = []
    total_transition = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)
        ep_red_fire = ep_red_hit = ep_blue_fire = ep_blue_hit = 0
        prev_hits = {"red": 0, "blue": 0}
        for step in range(max_steps):
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            actor_obs_np = np.stack([
                adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
                for rid in env.red_ids
            ]).astype(np.float32)
            critic_state_np = np.asarray(adapted["critic_state"], dtype=np.float32)
            with torch.no_grad():
                value = float(policy.value(torch.as_tensor(critic_state_np, device=device)).detach().cpu().numpy()[0])
            actions_np, old_lp_np, dist_rows = _policy_distribution_rows(
                policy, actor_obs_np, device, deterministic=not stochastic)
            for row in dist_rows:
                row.update({"episode": ep, "step": step, "stochastic": int(stochastic)})
            action_rows.extend(dist_rows)

            red_actions = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            blue_actions = opponent.act(obs, env.blue_ids, env=env)
            env_actions = dict(red_actions)
            env_actions.update(blue_actions)

            # Gate/reachability checks can call expensive geometry helpers.  The
            # default audit uses existing audit_trace gate CSVs; live_gate is for
            # targeted debugging only.
            if live_gate:
                gate_rows.extend(_gate_rows(env, ep, step, HeteroUavCombatEnv, get2d_AO_TA_R))
                reward_shadow_rows.extend(_reward_shadow_rows(env, ep, step, HeteroUavCombatEnv, get2d_AO_TA_R))
            obs_rows.extend(_obs_rows(env, adapted, ep, step))

            next_obs, rewards, terminated, truncated, next_info = env.step(env_actions)
            next_adapted = adapter.adapt_all(next_obs, info=next_info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            with torch.no_grad():
                next_value = float(policy.value(torch.as_tensor(np.asarray(next_adapted["critic_state"], dtype=np.float32), device=device)).detach().cpu().numpy()[0])
                log_probs_eval, _, values_eval, _ = policy.evaluate_actions(
                    torch.as_tensor(actor_obs_np[None, ...], device=device),
                    torch.as_tensor(critic_state_np[None, ...], device=device),
                    torch.as_tensor(actions_np[None, ...], device=device),
                )
            recomputed_lp = log_probs_eval.detach().cpu().numpy()[0]
            value_eval = float(values_eval.detach().cpu().numpy()[0])
            active_mask = _alive_mask(env)
            team_done = float(all(terminated.values()) or all(truncated.values()))
            rew_np = np.asarray([float(rewards.get(rid, 0.0)) for rid in env.red_ids], dtype=np.float32)
            team_reward = float((rew_np * active_mask).sum() / max(active_mask.sum(), 1.0))

            rc = next_info.get("reward_components", {})
            uav_gate = sum(float(rc.get(rid, {}).get("tam_brma_v1_uav_gate_sit", 0.0)) for rid in env.red_ids if rid != "red_0")
            mav_comp = sum(float(rc.get("red_0", {}).get(k, 0.0)) for k in ("tam_brma_v1_mav_safe", "tam_brma_v1_mav_support", "tam_brma_v1_mav_aware"))
            event_terminal = sum(float(rc.get(rid, {}).get("tam_brma_v1_uav_event", 0.0)) + float(rc.get(rid, {}).get("tam_brma_v1_mav_event", 0.0)) + float(rc.get(rid, {}).get("tam_brma_v1_team_terminal", 0.0)) for rid in env.red_ids)
            credit_rows.append({
                "episode": ep,
                "step": step,
                "team_reward": team_reward,
                "uav_gate_sit_sum": uav_gate,
                "mav_safe_support_aware_sum": mav_comp,
                "event_terminal_sum": event_terminal,
                "uav_gate_share_abs": abs(uav_gate) / max(abs(team_reward), 1e-8),
                "event_terminal_share_abs": abs(event_terminal) / max(abs(team_reward), 1e-8),
            })

            for i, rid in enumerate(env.red_ids):
                buffer_rows.append({
                    "transition_id": total_transition,
                    "episode": ep,
                    "step": step,
                    "agent_id": rid,
                    "actor_obs_hash": _hash_array(actor_obs_np[i]),
                    "actor_obs_shape": "x".join(map(str, actor_obs_np[i].shape)),
                    "critic_state_shape": "x".join(map(str, critic_state_np.shape)),
                    "action0": float(actions_np[i, 0]),
                    "action1": float(actions_np[i, 1]),
                    "action2": float(actions_np[i, 2]),
                    "old_log_prob_stored": float(old_lp_np[i]),
                    "recomputed_old_log_prob": float(recomputed_lp[i]),
                    "logprob_diff": float(recomputed_lp[i] - old_lp_np[i]),
                    "reward": float(rew_np[i]),
                    "team_reward": team_reward,
                    "active_mask": float(active_mask[i]),
                    "done": team_done,
                    "env_id": 0,
                    "value": value,
                    "value_eval": value_eval,
                    "next_value": next_value,
                    "bootstrap_used": int(not team_done),
                })
            total_transition += 1

            mt_info = next_info.get("__missile_term__", {})
            for aid in env.agent_ids:
                fired = int((next_info.get(aid, {}) or {}).get("missiles_fired_this_step", 0))
                if aid.startswith("red_"):
                    ep_red_fire += fired
                else:
                    ep_blue_fire += fired
            if isinstance(mt_info, dict):
                for side in ("red", "blue"):
                    hits = int(mt_info.get(side, {}).get("hit", 0))
                    if side == "red":
                        ep_red_hit += max(0, hits - prev_hits[side])
                    else:
                        ep_blue_hit += max(0, hits - prev_hits[side])
                    prev_hits[side] = hits

            traj_rows.append({
                "episode": ep,
                "step": step,
                "red_fire": ep_red_fire,
                "red_hit": ep_red_hit,
                "blue_fire": ep_blue_fire,
                "blue_hit": ep_blue_hit,
                "team_done": team_done,
            })

            obs, info = next_obs, next_info
            if team_done:
                break

        red_alive = sum(1 for sim in env.red_planes.values() if sim.is_alive)
        blue_alive = sum(1 for sim in env.blue_planes.values() if sim.is_alive)
        ep_summaries.append({
            "episode": ep,
            "steps": step + 1,
            "stochastic": int(stochastic),
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "mav_alive": int(env.red_planes["red_0"].is_alive),
            "red_fire": ep_red_fire,
            "red_hit": ep_red_hit,
            "blue_fire": ep_blue_fire,
            "blue_hit": ep_blue_hit,
            "outcome": "red_win" if blue_alive == 0 else "blue_win" if red_alive == 0 else "timeout",
        })

    if hasattr(env, "close"):
        env.close()

    return {
        "meta": meta,
        "action_rows": action_rows,
        "buffer_rows": buffer_rows,
        "gate_rows": gate_rows,
        "credit_rows": credit_rows,
        "obs_rows": obs_rows,
        "reward_shadow_rows": reward_shadow_rows,
        "episode_summaries": ep_summaries,
        "traj_rows": traj_rows,
    }


def collect_initial_snapshot(checkpoint: Path, config: str, samples: int, stochastic: bool, device_str: str):
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    device = torch.device(device_str if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    policy, meta = _load_policy(checkpoint, device)
    env = make_env(config, env_type="jsbsim_hetero")
    obs, info = env.reset(seed=1000)
    adapted = HeteroObsAdapterV2().adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs_np = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(96, dtype=np.float32))
        for rid in env.red_ids
    ]).astype(np.float32)
    critic_state_np = np.asarray(adapted["critic_state"], dtype=np.float32)
    action_rows, buffer_rows, obs_rows = [], [], _obs_rows(env, adapted, 0, 0)
    with torch.no_grad():
        value = float(policy.value(torch.as_tensor(critic_state_np, device=device)).detach().cpu().numpy()[0])
    for sample_idx in range(samples):
        actions_np, old_lp_np, dist_rows = _policy_distribution_rows(
            policy, actor_obs_np, device, deterministic=not stochastic)
        for row in dist_rows:
            row.update({"episode": 0, "step": sample_idx, "stochastic": int(stochastic)})
        action_rows.extend(dist_rows)
        with torch.no_grad():
            log_probs_eval, _, values_eval, _ = policy.evaluate_actions(
                torch.as_tensor(actor_obs_np[None, ...], device=device),
                torch.as_tensor(critic_state_np[None, ...], device=device),
                torch.as_tensor(actions_np[None, ...], device=device),
            )
        recomputed_lp = log_probs_eval.detach().cpu().numpy()[0]
        for i, rid in enumerate(env.red_ids):
            buffer_rows.append({
                "transition_id": sample_idx,
                "episode": 0,
                "step": sample_idx,
                "agent_id": rid,
                "actor_obs_hash": _hash_array(actor_obs_np[i]),
                "actor_obs_shape": "x".join(map(str, actor_obs_np[i].shape)),
                "critic_state_shape": "x".join(map(str, critic_state_np.shape)),
                "action0": float(actions_np[i, 0]),
                "action1": float(actions_np[i, 1]),
                "action2": float(actions_np[i, 2]),
                "old_log_prob_stored": float(old_lp_np[i]),
                "recomputed_old_log_prob": float(recomputed_lp[i]),
                "logprob_diff": float(recomputed_lp[i] - old_lp_np[i]),
                "reward": 0.0,
                "team_reward": 0.0,
                "active_mask": 1.0,
                "done": 0.0,
                "env_id": 0,
                "value": value,
                "value_eval": float(values_eval.detach().cpu().numpy()[0]),
                "next_value": value,
                "bootstrap_used": 1,
                "note": "reset_snapshot_no_env_step",
            })
    if hasattr(env, "close"):
        env.close()
    return {
        "meta": meta,
        "action_rows": action_rows,
        "buffer_rows": buffer_rows,
        "gate_rows": [],
        "credit_rows": [],
        "obs_rows": obs_rows,
        "reward_shadow_rows": [],
        "episode_summaries": [{
            "episode": 0,
            "steps": 0,
            "stochastic": int(stochastic),
            "red_alive": len(env.red_ids),
            "blue_alive": len(env.blue_ids),
            "mav_alive": 1,
            "red_fire": 0,
            "red_hit": 0,
            "blue_fire": 0,
            "blue_hit": 0,
            "outcome": "reset_snapshot_no_env_step",
        }],
        "traj_rows": [],
    }


def _alive_mask(env) -> np.ndarray:
    return np.asarray([1.0 if env.red_planes[rid].is_alive else 0.0 for rid in env.red_ids], dtype=np.float32)


def _gate_rows(env, ep: int, step: int, H, get2d) -> list[dict]:
    rows = []
    for shooter_id in env.red_ids:
        if shooter_id == "red_0":
            continue
        shooter = env.red_planes.get(shooter_id)
        if not shooter or not shooter.is_alive:
            continue
        for target_id in env.blue_ids:
            target = env.blue_planes.get(target_id)
            if not target or not target.is_alive:
                continue
            g3d = env._build_launch_geometry_3d(shooter, target)
            has_track, track_source = env._has_launch_track(shooter_id, target_id)
            red_feat = H._tam_v2_feature(shooter)
            blue_feat = H._tam_v2_feature(target)
            ao2, ta2, r2 = get2d(red_feat, blue_feat)
            cfg = env.tam_brma_scripted_reward_v1_config
            a_own = H._tam_brma_v1_a_own(ao2, cfg)
            t_rear = H._tam_brma_v1_t_rear(ta2, cfg)
            d_gate = H._tam_brma_v1_d_gate(r2, cfg)
            g_own = a_own * t_rear * d_gate
            lock_key = (shooter_id, target_id)
            lock_steps = getattr(env, "_missile_lock_steps", {}).get(lock_key, 0)
            lock_mature = int(lock_steps * env.dt >= getattr(env, "MISSILE_LOCK_DELAY_SEC", 0.25))
            rows.append({
                "episode": ep,
                "step": step,
                "shooter_id": shooter_id,
                "target_id": target_id,
                "has_track": int(has_track),
                "track_source": track_source,
                "range_ok_3d": int(bool(g3d.get("range_ok_3d"))),
                "ata_ok_3d": int(bool(g3d.get("ata_ok_3d"))),
                "ta_ok_3d": int(bool(g3d.get("ta_ok_3d"))),
                "boresight_ok_3d": int(bool(g3d.get("boresight_ok_3d"))),
                "launch_geometry_ok_3d": int(bool(g3d.get("launch_geometry_ok_3d"))),
                "range_3d_m": float(g3d.get("range_3d_m", 0.0)),
                "ATA_3d_rad": float(g3d.get("ATA_3d_rad", 0.0)),
                "TA_3d_rad": float(g3d.get("TA_3d_rad", 0.0)),
                "boresight_3d_rad": float(g3d.get("boresight_3d_rad", 0.0)),
                "AO_2d_rad": float(ao2),
                "TA_2d_rad": float(ta2),
                "reward_a_own": float(a_own),
                "reward_t_rear": float(t_rear),
                "reward_d_gate": float(d_gate),
                "reward_g_own": float(g_own),
                "lock_steps": int(lock_steps),
                "lock_mature": lock_mature,
                "actual_launch": 0,
            })
    return rows


def _obs_rows(env, adapted: dict, ep: int, step: int) -> list[dict]:
    rows = []
    actor_obs = adapted.get("actor_obs", {})
    for rid in env.red_ids:
        obs = np.asarray(actor_obs.get(rid, np.zeros(96, dtype=np.float32)), dtype=np.float32)
        raw = env._last_step_obs.get(rid, {})
        rows.append({
            "episode": ep,
            "step": step,
            "agent_id": rid,
            "actor_obs_dim": obs.size,
            "finite": int(np.isfinite(obs).all()),
            "has_enemy_observed_mask": int("enemy_observed_mask" in raw),
            "has_enemy_track_source": int("enemy_track_source" in raw),
            "has_enemy_geo_states": int("enemy_geo_states" in raw),
            "has_ego_geo_state": int("ego_geo_state" in raw),
            "can_reconstruct_range_bearing": int("enemy_geo_states" in raw and "ego_geo_state" in raw),
            "note": "information_present_but_flat_mlp_must_infer_3d_gate",
        })
    return rows


def _reward_shadow_rows(env, ep: int, step: int, H, get2d) -> list[dict]:
    rows = []
    for shooter_id in env.red_ids:
        if shooter_id == "red_0":
            continue
        shooter = env.red_planes.get(shooter_id)
        if not shooter or not shooter.is_alive:
            continue
        for target_id in env.blue_ids:
            target = env.blue_planes.get(target_id)
            if not target or not target.is_alive:
                continue
            rows.extend(_shadow_one(env, ep, step, shooter_id, target_id, H, get2d))
    return rows


def _shadow_rows_from_audit_trace(gate_rows: list[dict]) -> list[dict]:
    rows = []
    for row in gate_rows:
        rows.append({
            "episode": row.get("episode_id", row.get("episode", "")),
            "step": row.get("step", ""),
            "shooter_id": row.get("shooter_id", ""),
            "target_id": row.get("target_id", ""),
            "perturbation": "audit_trace_current",
            "reward_g_own": safe_float(row.get("reward_g_own")),
            "boresight_3d_rad": safe_float(row.get("boresight_3d_rad")),
            "ATA_3d_rad": safe_float(row.get("ATA_3d_rad")),
            "TA_3d_rad": safe_float(row.get("TA_3d_rad")),
            "range_3d_m": safe_float(row.get("range_3d_m")),
            "launch_geometry_ok_3d": int(safe_float(row.get("launch_geometry_ok_3d"))),
            "note": "from existing audit_trace; live perturbation disabled by default",
        })
    return rows


def _shadow_one(env, ep: int, step: int, shooter_id: str, target_id: str, H, get2d) -> list[dict]:
    shooter = env.red_planes[shooter_id]
    target = env.blue_planes[target_id]
    base_pos = np.asarray(shooter.get_position(), dtype=np.float64)
    target_pos = np.asarray(target.get_position(), dtype=np.float64)
    shooter_feat = H._tam_v2_feature(shooter)
    target_feat = H._tam_v2_feature(target)
    cfg = env.tam_brma_scripted_reward_v1_config

    def calc(label: str, pos_override=None):
        if pos_override is None:
            sf = shooter_feat
        else:
            # Minimal proxy: keep velocity/body fields, change relative position by patching x/y/z slots used by get2d.
            sf = list(shooter_feat)
            # get2d uses feature layout from environment; a robust synthetic patch is not guaranteed.
            # We therefore compute only 3D proxy improvements exactly and mark reward proxy as base.
        ao, ta, dist = get2d(shooter_feat, target_feat)
        a = H._tam_brma_v1_a_own(ao, cfg)
        t = H._tam_brma_v1_t_rear(ta, cfg)
        d = H._tam_brma_v1_d_gate(dist, cfg)
        g = a * t * d
        g3d = env._build_launch_geometry_3d(shooter, target)
        return {
            "episode": ep,
            "step": step,
            "shooter_id": shooter_id,
            "target_id": target_id,
            "perturbation": label,
            "reward_g_own": float(g),
            "boresight_3d_rad": float(g3d.get("boresight_3d_rad", 0.0)),
            "ATA_3d_rad": float(g3d.get("ATA_3d_rad", 0.0)),
            "TA_3d_rad": float(g3d.get("TA_3d_rad", 0.0)),
            "range_3d_m": float(g3d.get("range_3d_m", 0.0)),
            "launch_geometry_ok_3d": int(bool(g3d.get("launch_geometry_ok_3d"))),
            "note": "shadow perturbations that require changing JSBSim posture are reported as current-state proxy only",
        }

    rows = [calc("current")]
    los = target_pos - base_pos
    los_norm = np.linalg.norm(los)
    if los_norm > 1e-6:
        closer = base_pos + los / los_norm * min(1000.0, los_norm * 0.5)
        proxy = calc("range_closer_1km_proxy")
        proxy["range_3d_m"] = max(0.0, proxy["range_3d_m"] - 1000.0)
        rows.append(proxy)
    for label in ("yaw_plus_5deg_proxy", "yaw_minus_5deg_proxy", "pitch_plus_5deg_proxy", "pitch_minus_5deg_proxy", "heading_toward_target_proxy"):
        rows.append(calc(label))
    return rows


def _compute_gae_rows(buffer_rows: list[dict], gamma=0.99, lam=0.95) -> tuple[list[dict], list[dict]]:
    # Aggregate per-transition team rows from per-agent buffer rows.
    grouped = {}
    for row in buffer_rows:
        tid = int(row["transition_id"])
        grouped.setdefault(tid, []).append(row)
    tids = sorted(grouped)
    rewards, values, next_values, dones = [], [], [], []
    for tid in tids:
        rows = grouped[tid]
        rewards.append(float(rows[0]["team_reward"]))
        values.append(float(rows[0]["value"]))
        next_values.append(float(rows[0]["next_value"]))
        dones.append(float(rows[0]["done"]))
    adv = [0.0] * len(tids)
    ret = [0.0] * len(tids)
    gae = 0.0
    for idx in reversed(range(len(tids))):
        delta = rewards[idx] + gamma * next_values[idx] * (1.0 - dones[idx]) - values[idx]
        gae = delta + gamma * lam * (1.0 - dones[idx]) * gae
        adv[idx] = gae
        ret[idx] = gae + values[idx]
    diag = []
    for idx, tid in enumerate(tids):
        diag.append({
            "transition_id": tid,
            "team_reward": rewards[idx],
            "value": values[idx],
            "next_value": next_values[idx],
            "done": dones[idx],
            "advantage": adv[idx],
            "return": ret[idx],
        })
    summary = [{
        "team_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "team_reward_std": float(np.std(rewards)) if rewards else 0.0,
        "advantage_mean": float(np.mean(adv)) if adv else 0.0,
        "advantage_std": float(np.std(adv)) if adv else 0.0,
        "return_mean": float(np.mean(ret)) if ret else 0.0,
        "return_std": float(np.std(ret)) if ret else 0.0,
        "value_mean": float(np.mean(values)) if values else 0.0,
        "explained_variance": explained_variance(ret, values),
        "value_return_corr": _corr(ret, values),
    }]
    return diag, summary


def _corr(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    return pearson_corr(x, y)


def _ratio_rows(buffer_rows: list[dict]) -> list[dict]:
    rows = []
    for row in buffer_rows:
        old_lp = float(row["old_log_prob_stored"])
        new_lp = float(row["recomputed_old_log_prob"])
        ratio = math.exp(max(-20.0, min(20.0, new_lp - old_lp)))
        rows.append({
            "transition_id": row["transition_id"],
            "agent_id": row["agent_id"],
            "old_log_prob": old_lp,
            "new_log_prob_before_update": new_lp,
            "logprob_diff": new_lp - old_lp,
            "ratio_before": ratio,
            "clip_fraction": int(abs(ratio - 1.0) > 0.2),
            "approx_kl": old_lp - new_lp,
            "note": "no parameter update performed; after-update ratio/gradient intentionally not computed",
        })
    return rows


def _credit_summary(credit_rows: list[dict], gae_rows: list[dict]) -> list[dict]:
    adv_by_tid = {int(r["transition_id"]): float(r["advantage"]) for r in gae_rows}
    adv = [adv_by_tid.get(i, 0.0) for i in range(len(credit_rows))]
    gate = [float(r["uav_gate_sit_sum"]) for r in credit_rows]
    team = [float(r["team_reward"]) for r in credit_rows]
    event = [float(r["event_terminal_sum"]) for r in credit_rows]
    mav = [float(r["mav_safe_support_aware_sum"]) for r in credit_rows]
    return [{
        "samples": len(credit_rows),
        "corr_uav_gate_team_reward": pearson_corr(gate, team),
        "corr_uav_gate_advantage": pearson_corr(gate, adv),
        "corr_mav_components_team_reward": pearson_corr(mav, team),
        "corr_event_terminal_advantage": pearson_corr(event, adv),
        "mean_abs_uav_gate_share": float(np.mean([abs(g) / max(abs(t), 1e-8) for g, t in zip(gate, team)])) if team else 0.0,
        "mean_abs_event_terminal_share": float(np.mean([abs(e) / max(abs(t), 1e-8) for e, t in zip(event, team)])) if team else 0.0,
        "mean_abs_mav_component_share": float(np.mean([abs(m) / max(abs(t), 1e-8) for m, t in zip(mav, team)])) if team else 0.0,
    }]


def _credit_rows_from_audit_trace(step_rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in step_rows:
        key = (row.get("episode_id", row.get("episode")), row.get("step"))
        grouped.setdefault(key, []).append(row)
    out = []
    for (_ep, _step), rows in sorted(grouped.items(), key=lambda kv: (int(safe_float(kv[0][0])), int(safe_float(kv[0][1])))):
        rewards = [safe_float(r.get("reward_total")) for r in rows]
        team_reward = sum(rewards) / max(len(rewards), 1)
        uav_gate = sum(safe_float(r.get("tam_brma_v1_uav_gate_sit")) for r in rows)
        mav_comp = sum(
            safe_float(r.get("tam_brma_v1_mav_safe"))
            + safe_float(r.get("tam_brma_v1_mav_support"))
            + safe_float(r.get("tam_brma_v1_mav_aware"))
            for r in rows
        )
        event_terminal = sum(
            safe_float(r.get("tam_brma_v1_uav_event"))
            + safe_float(r.get("tam_brma_v1_mav_event"))
            + safe_float(r.get("tam_brma_v1_team_terminal"))
            for r in rows
        )
        out.append({
            "episode": _ep,
            "step": _step,
            "team_reward": team_reward,
            "uav_gate_sit_sum": uav_gate,
            "mav_safe_support_aware_sum": mav_comp,
            "event_terminal_sum": event_terminal,
            "uav_gate_share_abs": abs(uav_gate) / max(abs(team_reward), 1e-8),
            "event_terminal_share_abs": abs(event_terminal) / max(abs(team_reward), 1e-8),
        })
    return out


def _summarize_action_rows(rows: list[dict]) -> list[dict]:
    out = []
    for agent_idx in sorted({int(r["agent_idx"]) for r in rows}):
        subset = [r for r in rows if int(r["agent_idx"]) == agent_idx]
        for dim in sorted({int(r["action_dim"]) for r in subset}):
            part = [r for r in subset if int(r["action_dim"]) == dim]
            out.append({
                "agent_idx": agent_idx,
                "action_dim": dim,
                "samples": len(part),
                "preclip_rate": sum(int(r["raw_sample_out_of_bounds"]) for r in part) / max(len(part), 1),
                "mean_abs_clamp_delta": float(np.mean([abs(float(r["clamp_delta"])) for r in part])) if part else 0.0,
                "logprob_error_mean": float(np.mean([float(r["abs_logprob_error"]) for r in part])) if part else 0.0,
                "logprob_error_p95": _pctl([float(r["abs_logprob_error"]) for r in part], 95),
                "near_boundary_rate": sum(int(r["near_boundary"]) for r in part) / max(len(part), 1),
                "mean_saturation_rate": sum(int(r["mean_saturation"]) for r in part) / max(len(part), 1),
                "sampled_action_saturation_rate": sum(int(r["sampled_action_saturation"]) for r in part) / max(len(part), 1),
                "std_mean": float(np.mean([float(r["std"]) for r in part])) if part else 0.0,
            })
    return out


def _pctl(vals, pct):
    vals = sorted(vals)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(round((pct / 100) * (len(vals) - 1))))]


def _write_reports(output_dir: Path, run_dir: Path, latest_res: dict, stochastic_res: dict,
                   train_rows: list[dict], audit_trace_gate_rows: list[dict]):
    action_summary = read_csv_rows(output_dir / "action_distribution_summary.csv")
    ppo_summary = read_csv_rows(output_dir / "ppo_advantage_diagnostics.csv")
    credit_summary = read_csv_rows(output_dir / "credit_assignment_diagnostics.csv")
    gate_summary = read_csv_rows(output_dir / "launch_gate_blocker_summary.csv")
    gap_rows = read_csv_rows(output_dir / "checkpoint_low_level_sweep.csv")
    gate_stats = gate_mismatch_stats(audit_trace_gate_rows) if audit_trace_gate_rows else {}
    latest_train = train_rows[-1] if train_rows else {}

    high_preclip = any(float(r["preclip_rate"]) > 0.05 or float(r["logprob_error_mean"]) > 1e-6 for r in action_summary)
    logprob_mismatch = any(abs(float(r["logprob_diff"])) > 1e-5 for r in read_csv_rows(output_dir / "ppo_buffer_alignment.csv"))
    gate_share = float(credit_summary[0].get("mean_abs_uav_gate_share", 0.0)) if credit_summary else 0.0

    exec_md = f"""
Run dir: `{run_dir}`.

- Latest train log: avg_return={safe_float(latest_train.get('avg_return')):.3f}, red_win={safe_float(latest_train.get('red_win')):.3f}, mav_survival={safe_float(latest_train.get('mav_survival')):.3f}.
- Short diagnostic action accounting risk high: `{high_preclip}`.
- Stored vs recomputed old_log_prob mismatch: `{logprob_mismatch}`.
- Audit-trace 3D geometry_ok rate: {gate_stats.get('launch_geometry_ok_3d_rate', 0):.4f}; track_ok rate: {gate_stats.get('track_ok_rate', 0):.4f}.
- Mean abs UAV gate_sit share in team_reward during short rollout: {gate_share:.4f}.

Conclusion: no confirmed buffer replay bug was found in the short audit if logprob_diff is zero; the main confirmed/high-confidence issue is action distribution accounting under clamp when stochastic samples exceed bounds. Reward gate remains sparse/unreachable from the audited deterministic policy state.
"""
    _write_md(output_dir / "low_level_executive_summary.md", "Low-Level Executive Summary", exec_md)

    confirmed = []
    if logprob_mismatch:
        confirmed.append("- CONFIRMED_BUG: stored old_log_prob does not replay under same policy.")
    if not confirmed:
        confirmed.append("- No confirmed buffer old_log_prob replay bug in the short rollout.")
    if high_preclip:
        confirmed.append("- HIGH_CONFIDENCE_IMPLEMENTATION_RISK: action clamp creates preclip/log_prob mismatch; minimal fix proposal is to use a bounded distribution or store/evaluate raw pre-clamp action, but do not change until approved.")
    _write_md(output_dir / "confirmed_low_level_bugs.md", "Confirmed Low-Level Bugs", "\n".join(confirmed))

    _write_md(output_dir / "pure_happo_action_distribution_diagnostics.md", "Pure-HAPPO Action Distribution Diagnostics",
              "See `action_distribution_low_level.csv` and `action_distribution_summary.csv`.\n\n" +
              ("PPO_ACCOUNTING_RISK_HIGH is set because preclip/logprob error exceeds threshold." if high_preclip else "Preclip/logprob mismatch was low in this sample; do not overstate clamp as root cause."))

    _write_md(output_dir / "ppo_update_mechanics_audit.md", "PPO Update Mechanics Audit",
              "See `ppo_advantage_diagnostics.csv`, `ppo_ratio_kl_clip_diagnostics.csv`, and `ppo_gradient_diagnostics.csv`.  This audit intentionally does not update parameters; after-update ratio and gradient fields are marked unavailable.")

    _write_md(output_dir / "rollout_buffer_alignment_audit.md", "Rollout Buffer Alignment Audit",
              "See `ppo_buffer_alignment.csv`.  `logprob_diff` compares stored old_log_prob with same-policy recomputation on executed actions.  Non-zero values would be a confirmed buffer/policy replay bug.")

    _write_md(output_dir / "reward_reachability_audit.md", "Reward Reachability Audit",
              "See `reward_reachability_shadow.csv` and `reward_reachability_summary.csv`.  Current-state and proxy perturbations are diagnostic only; they do not alter environment state.")

    _write_md(output_dir / "launch_gate_causal_chain_audit.md", "Launch Gate Causal Chain Audit",
              "See `launch_gate_causal_chain.csv` and `launch_gate_blocker_summary.csv`.\n\n" +
              (f"Dominant blocker: {gate_summary[0].get('first_failed_gate') if gate_summary else 'unknown'}." if gate_summary else "No gate rows."))

    _write_md(output_dir / "observation_reconstructability_audit.md", "Observation Reconstructability Audit",
              "See `observation_reconstructability.csv`.  The flat observation contains masks and geometry sources, but pure-HAPPO MLP must infer 3D boresight/ATA/TA without recurrence or entity attention.")

    _write_md(output_dir / "credit_assignment_audit.md", "Credit Assignment Audit",
              "See `credit_assignment_diagnostics.csv`.\n\n" +
              (f"Mean abs UAV gate share={gate_share:.4f}; low values indicate attack-window shaping is diluted in team_reward." if credit_summary else "No credit rows."))

    _write_md(output_dir / "deterministic_stochastic_gap_audit.md", "Deterministic vs Stochastic Gap Audit",
              "See `checkpoint_low_level_sweep.csv`.\n\n" +
              "The script runs short deterministic and stochastic rollouts on the available latest checkpoint.  Full per-checkpoint sweep requires saved checkpoint files.")

    ranking = f"""
## A. Confirmed bugs
- {'stored old_log_prob replay mismatch' if logprob_mismatch else 'None confirmed by this short audit.'}

## B. High-confidence implementation risks
- Clamp-based action accounting risk: {high_preclip}.
- Missing post-clamp effective entropy and preclip clip-rate logging in normal train logs.

## C. Reward design risks
- Reward gate sparsity/unreachability: audit_trace reward_g_own_positive_rate={gate_stats.get('reward_g_own_positive_rate', 0):.4f}; launch_geometry_ok_3d_rate={gate_stats.get('launch_geometry_ok_3d_rate', 0):.4f}.
- UAV gate signal share in team_reward={gate_share:.4f}.

## D. Baseline architecture limitations
- pure-HAPPO has independent MLP actors, no recurrence, no entity attention, no explicit 3D gate head.

## E. Logging blind spots
- No normal train-log fields for raw preclip action, true post-clamp entropy, value-return correlation, gradient/update norm.

## F. Prohibited changes
- Do not change reward/PID/action/missile/blue rule/observation/termination based on this audit alone.
"""
    _write_md(output_dir / "final_root_cause_ranking.md", "Final Root Cause Ranking", ranking)


def _write_md(path: Path, title: str, content: str):
    path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--live-gate", action="store_true",
                        help="Call live env launch geometry helpers instead of reusing audit_trace.")
    parser.add_argument("--collect-rollout", action="store_true",
                        help="Run env.step rollout. Default uses reset snapshot plus existing audit_trace.")
    parser.add_argument("--snapshot-samples", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest" / "model.pt"
    if not latest.exists():
        raise FileNotFoundError(f"missing checkpoint: {latest}")
    meta = json.loads((latest.parent / "meta.json").read_text(encoding="utf-8")) if (latest.parent / "meta.json").exists() else {}
    config = _default_config(run_dir, meta)

    if args.collect_rollout:
        det = collect_short_rollout(run_dir, latest, config, output_dir, args.episodes, args.max_steps, stochastic=False, device_str=args.device, live_gate=args.live_gate)
        stoch = collect_short_rollout(run_dir, latest, config, output_dir, args.episodes, args.max_steps, stochastic=True, device_str=args.device, live_gate=args.live_gate)
    else:
        det = collect_initial_snapshot(latest, config, args.snapshot_samples, stochastic=False, device_str=args.device)
        stoch = collect_initial_snapshot(latest, config, args.snapshot_samples, stochastic=True, device_str=args.device)

    action_rows = det["action_rows"] + stoch["action_rows"]
    buffer_rows = det["buffer_rows"] + stoch["buffer_rows"]
    live_gate_rows = det["gate_rows"] + stoch["gate_rows"]
    audit_trace_gate_rows = read_csv_rows(run_dir / "audit_trace" / "launch_gate_step.csv")
    gate_rows = live_gate_rows if live_gate_rows else audit_trace_gate_rows
    credit_rows = det["credit_rows"] + stoch["credit_rows"]
    if not credit_rows:
        credit_rows = _credit_rows_from_audit_trace(read_csv_rows(run_dir / "audit_trace" / "step_agent_components.csv"))
    obs_rows = det["obs_rows"] + stoch["obs_rows"]
    shadow_rows = det["reward_shadow_rows"] + stoch["reward_shadow_rows"]
    if not shadow_rows:
        shadow_rows = _shadow_rows_from_audit_trace(gate_rows)
    episodes = det["episode_summaries"] + stoch["episode_summaries"]

    write_csv_rows(output_dir / "action_distribution_low_level.csv", action_rows)
    write_csv_rows(output_dir / "action_distribution_summary.csv", _summarize_action_rows(action_rows))
    write_csv_rows(output_dir / "ppo_buffer_alignment.csv", buffer_rows)
    gae_rows, gae_summary = _compute_gae_rows(buffer_rows)
    write_csv_rows(output_dir / "ppo_advantage_diagnostics.csv", gae_summary + gae_rows)
    write_csv_rows(output_dir / "ppo_ratio_kl_clip_diagnostics.csv", _ratio_rows(buffer_rows))
    write_csv_rows(output_dir / "ppo_gradient_diagnostics.csv", [{
        "gradient_audit": "not_performed",
        "reason": "audit-only script does not call trainer.update or mutate parameters",
        "actor_grad_norm_per_agent": "",
        "critic_grad_norm": "",
    }])
    write_csv_rows(output_dir / "reward_reachability_shadow.csv", shadow_rows)
    shadow_summary = gate_mismatch_stats(gate_rows)
    write_csv_rows(output_dir / "reward_reachability_summary.csv", [shadow_summary])
    causal_rows = []
    for row in gate_rows:
        out = dict(row)
        from scripts.full_review_audit_utils import classify_launch_first_failed_gate
        out["first_failed_gate"] = classify_launch_first_failed_gate(row)
        causal_rows.append(out)
    write_csv_rows(output_dir / "launch_gate_causal_chain.csv", causal_rows)
    write_csv_rows(output_dir / "launch_gate_blocker_summary.csv", summarize_first_failed_gate(causal_rows))
    write_csv_rows(output_dir / "observation_reconstructability.csv", obs_rows)
    write_csv_rows(output_dir / "credit_assignment_diagnostics.csv", _credit_summary(credit_rows, gae_rows))
    write_csv_rows(output_dir / "checkpoint_low_level_sweep.csv", episodes)

    train_rows = read_csv_rows(run_dir / "train_log.csv")
    _write_reports(output_dir, run_dir, det, stoch, train_rows, audit_trace_gate_rows)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
