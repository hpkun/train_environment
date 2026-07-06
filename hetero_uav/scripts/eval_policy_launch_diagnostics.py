"""Evaluate a learned policy and record launch-envelope diagnostics.

This script performs rollout diagnostics only. It does not train, save model
updates, or modify environment mechanics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_launch_envelope_oracle import _diagnose_red_shooter, _terminal_reason
from red_attack_audit_utils import alive_counts, collect_step_counts, team_done


SCENARIO_CONFIGS = {
    "3v2": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
    "5v4": "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
}

DIAG_FIELDS = [
    "model_label",
    "scenario",
    "episode_id",
    "step",
    "red_id",
    "shooter_alive",
    "shooter_role",
    "shooter_has_missile",
    "nearest_blue_id",
    "nearest_blue_alive",
    "nearest_blue_range_m",
    "selected_target_id",
    "selected_target_range_m",
    "selected_target_AO_rad",
    "selected_target_TA_rad",
    "candidate_count",
    "alive_blue_count",
    "unengaged_alive_blue_count",
    "track_available",
    "direct_track_available",
    "mav_shared_track_available",
    "launch_track_source",
    "lock_target",
    "lock_timer",
    "lock_delay_frames",
    "cooldown_frames_left",
    "boresight_ok",
    "boresight_status",
    "target_id",
    "range_m",
    "AO_rad",
    "TA_rad",
    "lock_ready",
    "cooldown_ready",
    "deconflict_ok",
    "has_missile",
    "target_alive",
    "range_ok",
    "ao_ok",
    "ta_ok",
    "final_launch_allowed",
    "actual_missiles_fired_this_step",
    "actual_red_hit_delta_this_step",
    "predicted_allowed_but_not_fired",
    "fired_without_predicted_allowed",
    "predicted_vs_final_mismatch",
    "predicted_launch_allowed_raw",
    "launch_block_reason_primary",
    "launch_block_reason_all",
    "launch_allowed",
    "launch_block_reason",
    "action_pitch",
    "action_heading",
    "action_speed",
    "missiles_fired",
    "missile_hits",
    "blue_dead",
    "terminal_reason",
]

SUMMARY_FIELDS = [
    "model_label",
    "scenario",
    "policy_arch",
    "episodes",
    "steps",
    "red_missiles_fired",
    "missile_hits",
    "blue_dead_mean",
    "range_ok_rate",
    "ao_ok_rate",
    "ta_ok_rate",
    "lock_ready_rate",
    "cooldown_ready_rate",
    "deconflict_ok_rate",
    "track_available_rate",
    "direct_track_available_rate",
    "mav_shared_track_available_rate",
    "final_launch_allowed_rate",
    "actual_fire_rate",
    "predicted_allowed_but_not_fired_count",
    "fired_without_predicted_allowed_count",
    "predicted_vs_final_mismatch_count",
    "recurrent_eval_used",
    "launch_allowed_rate",
    "action_mean_pitch",
    "action_mean_heading",
    "action_mean_speed",
    "action_saturation_rate",
    "dominant_block_reason",
    "block_reason_counts_json",
]


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_output_dir(path: str | Path) -> Path:
    out = _as_path(path)
    if out.exists() and any(out.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out.with_name(f"{out.name}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_model(args) -> Path:
    if args.checkpoint:
        cp = Path(args.checkpoint)
        if cp.name == "model.pt" or cp.suffix == ".pt":
            return _as_path(cp)
        if args.output_dir:
            return _as_path(args.output_dir) / cp / "model.pt"
        return _as_path(cp)
    if not args.output_dir:
        raise ValueError("either --output-dir or --checkpoint is required")
    name = args.checkpoint_name
    return _as_path(args.output_dir) / name / "model.pt"


def _load_meta(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.parent / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _build_policy(meta: dict[str, Any], device: torch.device):
    from algorithms.happo import (
        BRMAEntityHAPPOReferencePolicy,
        BRMARecurrentMaskedHAPPOReferencePolicy,
        BRMARecurrentHAPPOReferencePolicy,
        EntityHAPPOReferencePolicy,
        HAPPOReferencePolicy,
    )

    arch = str(meta.get("policy_arch", "flat"))
    if arch == "entity_attention":
        return EntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if arch == "brma_entity":
        return BRMAEntityHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
        ).to(device)
    if arch == "brma_recurrent":
        return BRMARecurrentHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
        ).to(device)
    if arch == "brma_recurrent_masked":
        return BRMARecurrentMaskedHAPPOReferencePolicy(
            entity_dim=int(meta.get("entity_dim", 19)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3,
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            random_scale_mask=bool(meta.get("random_scale_mask", False)),
            random_mask_prob=float(meta.get("random_mask_prob", 0.25)),
            biased_mask=bool(meta.get("biased_mask", False)),
        ).to(device)
    if arch == "flat":
        return HAPPOReferencePolicy(
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
        ).to(device)
    raise ValueError(f"unsupported policy_arch: {arch}")


def _role_ids(env) -> list[int]:
    return [0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids]


def _policy_actions(policy, adapter, env, obs, info, device: torch.device, rnn_hidden=None):
    adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
    actor_obs = np.stack([
        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
        for rid in env.red_ids
    ])
    with torch.no_grad():
        kwargs = {
            "roles": _role_ids(env),
            "critic_state": torch.as_tensor(adapted["critic_state"], dtype=torch.float32, device=device),
            "deterministic": True,
        }
        if rnn_hidden is not None:
            kwargs["rnn_hidden"] = torch.as_tensor(rnn_hidden, dtype=torch.float32, device=device)
        out = policy.act(
            torch.as_tensor(actor_obs, dtype=torch.float32, device=device),
            **kwargs,
        )
    next_hidden = out.get("rnn_hidden", out.get("next_rnn_hidden"))
    return (
        out["action"].detach().cpu().numpy().astype(np.float32),
        next_hidden.detach().cpu().numpy().astype(np.float32) if next_hidden is not None else rnn_hidden,
    )


def _summarize(rows: list[dict[str, Any]], episodes: int, label: str, scenario: str, arch: str) -> dict[str, Any]:
    if not rows:
        return {
            "model_label": label,
            "scenario": scenario,
            "policy_arch": arch,
            "episodes": episodes,
            "steps": 0,
        }
    block_counts = Counter(str(r.get("launch_block_reason_primary") or r.get("launch_block_reason", "")) for r in rows)
    actions = np.array([
        [float(r["action_pitch"]), float(r["action_heading"]), float(r["action_speed"])]
        for r in rows
    ], dtype=np.float32)
    blue_dead_by_ep: dict[int, float] = {}
    for r in rows:
        blue_dead_by_ep[int(r["episode_id"])] = max(
            blue_dead_by_ep.get(int(r["episode_id"]), 0.0),
            float(r["blue_dead"] or 0.0),
        )
    has_hit_delta = any("actual_red_hit_delta_this_step" in r for r in rows)
    hit_delta_by_step: dict[tuple[int, int], int] = {}
    if has_hit_delta:
        for r in rows:
            key = (int(r.get("episode_id", 0)), int(r.get("step", 0)))
            hit_delta_by_step[key] = max(
                hit_delta_by_step.get(key, 0),
                int(r.get("actual_red_hit_delta_this_step", 0) or 0),
            )
        missile_hits = int(sum(hit_delta_by_step.values()))
    else:
        missile_hits = int(max(int(r.get("missile_hits", 0) or 0) for r in rows))
    return {
        "model_label": label,
        "scenario": scenario,
        "policy_arch": arch,
        "episodes": episodes,
        "steps": int(max(int(r.get("step", 0) or 0) for r in rows)),
        "red_missiles_fired": int(sum(int(r.get("actual_missiles_fired_this_step", r.get("missiles_fired", 0)) or 0) for r in rows)),
        "missile_hits": missile_hits,
        "blue_dead_mean": float(np.mean(list(blue_dead_by_ep.values()))) if blue_dead_by_ep else 0.0,
        "range_ok_rate": float(np.mean([bool(r["range_ok"]) for r in rows])),
        "ao_ok_rate": float(np.mean([bool(r["ao_ok"]) for r in rows])),
        "ta_ok_rate": float(np.mean([bool(r["ta_ok"]) for r in rows])),
        "lock_ready_rate": float(np.mean([bool(r.get("lock_ready")) for r in rows])),
        "cooldown_ready_rate": float(np.mean([bool(r.get("cooldown_ready")) for r in rows])),
        "deconflict_ok_rate": float(np.mean([bool(r.get("deconflict_ok")) for r in rows])),
        "track_available_rate": float(np.mean([bool(r.get("track_available")) for r in rows])),
        "direct_track_available_rate": float(np.mean([bool(r.get("direct_track_available")) for r in rows])),
        "mav_shared_track_available_rate": float(np.mean([bool(r.get("mav_shared_track_available")) for r in rows])),
        "final_launch_allowed_rate": float(np.mean([bool(r.get("final_launch_allowed", r.get("launch_allowed"))) for r in rows])),
        "actual_fire_rate": float(np.mean([float(r.get("actual_missiles_fired_this_step", r.get("missiles_fired", 0)) or 0) > 0 for r in rows])),
        "predicted_allowed_but_not_fired_count": int(sum(int(r.get("predicted_allowed_but_not_fired", 0) or 0) for r in rows)),
        "fired_without_predicted_allowed_count": int(sum(int(r.get("fired_without_predicted_allowed", 0) or 0) for r in rows)),
        "predicted_vs_final_mismatch_count": int(sum(int(r.get("predicted_vs_final_mismatch", 0) or 0) for r in rows)),
        "recurrent_eval_used": arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"},
        "launch_allowed_rate": float(np.mean([bool(r.get("launch_allowed")) for r in rows])),
        "action_mean_pitch": float(actions[:, 0].mean()),
        "action_mean_heading": float(actions[:, 1].mean()),
        "action_mean_speed": float(actions[:, 2].mean()),
        "action_saturation_rate": float(np.mean(np.any(np.abs(actions) > 0.95, axis=1))),
        "dominant_block_reason": block_counts.most_common(1)[0][0] if block_counts else "",
        "block_reason_counts_json": json.dumps(dict(block_counts), sort_keys=True),
    }


def _track_flags(obs: dict[str, Any], env, rid: str, target_id: str) -> dict[str, Any]:
    red_obs = obs.get(rid, {}) if isinstance(obs, dict) else {}
    src = np.asarray(red_obs.get("enemy_track_source", []), dtype=np.float32)
    direct = False
    shared = False
    target_idx = -1
    if target_id in getattr(env, "blue_ids", []):
        target_idx = list(env.blue_ids).index(target_id)
    if src.ndim == 2 and 0 <= target_idx < src.shape[0]:
        direct = bool(src[target_idx, 0] > 0.5) if src.shape[1] > 0 else False
        shared = bool(src[target_idx, 1] > 0.5) if src.shape[1] > 1 else False
    if direct and shared:
        source = "mixed"
    elif direct:
        source = "direct"
    elif shared:
        source = "mav_shared"
    else:
        source = "none"
    return {
        "track_available": direct or shared,
        "direct_track_available": direct,
        "mav_shared_track_available": shared,
        "launch_track_source": source,
    }


def _target_candidate_stats(env, rid: str, selected_target_id: str = "") -> dict[str, Any]:
    shooter = getattr(env, "red_planes", {}).get(rid)
    engaged = set(getattr(env, "_engaged_targets", set()) or set())
    alive = []
    nearest_id = ""
    nearest_range = ""
    selected_range = ""
    if shooter is not None:
        try:
            shooter_pos = np.asarray(shooter.get_position(), dtype=np.float64)
        except Exception:
            shooter_pos = None
    else:
        shooter_pos = None
    for bid in getattr(env, "blue_ids", []):
        blue = getattr(env, "blue_planes", {}).get(bid)
        if blue is None or not bool(getattr(blue, "is_alive", False)):
            continue
        alive.append(bid)
        if shooter_pos is None:
            continue
        try:
            rng = float(np.linalg.norm(shooter_pos - np.asarray(blue.get_position(), dtype=np.float64)))
        except Exception:
            continue
        if nearest_range == "" or rng < float(nearest_range):
            nearest_id = bid
            nearest_range = rng
        if bid == selected_target_id:
            selected_range = rng
    unengaged = [bid for bid in alive if bid not in engaged]
    return {
        "alive_blue_count": int(len(alive)),
        "unengaged_alive_blue_count": int(len(unengaged)),
        "candidate_count": int(len(alive)),
        "nearest_blue_id": nearest_id,
        "nearest_blue_alive": bool(nearest_id),
        "nearest_blue_range_m": nearest_range,
        "selected_target_id": selected_target_id,
        "selected_target_range_m": selected_range,
    }


def _block_reasons(diag: dict[str, Any], track_available: bool, boresight_ok: bool) -> list[str]:
    reasons = []
    if not diag.get("shooter_alive", True):
        reasons.append("shooter_dead")
    if not diag.get("has_missile", False):
        reasons.append("no_missile")
    if not diag.get("target_alive", False):
        reasons.append("no_alive_target")
    if not track_available:
        reasons.append("no_track")
    if not diag.get("range_ok", False):
        reasons.append("out_of_range")
    if not diag.get("ao_ok", False):
        reasons.append("ao_blocked")
    if not diag.get("ta_ok", False):
        reasons.append("ta_blocked")
    if not diag.get("lock_ready", False):
        reasons.append("lock_delay")
    if not diag.get("cooldown_ready", False):
        reasons.append("cooldown")
    if diag.get("kill_cooldown", False):
        reasons.append("kill_cooldown")
    if not diag.get("deconflict_ok", False):
        reasons.append("engaged_deconflict")
    if not boresight_ok:
        reasons.append("boresight_blocked")
    return reasons or ["allowed"]


def _pre_step_launch_snapshot(env, obs: dict[str, Any], rid: str, diag: dict[str, Any]) -> dict[str, Any]:
    target_id = str(diag.get("target_id", ""))
    track = _track_flags(obs, env, rid, target_id)
    candidate = _target_candidate_stats(env, rid, target_id)
    boresight_enabled = bool(getattr(env, "use_boresight_launch_gate", False))
    boresight_ok = True if not boresight_enabled else bool(diag.get("boresight_ok", False))
    kill_cooldown = rid in set(getattr(env, "_agents_deny_kill", set()) or set())
    enriched = dict(diag)
    enriched["kill_cooldown"] = kill_cooldown
    reasons = _block_reasons(enriched, bool(track["track_available"]), boresight_ok)
    final_allowed = (
        bool(diag.get("shooter_alive", True))
        and bool(diag.get("has_missile", False))
        and bool(diag.get("target_alive", False))
        and bool(track["track_available"])
        and bool(diag.get("range_ok", False))
        and bool(diag.get("ao_ok", False))
        and bool(diag.get("ta_ok", False))
        and bool(diag.get("lock_ready", False))
        and bool(diag.get("cooldown_ready", False))
        and bool(diag.get("deconflict_ok", False))
        and bool(boresight_ok)
        and not kill_cooldown
    )
    predicted_raw = bool(diag.get("launch_allowed_predicted", False))
    return {
        **candidate,
        **track,
        "boresight_ok": int(boresight_ok),
        "boresight_status": "enabled" if boresight_enabled else "not_enabled",
        "kill_cooldown": kill_cooldown,
        "final_launch_allowed": bool(final_allowed),
        "predicted_launch_allowed_raw": predicted_raw,
        "predicted_vs_final_mismatch": int(predicted_raw != bool(final_allowed)),
        "launch_block_reason_primary": reasons[0],
        "launch_block_reason_all": ";".join(reasons),
    }


def _zero_dead_recurrent_hidden(rnn_hidden, env, info):
    if rnn_hidden is None:
        return None
    for idx, rid in enumerate(getattr(env, "red_ids", [])):
        if not bool(info.get(rid, {}).get("alive", False)):
            rnn_hidden[idx, :] = 0.0
    return rnn_hidden


def run_diagnostics(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from algorithms.mappo.opponent_policy import OpponentPolicy
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2

    model_path = _resolve_model(args)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    meta = _load_meta(model_path)
    device = torch.device(args.device)
    policy = _build_policy(meta, device)
    policy.load(model_path, map_location=device)
    policy.eval()
    adapter = HeteroObsAdapterV2()
    config = args.config or SCENARIO_CONFIGS[args.scenario]
    label = args.label or (
        Path(args.output_dir).name if args.output_dir else model_path.parent.parent.name
    )
    arch = str(meta.get("policy_arch", "flat"))
    recurrent_eval_used = arch in {"brma_recurrent", "brma_recurrent_masked", "hetero_entity_recurrent"}
    rows: list[dict[str, Any]] = []

    for ep in range(args.episodes):
        env = make_env(config, env_type="jsbsim_hetero", suppress_jsbsim_output=True)
        opponent = OpponentPolicy(mode=args.opponent_policy, seed=args.seed + ep + 17)
        try:
            obs, info = env.reset(seed=args.seed + ep)
            rnn_hidden = None
            if recurrent_eval_used:
                rnn_hidden = np.zeros((len(env.red_ids), int(getattr(policy, "rnn_hidden_size", 128))), dtype=np.float32)
            prev_hits = {"red": 0, "blue": 0}
            terminated = {aid: False for aid in env.agent_ids}
            truncated = {aid: False for aid in env.agent_ids}
            for step in range(1, args.max_steps + 1):
                actions_np, rnn_hidden = _policy_actions(policy, adapter, env, obs, info, device, rnn_hidden)
                action_dict = {rid: actions_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
                action_dict.update(opponent.act(obs, env.blue_ids, env=env))

                before = {}
                for i, rid in enumerate(env.red_ids):
                    if env.agent_roles.get(rid) == "mav":
                        continue
                    diag = _diagnose_red_shooter(env, rid)
                    diag["shooter_alive"] = bool(env.red_planes.get(rid) and env.red_planes[rid].is_alive)
                    diag["shooter_role"] = env.agent_roles.get(rid, "")
                    diag["cooldown_frames_left"] = int(getattr(env, "_missile_cooldown", {}).get(rid, 0))
                    diag["lock_target"] = getattr(env, "_lock_target", {}).get(rid, "")
                    diag["lock_delay_frames"] = int(getattr(env, "missile_lock_delay_frames", 0))
                    diag["kill_cooldown"] = rid in set(getattr(env, "_agents_deny_kill", set()) or set())
                    diag["action"] = actions_np[i].tolist()
                    diag["pre_step"] = _pre_step_launch_snapshot(env, obs, rid, diag)
                    before[rid] = diag

                obs, _rewards, terminated, truncated, info = env.step(action_dict)
                if rnn_hidden is not None:
                    rnn_hidden = _zero_dead_recurrent_hidden(rnn_hidden, env, info)
                counts = collect_step_counts(info)
                mt = info.get("__missile_term__", {})
                red_hit_total = int(mt.get("red", {}).get("hit", 0)) if isinstance(mt, dict) else 0
                red_hit_delta = max(red_hit_total - prev_hits["red"], 0)
                prev_hits["red"] = red_hit_total
                terminal = _terminal_reason(env, terminated, truncated)
                blue_dead = alive_counts(env)["blue_dead"]
                fired_by_red = {
                    aid: int(agent_info.get("missiles_fired_this_step", 0) or 0)
                    for aid, agent_info in info.items()
                    if isinstance(agent_info, dict) and aid.startswith("red_")
                }
                for rid, diag in before.items():
                    action = diag.pop("action")
                    pre_step = diag.pop("pre_step")
                    target_id = str(diag.get("target_id", ""))
                    fired_now = int(fired_by_red.get(rid, 0))
                    final_allowed = bool(pre_step["final_launch_allowed"])
                    rows.append({
                        "model_label": label,
                        "scenario": args.scenario,
                        "episode_id": ep,
                        "step": step,
                        "red_id": rid,
                        "shooter_alive": diag.get("shooter_alive", False),
                        "shooter_role": diag.get("shooter_role", ""),
                        "shooter_has_missile": diag.get("has_missile", False),
                        "nearest_blue_id": pre_step.get("nearest_blue_id", ""),
                        "nearest_blue_alive": pre_step.get("nearest_blue_alive", False),
                        "nearest_blue_range_m": pre_step.get("nearest_blue_range_m", ""),
                        "selected_target_id": pre_step.get("selected_target_id", target_id),
                        "selected_target_range_m": pre_step.get("selected_target_range_m", diag.get("range_m", "")),
                        "selected_target_AO_rad": diag.get("ao_rad", ""),
                        "selected_target_TA_rad": diag.get("ta_rad", ""),
                        "candidate_count": pre_step.get("candidate_count", 0),
                        "alive_blue_count": pre_step.get("alive_blue_count", 0),
                        "unengaged_alive_blue_count": pre_step.get("unengaged_alive_blue_count", 0),
                        "track_available": pre_step.get("track_available", False),
                        "direct_track_available": pre_step.get("direct_track_available", False),
                        "mav_shared_track_available": pre_step.get("mav_shared_track_available", False),
                        "launch_track_source": pre_step.get("launch_track_source", "none"),
                        "lock_target": diag.get("lock_target", ""),
                        "lock_timer": diag.get("lock_timer", 0),
                        "lock_delay_frames": diag.get("lock_delay_frames", 0),
                        "cooldown_frames_left": diag.get("cooldown_frames_left", 0),
                        "boresight_ok": pre_step.get("boresight_ok", 1),
                        "boresight_status": pre_step.get("boresight_status", "not_enabled"),
                        "target_id": diag.get("target_id", ""),
                        "range_m": diag.get("range_m", ""),
                        "AO_rad": diag.get("ao_rad", ""),
                        "TA_rad": diag.get("ta_rad", ""),
                        "lock_ready": diag.get("lock_ready", False),
                        "cooldown_ready": diag.get("cooldown_ready", False),
                        "deconflict_ok": diag.get("deconflict_ok", False),
                        "has_missile": diag.get("has_missile", False),
                        "target_alive": diag.get("target_alive", False),
                        "range_ok": diag.get("range_ok", False),
                        "ao_ok": diag.get("ao_ok", False),
                        "ta_ok": diag.get("ta_ok", False),
                        "final_launch_allowed": final_allowed,
                        "actual_missiles_fired_this_step": fired_now,
                        "actual_red_hit_delta_this_step": red_hit_delta,
                        "predicted_allowed_but_not_fired": int(final_allowed and fired_now <= 0),
                        "fired_without_predicted_allowed": int((not final_allowed) and fired_now > 0),
                        "predicted_vs_final_mismatch": pre_step.get("predicted_vs_final_mismatch", 0),
                        "predicted_launch_allowed_raw": pre_step.get("predicted_launch_allowed_raw", False),
                        "launch_block_reason_primary": pre_step.get("launch_block_reason_primary", ""),
                        "launch_block_reason_all": pre_step.get("launch_block_reason_all", ""),
                        "launch_allowed": diag.get("launch_allowed_predicted", False),
                        "launch_block_reason": diag.get("launch_block_reason", ""),
                        "action_pitch": float(action[0]),
                        "action_heading": float(action[1]),
                        "action_speed": float(action[2]),
                        "missiles_fired": fired_now,
                        "missile_hits": red_hit_total,
                        "blue_dead": blue_dead,
                        "terminal_reason": terminal,
                    })
                if red_hit_delta:
                    pass
                if team_done(terminated, truncated):
                    break
            if rnn_hidden is not None:
                rnn_hidden[:] = 0.0
        finally:
            env.close()
    return rows, _summarize(rows, args.episodes, label, args.scenario, arch)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Learned Policy Launch Diagnostics",
        "",
        f"- model: `{summary.get('model_label')}`",
        f"- scenario: `{summary.get('scenario')}`",
        f"- policy_arch: `{summary.get('policy_arch')}`",
        f"- episodes: `{summary.get('episodes')}`",
        f"- red_missiles_fired: `{summary.get('red_missiles_fired')}`",
        f"- missile_hits: `{summary.get('missile_hits')}`",
        f"- blue_dead_mean: `{summary.get('blue_dead_mean')}`",
        f"- range_ok_rate: `{summary.get('range_ok_rate')}`",
        f"- ao_ok_rate: `{summary.get('ao_ok_rate')}`",
        f"- ta_ok_rate: `{summary.get('ta_ok_rate')}`",
        f"- lock_ready_rate: `{summary.get('lock_ready_rate')}`",
        f"- cooldown_ready_rate: `{summary.get('cooldown_ready_rate')}`",
        f"- deconflict_ok_rate: `{summary.get('deconflict_ok_rate')}`",
        f"- track_available_rate: `{summary.get('track_available_rate')}`",
        f"- mav_shared_track_available_rate: `{summary.get('mav_shared_track_available_rate')}`",
        f"- final_launch_allowed_rate: `{summary.get('final_launch_allowed_rate')}`",
        f"- actual_fire_rate: `{summary.get('actual_fire_rate')}`",
        f"- predicted_allowed_but_not_fired_count: `{summary.get('predicted_allowed_but_not_fired_count')}`",
        f"- fired_without_predicted_allowed_count: `{summary.get('fired_without_predicted_allowed_count')}`",
        f"- predicted_vs_final_mismatch_count: `{summary.get('predicted_vs_final_mismatch_count')}`",
        f"- dominant_block_reason: `{summary.get('dominant_block_reason')}`",
        f"- block_reason_counts: `{summary.get('block_reason_counts_json')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate learned policy launch-envelope diagnostics")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model.pt, or checkpoint name under --output-dir.")
    parser.add_argument("--checkpoint-name", choices=["best", "latest"], default="best")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scenario", choices=["3v2", "5v4"], default="3v2")
    parser.add_argument("--config", default=None)
    parser.add_argument("--diagnostic-output-dir", default="outputs/learned_policy_launch_diagnostics")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-policy", default="brma_rule",
                        choices=["zero", "random", "rule_nearest", "greedy_fsm", "brma_rule", "brma_rule_safe_pursuit", "tam_greedy_easy", "brma_rule_safe_pursuit_easy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    out_dir = _safe_output_dir(args.diagnostic_output_dir)
    rows, summary = run_diagnostics(args)
    detail_csv = out_dir / "launch_diagnostics.csv"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    _write_csv(detail_csv, rows, DIAG_FIELDS)
    _write_csv(summary_csv, [summary], SUMMARY_FIELDS)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_md(summary_md, summary)
    for line in [
        f"output_dir: {out_dir}",
        f"summary_json: {summary_json}",
        f"detail_csv: {detail_csv}",
        f"dominant_block_reason: {summary.get('dominant_block_reason')}",
        f"range_ok_rate: {summary.get('range_ok_rate')}",
        f"ao_ok_rate: {summary.get('ao_ok_rate')}",
        f"ta_ok_rate: {summary.get('ta_ok_rate')}",
        f"track_available_rate: {summary.get('track_available_rate')}",
        f"mav_shared_track_available_rate: {summary.get('mav_shared_track_available_rate')}",
        f"final_launch_allowed_rate: {summary.get('final_launch_allowed_rate')}",
        f"actual_fire_rate: {summary.get('actual_fire_rate')}",
        f"predicted_allowed_but_not_fired_count: {summary.get('predicted_allowed_but_not_fired_count')}",
        f"fired_without_predicted_allowed_count: {summary.get('fired_without_predicted_allowed_count')}",
        f"red_missiles_fired: {summary.get('red_missiles_fired')}",
        f"missile_hits: {summary.get('missile_hits')}",
    ]:
        try:
            print(line, flush=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
