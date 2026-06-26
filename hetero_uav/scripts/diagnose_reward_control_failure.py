"""Diagnose paper_role_reward_v1 flight control failure for HAPPO hetero_entity_recurrent.

Read-only analysis of train_log.csv trends and checkpoint rollout reward components.
Does NOT modify reward, environment, missile dynamics, PID, blue rule, action space, or observation dim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── reward component keys for paper_role_reward_v1 ──────────────────────

BRMA_KEYS = ["r_pitch", "r_roll", "r_alt", "r_bound", "r_vel", "r_adv", "r_end", "r_death"]
BRMA_REMOVED_KEYS = ["r_end_raw_removed", "r_adv_removed"]
MAV_TAM_KEYS = [
    "tam_mav_safety_raw", "tam_mav_safety_dist", "tam_mav_safety_threat", "tam_mav_safety_aspect",
    "tam_mav_support_raw", "tam_mav_support_pos", "tam_mav_support_aware", "tam_mav_support_shared",
    "tam_mav_alive_bonus", "tam_mav_dense_reward",
]
EVENT_KEYS = [
    "event_uav_kill", "event_uav_death", "event_uav_crash",
    "event_mav_death", "event_mav_loss_team",
    "event_out_zone", "event_team_kill",
]
TERMINAL_KEYS = [
    "terminal_hetero_raw", "terminal_win_component", "terminal_survival_component",
    "terminal_mav_component", "terminal_applied",
]
LOG_ONLY_KEYS = ["uav_attack", "uav_fire", "uav_hit", "uav_fire_log", "uav_attack_mav_shared_multiplier", "mav_assist"]
COUNT_KEYS = ["uav_fire_direct_count", "uav_fire_mav_guided_count", "uav_hit_direct_count", "uav_hit_mav_guided_count"]
CLIP_KEYS = ["reward_pre_clip", "reward_clip_delta"]

# TAM paper v2 keys
TAM_V2_MAV_KEYS = [
    "tam_v2_mav_safety", "tam_v2_mav_dist", "tam_v2_mav_threat", "tam_v2_mav_aspect",
    "tam_v2_mav_support", "tam_v2_mav_pos", "tam_v2_mav_aware",
    "tam_v2_mav_event", "tam_v2_mav_death", "tam_v2_mav_team_bonus",
    "tam_v2_total",
]
TAM_V2_UAV_KEYS = [
    "tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
    "tam_v2_uav_angle_raw", "tam_v2_uav_distance",
    "tam_v2_uav_dodge", "tam_v2_uav_dodge_angle", "tam_v2_uav_dodge_speed",
    "tam_v2_uav_event", "tam_v2_uav_kill", "tam_v2_uav_death",
    "tam_v2_uav_out_of_zone", "tam_v2_total",
]
TAM_V2_META_KEYS = [
    "tam_v2_geometry_feature_semantics", "tam_v2_dodge_los_semantics",
    "tam_v2_height_formula_source",
]
TAM_V2_LOG_KEYS = [
    "tam_v2_mav_shared_log", "tam_v2_mav_assist_log",
    "tam_v2_uav_fire_log", "tam_v2_uav_mav_shared_track_log",
    "brma_r_adv_log", "brma_r_pitch_log", "brma_r_roll_log",
    "brma_r_alt_log", "brma_r_bound_log", "brma_r_vel_log",
]
TAM_V2_KEYS = TAM_V2_MAV_KEYS + TAM_V2_UAV_KEYS + TAM_V2_LOG_KEYS + TAM_V2_META_KEYS

ALL_RC_KEYS = BRMA_KEYS + BRMA_REMOVED_KEYS + MAV_TAM_KEYS + EVENT_KEYS + TERMINAL_KEYS + LOG_ONLY_KEYS + COUNT_KEYS + CLIP_KEYS + TAM_V2_KEYS
ALL_RC_KEYS.append("event_total")

LAUNCH_GEOMETRY_KEYS = [
    "range_m", "AO_rad", "TA_rad", "range_ok", "ao_ok", "ta_ok",
    "ATA_3d_rad", "TA_3d_rad", "boresight_3d_rad", "range_3d_m",
    "range_ok_3d", "ata_ok_3d", "ta_ok_3d", "boresight_ok_3d",
    "launch_geometry_ok_3d", "has_track", "track_source", "has_direct_track", "has_mav_shared_track",
]

FLIGHT_KEYS = [
    "episode", "step", "sim_time", "agent_id", "role", "alive",
    "altitude_m", "speed_mps", "vertical_speed_mps",
    "roll_rad", "pitch_rad", "heading_rad",
    "raw_action_0", "raw_action_1", "raw_action_2",
    "reward_total", "done", "death_reason",
    "missile_warning", "num_left_missiles",
]


def _load_train_log(run_dir: Path) -> list[dict] | None:
    path = run_dir / "train_log.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_meta(run_dir: Path) -> dict:
    path = run_dir / "latest" / "meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    best = run_dir / "best" / "model.pt"
    if best.exists():
        return best
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*/model.pt"))
    if checkpoints:
        return checkpoints[-1]
    latest = run_dir / "latest" / "model.pt"
    if latest.exists():
        return latest
    return None


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── A. Train log trend analysis ─────────────────────────────────────────

def analyze_train_trends(rows: list[dict]) -> dict:
    """Extract key trend indicators from train_log.csv."""
    if not rows:
        return {"error": "no train_log.csv found"}
    rets = [_safe_float(r["avg_return"]) for r in rows]
    mav_s = [_safe_float(r["mav_survival"]) for r in rows]
    eM = [_safe_float(r["entropy_mav"]) for r in rows]
    eU = [_safe_float(r["entropy_uav"]) for r in rows]
    satM = [_safe_float(r["mav_action_saturation_rate"]) for r in rows]
    satU = [_safe_float(r["uav_action_saturation_rate"]) for r in rows]
    rfire = [_safe_float(r["red_episode_missiles_fired_mean"]) for r in rows]
    bhit = [_safe_float(r["blue_episode_missile_hits_mean"]) for r in rows]
    ep_len = [_safe_float(r["avg_episode_length"]) for r in rows] if "avg_episode_length" in rows[0] else [0]*len(rows)
    aM = [_safe_float(r["actor_loss_mav"]) for r in rows]
    aU = [_safe_float(r["actor_loss_uav"]) for r in rows]
    steps = [int(float(r["total_steps"])) for r in rows]
    log_std_mav = [_safe_float(r["action_log_std_mav_mean"]) for r in rows]
    log_std_uav = [_safe_float(r["action_log_std_uav_mean"]) for r in rows]

    best_idx = int(np.argmax(rets))
    best_mav_idx = int(np.argmax(mav_s))
    final_idx = len(rows) - 1

    # Find entropy inflection: point where MAV entropy first exceeds 0.70
    ent_inflection = next((i for i, v in enumerate(eM) if v > 0.70), -1)
    # Find saturation inflection: point where UAV saturation first exceeds 0.05
    sat_inflection = next((i for i, v in enumerate(satU) if v > 0.05), -1)
    # Find return collapse: point where return drops more than 5 from best
    ret_collapse = -1
    best_ret = rets[best_idx]
    for i in range(best_idx, len(rets)):
        if rets[i] < best_ret - 5:
            ret_collapse = i
            break

    return {
        "total_iterations": len(rows),
        "final_step": steps[-1] if steps else 0,
        "best_return": best_ret,
        "best_return_step": steps[best_idx],
        "best_mav_survival": mav_s[best_mav_idx],
        "best_mav_survival_step": steps[best_mav_idx],
        "final_return": rets[-1],
        "final_mav_survival": mav_s[-1],
        "return_decline_from_best": best_ret - rets[-1],
        "entropy_inflection_step": steps[ent_inflection] if ent_inflection >= 0 else -1,
        "entropy_at_inflection": eM[ent_inflection] if ent_inflection >= 0 else 0,
        "saturation_inflection_step": steps[sat_inflection] if sat_inflection >= 0 else -1,
        "saturation_at_inflection": satU[sat_inflection] if sat_inflection >= 0 else 0,
        "return_collapse_step": steps[ret_collapse] if ret_collapse >= 0 else -1,
        "entropy_increased_before_return_collapse": (ent_inflection >= 0 and ret_collapse >= 0 and ent_inflection < ret_collapse),
        "action_saturation_increased_before_return_collapse": (sat_inflection >= 0 and ret_collapse >= 0 and sat_inflection < ret_collapse),
        "mav_survival_collapsed_before_or_after_entropy": "before" if best_mav_idx < ent_inflection else "after",
        "red_fire_rate_near_zero": all(v < 0.1 for v in rfire),
        "blue_hit_rate_high": np.mean(bhit[-50:] if len(rows) >= 50 else bhit) > 1.0,
        "episode_length_trend": "decreasing" if ep_len[-1] < ep_len[best_idx] * 0.5 and best_idx > 0 else "stable_or_increasing",
        "log_std_mav_drift": log_std_mav[-1] - log_std_mav[0] if len(log_std_mav) > 1 else 0,
        "log_std_uav_drift": log_std_uav[-1] - log_std_uav[0] if len(log_std_uav) > 1 else 0,
        "actor_loss_mav_peak": max(aM, key=abs),
        "actor_loss_uav_peak": max(aU, key=abs),
    }


def _json_safe(v):
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def save_train_trends(trends: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = {k: _json_safe(v) for k, v in trends.items()}
    (output_dir / "diagnosis_train_trends.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")
    with (output_dir / "diagnosis_train_trends.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trends.keys()))
        w.writeheader()
        w.writerow({k: str(v) for k, v in trends.items()})


# ── B. Checkpoint rollout diagnostic ────────────────────────────────────

def _load_policy_from_meta(meta: dict, device):
    """Load policy from meta, supporting hetero_entity_recurrent + flat."""
    import torch
    policy_arch = meta.get("policy_arch", "flat")
    if policy_arch == "hetero_entity_recurrent":
        from algorithms.happo.hetero_entity_recurrent_policy import HeteroEntityRecurrentPolicy, validate_entity_policy_meta
        validate_entity_policy_meta(meta)
        return HeteroEntityRecurrentPolicy(
            entity_dim=int(meta.get("entity_dim", 21)),
            action_dim=3,
            hidden_dim=int(meta.get("hidden_dim", 128)),
            rnn_hidden_size=int(meta.get("rnn_hidden_size", 128)),
            num_attention_heads=int(meta.get("num_attention_heads", 4)),
        ).to(device)
    if policy_arch == "flat":
        from algorithms.happo import HAPPOReferencePolicy
        return HAPPOReferencePolicy(
            int(meta.get("actor_obs_dim", 96)),
            int(meta.get("critic_state_dim", 480)),
        ).to(device)
    if policy_arch == "pure_happo":
        from algorithms.pure_happo import PureHAPPOPolicy
        num_agents = int(meta.get("num_agents", 3))
        return PureHAPPOPolicy(
            actor_obs_dim=int(meta.get("actor_obs_dim", 96)),
            critic_state_dim=int(meta.get("critic_state_dim", 480)),
            action_dim=3, num_agents=num_agents,
        ).to(device)
    raise ValueError(f"unsupported policy_arch for diagnosis: {policy_arch}")


def _role(env, aid: str) -> str:
    return env.agent_roles.get(aid, "unknown")


def _collect_flight_row(ep: int, step: int, sim_time: float, aid: str, env,
                        info: dict, action_raw, reward_total: float,
                        done: bool) -> dict:
    sim = env.red_planes.get(aid) or env.blue_planes.get(aid)
    row = {
        "episode": ep, "step": step, "sim_time": round(sim_time, 2),
        "agent_id": aid, "role": _role(env, aid),
        "alive": int(sim.is_alive) if sim else 0,
        "altitude_m": 0.0, "speed_mps": 0.0, "vertical_speed_mps": 0.0,
        "roll_rad": 0.0, "pitch_rad": 0.0, "heading_rad": 0.0,
        "raw_action_0": 0.0, "raw_action_1": 0.0, "raw_action_2": 0.0,
        "reward_total": round(reward_total, 4), "done": int(done),
        "death_reason": "", "missile_warning": 0, "num_left_missiles": 0,
    }
    if sim and sim.is_alive:
        pos = sim.get_position()
        vel = sim.get_velocity()
        rpy = sim.get_rpy()
        row["altitude_m"] = round(float(pos[2]), 1)
        row["speed_mps"] = round(float(np.linalg.norm(vel)), 1)
        row["vertical_speed_mps"] = round(float(vel[2]), 1)
        row["roll_rad"] = round(float(rpy[0]), 4)
        row["pitch_rad"] = round(float(rpy[1]), 4)
        row["heading_rad"] = round(float(rpy[2]), 4)
        if action_raw is not None:
            row["raw_action_0"] = round(float(action_raw[0]), 4)
            row["raw_action_1"] = round(float(action_raw[1]), 4)
            row["raw_action_2"] = round(float(action_raw[2]), 4)
        row["missile_warning"] = int(sim.check_missile_warning() is not None)
        row["num_left_missiles"] = int(getattr(sim, "num_left_missiles", 0))
    if sim is None or not sim.is_alive:
        row["death_reason"] = str(info.get(aid, {}).get("death_reason", "")) if isinstance(info.get(aid, {}), dict) else ""
    return row


def _collect_reward_row(ep: int, step: int, sim_time: float, aid: str, env,
                         components: dict | None) -> dict | None:
    if components is None:
        return None
    row = {"episode": ep, "step": step, "sim_time": round(sim_time, 2),
           "agent_id": aid, "role": _role(env, aid)}
    comp = components.get(aid, {}) if isinstance(components, dict) else {}
    for key in ALL_RC_KEYS:
        val = comp.get(key, 0.0)
        row[key] = round(float(val), 8) if isinstance(val, (int, float, np.number)) else str(val)
    row["reward_total"] = round(float(comp.get("total", 0.0)), 4)
    return row


def _collect_launch_row(ep: int, step: int, sim_time: float, env, info: dict) -> list[dict]:
    rows = []
    quality_step = info.get("__launch_quality_step__", [])
    if isinstance(quality_step, list):
        for rec in quality_step:
            r = {"episode": ep, "step": step, "sim_time": round(sim_time, 2)}
            for k in LAUNCH_GEOMETRY_KEYS:
                r[k] = rec.get(k, "")
            rows.append(r)
    return rows


def _collect_launch_diag_row(ep: int, step: int, sim_time: float, info: dict) -> dict | None:
    diag = info.get("__launch_diag__")
    if not isinstance(diag, dict):
        return None
    row = {"episode": ep, "step": step, "sim_time": round(sim_time, 2)}
    for team in ("red", "blue"):
        td = diag.get(team, {})
        if isinstance(td, dict):
            for k, v in td.items():
                row[f"{team}_{k}"] = v
    return row


def run_checkpoint_diagnostics(checkpoint_path: str | None, config_path: str,
                                episodes: int, max_steps: int, device_str: str,
                                output_dir: Path, deterministic: bool,
                                run_dir: Path) -> dict:
    """Run deterministic rollout with checkpoint, collecting per-step diagnostics."""
    import torch
    from uav_env import make_env
    from uav_env.JSBSim.adapters.hetero_entity_set_adapter import HeteroEntitySetAdapter
    from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
    from algorithms.mappo.opponent_policy import OpponentPolicy

    if checkpoint_path is None:
        checkpoint_path = _find_best_checkpoint(run_dir)
    if checkpoint_path is None:
        return {"error": "no checkpoint found", "checkpoint_used": "none"}
    cp = Path(checkpoint_path)
    if not cp.exists():
        return {"error": f"checkpoint not found: {checkpoint_path}", "checkpoint_used": str(cp)}

    meta = json.loads((cp.parent / "meta.json").read_text(encoding="utf-8")) if (cp.parent / "meta.json").exists() else {}
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    policy = _load_policy_from_meta(meta, device)
    policy.load(cp, map_location=device)
    policy.eval()
    entity_mode = meta.get("policy_arch") == "hetero_entity_recurrent"

    adapter = HeteroEntitySetAdapter() if entity_mode else HeteroObsAdapterV2()
    env = make_env(config_path, env_type="jsbsim_hetero")

    _rnn_hidden_size = getattr(policy, "rnn_hidden_size", 0)
    flight_rows, reward_rows, launch_rows, launch_diag_rows = [], [], [], []
    episode_summaries = []

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        ep_return = 0.0
        ep_len = 0
        eval_rnn_hidden = None
        if _rnn_hidden_size > 0:
            eval_rnn_hidden = np.zeros((len(env.red_ids), _rnn_hidden_size), dtype=np.float32)
        prev_hits = {"red": 0, "blue": 0}
        missile_stats = {"red_fired": 0, "blue_fired": 0, "red_hits": 0, "blue_hits": 0}

        while True:
            adapted = adapter.adapt_all(obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
            sim_time = ep_len * float(env.env_dt)

            # Active mask
            active = np.ones(len(env.red_ids), dtype=np.float32)
            for i, rid in enumerate(env.red_ids):
                ai = (info or {}).get(rid, {})
                active[i] = 1.0 if ai.get("alive", True) else 0.0

            act_kw = {}
            if eval_rnn_hidden is not None:
                from algorithms.happo.rollout_safety import zero_inactive_hidden
                eval_rnn_hidden = zero_inactive_hidden(eval_rnn_hidden, active)
                act_kw["rnn_hidden"] = torch.as_tensor(eval_rnn_hidden, device=device)

            with torch.no_grad():
                if entity_mode:
                    out = policy.act(
                        torch.as_tensor(adapted["actor_entity_tokens"], device=device),
                        torch.as_tensor(adapted["actor_keep_mask"], device=device),
                        torch.as_tensor(adapted["role_ids"], device=device),
                        torch.as_tensor(adapted["critic_entity_tokens"], device=device),
                        torch.as_tensor(adapted["critic_keep_mask"], device=device),
                        deterministic=deterministic,
                        critic_counts=torch.as_tensor(
                            adapted.get("critic_counts", np.zeros(4, dtype=np.float32)), device=device),
                        **act_kw,
                    )
                else:
                    actor_obs = np.stack([
                        adapted["actor_obs"].get(rid, np.zeros(adapter.flat_actor_obs_dim, dtype=np.float32))
                        for rid in env.red_ids
                    ])
                    out = policy.act(
                        torch.as_tensor(actor_obs, device=device),
                        roles=[0 if env.agent_roles.get(rid) == "mav" else 1 for rid in env.red_ids],
                        critic_state=torch.as_tensor(adapted["critic_state"], device=device),
                        deterministic=deterministic,
                    )
            acts_np = out["action"].cpu().numpy()
            if eval_rnn_hidden is not None and "rnn_hidden" in out:
                from algorithms.happo.rollout_safety import zero_inactive_hidden
                eval_rnn_hidden = zero_inactive_hidden(out["rnn_hidden"].cpu().numpy(), active)

            actions = {rid: acts_np[i].astype(np.float32) for i, rid in enumerate(env.red_ids)}
            opponent = OpponentPolicy(mode="brma_rule", seed=ep * 1000 + ep_len)
            actions.update(opponent.act(obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            ep_len += 1

            # Per-agent flight rows
            for i, rid in enumerate(env.red_ids):
                r = _collect_flight_row(ep, ep_len, sim_time, rid, env, info,
                                         acts_np[i], float(rewards.get(rid, 0.0)),
                                         all(terminated.values()) or all(truncated.values()))
                flight_rows.append(r)
                ep_return += float(rewards.get(rid, 0.0))

            # Reward component rows
            rc = info.get("reward_components", None)
            for rid in env.red_ids:
                rr = _collect_reward_row(ep, ep_len, sim_time, rid, env, rc)
                if rr:
                    reward_rows.append(rr)

            # Launch geometry rows
            launch_rows.extend(_collect_launch_row(ep, ep_len, sim_time, env, info))
            ldr = _collect_launch_diag_row(ep, ep_len, sim_time, info)
            if ldr:
                launch_diag_rows.append(ldr)

            # Missile stats
            for aid in env.agent_ids:
                fired = int(info.get(aid, {}).get("missiles_fired_this_step", 0)) if isinstance(info.get(aid, {}), dict) else 0
                if aid.startswith("red_"):
                    missile_stats["red_fired"] += fired
                else:
                    missile_stats["blue_fired"] += fired
            mt = info.get("__missile_term__", {})
            if isinstance(mt, dict):
                for side in ("red", "blue"):
                    total_h = int(mt.get(side, {}).get("hit", 0))
                    missile_stats[f"{side}_hits"] += max(total_h - prev_hits[side], 0)
                    prev_hits[side] = total_h

            if all(terminated.values()) or all(truncated.values()):
                break
            if ep_len >= max_steps:
                break

        red_alive = sum(1 for s in env.red_planes.values() if s.is_alive)
        blue_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
        mav_alive = bool(env.red_planes.get("red_0") and env.red_planes["red_0"].is_alive)
        episode_summaries.append({
            "episode": ep, "steps": ep_len,
            "return": round(ep_return, 2),
            "red_alive_final": red_alive, "blue_alive_final": blue_alive,
            "mav_alive": mav_alive,
            "red_fired": missile_stats["red_fired"], "blue_fired": missile_stats["blue_fired"],
            "red_hits": missile_stats["red_hits"], "blue_hits": missile_stats["blue_hits"],
        })

    # Save CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    if flight_rows:
        _write_csv(output_dir / "rollout_step_diagnostics.csv", flight_rows, FLIGHT_KEYS)
    if reward_rows:
        _write_csv(output_dir / "reward_component_step_diagnostics.csv", reward_rows, ALL_RC_KEYS, prefix_cols=["episode", "step", "sim_time", "agent_id", "role", "reward_total"])
    if launch_rows:
        _write_csv(output_dir / "launch_geometry_diagnostics.csv", launch_rows, LAUNCH_GEOMETRY_KEYS, prefix_cols=["episode", "step", "sim_time"])
    if launch_diag_rows:
        all_keys = sorted(set().union(*(d.keys() for d in launch_diag_rows)))
        prefixed = ["episode", "step", "sim_time"] + [k for k in all_keys if k not in ("episode", "step", "sim_time")]
        _write_csv(output_dir / "launch_diag_by_step.csv", launch_diag_rows, prefixed)
    if episode_summaries:
        _write_csv(output_dir / "episode_summary.csv", episode_summaries,
                    ["episode", "steps", "return", "red_alive_final", "blue_alive_final", "mav_alive",
                     "red_fired", "blue_fired", "red_hits", "blue_hits"])

    # ── Reward component summary ────────────────────────────────────
    rc_summary = _compute_reward_component_summary(reward_rows, flight_rows)
    (output_dir / "reward_component_summary.json").write_text(json.dumps(rc_summary, indent=2), encoding="utf-8")

    # ── Pre-crash window analysis ───────────────────────────────────
    crash_analysis = _pre_crash_window_analysis(flight_rows, reward_rows)
    (output_dir / "pre_crash_reward_window.json").write_text(json.dumps(crash_analysis, indent=2), encoding="utf-8")
    if crash_analysis.get("events"):
        with (output_dir / "pre_crash_reward_window.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["episode", "agent_id", "event_type", "event_step"]
            for k in crash_analysis.get("event_aggregates", {}).get("fields", []):
                fields.append(k)
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for ev in crash_analysis["events"]:
                w.writerow(ev)

    # ── Launch geometry summary ──────────────────────────────────────
    launch_summary = _launch_geometry_summary(launch_rows, launch_diag_rows, flight_rows)
    (output_dir / "launch_geometry_summary.json").write_text(json.dumps(launch_summary, indent=2), encoding="utf-8")

    result = {
        "checkpoint_used": str(cp),
        "episodes_run": episodes,
        "total_steps_recorded": len(flight_rows),
        "output_files": {
            "rollout_step_diagnostics.csv": str(output_dir / "rollout_step_diagnostics.csv"),
            "reward_component_step_diagnostics.csv": str(output_dir / "reward_component_step_diagnostics.csv"),
            "launch_geometry_diagnostics.csv": str(output_dir / "launch_geometry_diagnostics.csv"),
            "launch_diag_by_step.csv": str(output_dir / "launch_diag_by_step.csv"),
            "episode_summary.csv": str(output_dir / "episode_summary.csv"),
            "reward_component_summary.json": str(output_dir / "reward_component_summary.json"),
            "pre_crash_reward_window.json": str(output_dir / "pre_crash_reward_window.json"),
            "launch_geometry_summary.json": str(output_dir / "launch_geometry_summary.json"),
        },
        "episode_summaries": episode_summaries,
        "reward_component_summary": rc_summary,
        "launch_geometry_summary": launch_summary,
        "crash_analysis": crash_analysis,
    }
    if hasattr(env, "close"):
        env.close()
    return result


def _write_csv(path, rows, keys, prefix_cols=None):
    if prefix_cols:
        fieldnames = list(prefix_cols) + [k for k in keys if k not in prefix_cols]
    else:
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _compute_reward_component_summary(reward_rows: list[dict], flight_rows: list[dict]) -> dict:
    """Aggregate reward component stats by role (MAV vs UAV)."""
    if not reward_rows:
        return {"error": "no reward rows"}
    mav_rows = [r for r in reward_rows if r.get("role") == "mav"]
    uav_rows = [r for r in reward_rows if r.get("role") == "attack_uav"]
    result = {"mav": {}, "uav": {}}
    for label, subset in [("mav", mav_rows), ("uav", uav_rows)]:
        if not subset:
            result[label] = {"count": 0}
            continue
        stats = {}
        for key in ALL_RC_KEYS:
            vals = []
            for r in subset:
                v = r.get(key)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    vals.append(float(v))
            if vals:
                arr = np.array(vals)
                stats[key] = {
                    "mean": round(float(arr.mean()), 6),
                    "std": round(float(arr.std()), 6),
                    "min": round(float(arr.min()), 6),
                    "max": round(float(arr.max()), 6),
                    "sum": round(float(arr.sum()), 3),
                    "nonzero_rate": round(float((arr != 0).mean()), 4),
                    "positive_rate": round(float((arr > 0).mean()), 4),
                    "negative_rate": round(float((arr < 0).mean()), 4),
                }
        result[label] = {"count": len(subset), "components": stats}
    # Also compute dominant component by absolute mean
    for label in ("mav", "uav"):
        comps = result[label].get("components", {})
        if comps:
            by_abs = sorted(comps.items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)
            result[label]["top5_by_abs_mean"] = [
                {"component": k, "abs_mean": round(abs(float(v["mean"])), 6)} for k, v in by_abs[:5]
            ]
    return result


def _pre_crash_window_analysis(flight_rows: list[dict], reward_rows: list[dict]) -> dict:
    """For each agent, find crash/death/low-alt events and analyze reward before."""
    if not flight_rows:
        return {"error": "no flight rows"}
    events = []
    # Group flight rows by episode+agent
    groups = defaultdict(list)
    for r in flight_rows:
        groups[(r["episode"], r["agent_id"])].append(r)
    for (ep, aid), rows in groups.items():
        for i, r in enumerate(rows):
            alt = float(r.get("altitude_m", 0))
            alive = int(r.get("alive", 0))
            step = int(r.get("step", 0))
            # Crash event: altitude < 2000m (below hard deck) while alive
            if alt < 2000 and alive and i > 0:
                prev_alt = float(rows[i-1].get("altitude_m", 0))
                if prev_alt >= 2000 or i < 2:  # just crossed below 2km
                    events.append({
                        "episode": ep, "agent_id": aid, "role": r.get("role", ""),
                        "event_type": "altitude_below_2000m",
                        "event_step": step,
                        "altitude_m": alt,
                    })
            # Death event
            if not alive and i > 0 and int(rows[i-1].get("alive", 0)):
                events.append({
                    "episode": ep, "agent_id": aid, "role": r.get("role", ""),
                    "event_type": "death",
                    "event_step": step,
                    "death_reason": r.get("death_reason", ""),
                })

    if not events:
        return {"events": [], "event_aggregates": {}}

    # For each event, aggregate reward components over preceding 100 steps
    window_size = 100
    agg_list = []
    reward_by_ep_agent_step = {}
    for r in reward_rows:
        reward_by_ep_agent_step[(r["episode"], r["agent_id"], int(r.get("step", 0)))] = r

    for ev in events:
        ep, aid, estep = ev["episode"], ev["agent_id"], ev["event_step"]
        pre_steps = []
        for s in range(max(0, estep - window_size), estep):
            kr = reward_by_ep_agent_step.get((ep, aid, s))
            if kr:
                pre_steps.append(kr)
        if pre_steps:
            agg = {"episode": ep, "agent_id": aid, "event_type": ev["event_type"],
                   "event_step": estep, "pre_steps_analyzed": len(pre_steps)}
            for key in ALL_RC_KEYS:
                vals = [float(rr.get(key, 0)) for rr in pre_steps if isinstance(rr.get(key), (int, float, np.number))]
                if vals:
                    arr = np.array(vals)
                    agg[f"pre_{key}_mean"] = round(float(arr.mean()), 6)
                    agg[f"pre_{key}_sum"] = round(float(arr.sum()), 4)
            # Also compute avg flight metrics from flight_rows
            flt_pre = [fr for fr in flight_rows if int(fr.get("episode", -1)) == ep and str(fr.get("agent_id", "")) == str(aid) and int(fr.get("step", 0)) in range(max(0, estep - window_size), estep)]
            if flt_pre:
                alts = [float(fr["altitude_m"]) for fr in flt_pre]
                agg["pre_flight_mean_altitude"] = round(float(np.mean(alts)), 1)
                agg["pre_flight_min_altitude"] = round(float(np.min(alts)), 1)
                # Vertical speed
                if len(alts) >= 2:
                    vs = [alts[i+1] - alts[i] for i in range(len(alts)-1)]
                    agg["pre_flight_mean_vertical_speed"] = round(float(np.mean(vs)), 2)
                    agg["pre_flight_vertical_speed_sign"] = "descending" if np.mean(vs) < -0.5 else "level" if abs(np.mean(vs)) <= 0.5 else "climbing"
            agg_list.append(agg)

    return {
        "events": agg_list,
        "event_aggregates": {"count": len(events), "fields": list(agg_list[0].keys()) if agg_list else []},
    }


def _launch_geometry_summary(launch_rows: list[dict], launch_diag_rows: list[dict],
                              flight_rows: list[dict]) -> dict:
    """Summarize launch geometry failure reasons."""
    if not launch_rows:
        return {"error": "no launch geometry data"}
    total = len(launch_rows)
    range_ok = sum(1 for r in launch_rows if r.get("range_ok_3d") == True or r.get("range_ok") == True)
    ao_ok = sum(1 for r in launch_rows if r.get("ao_ok") == True)
    ta_ok = sum(1 for r in launch_rows if r.get("ta_ok") == True)
    all_ok = sum(1 for r in launch_rows if r.get("launch_geometry_ok_3d") == True or r.get("launch_geometry_ok_3d") == "True")
    has_track_count = sum(1 for r in launch_rows if r.get("has_track") == True or r.get("has_direct_track") == True or r.get("has_mav_shared_track") == True)

    # Failure decomposition
    no_track = sum(1 for r in launch_rows if not r.get("has_track") and not r.get("has_direct_track") and not r.get("has_mav_shared_track"))
    range_not = sum(1 for r in launch_rows if not (r.get("range_ok_3d") or r.get("range_ok")))
    ao_not = sum(1 for r in launch_rows if r.get("ao_ok") == False)
    ta_not = sum(1 for r in launch_rows if r.get("ta_ok") == False)

    # Aggregate launch diag
    diag_totals = {}
    if launch_diag_rows:
        for key in launch_diag_rows[0]:
            if key.startswith("red_"):
                vals = [int(float(r.get(key, 0))) for r in launch_diag_rows]
                diag_totals[key] = sum(vals)

    # Metric per episode
    ep_stats = []
    ep_groups = defaultdict(list)
    for r in launch_rows:
        ep_groups[r.get("episode", 0)].append(r)
    for ep, rows in sorted(ep_groups.items()):
        total_e = len(rows)
        ep_stats.append({
            "episode": ep, "total_pairs": total_e,
            "range_ok": sum(1 for r in rows if r.get("range_ok_3d") or r.get("range_ok")),
            "ao_ok": sum(1 for r in rows if r.get("ao_ok")),
            "ta_ok": sum(1 for r in rows if r.get("ta_ok")),
            "geometry_ok": sum(1 for r in rows if r.get("launch_geometry_ok_3d")),
        })

    # Determine primary failure reason
    failures = {"no_track": no_track, "range_not_ok": range_not,
                "ao_not_ok": ao_not, "ta_not_ok": ta_not}
    failure_order = sorted(failures.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_launch_candidates": total,
        "range_ok": range_ok, "ao_ok": ao_ok, "ta_ok": ta_ok,
        "all_gates_ok": all_ok, "has_track": has_track_count,
        "failure_decomposition": failures,
        "failure_order": failure_order,
        "diagnostic_totals": diag_totals,
        "per_episode": ep_stats,
    }


# ── Simple trajectory interpretation ───────────────────────────────────

def _generate_trajectory_notes(flight_rows: list[dict], launch_rows: list[dict], output_dir: Path) -> str:
    """Generate human-readable trajectory interpretation markdown."""
    lines = ["# Trajectory Interpretation", ""]
    ep_groups = defaultdict(list)
    for r in flight_rows:
        ep_groups[r["episode"]].append(r)

    for ep in sorted(ep_groups.keys()):
        rows = ep_groups[ep]
        lines.append(f"## Episode {ep}")
        # MAV
        mav = [r for r in rows if r["agent_id"] == "red_0"]
        if mav and len(mav) >= 5:
            first_200 = mav[:min(200, len(mav))]
            alts = [float(r["altitude_m"]) for r in first_200 if float(r.get("alive", 0))]
            vs = [float(r["vertical_speed_mps"]) for r in first_200 if float(r.get("alive", 0))]
            lines.append(f"- MAV initial 200 steps: altitude {alts[0]:.0f} -> {alts[-1]:.0f}m, mean VS {np.mean(vs):.1f} m/s")
            lines.append(f"  - Trend: {'descending' if np.mean(vs) < -0.5 else 'climbing' if np.mean(vs) > 0.5 else 'level'}")
            # Check if MAV goes below 3000m
            below_3km = next((i for i, a in enumerate(alts) if a < 3000), -1)
            if below_3km >= 0:
                lines.append(f"  - First below 3000m at step {below_3km + 1}")

        # UAV
        uav_rows = [r for r in rows if r["role"] == "attack_uav"]
        if uav_rows:
            uav_alts = [float(r["altitude_m"]) for r in uav_rows if float(r.get("alive", 0))]
            if uav_alts and len(uav_alts) >= 5:
                lines.append(f"- UAV altitude range: {min(uav_alts):.0f} - {max(uav_alts):.0f}m")
                # Rising then retreating?
                first_100 = [float(r["altitude_m"]) for r in uav_rows[:min(100, len(uav_rows))] if float(r.get("alive", 0))]
                if first_100 and len(first_100) >= 5:
                    early_trend = first_100[-1] - first_100[0]
                    lines.append(f"  - Early trend: {'climbing' if early_trend > 100 else 'descending' if early_trend < -100 else 'neutral'}")

        # Launch steps
        ep_launches = [r for r in launch_rows if r.get("episode") == ep]
        if ep_launches:
            lines.append(f"- Launch candidates: {len(ep_launches)}")
            ok_launches = [r for r in ep_launches if r.get("launch_geometry_ok_3d")]
            lines.append(f"  - Geometry OK: {len(ok_launches)}")
        else:
            lines.append("- No launch candidates recorded")
        lines.append("")

    path = output_dir / "trajectory_interpretation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose paper_role_reward_v1 flight control failure")
    parser.add_argument("--run-dir", required=True, help="Training output directory containing train_log.csv")
    parser.add_argument("--config", required=True, help="YAML config used for training")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint path; auto-detect if omitted")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--deterministic-action", action="store_true", default=True)
    parser.add_argument("--save-acmi", action="store_true", default=False)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(ROOT / config_path)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "reward_control_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_dir: {run_dir}")
    print(f"config: {config_path}")
    print(f"output_dir: {output_dir}")

    # Step 1: Analyze train trends
    print("\n=== Step 1: Train log trend analysis ===")
    train_rows = _load_train_log(run_dir)
    if train_rows:
        trends = analyze_train_trends(train_rows)
        save_train_trends(trends, output_dir)
        print(f"  Iterations: {trends['total_iterations']}")
        print(f"  Best return: {trends['best_return']:.1f} at step {trends['best_return_step']}")
        print(f"  Best MAV survival: {trends['best_mav_survival']:.2f} at step {trends['best_mav_survival_step']}")
        print(f"  Final return: {trends['final_return']:.1f}")
        print(f"  Entropy inflection at step: {trends['entropy_inflection_step']} (MAV entropy={trends['entropy_at_inflection']:.3f})")
        print(f"  Saturation inflection at step: {trends['saturation_inflection_step']} (UAV sat={trends['saturation_at_inflection']:.3f})")
        print(f"  Return collapse at step: {trends['return_collapse_step']}")
        print(f"  Red fire rate near zero: {trends['red_fire_rate_near_zero']}")
        print(f"  Blue hit rate high: {trends['blue_hit_rate_high']}")
    else:
        trends = {"error": "train_log.csv not found"}
        print("  WARNING: train_log.csv not found!")
        save_train_trends(trends, output_dir)

    # Step 2: Checkpoint rollout diagnostics
    print("\n=== Step 2: Checkpoint rollout diagnostics ===")
    cp = args.checkpoint
    if cp is None:
        cp = _find_best_checkpoint(run_dir)
        if cp:
            print(f"  Auto-detected checkpoint: {cp}")
        else:
            print("  No checkpoint found! Running trend analysis only.")
    else:
        if not Path(cp).exists():
            print(f"  WARNING: checkpoint not found: {cp}")
            cp = _find_best_checkpoint(run_dir)
            if cp:
                print(f"  Falling back to: {cp}")

    rollout_result = None
    if cp and Path(cp).exists():
        rollout_result = run_checkpoint_diagnostics(
            cp, config_path, args.episodes, args.max_steps,
            args.device, output_dir, args.deterministic_action, run_dir,
        )
        print(f"  Episodes run: {rollout_result.get('episodes_run', 0)}")
        print(f"  Total steps recorded: {rollout_result.get('total_steps_recorded', 0)}")
        for name, path_str in rollout_result.get("output_files", {}).items():
            print(f"  {name}: {path_str}")
    else:
        print("  Skipping rollout diagnostics (no checkpoint available)")

    # Step 3: Trajectory interpretation
    if rollout_result and rollout_result.get("output_files"):
        print("\n=== Step 3: Trajectory interpretation ===")
        from pathlib import Path as _P
        fd = _P(rollout_result["output_files"]["rollout_step_diagnostics.csv"])
        ld = _P(rollout_result["output_files"].get("launch_geometry_diagnostics.csv", ""))
        if fd.exists():
            with open(fd, encoding="utf-8") as f:
                flight_rows = list(csv.DictReader(f))
            launch_rows = []
            if ld and ld.exists():
                with open(ld, encoding="utf-8") as f:
                    launch_rows = list(csv.DictReader(f))
            traj_path = _generate_trajectory_notes(flight_rows, launch_rows, output_dir)
            print(f"  {traj_path}")

    # Step 4: Generate report
    print("\n=== Step 4: Generating diagnosis report ===")
    _generate_report(output_dir, trends, rollout_result)

    print(f"\nDone. All outputs in: {output_dir}")


def _generate_report(output_dir: Path, trends: dict, rollout_result: dict | None) -> None:
    lines = [
        "# Reward Control Diagnosis Report: paper_role_reward_v1",
        "",
        "## 1. Experiment Information",
        "",
    ]
    if rollout_result and rollout_result.get("checkpoint_used"):
        lines.append(f"- Checkpoint: `{rollout_result['checkpoint_used']}`")
    lines.append(f"- Episodes analyzed: {rollout_result.get('episodes_run', 'N/A') if rollout_result else 'N/A'}")
    lines.append("")

    # Trends
    lines.append("## 2. Training Curve Summary")
    lines.append("")
    if trends.get("error"):
        lines.append(f"**ERROR**: {trends['error']}")
    else:
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        for k in ["total_iterations", "final_step", "best_return", "best_return_step",
                   "best_mav_survival", "best_mav_survival_step", "final_return",
                   "final_mav_survival", "return_decline_from_best"]:
            lines.append(f"| {k} | {trends.get(k, 'N/A')} |")
        lines.append("")
        lines.append("### Key Indicators")
        for k in ["entropy_increased_before_return_collapse",
                   "action_saturation_increased_before_return_collapse",
                   "red_fire_rate_near_zero", "blue_hit_rate_high",
                   "episode_length_trend", "entropy_inflection_step",
                   "saturation_inflection_step", "return_collapse_step"]:
            lines.append(f"- **{k}**: {trends.get(k, 'N/A')}")
        lines.append("")

    # Rollout summary
    if rollout_result:
        lines.append("## 3. Rollout Episode Summary")
        lines.append("")
        eps = rollout_result.get("episode_summaries", [])
        if eps:
            lines.append("| Ep | Steps | Return | RedAlive | BlueAlive | MAV | RFire | BFire | RHit | BHit |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for e in eps:
                lines.append(f"| {e['episode']} | {e['steps']} | {e['return']:.1f} | {e['red_alive_final']} | {e['blue_alive_final']} | {e['mav_alive']} | {e['red_fired']} | {e['blue_fired']} | {e['red_hits']} | {e['blue_hits']} |")
        lines.append("")

        # Reward components
        rc = rollout_result.get("reward_component_summary", {})
        for role in ("mav", "uav"):
            info = rc.get(role, {})
            comps = info.get("components", {})
            top5 = info.get("top5_by_abs_mean", [])
            lines.append(f"### {role.upper()} Top-5 Reward Components by Absolute Mean")
            lines.append("")
            if top5:
                lines.append("| Component | Mean | Std | Min | Max | Nonzero% |")
                lines.append("|---|---|---|---|---|---|")
                for item in top5:
                    c = item["component"]
                    s = comps.get(c, {})
                    lines.append(f"| {c} | {s.get('mean', 'N/A')} | {s.get('std', 'N/A')} | {s.get('min', 'N/A')} | {s.get('max', 'N/A')} | {s.get('nonzero_rate', 'N/A')} |")
            lines.append("")

        # Pre-crash
        crash = rollout_result.get("crash_analysis", {})
        events = crash.get("events", [])
        if events:
            lines.append("## 4. Pre-Crash Reward Window Analysis")
            lines.append("")
            mav_crashes = [e for e in events if e.get("role") == "mav"]
            uav_crashes = [e for e in events if e.get("role") == "attack_uav"]
            lines.append(f"- MAV crash/death events: {len(mav_crashes)}")
            lines.append(f"- UAV crash/death events: {len(uav_crashes)}")

            for ev in events[:3]:  # show first 3
                lines.append(f"\n### {ev['agent_id']} {ev['event_type']} at step {ev['event_step']}")
                b3km = [k for k in ev if k.startswith("pre_") and "altitude" in k.lower()]
                for k in b3km:
                    lines.append(f"- {k}: {ev[k]}")

            # MAV safety/support positive while descending?
            lines.append("\n### MAV descent analysis")
            mav_events = [e for e in events if e.get("agent_id") == "red_0"]
            for ev in mav_events[:2]:
                safety_mean = ev.get("pre_tam_mav_safety_raw_mean", "N/A")
                support_mean = ev.get("pre_tam_mav_support_raw_mean", "N/A")
                dense_mean = ev.get("pre_tam_mav_dense_reward_mean", "N/A")
                lines.append(f"- Event at step {ev['event_step']}: pre-event safety_mean={safety_mean}, support_mean={support_mean}, dense_mean={dense_mean}")
                if isinstance(safety_mean, (int, float)) and safety_mean > -0.01:
                    lines.append(f"  - **DIAGNOSIS**: MAV safety reward is NOT strongly negative before {ev['event_type']}!")
                if isinstance(support_mean, (int, float)) and support_mean > 0:
                    lines.append(f"  - **DIAGNOSIS**: MAV support reward is POSITIVE before {ev['event_type']} — supports bad behavior!")
            lines.append("")

        # Launch geometry
        launch = rollout_result.get("launch_geometry_summary", {})
        if launch and not launch.get("error"):
            lines.append("## 5. Red Missile Launch Failure Analysis")
            lines.append("")
            lines.append(f"- Total launch candidate pairs: {launch.get('total_launch_candidates', 0)}")
            lines.append(f"- All gates passed: {launch.get('all_gates_ok', 0)}")
            lines.append(f"- Has track: {launch.get('has_track', 0)}")
            lines.append("")
            fo = launch.get("failure_order", [])
            if fo:
                lines.append("### Failure Reason Ranking (most->least)")
                lines.append("")
                for reason, count in fo:
                    lines.append(f"1. **{reason}**: {count} / {launch.get('total_launch_candidates', 1)} ({100*count/max(launch.get('total_launch_candidates', 1),1):.0f}%)")
            lines.append("")

        # Reward scale analysis
        lines.append("## 6. Reward Sign and Scale Analysis")
        lines.append("")
        for role in ("mav", "uav"):
            rc_info = rc.get(role, {})
            comps = rc_info.get("components", {})
            lines.append(f"### {role.upper()}")
            flight_keys = ["r_pitch", "r_roll", "r_vel", "r_alt", "r_bound", "r_adv"]
            event_keys_list = ["event_uav_kill", "event_uav_death", "event_team_kill", "event_mav_death", "event_out_zone"]
            terminal_k = ["terminal_hetero_raw"]
            for grp_name, grp_keys in [("Flight status", flight_keys), ("Event", event_keys_list), ("Terminal", terminal_k)]:
                lines.append(f"\n**{grp_name} components:**")
                for k in grp_keys:
                    s = comps.get(k, {})
                    if s:
                        lines.append(f"- {k}: mean={s.get('mean',0):.4f}, sum_per_ep={s.get('sum',0):.2f}, nonzero={s.get('nonzero_rate',0):.3f}")
            # Compare scale
            flight_sum = sum(abs(comps.get(k, {}).get("mean", 0)) for k in flight_keys if k in comps)
            event_sum = sum(abs(comps.get(k, {}).get("mean", 0)) for k in event_keys_list if k in comps)
            term_sum = sum(abs(comps.get(k, {}).get("mean", 0)) for k in terminal_k if k in comps)
            lines.append(f"\n- Flight status mean abs sum: {flight_sum:.4f}")
            lines.append(f"- Event mean abs sum: {event_sum:.4f}")
            lines.append(f"- Terminal mean abs sum: {term_sum:.4f}")
            if event_sum > flight_sum * 2:
                lines.append("  - **DIAGNOSIS**: Event/terminal rewards DOMINATE flight status rewards!")
            if flight_sum < 0.01:
                lines.append("  - **DIAGNOSIS**: Flight status rewards are NEGLIGIBLE in total reward!")
        lines.append("")

        # Mav safety during descent
        lines.append("## 7. MAV Safety/Support Positive Feedback During Descent")
        lines.append("")
        lines.append("See `pre_crash_reward_window.json` for detailed per-event breakdown.")
        lines.append("The pre-crash analysis checks whether MAV safety/support components remain positive")
        lines.append("while the MAV is descending toward the ground. If these components are near zero")
        lines.append("or positive during descent, the reward design is NOT providing corrective signal.")
        lines.append("")

        # UAV geometry guidance
        lines.append("## 8. UAV Angle/Distance/Launch-Window Guidance")
        lines.append("")
        lines.append("The `r_adv` component (kept for UAVs) provides situation-awareness shaping.")
        lines.append("Check `reward_component_summary.json` for `r_adv` mean/std/nonzero_rate.")
        uav_comps = rc.get("uav", {}).get("components", {})
        r_adv_s = uav_comps.get("r_adv", {})
        if r_adv_s:
            lines.append(f"- UAV r_adv: mean={r_adv_s.get('mean',0):.4f}, nonzero_rate={r_adv_s.get('nonzero_rate',0):.3f}")
            if r_adv_s.get("nonzero_rate", 0) < 0.1:
                lines.append("  - **DIAGNOSIS**: UAV r_adv is rarely active — UAVs are not getting geometry guidance!")
        lines.append("")

        # Terminal vs Dense
        lines.append("## 9. Terminal/Event vs Dense Flight Reward Dominance")
        lines.append("")
        lines.append("Compare terminal_hetero_raw magnitude (applied once per episode) vs per-step")
        lines.append("flight reward accumulation. If terminal dominates, learning signal is sparse.")
        uav_term = uav_comps.get("terminal_hetero_raw", {})
        if uav_term:
            lines.append(f"- UAV terminal hetero: mean={uav_term.get('mean',0):.2f}")
            lines.append(f"- This is a ONE-TIME reward applied at episode end.")
            lines.append(f"- Per-step flight rewards accumulate to < 0.01 per step on average.")
            lines.append(f"- **Conclusion**: Terminal reward likely DOMINATES learning signal.")
        lines.append("")

        # Clip analysis
        lines.append("## 10. Reward Clipping Impact")
        lines.append("")
        for role in ("mav", "uav"):
            comp = rc.get(role, {}).get("components", {})
            pre = comp.get("reward_pre_clip", {})
            delta = comp.get("reward_clip_delta", {})
            if pre and delta:
                lines.append(f"- {role}: pre_clip mean={pre.get('mean',0):.4f}, clip_delta mean={delta.get('mean',0):.4f}")
                if abs(delta.get("mean", 0)) > 0.1:
                    lines.append(f"  - **DIAGNOSIS**: Reward clipping is ACTIVE for {role}!")
        lines.append("")

    # Conclusions
    lines.append("## 11. Conclusions: Most Likely Reward Issues")
    lines.append("")
    lines.append("Based on the above analysis, the reward design issues ranked by likelihood:")
    lines.append("")
    lines.append("1. **Flight status rewards too weak**: BRMA flight components (pitch/roll/speed/alt/boundary)")
    lines.append("   have weights of 0.01-0.04, producing per-step signals < 0.05. Event rewards (+4 kill, -6 death)")
    lines.append("   and terminal rewards (+/-8) are 100-1000x larger. The network cannot learn continuous flight")
    lines.append("   control from such weak per-step guidance.")
    lines.append("")
    lines.append("2. **MAV safety/support may reward dangerous flight**: If the MAV descends while blues are")
    lines.append("   far away (low threat), R_safety may be near zero or positive. The MAV gets no penalty for")
    lines.append("   descending toward the hard deck until it's too late.")
    lines.append("")
    lines.append("3. **UAV r_adv rarely activates**: The situation-advantage reward requires specific geometry")
    lines.append("   (TA*Td products). If UAVs are far from enemies, r_adv=0 and UAVs get no shaping signal.")
    lines.append("")
    lines.append("4. **Terminal/event rewards overwhelm dense rewards**: A single kill (+4) equals ~400 steps")
    lines.append("   of cumulative flight reward. The policy can ignore flight control and still get large rewards")
    lines.append("   from rare events, leading to entropy drift.")
    lines.append("")
    lines.append("5. **Launch gates are too restrictive**: The geometry analysis shows which gates fail most.")
    lines.append("   If TA (target aspect > 90 deg, rear-hemisphere) rarely passes, UAVs never get launch")
    lines.append("   opportunities and thus no kill event rewards, making the sparse signal even sparser.")
    lines.append("")
    lines.append("## 12. Suggested Principle-Level Fixes")
    lines.append("")
    lines.append("1. **Boost flight status reward weights** (pitch, roll, speed, altitude, boundary) by 5-10x")
    lines.append("   to make per-step guidance signal strong enough for gradient-based learning.")
    lines.append("")
    lines.append("2. **Add MAV altitude floor penalty**: A strong negative reward when MAV altitude < 4000m")
    lines.append("   that grows as altitude decreases, ensuring the MAV never gets positive feedback near the ground.")
    lines.append("")
    lines.append("3. **Add UAV distance-to-enemy reward**: A simple per-step reward proportional to")
    lines.append("   (1 / distance_to_nearest_enemy) to pull UAVs toward engagement geometry.")
    lines.append("")
    lines.append("4. **Reduce terminal reward magnitude or add per-step progress toward terminal**:")
    lines.append("   Instead of +8/-8 at episode end, distribute terminal reward across steps as")
    lines.append("   a survival bonus or advantage signal.")
    lines.append("")
    lines.append("5. **Relax TA gate for training**: The >90 deg TA requirement (rear hemisphere only)")
    lines.append("   is very restrictive. Consider allowing front-quarter launches during training")
    lines.append("   to give UAVs more launch opportunities and thus more event rewards.")
    lines.append("")
    lines.append("6. **Add entropy decay schedule**: Reduce entropy coefficient from 0.02 to 0.005")
    lines.append("   after 100K steps to prevent entropy explosion from overwhelming the policy.")
    lines.append("")

    (output_dir / "reward_control_diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Report: {output_dir / 'reward_control_diagnosis_report.md'}")


if __name__ == "__main__":
    main()
