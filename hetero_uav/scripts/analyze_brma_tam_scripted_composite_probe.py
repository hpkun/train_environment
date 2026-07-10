"""Analyze brma_tam_scripted_composite_v1 learnability probe output.

Reads a training output directory and produces:
  - probe_summary.json
  - probe_report.md
  - probe_window_metrics.csv
  - probe_component_scale.csv
  - probe_eval_comparison.csv

Compatible with 10K, 50K, 100K+ outputs.
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

# ── helpers ──────────────────────────────────────────────────────────
def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _mean_std(vals):
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _percentile(vals, p):
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


# ── window division ──────────────────────────────────────────────────
def _windows(total_steps: float, train_rows: list[dict]) -> list[tuple[str, float, float]]:
    if not train_rows:
        return []
    steps = [_safe(r.get("total_steps", 0)) for r in train_rows]
    min_s, max_s = min(steps), max(steps)
    if max_s - min_s < 1:
        return [("all", min_s, max_s)]
    span = max_s - min_s
    return [
        ("early", min_s, min_s + span * 0.2),
        ("middle", min_s + span * 0.4, min_s + span * 0.6),
        ("late", max_s - span * 0.2, max_s),
    ]


def _in_window(row, w_start, w_end):
    s = _safe(row.get("total_steps", 0))
    return w_start <= s <= w_end


# ── main ─────────────────────────────────────────────────────────────
def analyze(output_dir: str) -> dict:
    root = Path(output_dir)
    train_path = root / "train_log.csv"
    ep_comp_path = root / "rich_logs" / "episode_reward_components.csv"
    step_comp_path = root / "rich_logs" / "reward_components.csv"
    missile_path = root / "rich_logs" / "missile_events.csv"
    eval_ep_path = root / "rich_logs" / "eval_episode_metrics.csv"
    eval_summ_path = root / "rich_logs" / "eval_summary_metrics.csv"

    train_rows = list(csv.DictReader(train_path.open(encoding="utf-8"))) if train_path.exists() else []
    ep_rows = list(csv.DictReader(ep_comp_path.open(encoding="utf-8"))) if ep_comp_path.exists() else []
    step_rows = list(csv.DictReader(step_comp_path.open(encoding="utf-8"))) if step_comp_path.exists() else []
    missile_rows = list(csv.DictReader(missile_path.open(encoding="utf-8"))) if missile_path.exists() else []
    eval_ep_rows = list(csv.DictReader(eval_ep_path.open(encoding="utf-8"))) if eval_ep_path.exists() else []
    eval_summ_rows = list(csv.DictReader(eval_summ_path.open(encoding="utf-8"))) if eval_summ_path.exists() else []

    if not train_rows:
        return {"error": "no train_log.csv found"}

    total_steps = max(_safe(r.get("total_steps", 0)) for r in train_rows)
    windows = _windows(total_steps, train_rows)

    # ── per-window metrics ──────────────────────────────────────────
    window_metrics = []
    for wname, wstart, wend in windows:
        w_train = [r for r in train_rows if _in_window(r, wstart, wend)]
        w_ep = [r for r in ep_rows if _safe(r.get("episode_id", -1)) >= 0]
        w_step = [r for r in step_rows
                  if _safe(r.get("episode_id", -1)) >= 0 and wstart <= _safe(r.get("step", 0)) <= wend]
        # Filter episodes that ended in this window
        ep_ids_in_window = set()
        for sr in w_step:
            ep_ids_in_window.add(int(_safe(sr.get("episode_id", -1))))
        w_ep_filtered = [r for r in w_ep if int(_safe(r.get("episode_id", -1))) in ep_ids_in_window]

        m = {"window": wname}
        # training
        if w_train:
            m["avg_return"] = float(np.mean([_safe(r.get("avg_return", 0)) for r in w_train]))
            m["red_win"] = float(np.mean([_safe(r.get("red_win", 0)) for r in w_train]))
            m["blue_win"] = float(np.mean([_safe(r.get("blue_win", 0)) for r in w_train]))
            m["draw"] = float(np.mean([_safe(r.get("draw", 0)) for r in w_train]))
            m["timeout"] = float(np.mean([_safe(r.get("timeout", 0)) for r in w_train]))
            m["mav_survival"] = float(np.mean([_safe(r.get("mav_survival", 0)) for r in w_train]))
            m["red_alive_final"] = float(np.mean([_safe(r.get("red_alive_final", 0)) for r in w_train]))
            m["blue_alive_final"] = float(np.mean([_safe(r.get("blue_alive_final", 0)) for r in w_train]))
            m["red_missiles_fired"] = float(np.sum([_safe(r.get("red_missiles_fired", 0)) for r in w_train]))
            m["red_missile_hits"] = float(np.sum([_safe(r.get("red_missile_hits", 0)) for r in w_train]))
            m["blue_missiles_fired"] = float(np.sum([_safe(r.get("blue_missiles_fired", 0)) for r in w_train]))
            m["blue_missile_hits"] = float(np.sum([_safe(r.get("blue_missile_hits", 0)) for r in w_train]))
            m["actor_loss_mav"] = float(np.mean([_safe(r.get("actor_loss_mav", 0)) for r in w_train]))
            m["actor_loss_uav"] = float(np.mean([_safe(r.get("actor_loss_uav", 0)) for r in w_train]))
            m["critic_loss"] = float(np.mean([_safe(r.get("critic_loss", 0)) for r in w_train]))
            m["entropy_mav"] = float(np.mean([_safe(r.get("entropy_mav", 0)) for r in w_train]))
            m["entropy_uav"] = float(np.mean([_safe(r.get("entropy_uav", 0)) for r in w_train]))
            m["approx_kl_mav"] = float(np.mean([_safe(r.get("approx_kl_mav", 0)) for r in w_train]))
            m["approx_kl_uav"] = float(np.mean([_safe(r.get("approx_kl_uav", 0)) for r in w_train]))
            m["value_explained_variance"] = float(np.mean([_safe(r.get("value_explained_variance", 0)) for r in w_train]))
            m["return_mean"] = float(np.mean([_safe(r.get("return_mean", 0)) for r in w_train]))
            m["return_std"] = float(np.mean([_safe(r.get("return_std", 0)) for r in w_train]))
            m["mav_action_saturation"] = float(np.mean([_safe(r.get("mav_action_saturation_rate", 0)) for r in w_train]))
            m["uav_action_saturation"] = float(np.mean([_safe(r.get("uav_action_saturation_rate", 0)) for r in w_train]))
            m["approx_kl_abs_mav"] = float(np.mean([_safe(r.get("approx_kl_abs_mav", 0)) for r in w_train]))
            m["approx_kl_abs_uav"] = float(np.mean([_safe(r.get("approx_kl_abs_uav", 0)) for r in w_train]))
            m["clip_fraction_mav"] = float(np.mean([_safe(r.get("clip_fraction_mav", 0)) for r in w_train]))
            m["clip_fraction_uav"] = float(np.mean([_safe(r.get("clip_fraction_uav", 0)) for r in w_train]))
            m["critic_grad_norm"] = float(np.mean([_safe(r.get("critic_grad_norm", 0)) for r in w_train]))

        # episode-level
        if w_ep_filtered:
            rets = [_safe(r.get("episode_return", 0)) for r in w_ep_filtered]
            lengths = [_safe(r.get("episode_length", 0)) for r in w_ep_filtered]
            m["episode_count"] = len(w_ep_filtered)
            m["episode_return_mean"] = float(np.mean(rets))
            m["episode_return_std"] = float(np.std(rets))
            m["episode_length_mean"] = float(np.mean(lengths))
            # red episodes only
            red_eps = [r for r in w_ep_filtered if str(r.get("team", "")).lower() == "red"]
            if red_eps:
                m["red_episode_return_mean"] = float(np.mean([_safe(r.get("episode_return", 0)) for r in red_eps]))
        else:
            m["episode_count"] = 0

        # reward components from per-step data
        if w_step:
            comp_keys = [
                "brma_pitch", "brma_roll", "brma_vel",
                "tam_speed_weighted", "tam_angle_weighted", "tam_distance_weighted",
                "uav_event_total",
                "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
                "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
                "total",
            ]
            for ck in comp_keys:
                vals = [_safe(r.get(ck, 0)) for r in w_step]
                m[f"{ck}_per_step"] = float(np.mean(vals)) if vals else 0.0
                m[f"{ck}_sum"] = float(np.sum(vals))

            # UAV components for role=attack_uav
            uav_step = [r for r in w_step if str(r.get("role", "")) == "attack_uav"]
            mav_step = [r for r in w_step if str(r.get("role", "")) == "mav"]
            if uav_step:
                m["uav_dense_per_step"] = float(np.mean([
                    _safe(r.get("tam_speed_weighted", 0)) + _safe(r.get("tam_angle_weighted", 0))
                    + _safe(r.get("tam_distance_weighted", 0)) + _safe(r.get("brma_pitch", 0))
                    + _safe(r.get("brma_roll", 0)) + _safe(r.get("brma_vel", 0))
                    for r in uav_step
                ]))
                m["uav_event_per_step"] = float(np.mean([_safe(r.get("uav_event_total", 0)) for r in uav_step]))
                m["uav_distance_neg_ratio"] = float(
                    sum(1 for r in uav_step if _safe(r.get("tam_distance_weighted", 0)) < 0)
                    / max(len(uav_step), 1)
                )
            if mav_step:
                m["mav_dense_per_step"] = float(np.mean([
                    _safe(r.get("mav_dist_weighted", 0)) + _safe(r.get("mav_threat_weighted", 0))
                    + _safe(r.get("mav_aspect_weighted", 0)) + _safe(r.get("mav_pos_weighted", 0))
                    + _safe(r.get("mav_aware_weighted", 0)) + _safe(r.get("brma_pitch", 0))
                    + _safe(r.get("brma_roll", 0)) + _safe(r.get("brma_vel", 0))
                    for r in mav_step
                ]))
                m["mav_event_per_step"] = float(np.mean([_safe(r.get("mav_event_total", 0)) for r in mav_step]))

            # component scale comparison
            abs_sum = 0.0
            abs_keys = [
                "brma_pitch", "brma_roll", "brma_vel",
                "tam_speed_weighted", "tam_angle_weighted", "tam_distance_weighted",
                "uav_event_total",
                "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
                "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
            ]
            for ck in abs_keys:
                abs_sum += float(np.sum([abs(_safe(r.get(ck, 0))) for r in w_step]))
            if abs_sum > 1e-9:
                for ck in abs_keys:
                    m[f"{ck}_frac_of_abs"] = float(np.sum([abs(_safe(r.get(ck, 0))) for r in w_step])) / abs_sum

            # diagnostic fields
            diag_keys = [
                "reward_target_observed", "reward_target_direct_visible",
                "reward_target_mav_shared_visible", "reward_target_matches_lock",
                "reward_target_matches_launch", "evasion_override_agent_steps",
                "evasion_override_env_steps", "above_altitude_max_agent_steps",
                "above_altitude_max_env_steps",
                "mav_support_after_all_attack_uav_dead",
                "mav_safety_after_all_attack_uav_dead",
                "mav_flight_after_all_attack_uav_dead",
                "mav_event_after_all_attack_uav_dead",
                "mav_total_after_all_attack_uav_dead",
            ]
            for dk in diag_keys:
                vals = [_safe(r.get(dk, 0)) for r in w_step]
                m[f"{dk}_mean"] = float(np.mean(vals)) if vals else 0.0
                m[f"{dk}_sum"] = float(np.sum(vals))

            # MAV after all attack UAV dead total
            mav_after_dead = [_safe(r.get("mav_total_after_all_attack_uav_dead", 0)) for r in w_step]
            m["mav_total_after_all_attack_uav_dead_sum"] = float(np.sum(mav_after_dead))

        # kill episode vs no-kill timeout comparison
        kill_eps = [r for r in w_ep_filtered if _safe(r.get("red_hit_count", 0)) > 0]
        no_kill_timeout_eps = [r for r in w_ep_filtered
                               if _safe(r.get("red_hit_count", 0)) == 0
                               and str(r.get("end_reason", "")) == "timeout"]
        if kill_eps:
            m["kill_episode_return_mean"] = float(np.mean([_safe(r.get("episode_return", 0)) for r in kill_eps]))
            m["kill_episode_count"] = len(kill_eps)
        if no_kill_timeout_eps:
            m["no_kill_timeout_return_mean"] = float(np.mean([_safe(r.get("episode_return", 0)) for r in no_kill_timeout_eps]))
            m["no_kill_timeout_count"] = len(no_kill_timeout_eps)
        red_loss_eps = [r for r in w_ep_filtered if _safe(r.get("red_alive_final", 0)) == 0]
        if red_loss_eps:
            m["red_loss_return_mean"] = float(np.mean([_safe(r.get("episode_return", 0)) for r in red_loss_eps]))

        # first red launch / hit step
        m["first_red_launch_step"] = 0.0
        m["first_red_hit_step"] = 0.0
        red_launches = [r for r in missile_rows if "red_" in str(r.get("shooter_id", ""))
                        and r.get("launch_step")]
        red_hits = [r for r in missile_rows if "red_" in str(r.get("shooter_id", ""))
                    and str(r.get("termination_reason", "")) == "hit"]
        if red_launches:
            m["first_red_launch_step"] = float(min(_safe(r.get("launch_step", float("inf"))) for r in red_launches))
        if red_hits:
            m["first_red_hit_step"] = float(min(_safe(r.get("launch_step", float("inf"))) for r in red_hits))

        # launch blocking diagnostics
        diag_records = []
        for sr in w_step:
            diag = {}
            for k in sr:
                if k.startswith("diag_"):
                    diag[k[5:]] = _safe(sr[k])
            if diag:
                diag_records.append(diag)
        if diag_records:
            m["track_unobserved_blocked"] = float(np.sum([r.get("track_unobserved_blocked", 0) for r in diag_records]))
            m["range_ok_pairs"] = float(np.sum([r.get("range_ok_pairs", 0) for r in diag_records]))
            m["ao_ok_pairs"] = float(np.sum([r.get("ao_ok_pairs", 0) for r in diag_records]))
            m["ta_ok_pairs"] = float(np.sum([r.get("ta_ok_pairs", 0) for r in diag_records]))
            m["geometry_ok_pairs"] = float(np.sum([r.get("geometry_ok_pairs", 0) for r in diag_records]))
            m["lock_mature_pairs"] = float(np.sum([r.get("lock_mature_pairs", 0) for r in diag_records]))
            m["launches"] = float(np.sum([r.get("launches", 0) for r in diag_records]))
            m["cooldown_blocked"] = float(np.sum([r.get("cooldown_blocked", 0) for r in diag_records]))
            m["ammo_empty_blocked"] = float(np.sum([r.get("ammo_empty_blocked", 0) for r in diag_records]))

        # NaN/Inf check
        nan_count = 0
        for tr in w_train:
            for k, v in tr.items():
                try:
                    f = float(v)
                    if not math.isfinite(f):
                        nan_count += 1
                except (ValueError, TypeError):
                    pass
        m["nan_inf_field_count"] = float(nan_count)

        window_metrics.append(m)

    # ── eval comparison ──────────────────────────────────────────────
    eval_comparison = []
    if eval_summ_rows:
        for row in eval_summ_rows:
            eval_comparison.append({
                "checkpoint_step": _safe(row.get("checkpoint_step", 0)),
                "config": str(row.get("config", "")),
                "red_win": _safe(row.get("red_win", 0)),
                "blue_win": _safe(row.get("blue_win", 0)),
                "draw": _safe(row.get("draw", 0)),
                "timeout": _safe(row.get("timeout", 0)),
                "red_elimination_win": _safe(row.get("red_elimination_win", 0)),
                "red_timeout_alive_advantage": _safe(row.get("red_timeout_alive_advantage", 0)),
                "red_kill_fraction": _safe(row.get("red_kill_fraction", 0)),
                "net_kill_fraction": _safe(row.get("net_kill_fraction", 0)),
                "mav_survival": _safe(row.get("mav_survival", 0)),
                "red_alive_mean": _safe(row.get("red_alive_final", _safe(row.get("red_alive_mean", 0)))),
                "blue_alive_mean": _safe(row.get("blue_alive_final", _safe(row.get("blue_alive_mean", 0)))),
                "red_launch_mean": _safe(row.get("red_missiles_fired", 0)),
                "red_hit_mean": _safe(row.get("red_missile_hits", 0)),
                "blue_launch_mean": _safe(row.get("blue_missiles_fired", 0)),
                "blue_hit_mean": _safe(row.get("blue_missile_hits", 0)),
                "episode_length": _safe(row.get("episode_length", 0)),
                "avg_return": _safe(row.get("avg_return", 0)),
            })
    elif eval_ep_rows:
        # Aggregate from per-episode eval metrics
        checkpoints = sorted(set(_safe(r.get("checkpoint_step", 0)) for r in eval_ep_rows))
        for ckpt in checkpoints:
            ck_eps = [r for r in eval_ep_rows if _safe(r.get("checkpoint_step", 0)) == ckpt]
            configs = sorted(set(str(r.get("scenario", "")) for r in ck_eps))
            for cfg in configs:
                cfg_eps = [r for r in ck_eps if str(r.get("scenario", "")) == cfg]
                if not cfg_eps:
                    continue
                eval_comparison.append({
                    "checkpoint_step": ckpt,
                    "config": cfg,
                    "red_win": float(np.mean([_safe(r.get("red_win", 0)) for r in cfg_eps])),
                    "blue_win": float(np.mean([_safe(r.get("blue_win", 0)) for r in cfg_eps])),
                    "draw": float(np.mean([_safe(r.get("draw", 0)) for r in cfg_eps])),
                    "timeout": float(np.mean([_safe(r.get("timeout", 0)) for r in cfg_eps])),
                    "mav_survival": float(np.mean([_safe(r.get("mav_survival", 0)) for r in cfg_eps])),
                    "red_alive_mean": float(np.mean([_safe(r.get("red_alive_final", 0)) for r in cfg_eps])),
                    "blue_alive_mean": float(np.mean([_safe(r.get("blue_alive_final", 0)) for r in cfg_eps])),
                    "red_launch_mean": float(np.mean([_safe(r.get("red_launch_count", 0)) for r in cfg_eps])),
                    "red_hit_mean": float(np.mean([_safe(r.get("red_hit_count", 0)) for r in cfg_eps])),
                    "blue_launch_mean": float(np.mean([_safe(r.get("blue_launch_count", 0)) for r in cfg_eps])),
                    "blue_hit_mean": float(np.mean([_safe(r.get("blue_hit_count", 0)) for r in cfg_eps])),
                    "episode_length": float(np.mean([_safe(r.get("episode_length", 0)) for r in cfg_eps])),
                    "avg_return": float(np.mean([_safe(r.get("episode_return", 0)) for r in cfg_eps])),
                })

    # ── component scale comparison ───────────────────────────────────
    component_scale = []
    all_step_rows = step_rows
    if all_step_rows:
        abs_sum = 0.0
        abs_keys = [
            "brma_pitch", "brma_roll", "brma_vel",
            "tam_speed_weighted", "tam_angle_weighted", "tam_distance_weighted",
            "uav_event_total",
            "mav_dist_weighted", "mav_threat_weighted", "mav_aspect_weighted",
            "mav_pos_weighted", "mav_aware_weighted", "mav_event_total",
        ]
        for ck in abs_keys:
            abs_sum += float(np.sum([abs(_safe(r.get(ck, 0))) for r in all_step_rows]))
        for ck in abs_keys:
            comp_sum = float(np.sum([abs(_safe(r.get(ck, 0))) for r in all_step_rows]))
            component_scale.append({
                "component": ck,
                "abs_sum": comp_sum,
                "frac_of_total_abs": comp_sum / max(abs_sum, 1e-9),
                "per_step_mean": float(np.mean([_safe(r.get(ck, 0)) for r in all_step_rows])),
            })

    # ── learnability answers ─────────────────────────────────────────
    red_launch_steps = []
    red_hit_steps = []
    for r in missile_rows:
        if "red_" in str(r.get("shooter_id", "")) and r.get("launch_step"):
            red_launch_steps.append(_safe(r.get("launch_step", 0)))
        if "red_" in str(r.get("shooter_id", "")) and str(r.get("termination_reason", "")) == "hit":
            red_hit_steps.append(_safe(r.get("launch_step", 0)))

    learnability = {
        "any_red_launch": len(red_launch_steps) > 0,
        "first_red_launch_step": float(min(red_launch_steps)) if red_launch_steps else 0.0,
        "total_red_launches": len(red_launch_steps),
        "any_red_hit": len(red_hit_steps) > 0,
        "total_red_hits": len(red_hit_steps),
        "first_red_hit_step": float(min(red_hit_steps)) if red_hit_steps else 0.0,
        "any_blue_death": any(
            "blue_" in str(r.get("target_id", "")) and str(r.get("termination_reason", "")) == "hit"
            for r in missile_rows
        ),
    }

    # kill event trainer contribution estimate
    uav_step_rows = [r for r in all_step_rows if str(r.get("role", "")) == "attack_uav"]
    kill_steps = [r for r in uav_step_rows if _safe(r.get("uav_event_kill", 0)) > 0]
    learnability["kill_event_mean_trainer_contribution"] = (
        float(np.mean([_safe(r.get("uav_event_total", 0)) for r in kill_steps]))
        if kill_steps else 0.0
    )

    # UAV distance negative vs kill positive ratio
    uav_dist_neg = float(np.sum([abs(_safe(r.get("tam_distance_weighted", 0)))
                                  for r in uav_step_rows if _safe(r.get("tam_distance_weighted", 0)) < 0]))
    uav_event_pos = float(np.sum([_safe(r.get("uav_event_kill", 0)) for r in uav_step_rows
                                   if _safe(r.get("uav_event_kill", 0)) > 0]))
    learnability["uav_distance_neg_vs_kill_pos_ratio"] = (
        uav_dist_neg / max(uav_event_pos, 1e-9)
    )

    # late vs early comparison
    early = next((m for m in window_metrics if m.get("window") == "early"), None)
    late = next((m for m in window_metrics if m.get("window") == "late"), None)
    if early and late:
        for key in ("avg_return", "red_win", "uav_distance_weighted_per_step",
                     "uav_event_total_per_step", "mav_total_after_all_attack_uav_dead_sum"):
            ev = early.get(key, 0.0)
            lv = late.get(key, 0.0)
            learnability[f"{key}_early"] = ev
            learnability[f"{key}_late"] = lv
            learnability[f"{key}_delta"] = lv - ev

    # ── assemble output ─────────────────────────────────────────────
    result = {
        "output_dir": str(root),
        "total_env_steps": total_steps,
        "train_iterations": len(train_rows),
        "episodes_logged": len(ep_rows),
        "missile_events": len(missile_rows),
        "windows": window_metrics,
        "eval_comparison": eval_comparison,
        "component_scale": component_scale,
        "learnability": learnability,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Training output directory to analyze")
    parser.add_argument("--report-dir", default=None,
                        help="Directory to write reports (defaults to output-dir)")
    args = parser.parse_args()
    root = Path(args.output_dir)
    if not root.exists():
        print(f"ERROR: output directory not found: {args.output_dir}")
        sys.exit(1)
    report_dir = Path(args.report_dir) if args.report_dir else root

    result = analyze(str(root))

    # Write JSON
    (report_dir / "probe_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Write window metrics CSV
    if result["windows"]:
        wm_path = report_dir / "probe_window_metrics.csv"
        fields = list(result["windows"][0].keys())
        with wm_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["windows"])

    # Write component scale CSV
    if result["component_scale"]:
        cs_path = report_dir / "probe_component_scale.csv"
        fields = list(result["component_scale"][0].keys())
        with cs_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["component_scale"])

    # Write eval comparison CSV
    if result["eval_comparison"]:
        ec_path = report_dir / "probe_eval_comparison.csv"
        fields = list(result["eval_comparison"][0].keys())
        with ec_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["eval_comparison"])

    # Write markdown report
    L = result["learnability"]
    W = result["windows"]
    lines = [
        "# BRMA/TAM Scripted Composite v1 — Learnability Probe Report",
        "",
        f"- Output: `{result['output_dir']}`",
        f"- Total env steps: {result['total_env_steps']}",
        f"- Train iterations: {result['train_iterations']}",
        f"- Episodes logged: {result['episodes_logged']}",
        f"- Missile events: {result['missile_events']}",
        "",
        "## Learnability Questions",
        "",
        f"1. Red launch occurred: **{L['any_red_launch']}**",
        f"2. First red launch step: **{L['first_red_launch_step']}**",
        f"3. Red hits / blue deaths: **{L['any_red_hit']}** (total hits: {L['total_red_hits']})",
        f"4. First red hit step: **{L['first_red_hit_step']}**",
    ]
    if not L["any_red_launch"]:
        lines.append("")
        lines.append("### Launch blocking analysis")
        lines.append("| Blocker | Count |")
        lines.append("|---------|-------|")
        for wk in ["track_unobserved_blocked", "range_ok_pairs", "ao_ok_pairs",
                    "ta_ok_pairs", "geometry_ok_pairs", "lock_mature_pairs",
                    "cooldown_blocked", "ammo_empty_blocked"]:
            if W:
                late = W[-1]
                v = late.get(wk, 0)
                lines.append(f"| {wk} | {v:.0f} |")

    lines += [
        "",
        f"6. Kill event mean trainer contribution: **{L['kill_event_mean_trainer_contribution']:.2f}**",
        f"7. UAV distance neg vs kill pos ratio: **{L['uav_distance_neg_vs_kill_pos_ratio']:.2f}**",
        "",
        "## Window Comparison",
        "",
        "| Metric | Early | Late | Delta |",
        "|--------|-------|------|-------|",
    ]
    for key in ["avg_return", "red_win", "uav_distance_weighted_per_step",
                 "uav_event_total_per_step", "mav_total_after_all_attack_uav_dead_sum"]:
        ev = L.get(f"{key}_early", 0)
        lv = L.get(f"{key}_late", 0)
        dv = L.get(f"{key}_delta", 0)
        lines.append(f"| {key} | {ev:.4f} | {lv:.4f} | {dv:+.4f} |")

    lines += [
        "",
        "## Evaluation Comparison (3V2 train / 5V4 zero-shot)",
        "",
    ]
    for ec in result["eval_comparison"]:
        lines.append(
            f"- step={ec['checkpoint_step']} cfg={ec['config']}: "
            f"R={ec['red_win']:.2f} B={ec['blue_win']:.2f} D={ec['draw']:.2f} "
            f"T={ec['timeout']:.2f} MAV={ec['mav_survival']:.2f} "
            f"RL={ec['red_launch_mean']:.1f} RH={ec['red_hit_mean']:.1f}"
        )

    # determine verdict
    verdict = "NEEDS_TRAINING_STABILITY_REVIEW"
    nan_ok = all(w.get("nan_inf_field_count", 999) == 0 for w in W)
    _early = next((m for m in W if m.get("window") == "early"), None)
    _late = next((m for m in W if m.get("window") == "late"), None)
    if not nan_ok:
        verdict = "IMPLEMENTATION_ERROR"
    elif L["any_red_launch"]:
        verdict = "READY_FOR_50K"
    elif not L["any_red_launch"]:
        # Check if there was observability improvement
        if _late and _early:
            if _late.get("reward_target_observed_mean", 0) > _early.get("reward_target_observed_mean", 0):
                verdict = "NEEDS_REWARD_SCALE_REVIEW"
            else:
                verdict = "NEEDS_FIRE_CONTROL_ALIGNMENT_REVIEW"

    lines += [
        "",
        f"## Verdict: **{verdict}**",
        "",
        "---",
        f"*Generated from {result['output_dir']}*",
    ]

    (report_dir / "probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "learnability": L, "outputs": [
        str(report_dir / "probe_summary.json"),
        str(report_dir / "probe_report.md"),
        str(report_dir / "probe_window_metrics.csv"),
        str(report_dir / "probe_component_scale.csv"),
        str(report_dir / "probe_eval_comparison.csv"),
    ]}, indent=2, default=str))


if __name__ == "__main__":
    main()
