"""Audit role_v1 reward effects from existing 50k runs.

This script is diagnostic only. It reads completed experiment logs, inspects the
role_v1 reward overlay, and samples a short rollout to estimate component
trigger rates. It does not train or modify reward code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


ROLE_COMPONENT_KEYS = [
    "r_role_mav_survival",
    "r_role_mav_death",
    "r_role_mav_support",
    "r_role_mav_team_contribution",
    "r_role_uav_attack_window",
    "r_role_uav_kill_bonus",
    "r_role_uav_death_penalty",
    "r_role_uav_missile_warning",
]

CURVE_FIELDS = [
    "average_team_return",
    "average_episode_length",
    "average_red_alive",
    "average_blue_alive",
    "train_red_win_rate_recent",
    "train_blue_win_rate_recent",
    "train_draw_rate_recent",
    "train_timeout_rate_recent",
    "train_mav_survival_rate_recent",
    "train_red_missiles_fired_recent",
    "train_blue_missiles_fired_recent",
    "train_missile_hit_rate_recent",
    "action_saturation_rate",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _nearest_row(rows: list[dict[str, str]], target_steps: int) -> dict[str, Any]:
    if not rows:
        return {}
    row = min(rows, key=lambda r: abs(int(float(r.get("total_steps", 0))) - target_steps))
    out: dict[str, Any] = {
        "iteration": int(float(row.get("iteration", 0))),
        "total_steps": int(float(row.get("total_steps", 0))),
    }
    for field in CURVE_FIELDS:
        out[field] = _to_float(row.get(field))
    return out


def _curve_snapshot(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    targets = [10_000, 20_000, 30_000, 40_000, 50_000]
    return {f"{target // 1000}k": _nearest_row(rows, target) for target in targets}


def _last_row_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    row = rows[-1]
    return {field: _to_float(row.get(field)) for field in CURVE_FIELDS}


def _eval_by_config(summary: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(summary, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in summary:
        if not isinstance(item, dict):
            continue
        cfg = str(item.get("eval_config") or item.get("config") or "")
        key = "5v4" if "5v4" in cfg else "3v2" if "3v2" in cfg else cfg
        out[key] = item
    return out


def _compare(role_dir: Path, legacy_dir: Path) -> dict[str, Any]:
    role_rows = _read_csv(role_dir / "train_log.csv")
    legacy_rows = _read_csv(legacy_dir / "train_log.csv")
    role_summary = _eval_by_config(_read_json(role_dir / "main_experiment_summary.json"))
    legacy_summary = _eval_by_config(_read_json(legacy_dir / "main_experiment_summary.json"))

    comparison = {
        "curve_snapshots": {
            "role_v1": _curve_snapshot(role_rows),
            "brma_legacy": _curve_snapshot(legacy_rows),
        },
        "final_train_metrics": {
            "role_v1": _last_row_metrics(role_rows),
            "brma_legacy": _last_row_metrics(legacy_rows),
        },
        "final_eval_summary": {
            "role_v1": role_summary,
            "brma_legacy": legacy_summary,
        },
    }

    role_3 = role_summary.get("3v2", {})
    legacy_3 = legacy_summary.get("3v2", {})
    role_5 = role_summary.get("5v4", {})
    legacy_5 = legacy_summary.get("5v4", {})
    role_weaker = (
        _to_float(role_3.get("red_win_rate")) < _to_float(legacy_3.get("red_win_rate"))
        and _to_float(role_5.get("red_win_rate")) < _to_float(legacy_5.get("red_win_rate"))
    )
    mav_improved = (
        _to_float(role_3.get("mav_survival_rate")) > _to_float(legacy_3.get("mav_survival_rate"))
        or _to_float(role_5.get("mav_survival_rate")) > _to_float(legacy_5.get("mav_survival_rate"))
    )
    comparison["judgement"] = {
        "role_v1_clearly_weaker_than_legacy": bool(role_weaker),
        "role_v1_improves_mav_survival": bool(mav_improved),
        "role_v1_may_reduce_uav_attack_effect": bool(
            _to_float(role_3.get("blue_alive_final_mean")) > _to_float(legacy_3.get("blue_alive_final_mean"))
            or _to_float(role_5.get("blue_alive_final_mean")) > _to_float(legacy_5.get("blue_alive_final_mean"))
        ),
    }
    return comparison


def _code_audit() -> dict[str, Any]:
    env_file = ROOT / "uav_env" / "JSBSim" / "envs" / "hetero_uav_combat_env.py"
    text = env_file.read_text(encoding="utf-8")
    role_block = text[text.find("# ---- role_v1 overlay ----") :]
    online_eval_issue = False
    eval_log = ROOT / "outputs" / "main_mappo_experiment_f22_50k_role_v1" / "eval_log.csv"
    if eval_log.exists():
        rows = _read_csv(eval_log)
        online_eval_issue = any("role_v1" not in str(r.get("eval_config", "")) for r in rows)

    support_uses_alive_mask = bool(
        re.search(r"r_role_mav_support[\s\S]*?enemy_alive_mask", role_block)
        or re.search(r"alive_mask\s*=\s*np\.asarray\(o\.get\(\"enemy_alive_mask\"", role_block)
    )
    return {
        "brma_legacy_unchanged_path": "role_v1 returns base rewards unchanged unless hetero_reward_mode is minimal_v1/role_v1",
        "role_v1_overlay_on_brma_legacy": "base_rewards, components = super()._compute_rewards()" in text,
        "mav_support_uses_enemy_alive_mask": support_uses_alive_mask,
        "mav_support_uses_observed_enemy_mask": False,
        "last_step_obs_one_step_lag_risk": "_last_step_obs" in role_block,
        "uav_death_penalty_first_death": "_uav_death_penalized" in role_block,
        "mav_death_penalty_first_death": "_mav_death_penalized" in role_block,
        "kill_bonus_depends_on_step_kill_count": "_step_kill_count" in role_block,
        "online_eval_used_non_role_configs": online_eval_issue,
    }


def _sample_components(config: str, steps: int, seed: int) -> dict[str, Any]:
    env = make_env(config, env_type="jsbsim_hetero", max_steps=max(steps, 5))
    try:
        obs, info = env.reset(seed=seed)
        sums = defaultdict(float)
        nonzero = defaultdict(int)
        counts = defaultdict(int)
        role_sums = defaultdict(float)
        nan_detected = False
        steps_executed = 0
        for _ in range(steps):
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps_executed += 1
            for aid in env.agent_ids:
                agent_info = info.get(aid, {})
                role = env.agent_roles.get(aid, "")
                for key in ROLE_COMPONENT_KEYS:
                    value = _to_float(agent_info.get(key))
                    sums[key] += value
                    counts[key] += 1
                    if abs(value) > 1e-9:
                        nonzero[key] += 1
                    if key in agent_info:
                        role_sums[f"{role}:{key}"] += value
                    if not math.isfinite(value):
                        nan_detected = True
            if all(terminated.values()) or all(truncated.values()):
                break
        component_stats = {}
        for key in ROLE_COMPONENT_KEYS:
            denom = max(counts[key], 1)
            component_stats[key] = {
                "sum": float(sums[key]),
                "mean": float(sums[key] / denom),
                "nonzero_count": int(nonzero[key]),
                "trigger_rate": float(nonzero[key] / denom),
            }
        return {
            "config": config,
            "steps_requested": steps,
            "steps_executed": steps_executed,
            "nan_detected": nan_detected,
            "component_stats": component_stats,
            "role_component_sums": dict(role_sums),
        }
    finally:
        env.close()


def _paper_alignment() -> dict[str, Any]:
    return {
        "brma_mappo": {
            "alignment": "brma_legacy keeps BRMA-style flight stability, posture/engagement shaping, and terminal combat outcome reward.",
            "gap": "It does not explicitly separate MAV support from UAV attack roles.",
        },
        "heterogeneous_mav_uav": {
            "alignment": "role_v1 attempts to add MAV survival/support/team contribution and UAV attack-window/kill/death signals.",
            "gaps": [
                "MAV support currently uses enemy_alive_mask, which is not equivalent to observed enemy support.",
                "No height/speed/dodge role reward is audited as effective.",
                "No complex missile-dodge reward is included, consistent with missing full missile geometry observation.",
                "The implementation is an ablation, not a full reproduction of the heterogeneous paper reward.",
            ],
        },
    }


def _recommended_changes(code_audit: dict[str, Any], comparison: dict[str, Any], component_audit: dict[str, Any]) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    if code_audit.get("mav_support_uses_enemy_alive_mask"):
        recs.append({
            "target": "r_role_mav_support",
            "reason": "Current support can reward merely alive enemies rather than enemies observed or shared by MAV.",
            "change": "Use enemy_observed/shared-track evidence instead of enemy_alive_mask.",
        })
    if code_audit.get("online_eval_used_non_role_configs"):
        recs.append({
            "target": "online eval config selection",
            "reason": "role_v1 online eval_log uses non-role configs, so the online curve is not a clean role_v1 evaluation.",
            "change": "Pass role_v1 eval configs during role_v1 training/eval-during-training.",
        })
    if comparison.get("judgement", {}).get("role_v1_clearly_weaker_than_legacy"):
        recs.append({
            "target": "role_v1 overlay scale/sign",
            "reason": "50k role_v1 is much weaker than brma_legacy and does not improve MAV survival.",
            "change": "Audit per-component magnitudes before any numeric tuning; preserve brma_legacy base behavior.",
        })
    stats = component_audit.get("component_stats", {})
    if stats.get("r_role_uav_kill_bonus", {}).get("trigger_rate", 0.0) <= 0:
        recs.append({
            "target": "r_role_uav_kill_bonus",
            "reason": "Short rollout saw no kill bonus trigger; sparse kill reward may not provide early learning signal.",
            "change": "Keep kill bonus but pair it with verified attack-window shaping that triggers before kills.",
        })
    if stats.get("r_role_uav_attack_window", {}).get("trigger_rate", 0.0) <= 0:
        recs.append({
            "target": "r_role_uav_attack_window",
            "reason": "Short rollout saw little or no attack-window signal, so UAV may not receive dense approach guidance.",
            "change": "Recheck distance/angle thresholds against observed normalized geometry before changing values.",
        })
    return recs[:5]


def _blocking_issues(code_audit: dict[str, Any], comparison: dict[str, Any], component_audit: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if code_audit.get("online_eval_used_non_role_configs"):
        issues.append("role_v1_online_eval_did_not_use_role_v1_configs")
    if code_audit.get("mav_support_uses_enemy_alive_mask"):
        issues.append("r_role_mav_support_uses_enemy_alive_mask_not_observed_mask")
    judgement = comparison.get("judgement", {})
    if judgement.get("role_v1_clearly_weaker_than_legacy") and not judgement.get("role_v1_improves_mav_survival"):
        issues.append("role_v1_weaker_than_brma_legacy_and_no_mav_survival_gain")
    stats = component_audit.get("component_stats", {})
    if stats.get("r_role_uav_kill_bonus", {}).get("trigger_rate", 0.0) <= 0:
        issues.append("r_role_uav_kill_bonus_sparse_or_not_triggered_in_short_rollout")
    return issues


def _markdown(data: dict[str, Any]) -> str:
    comp = data["comparison_summary"]
    role_eval = comp["final_eval_summary"]["role_v1"]
    legacy_eval = comp["final_eval_summary"]["brma_legacy"]
    lines = [
        "# role_v1 Reward Effects Audit",
        "",
        "## Summary",
        f"- role_v1 weaker than brma_legacy: {comp['judgement']['role_v1_clearly_weaker_than_legacy']}",
        f"- role_v1 improves MAV survival: {comp['judgement']['role_v1_improves_mav_survival']}",
        f"- blocking/high-priority issues: {', '.join(data['blocking_or_high_priority_issues']) or 'none'}",
        "",
        "## 50k Result Comparison",
        "",
        "| setting | 3v2 red_win | 3v2 MAV survival | 5v4 red_win | 5v4 MAV survival |",
        "|---|---:|---:|---:|---:|",
        (
            f"| brma_legacy | {legacy_eval.get('3v2', {}).get('red_win_rate', 0):.3f} | "
            f"{legacy_eval.get('3v2', {}).get('mav_survival_rate', 0):.3f} | "
            f"{legacy_eval.get('5v4', {}).get('red_win_rate', 0):.3f} | "
            f"{legacy_eval.get('5v4', {}).get('mav_survival_rate', 0):.3f} |"
        ),
        (
            f"| role_v1 | {role_eval.get('3v2', {}).get('red_win_rate', 0):.3f} | "
            f"{role_eval.get('3v2', {}).get('mav_survival_rate', 0):.3f} | "
            f"{role_eval.get('5v4', {}).get('red_win_rate', 0):.3f} | "
            f"{role_eval.get('5v4', {}).get('mav_survival_rate', 0):.3f} |"
        ),
        "",
        "## Role Component Audit",
    ]
    for key, stats in data["role_component_audit"]["component_stats"].items():
        lines.append(
            f"- {key}: sum={stats['sum']:.4f}, trigger_rate={stats['trigger_rate']:.4f}, mean={stats['mean']:.6f}"
        )
    lines.extend([
        "",
        "## Code Logic Findings",
        f"- role_v1 overlays brma_legacy: {data['code_logic_audit']['role_v1_overlay_on_brma_legacy']}",
        f"- brma_legacy kept separate: {data['code_logic_audit']['brma_legacy_unchanged_path']}",
        f"- MAV support uses enemy_alive_mask: {data['code_logic_audit']['mav_support_uses_enemy_alive_mask']}",
        f"- _last_step_obs one-step lag risk: {data['code_logic_audit']['last_step_obs_one_step_lag_risk']}",
        f"- online eval used non-role configs: {data['code_logic_audit']['online_eval_used_non_role_configs']}",
        "",
        "## Paper Alignment",
        f"- BRMA-MAPPO: {data['paper_alignment']['brma_mappo']['alignment']}",
        f"- BRMA-MAPPO gap: {data['paper_alignment']['brma_mappo']['gap']}",
        f"- heterogeneous MAV/UAV: {data['paper_alignment']['heterogeneous_mav_uav']['alignment']}",
        "",
        "## Recommended Changes",
    ])
    for idx, rec in enumerate(data["recommended_changes"], start=1):
        lines.append(f"{idx}. {rec['target']}: {rec['change']} Reason: {rec['reason']}")
    lines.extend([
        "",
        "This audit does not modify reward, termination, missile, action, PID, aircraft XML, or MAPPO network code.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit role_v1 reward effects from existing 50k results.")
    parser.add_argument("--role-dir", default="outputs/main_mappo_experiment_f22_50k_role_v1")
    parser.add_argument("--legacy-dir", default="outputs/main_mappo_experiment_f22_50k_rule_nearest_alive_done_fix")
    parser.add_argument("--output-json", default="outputs/reward_audit/role_v1_reward_effects_audit.json")
    parser.add_argument("--output-md", default="outputs/reward_audit/role_v1_reward_effects_audit.md")
    parser.add_argument("--component-config", default="uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_role_v1.yaml")
    parser.add_argument("--component-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    role_dir = ROOT / args.role_dir
    legacy_dir = ROOT / args.legacy_dir
    comparison = _compare(role_dir, legacy_dir)
    code_audit = _code_audit()
    component_audit = _sample_components(args.component_config, args.component_steps, args.seed)
    paper_alignment = _paper_alignment()
    recommended = _recommended_changes(code_audit, comparison, component_audit)
    issues = _blocking_issues(code_audit, comparison, component_audit)
    data = {
        "role_dir": str(role_dir),
        "legacy_dir": str(legacy_dir),
        "comparison_summary": comparison,
        "role_component_audit": component_audit,
        "code_logic_audit": code_audit,
        "paper_alignment": paper_alignment,
        "blocking_or_high_priority_issues": issues,
        "recommended_changes": recommended,
        "reward_code_modified_by_this_audit": False,
    }

    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")
    print(f"role_v1_weaker_than_legacy: {comparison['judgement']['role_v1_clearly_weaker_than_legacy']}", flush=True)
    print(f"role_v1_improves_mav_survival: {comparison['judgement']['role_v1_improves_mav_survival']}", flush=True)
    print(f"blocking_or_high_priority_issues: {issues}", flush=True)
    print(f"recommended_changes: {len(recommended)}", flush=True)
    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)


if __name__ == "__main__":
    main()
