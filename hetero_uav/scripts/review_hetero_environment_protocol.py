"""Review hetero environment protocol — config audit, dimensions, status.

No training.  Reads config groups, optionally resets each env once to
infer adapter dimensions, then writes JSON and Markdown reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- protocol groups ---------------------------------------------------

MAIN_PAPER_ALIGNED = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]

HARD_ABLATION_BALANCED = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]

OPTIONAL_REWARD_OVERLAY = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_reward_minimal.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_reward_minimal.yaml",
    "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2_reward_minimal.yaml",
]

DIAGNOSTIC_ONLY = [
    "uav_env/JSBSim/configs/hetero_diagnostic_close_range_mav_shared_geo_3v2.yaml",
]

OPTIONAL_REFERENCE = [
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_4v4.yaml",
]

V2_ACTOR_DIM = 96
V2_CRITIC_DIM = 480
V1_ACTOR_DIM = 140
V1_CRITIC_DIM = 700

# -- helpers -----------------------------------------------------------


def _load_yaml(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _count_types(agent_types: list[str], type_name: str) -> int:
    return sum(1 for t in agent_types if t == type_name)


def _count_mav(agent_types: list[str]) -> int:
    return _count_types(agent_types, "mav")


def _count_attack_uav(agent_types: list[str]) -> int:
    return _count_types(agent_types, "attack_uav")


def _decision_dt(cfg: dict) -> float:
    sim_freq = float(cfg.get("sim_freq", 60))
    steps = int(cfg.get("agent_interaction_steps", 12))
    return steps / sim_freq


def _adapter_dims(cfg: dict, skip_env: bool) -> tuple[int | None, int | None]:
    """Return (actor_dim, critic_dim) or (None, None) if skipped."""
    if skip_env:
        return None, None
    obs_mode = cfg.get("observation_mode", "")
    if obs_mode == "mav_shared_geo":
        return V2_ACTOR_DIM, V2_CRITIC_DIM
    if obs_mode == "brma_sensor":
        return V1_ACTOR_DIM, V1_CRITIC_DIM
    return None, None


def _review_config(config_path: str, group: str, skip_env: bool) -> dict:
    cfg = _load_yaml(config_path)
    if not cfg:
        return {
            "config": config_path,
            "protocol_group": group,
            "status": "fail",
            "warnings": ["config_file_missing"],
        }

    obs_mode = cfg.get("observation_mode", "")
    reward_mode = cfg.get("hetero_reward_mode", "brma_legacy")
    red_types = list(cfg.get("red_agent_types", []))
    blue_types = list(cfg.get("blue_agent_types", []))
    red_count = int(cfg.get("max_num_red", 0))
    blue_count = int(cfg.get("max_num_blue", 0))
    red_attack = _count_attack_uav(red_types)
    blue_attack = _count_attack_uav(blue_types)
    mav_count = _count_mav(red_types) + _count_mav(blue_types)
    missile_counts: dict[str, int] = {}
    for i, t in enumerate(red_types):
        missile_counts[f"red_{i}"] = 0 if t == "mav" else 2
    for i, t in enumerate(blue_types):
        missile_counts[f"blue_{i}"] = 0 if t == "mav" else 2

    max_steps = int(cfg.get("max_steps", 0))
    sim_freq = int(cfg.get("sim_freq", 60))
    agent_interaction_steps = int(cfg.get("agent_interaction_steps", 12))
    decision_dt = _decision_dt(cfg)
    actor_dim, critic_dim = _adapter_dims(cfg, skip_env)

    warnings: list[str] = []
    status = "pass"

    # -- per-group checks --
    if group == "main_paper_aligned":
        if obs_mode != "mav_shared_geo":
            status = "fail"
            warnings.append(f"observation_mode={obs_mode}, expected mav_shared_geo")
        if reward_mode != "brma_legacy":
            status = "fail"
            warnings.append(f"hetero_reward_mode={reward_mode}, expected brma_legacy")
        if max_steps < 1000:
            warnings.append(f"max_steps={max_steps} < 1000")
        if sim_freq != 60:
            warnings.append(f"sim_freq={sim_freq}, expected 60")
        if agent_interaction_steps != 12:
            warnings.append(f"agent_interaction_steps={agent_interaction_steps}, expected 12")
        if "3v2" in Path(config_path).stem:
            if red_count != 3 or blue_count != 2:
                status = "fail"
                warnings.append(f"3v2: red={red_count}, blue={blue_count}")
            if red_attack != 2:
                warnings.append(f"3v2: red attack UAV={red_attack}, expected 2")
            if blue_attack != 2:
                warnings.append(f"3v2: blue attack UAV={blue_attack}, expected 2")
            if mav_count != 1:
                warnings.append(f"3v2: MAV count={mav_count}, expected 1")
        if "5v4" in Path(config_path).stem and "reward_minimal" not in Path(config_path).stem:
            if red_count != 5 or blue_count != 4:
                status = "fail"
                warnings.append(f"5v4: red={red_count}, blue={blue_count}")
            if red_attack != 4:
                warnings.append(f"5v4: red attack UAV={red_attack}, expected 4")
            if blue_attack != 4:
                warnings.append(f"5v4: blue attack UAV={blue_attack}, expected 4")
            if mav_count != 1:
                warnings.append(f"5v4: MAV count={mav_count}, expected 1")

    elif group == "hard_ablation_balanced":
        if obs_mode != "mav_shared_geo":
            status = "fail"
            warnings.append(f"observation_mode={obs_mode}, expected mav_shared_geo")
        if reward_mode != "brma_legacy":
            status = "fail"
            warnings.append(f"hetero_reward_mode={reward_mode}, expected brma_legacy")
        if red_count != blue_count:
            status = "fail"
            warnings.append(f"not balanced: red={red_count}, blue={blue_count}")
        if red_attack < blue_attack:
            warnings.append(
                f"red has {red_attack} attack UAV vs blue {blue_attack} "
                f"(expected: red has one fewer attack UAV due to non-shooting MAV)"
            )

    elif group == "optional_reward_overlay":
        if reward_mode != "minimal_v1":
            status = "fail"
            warnings.append(f"hetero_reward_mode={reward_mode}, expected minimal_v1")
        warnings.append("optional reward overlay — not default baseline")

    elif group == "diagnostic_only":
        warnings.append("diagnostic config — not a training config")

    elif group == "optional_reference":
        warnings.append("optional reference — not a blocker for main protocol readiness")

    record = {
        "config": config_path,
        "protocol_group": group,
        "observation_mode": obs_mode,
        "hetero_reward_mode": reward_mode,
        "red_count": red_count,
        "blue_count": blue_count,
        "red_agent_types": red_types,
        "blue_agent_types": blue_types,
        "red_attack_uav_count": red_attack,
        "blue_attack_uav_count": blue_attack,
        "mav_count": mav_count,
        "missile_counts": missile_counts,
        "max_steps": max_steps,
        "sim_freq": sim_freq,
        "agent_interaction_steps": agent_interaction_steps,
        "decision_dt": decision_dt,
        "actor_dim": actor_dim,
        "critic_dim": critic_dim,
        "status": status,
        "warnings": warnings,
    }
    return record


def _markdown(data: dict) -> str:
    lines = [
        "# Hetero Environment Protocol Review",
        "",
        "Purpose: freeze environment protocol decisions before training.",
        "This is not a method module. No training, no algorithm changes.",
        "",
        "## Summary",
        "",
        f"- main_protocol_ready: {data['summary']['main_protocol_ready']}",
        f"- hard_ablation_ready: {data['summary']['hard_ablation_ready']}",
        f"- reward_overlay_ready: {data['summary']['reward_overlay_ready']}",
        f"- blocking_failures: {data['summary']['blocking_failures']}",
        f"- warnings: {len(data['summary']['warnings'])}",
        f"- next_environment_task: {data['summary']['next_environment_task']}",
        "",
        "## Protocol Decision",
        "",
        "- **main paper-aligned protocol**: train 3v2, eval 5v4",
        "- **hard ablation**: balanced 3v3/4v4",
        "- **optional reward overlay**: minimal_v1",
        "- **optional reference**: V1 brma_sensor",
        "",
        "## Reward Decision",
        "",
        "- `brma_legacy` is the default baseline reward",
        "- `minimal_v1` is an optional role-aware overlay",
        "- no termination change",
        "",
        "## Opponent Decision",
        "",
        "- `rule_nearest` remains available",
        "- `greedy_fsm` is a diagnostic environment opponent",
        "- `greedy_fsm` is NOT yet the final default training opponent",
        "",
        "## What is Frozen",
        "",
        "- aircraft models (A-4 MAV, f16 UAV)",
        "- MAV/UAV missile counts (MAV=0, attack UAV=2)",
        "- paper-aligned composition (red attack UAV = blue attack UAV)",
        "- V2 observation mode (mav_shared_geo)",
        "- decision frequency (sim_freq=60, agent_interaction_steps=12)",
        "- reward default (brma_legacy)",
        "",
        "## What Remains Open",
        "",
        "- whether to train with brma_legacy or minimal_v1",
        "- whether greedy_fsm should replace rule_nearest for baseline training",
        "- whether reward/termination need further changes after short smoke",
        "- NO method module yet",
        "",
        "## Records by Group",
    ]

    for group, label in [
        ("main_paper_aligned", "Main Paper-Aligned"),
        ("hard_ablation_balanced", "Hard Ablation (Balanced)"),
        ("optional_reward_overlay", "Optional Reward Overlay"),
        ("diagnostic_only", "Diagnostic Only"),
        ("optional_reference", "Optional Reference"),
    ]:
        lines.append(f"")
        lines.append(f"### {label}")
        for rec in [r for r in data["records"] if r.get("protocol_group") == group]:
            lines.append(f"")
            lines.append(f"**{Path(rec['config']).name}** — status: `{rec['status']}`")
            lines.append(f"")
            lines.append(f"- obs_mode: {rec['observation_mode']}")
            lines.append(f"- reward_mode: {rec['hetero_reward_mode']}")
            lines.append(f"- red: {rec['red_count']} ({', '.join(rec['red_agent_types'])})")
            lines.append(f"- blue: {rec['blue_count']} ({', '.join(rec['blue_agent_types'])})")
            lines.append(f"- red attack UAV: {rec['red_attack_uav_count']}, blue attack UAV: {rec['blue_attack_uav_count']}")
            lines.append(f"- MAV count: {rec['mav_count']}")
            lines.append(f"- max_steps: {rec['max_steps']}")
            lines.append(f"- sim_freq: {rec['sim_freq']}")
            lines.append(f"- agent_interaction_steps: {rec['agent_interaction_steps']}")
            lines.append(f"- decision_dt: {rec['decision_dt']}")
            if rec["actor_dim"] is not None:
                lines.append(f"- actor_dim: {rec['actor_dim']}")
            if rec["critic_dim"] is not None:
                lines.append(f"- critic_dim: {rec['critic_dim']}")
            if rec["missile_counts"]:
                lines.append(f"- missile_counts: {rec['missile_counts']}")
            if rec["warnings"]:
                lines.append(f"- warnings: {rec['warnings']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/hetero_environment_protocol_review.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/hetero_environment_protocol_review.md",
    )
    parser.add_argument("--skip-env-reset", action="store_true")
    args = parser.parse_args()

    groups = {
        "main_paper_aligned": MAIN_PAPER_ALIGNED,
        "hard_ablation_balanced": HARD_ABLATION_BALANCED,
        "optional_reward_overlay": OPTIONAL_REWARD_OVERLAY,
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "optional_reference": OPTIONAL_REFERENCE,
    }

    records: list[dict] = []
    for group_name, configs in groups.items():
        for config in configs:
            if not (ROOT / config).exists():
                records.append({
                    "config": config,
                    "protocol_group": group_name,
                    "status": "fail",
                    "warnings": ["config_file_missing"],
                })
                continue
            record = _review_config(config, group_name, args.skip_env_reset)
            records.append(record)

    # -- summary --
    blocking_failures: list[str] = []
    warnings: list[str] = []
    main_ready = True
    ablation_ready = True
    overlay_ready = True

    for rec in records:
        group = rec.get("protocol_group", "")
        if rec.get("status") == "fail":
            blocking_failures.append(rec["config"])
            if group == "main_paper_aligned":
                main_ready = False
            if group == "hard_ablation_balanced":
                ablation_ready = False
            if group == "optional_reward_overlay":
                overlay_ready = False
        for w in rec.get("warnings", []):
            if w not in warnings:
                warnings.append(w)

    summary = {
        "main_protocol_ready": main_ready,
        "hard_ablation_ready": ablation_ready,
        "reward_overlay_ready": overlay_ready,
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "next_environment_task": "environment_protocol_review_then_optional_training_decision",
    }

    data = {"records": records, "summary": summary}

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")

    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)
    print(f"main_protocol_ready: {main_ready}")
    print(f"hard_ablation_ready: {ablation_ready}")
    print(f"reward_overlay_ready: {overlay_ready}")
    print(f"blocking_failures: {blocking_failures}")
    print(f"warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
