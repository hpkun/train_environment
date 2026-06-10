"""Audit HAPPO 3v2 reference-validation alignment.

This is a protocol planning audit. It does not implement HAPPO, change reward,
change missiles, change PID, or train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_current_env_setup() -> dict[str, Any]:
    cfg_path = ROOT / "uav_env" / "JSBSim" / "configs" / "hetero_mav_shared_geo_3v2_no_mav_trim.yaml"
    text = _read_text(cfg_path)
    return {
        "config": "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_no_mav_trim.yaml",
        "red": ["red_0=f22 MAV missiles=0", "red_1=f16 attack_uav missiles=2", "red_2=f16 attack_uav missiles=2"],
        "blue": ["blue_0=f16 attack_uav missiles=2", "blue_1=f16 attack_uav missiles=2"],
        "action": "high-level [pitch, heading, speed]",
        "observation": "V2 mav_shared_geo",
        "reward": "brma_legacy in current validated baseline",
        "missile_launch": "current scripted BRMA-style fire-control",
        "missile_evasion": "current scripted evasion; red-only in current hetero env",
        "sim_freq_60": "sim_freq: 60" in text,
        "agent_interaction_steps_12": "agent_interaction_steps: 12" in text,
        "max_steps_1000": "max_steps: 1000" in text,
        "mav_trim_disabled": "pitch: 0.0" in text,
    }


def _baseline_evidence() -> dict[str, Any]:
    path = ROOT / "outputs" / "main_mappo_baseline_1m_fast_brma_rule_no_mav_trim" / "checkpoint_eval_100ep_best" / "baseline_1m_best_100ep_eval.json"
    data = _read_json(path) or {}
    return {
        "source": str(path),
        "summary": data,
        "interpretation": {
            "shared_mlp_survives_to_timeout": True,
            "effective_kill_policy": False,
            "mav_support_role_established": False,
            "reason": "100-episode best checkpoint has timeout=1.0, zero elimination wins, and MAV survival=0.",
        },
    }


def build_audit() -> dict[str, Any]:
    return {
        "paper_reference_setup": {
            "scope": "paper-informed HAPPO environment validation, not full TAM-HAPPO reproduction",
            "red_composition": "1 MAV + 2 UAV",
            "blue_composition": "2 UAV",
            "mav_role": [
                "no missiles",
                "battlefield information",
                "mission guidance",
                "support",
            ],
            "uav_role": [
                "attack",
                "missile engagement",
                "combat execution",
            ],
            "method_reference": [
                "HAPPO handles heterogeneous policies",
                "original paper includes temporal module and attention",
                "first validation stage does not implement full TAM-HAPPO",
            ],
        },
        "current_env_setup": _extract_current_env_setup(),
        "baseline_evidence": _baseline_evidence(),
        "gaps": [
            {
                "name": "action_interface_gap",
                "current": "high-level [pitch, heading, speed] + PID",
                "paper_reference": "lower-level control is closer to paper",
                "decision": "keep current interface for validation to avoid control confounders",
            },
            {
                "name": "reward_gap",
                "current": "brma_legacy baseline; role_v1 failed as current ablation",
                "paper_reference": "role-specific MAV/UAV reward",
                "decision": "prepare happo_ref_v0 design, but do not replace brma_legacy in this audit",
            },
            {
                "name": "opponent_policy_gap",
                "current": "brma_rule / scripted opponents",
                "paper_reference": "paper-informed adversary behavior",
                "decision": "keep opponent fixed for reference validation first",
            },
            {
                "name": "missile_model_gap",
                "current": "scripted BRMA-style launch/evasion",
                "paper_reference": "missile-aware combat and dodge signals",
                "decision": "keep current missile mechanics; document non-full reproduction",
            },
            {
                "name": "temporal_module_gap",
                "current": "shared MLP baseline lacks temporal module",
                "paper_reference": "temporal feature module",
                "decision": "HAPPO reference v0 can start without temporal module; add later if needed",
            },
            {
                "name": "attention_gap",
                "current": "no attention in shared MLP baseline",
                "paper_reference": "attention module",
                "decision": "do not implement attention in first HAPPO reference validation",
            },
            {
                "name": "happo_update_gap",
                "current": "shared-policy MAPPO",
                "paper_reference": "heterogeneous actor update",
                "decision": "future HAPPO smoke should add separate MAV/UAV actors and sequential update",
            },
            {
                "name": "aircraft_model_gap",
                "current": "F-22 MAV and F-16 UAV",
                "paper_reference": "paper MAV/UAV physical models are not exactly reproduced",
                "decision": "treat as engineering approximation and document it",
            },
        ],
        "success_criteria": [
            "MAV should not always die early.",
            "UAVs should produce effective missile launches.",
            "Blue deaths should not remain zero long-term.",
            "Episodes should not all be timeout draws.",
            "Death reasons should mainly come from combat/missiles rather than crash, over-g, or control anomalies.",
            "ACMI should show MAV maintaining a support posture while UAVs engage.",
        ],
        "happo_ref_v0_reward_design": {
            "implemented_this_round": False,
            "purpose": "paper-reference reward for validation only, not replacement for brma_legacy baseline",
            "mav_components": [
                "survival reward",
                "death penalty",
                "support reward based on enemy_observed_mask or enemy_track_source",
                "boundary/altitude/speed safety terms",
                "no kill-directed reward for MAV",
            ],
            "uav_components": [
                "altitude/speed/angle/distance terms",
                "missile-warning dodge term",
                "attack-window term",
                "kill bonus",
                "death penalty",
                "team event reward",
            ],
            "blocking_note": "Do not implement until reward scale and component audit criteria are fixed.",
        },
        "minimal_happo_v0_plan": {
            "implemented_this_round": False,
            "policy": [
                "separate MAV actor",
                "separate UAV actor",
                "centralized critic",
                "Gaussian action distribution",
                "action_dim=3",
                "actor_obs_dim=96",
                "critic_state_dim=480",
            ],
            "trainer": [
                "sequential actor update",
                "active mask preserved",
                "team done preserved",
                "no attention",
                "no GRU in first HAPPO reference v0",
                "no observation dim change",
            ],
            "smoke_runner": "Only after implementation is scoped as one coherent change.",
        },
        "next_steps": [
            "Use this audit to decide whether to implement HAPPO reference v0 smoke.",
            "If implemented, start with 64-step smoke only.",
            "If smoke passes, run 200k HAPPO reference validation before any 1M run.",
        ],
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# HAPPO 3v2 Reference Alignment Audit",
        "",
        "## Scope",
        "",
        data["paper_reference_setup"]["scope"],
        "",
        "## Paper Reference Setup",
        "",
        f"- Red: {data['paper_reference_setup']['red_composition']}",
        f"- Blue: {data['paper_reference_setup']['blue_composition']}",
        "- MAV: no missiles; battlefield information, mission guidance, support.",
        "- UAV: attack and missile engagement.",
        "- Method: HAPPO-style heterogeneous policy reference; temporal and attention are later stages.",
        "",
        "## Current Environment Setup",
        "",
        f"- Config: {data['current_env_setup']['config']}",
        f"- Red: {', '.join(data['current_env_setup']['red'])}",
        f"- Blue: {', '.join(data['current_env_setup']['blue'])}",
        f"- Action: {data['current_env_setup']['action']}",
        f"- Observation: {data['current_env_setup']['observation']}",
        f"- Reward: {data['current_env_setup']['reward']}",
        "",
        "## Baseline Evidence",
        "",
        "- 1M shared MLP best checkpoint survives to timeout but does not establish combat effectiveness.",
        "- 3v2 and 5v4 best checkpoint re-eval have zero elimination wins and MAV survival remains zero.",
        "- Timeout draw behavior must not be reported as successful air-combat strategy.",
        "",
        "## Gaps",
    ]
    for gap in data["gaps"]:
        lines.append(f"- {gap['name']}: {gap['current']} / decision: {gap['decision']}")
    lines.extend([
        "",
        "## Success Criteria",
    ])
    for item in data["success_criteria"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## HAPPO Reference v0 Plan",
        "",
        "- Separate MAV/UAV actors.",
        "- Centralized critic.",
        "- Sequential HAPPO update.",
        "- No attention in first stage.",
        "- Optional recurrent/temporal module later.",
        "- No reward, missile, PID, aircraft XML, or observation-dimension changes in this audit.",
        "",
        "## Next Steps",
    ])
    for item in data["next_steps"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HAPPO 3v2 reference-validation alignment.")
    parser.add_argument("--output-json", default="outputs/protocol_audit/happo_3v2_reference_alignment.json")
    parser.add_argument("--output-md", default="outputs/protocol_audit/happo_3v2_reference_alignment.md")
    args = parser.parse_args()

    data = build_audit()
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(data), encoding="utf-8")
    print("paper_reference_setup: HAPPO 3v2 validation", flush=True)
    print(f"current_config: {data['current_env_setup']['config']}", flush=True)
    print(f"happo_ref_v0_reward_implemented: {data['happo_ref_v0_reward_design']['implemented_this_round']}", flush=True)
    print(f"happo_smoke_implemented: {data['minimal_happo_v0_plan']['implemented_this_round']}", flush=True)
    print(f"output_json: {out_json}", flush=True)
    print(f"output_md: {out_md}", flush=True)


if __name__ == "__main__":
    main()
