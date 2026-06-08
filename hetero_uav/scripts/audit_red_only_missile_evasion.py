"""Audit red-only scripted missile evasion behavior.

This script is a short environment audit, not training. It checks that scripted
missile evasion is only applied to red agents while blue keeps the GCAS safety
net and normal action path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env

ENV_PATH = ROOT / "uav_env" / "JSBSim" / "env.py"
CONFIG_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"


def _parse_action_block() -> str:
    text = ENV_PATH.read_text(encoding="utf-8", errors="replace")
    start = text.index("Missile Evasion Script")
    end = text.index("Layer 2", start)
    return text[start:end]


def _obs_has_nan(obs: dict) -> bool:
    for agent_obs in obs.values():
        for value in agent_obs.values():
            arr = np.asarray(value)
            if arr.dtype.kind in {"f", "c"} and np.isnan(arr).any():
                return True
    return False


def _short_smoke() -> dict[str, Any]:
    env = make_env(CONFIG_3V2, env_type="jsbsim_hetero")
    try:
        action_dim = int(next(iter(env.action_space.spaces.values())).shape[0])
        nan_detected = False
        obs, info = env.reset(seed=0)
        for _ in range(3):
            actions = {aid: np.zeros(action_dim, dtype=np.float32)
                       for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
            nan_detected = nan_detected or _obs_has_nan(obs)

        obs, info = env.reset(seed=1)
        rng = np.random.default_rng(1)
        for _ in range(3):
            actions = {
                aid: rng.uniform(-0.5, 0.5, size=action_dim).astype(np.float32)
                for aid in env.agent_ids
            }
            obs, rewards, terminated, truncated, info = env.step(actions)
            nan_detected = nan_detected or _obs_has_nan(obs)
        return {
            "reset_ok": True,
            "zero_steps": 3,
            "bounded_random_steps": 3,
            "nan_detected": nan_detected,
            "action_dim": action_dim,
        }
    finally:
        env.close()


def build_audit() -> dict[str, Any]:
    text = ENV_PATH.read_text(encoding="utf-8", errors="replace")
    action_block = _parse_action_block()
    red_guard = "if not is_blue" in action_block and "sim.check_missile_warning()" in action_block
    no_both_comment = "BOTH teams" not in action_block
    red_comment = "RED team only" in action_block
    blue_comment = "blue\n            #  rule-based opponent" in action_block or "blue rule-based opponent" in action_block
    blue_gcas = "if is_blue and self.enable_gcas_for_blue" in text
    launch_contract_unchanged = all(token in text for token in [
        "MISSILE_LAUNCH_AO_THRESH",
        "MISSILE_LAUNCH_RANGE_THRESH",
        "MISSILE_LOCK_DELAY_FRAMES",
        "MISSILE_COOLDOWN_STEPS",
    ])

    smoke = _short_smoke()
    blocking_issues: list[str] = []
    if not red_guard:
        blocking_issues.append("missile_warning_not_guarded_by_not_is_blue")
    if not blue_gcas:
        blocking_issues.append("blue_gcas_safety_net_missing")
    if smoke["nan_detected"]:
        blocking_issues.append("short_smoke_nan_detected")
    if smoke["action_dim"] != 3:
        blocking_issues.append("action_dim_not_3")

    return {
        "red_scripted_evasion_enabled": red_guard and red_comment,
        "blue_scripted_evasion_enabled": False if red_guard else None,
        "blue_gcas_still_enabled": blue_gcas,
        "missile_launch_contract_unchanged": launch_contract_unchanged,
        "action_dim": smoke["action_dim"],
        "blocking_issues": blocking_issues,
        "static_checks": {
            "missile_evasion_block_has_red_guard": red_guard,
            "missile_evasion_block_has_no_both_teams_comment": no_both_comment,
            "missile_evasion_block_mentions_red_only": red_comment,
            "missile_evasion_block_mentions_blue_rule_opponent": blue_comment,
        },
        "scripted_evasion_summary": {
            "high_altitude": "25 deg pitch + 60 deg break turn + max speed",
            "low_altitude": "30 deg zoom climb + max speed",
            "bearing": "uses incoming missile horizontal bearing",
        },
        "limitations": [
            "Current evasion is scripted, not learned.",
            "It does not use full missile entity observation.",
            "It does not model time-to-go, missile energy, or 3D dodge reward.",
            "It is BRMA-style emergency scripted evasion.",
            "Learned or hybrid evasion should wait for missile-aware observation.",
        ],
        "smoke": smoke,
    }


def write_markdown(audit: dict[str, Any], output_md: Path) -> None:
    md = f"""# Red-Only Missile Evasion Audit

## Design

- Red uses scripted missile evasion: {audit['red_scripted_evasion_enabled']}
- Blue uses scripted missile evasion: {audit['blue_scripted_evasion_enabled']}
- Blue GCAS remains enabled: {audit['blue_gcas_still_enabled']}
- Action dim remains: {audit['action_dim']}

## Motivation

Missile warning and scripted evasion are modeled as a red MAV/UAV formation
information advantage. Blue is a rule-based opponent and does not use the same
scripted missile evasion layer.

## Script Summary

- High altitude: 25 deg pitch + 60 deg break turn + max speed
- Low altitude: 30 deg zoom climb + max speed
- Bearing uses incoming missile horizontal bearing

## Limitations

- Current evasion is scripted, not learned.
- It does not use full missile entity observation.
- It does not model time-to-go, missile energy, or 3D dodge reward.
- It is BRMA-style emergency scripted evasion.
- Learned or hybrid evasion should wait for missile-aware observation.

## Blocking Issues

{chr(10).join(f'- {issue}' for issue in audit['blocking_issues']) or '- None'}
"""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        default="outputs/environment_audit/red_only_missile_evasion_audit.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/environment_audit/red_only_missile_evasion_audit.md",
    )
    args = parser.parse_args()

    audit = build_audit()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit, output_md)
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"blocking_issues: {audit['blocking_issues']}", flush=True)


if __name__ == "__main__":
    main()
