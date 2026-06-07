"""Audit F-22 MAV high-level control response in the main hetero env.

This is diagnostic-only. It does not modify reward, termination, missile,
action space, evasion, PID, aircraft XML, or MAPPO.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env


MAIN_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
SCENARIOS = {
    "level": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "climb": np.array([0.4, 0.0, 0.0], dtype=np.float32),
    "descend": np.array([-0.4, 0.0, 0.0], dtype=np.float32),
    "turn_left": np.array([0.0, -0.4, 0.0], dtype=np.float32),
    "turn_right": np.array([0.0, 0.4, 0.0], dtype=np.float32),
    "speed_up": np.array([0.0, 0.0, 0.8], dtype=np.float32),
    "slow_down": np.array([0.0, 0.0, -0.8], dtype=np.float32),
}


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        return result.stdout + result.stderr
    return result.stdout


def _read_yaml(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _status_has(status: str, token: str) -> bool:
    return any(token in line for line in status.splitlines())


def _diff_state(path: str) -> dict:
    diff = _run_git(["diff", "--", path])
    summary = _run_git(["diff", "--stat", "--", path]).strip()
    return {
        "path": path,
        "modified": _status_has(_run_git(["status", "--short", "hetero_uav"]), path.replace("/", "\\")) or _status_has(_run_git(["status", "--short", "hetero_uav"]), path),
        "has_text_diff": bool(diff.strip()),
        "diff_summary": summary,
        "state": "modified_but_no_text_diff" if not diff.strip() else "modified_with_text_diff",
    }


def audit_resource_status() -> dict:
    status = _run_git(["status", "--short", "hetero_uav"])
    f100 = _diff_state("hetero_uav/uav_env/JSBSim/data/engine/F100-PW-229.xml")
    f119 = _diff_state("hetero_uav/uav_env/JSBSim/data/engine/F119-PW-1.xml")
    f22_xml = ROOT / "uav_env/JSBSim/data/aircraft/f22/f22.xml"
    f22_text = f22_xml.read_text(encoding="utf-8", errors="replace") if f22_xml.exists() else ""
    f15_refs = _run_git([
        "grep", "-n", "-i", "f15", "--",
        "hetero_uav/uav_env/JSBSim/configs",
        "hetero_uav/scripts",
        "hetero_uav/tests",
    ])
    f15_used = bool(f15_refs.strip())
    return {
        "git_status_short_hetero_uav": status.splitlines(),
        "f100_pw_229": {
            **f100,
            "recommendation": "exclude_or_revert_unless_user_confirms",
            "reason": "F-22 XML does not reference F100-PW-229",
        },
        "f119_pw_1": {
            **f119,
            "required_by_f22": "F119-PW-1" in f22_text,
        },
        "f15": {
            "folder_present": (ROOT / "uav_env/JSBSim/data/aircraft/f15").exists(),
            "unused": not f15_used,
            "references": f15_refs.splitlines(),
            "recommendation": "not part of main experiment",
        },
        "f22": {
            "folder_present": (ROOT / "uav_env/JSBSim/data/aircraft/f22").exists(),
            "xml_present": f22_xml.exists(),
        },
    }


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _red0_metrics(env) -> dict:
    sim = env.red_planes["red_0"]
    vel = sim.get_velocity()
    return {
        "altitude_m": float(sim.get_geodetic()[2]),
        "speed_mps": float(np.linalg.norm(vel)),
        "heading_rad": float(sim.get_rpy()[2]),
        "crash": bool(not sim.is_alive),
    }


def _scenario_actions(env, red0_action: np.ndarray) -> dict[str, np.ndarray]:
    actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    actions["red_0"] = red0_action.astype(np.float32)
    return actions


def run_control_response(steps: int) -> dict:
    records: dict[str, dict] = {}
    for name, action in SCENARIOS.items():
        env = make_env(MAIN_CONFIG, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
        try:
            obs, _info = env.reset(seed=0)
            initial = _red0_metrics(env)
            nan_detected = _contains_nan(obs)
            terminated = {}
            truncated = {}
            for _ in range(steps):
                obs, rewards, terminated, truncated, _info = env.step(
                    _scenario_actions(env, action)
                )
                nan_detected = nan_detected or _contains_nan(obs) or _contains_nan(rewards)
                if all(terminated.values()) or all(truncated.values()):
                    break
            final = _red0_metrics(env)
            records[name] = {
                "action": action.tolist(),
                "initial_altitude_m": initial["altitude_m"],
                "final_altitude_m": final["altitude_m"],
                "altitude_delta_m": final["altitude_m"] - initial["altitude_m"],
                "initial_speed_mps": initial["speed_mps"],
                "final_speed_mps": final["speed_mps"],
                "speed_delta_mps": final["speed_mps"] - initial["speed_mps"],
                "initial_heading_rad": initial["heading_rad"],
                "final_heading_rad": final["heading_rad"],
                "heading_delta_rad": _wrap_pi(final["heading_rad"] - initial["heading_rad"]),
                "crash": bool(final["crash"]),
                "nan_detected": bool(nan_detected),
                "terminated_all": bool(terminated and all(terminated.values())),
                "truncated_all": bool(truncated and all(truncated.values())),
            }
        finally:
            env.close()
    return records


def assess_response(records: dict[str, dict]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    blocking: list[str] = []
    if any(record["nan_detected"] for record in records.values()):
        blocking.append("nan_detected")
    if any(record["crash"] for record in records.values()):
        warnings.append("f22_mav_crash_in_short_response_diagnostic")

    deltas = [
        (
            round(record["altitude_delta_m"], 3),
            round(record["speed_delta_mps"], 3),
            round(record["heading_delta_rad"], 6),
        )
        for record in records.values()
    ]
    if len(set(deltas)) <= 1:
        blocking.append("f22_mav_high_level_action_response_not_verified")

    climb = records["climb"]["altitude_delta_m"]
    descend = records["descend"]["altitude_delta_m"]
    level_alt = records["level"]["altitude_delta_m"]
    if not (climb > level_alt and descend < level_alt):
        warnings.append("climb_descend_response_not_clearly_separated")

    speed_up = records["speed_up"]["speed_delta_mps"]
    slow_down = records["slow_down"]["speed_delta_mps"]
    level_speed = records["level"]["speed_delta_mps"]
    if not (speed_up > level_speed and slow_down < level_speed):
        warnings.append("speed_response_not_clearly_separated")

    left = records["turn_left"]["heading_delta_rad"]
    right = records["turn_right"]["heading_delta_rad"]
    if not (left < 0.0 and right > 0.0):
        warnings.append("turn_left_right_heading_direction_not_clearly_correct")
    return warnings, blocking


def _markdown(data: dict) -> str:
    lines = [
        "# F-22 MAV Control Response Audit",
        "",
        "Purpose: verify F-22 MAV control response in the main hetero env.",
        "This audit does not modify model files, PID, action space, reward,",
        "termination, missile, evasion, or MAPPO.",
        "",
        "missile audit should wait until F-22 control response is verified.",
        "",
        f"- action_trim_by_role.mav.pitch: {data['action_trim_by_role']['mav_pitch']}",
        "- note: this trim is an A-4 stability carryover and is not changed here.",
        "",
        "## Scenarios",
    ]
    for name, record in data["control_response"].items():
        lines.append(
            f"- {name}: altitude_delta={record['altitude_delta_m']:.2f} m, "
            f"speed_delta={record['speed_delta_mps']:.2f} m/s, "
            f"heading_delta={record['heading_delta_rad']:.4f} rad, "
            f"crash={record['crash']}, nan={record['nan_detected']}"
        )
    lines.extend([
        "",
        "## Resource Status",
        "",
        f"- F100 recommendation: {data['resource_status']['f100_pw_229']['recommendation']}",
        f"- F119 required_by_f22: {data['resource_status']['f119_pw_1']['required_by_f22']}",
        f"- f15 unused: {data['resource_status']['f15']['unused']}",
        "",
        "## Decision",
        "",
        f"- blocking_issues: {data['blocking_issues']}",
        f"- warnings: {data['warnings']}",
        f"- next_action: {data['next_action']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output-json", default="outputs/environment_audit/f22_mav_control_response.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/f22_mav_control_response.md")
    args = parser.parse_args()

    cfg = _read_yaml(MAIN_CONFIG)
    control_response = run_control_response(args.steps)
    warnings, blocking = assess_response(control_response)
    data = {
        "resource_status": audit_resource_status(),
        "action_trim_by_role": {
            "mav_pitch": cfg.get("action_trim_by_role", {}).get("mav", {}).get("pitch"),
            "note": "A-4 stability carryover; F-22 need is not verified here.",
        },
        "control_response": control_response,
        "warnings": warnings,
        "blocking_issues": blocking,
        "next_action": (
            "continue_to_missile_launch_contract_audit"
            if not blocking
            else "diagnose_f22_control_path_trim_throttle_before_missile_audit"
        ),
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(data), encoding="utf-8")
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"blocking_issues: {blocking}", flush=True)
    print(f"warnings: {warnings}", flush=True)
    print(f"next_action: {data['next_action']}", flush=True)


if __name__ == "__main__":
    main()
