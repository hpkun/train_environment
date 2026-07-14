"""Generate the single TAM paper-environment readiness report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.make_env import make_env


CONFIG_NAMES = ["tam_paper_env_v1_2v2.yaml", "tam_paper_env_v1_3v2.yaml",
                "tam_paper_env_v1_5v4.yaml"]
DISABLED_LEGACY = [
    "brma_legacy_reward", "happo_ref_v0_reward", "brma_11d_entity_observation",
    "scripted_evasion_red", "scripted_evasion_blue", "blue_gcas",
    "lock_delay_0.25s", "launch_cooldown_0.5s", "kill_cooldown",
    "multi_kill_block", "target_revival", "random_hit_probability",
    "engaged_target_gate", "nearest_target_rule", "legacy_action_trim",
]


def rollout(config: Path, backend: str, steps: int, seed: int, rule_red: bool = False) -> dict:
    env = make_env(str(config), dynamics_backend=backend,
                   reset_warmup_seconds=0.0 if backend == "jsbsim" else 35.0)
    obs, info = env.reset(seed=seed)
    finite = True
    completed = 0
    for _ in range(steps):
        if rule_red:
            actions = {aid: env.task.opponent.act(aid, obs[aid])[0] for aid in env.agent_ids}
        else:
            actions = {aid: env.action_space[aid].sample() for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        finite &= all(np.isfinite(value) for value in rewards.values())
        finite &= np.isfinite(env.get_state()).all()
        completed += 1
        if all(terminated.values()) or all(truncated.values()):
            break
    result = {"steps": completed, "finite": bool(finite),
              "missiles_fired": info["missiles_fired"],
              "missile_hits": info["missile_hits"], "winner": info["winner"],
              "termination_reason": info["termination_reason"]}
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/tam_paper_env_v1_audit.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--backend", choices=("simple", "jsbsim"), default="simple")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    config_dir = ROOT / "uav_env" / "JSBSim" / "configs"
    configs = [config_dir / name for name in CONFIG_NAMES]
    cfg = yaml.safe_load(configs[1].read_text(encoding="utf-8"))
    tests = {"passed": False, "returncode": None}
    if not args.skip_tests:
        command = [sys.executable, "-m", "pytest",
                   "tests/test_tam_paper_env_v1_contract.py",
                   "tests/test_tam_paper_env_v1_integration.py", "-q"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        tests = {"passed": result.returncode == 0, "returncode": result.returncode,
                 "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
    rollouts = {path.stem: rollout(path, args.backend, args.steps, 100 + idx)
                for idx, path in enumerate(configs)}
    rule_rollouts = {path.stem: rollout(path, args.backend, args.steps, 200 + idx, True)
                     for idx, path in enumerate(configs)}
    stable = all(item["finite"] and item["steps"] > 0 for item in rollouts.values())
    rule_combat = (all(item["missiles_fired"] > 0 for item in rule_rollouts.values())
                   and sum(item["missile_hits"] for item in rule_rollouts.values()) > 0
                   and any(item["termination_reason"] for item in rule_rollouts.values()))
    report = {
        "formal_environment_mode": "tam_paper_env_v1",
        "aircraft_and_roles": {
            "mav": {"model": "f22", "missiles": 0},
            "attack_uav": {"model": "f16", "missiles": 2},
        },
        "published_parameters": cfg["published_parameters"],
        "inferred_parameters": cfg["inferred_parameters"],
        "observation": {"ego_state": [7], "relative_entity": [5],
                        "incoming_missile": [5], "fixed_slots_and_masks": True},
        "action": {"space": "MultiDiscrete([40,40,40,40])",
                   "order": ["throttle", "aileron", "elevator", "rudder"],
                   "direct_fcs": True},
        "time_scale": {"physics_hz": 60, "decision_hz": 5,
                       "physics_frames_per_action": 12, "episode_steps": 1000},
        "missile": {key: cfg["published_parameters"][key] for key in
                    ("maximum_attack_range_m", "launch_interval_s", "maximum_overload_g",
                     "navigation_gain_y", "navigation_gain_z")},
        "launch_rules": {"automatic": True, "current_target_only": True,
                         "mav_unarmed": True, "random_hit": False,
                         "engaged_target_gate": False},
        "reward_mode": "paper_role_reward_v1",
        "disabled_legacy_mechanisms": DISABLED_LEGACY,
        "tests": tests,
        "random_rollout_statistics": rollouts,
        "rule_vs_rule_statistics": rule_rollouts,
    }
    report["READY_FOR_HAPPO"] = bool(tests["passed"] and stable and rule_combat)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["READY_FOR_HAPPO"] or args.skip_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
