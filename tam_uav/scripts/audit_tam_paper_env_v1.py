"""Audit the live TAM paper environment contract with real rollouts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.paper.env import TAMPaperEnv
from uav_env.JSBSim.paper.missile import PaperMissile
from uav_env.JSBSim.paper.observation import PaperObservation
from uav_env.JSBSim.paper.reward import PaperReward
from uav_env.JSBSim.paper.weapon import PaperWeaponManager
from uav_env.make_env import make_env


CONFIG_NAMES = (
    "tam_paper_env_v1_2v2.yaml",
    "tam_paper_env_v1_3v2.yaml",
    "tam_paper_env_v1_5v4.yaml",
)
LEGACY_ATTRIBUTES = {
    "legacy_reward", "legacy_observation", "legacy_missile", "blue_gcas",
    "scripted_evasion", "engaged_target_gate", "kill_cooldown",
    "target_revival", "direct_fcs_static_trim_by_model",
}
PARAMETER_CONSUMERS = {
    "simulation_frequency_hz": "core.scenario, paper.task",
    "decision_frequency_hz": "paper.task",
    "physics_frames_per_action": "paper.task, paper.opponent",
    "episode_limit_steps": "paper.task",
    "maximum_attack_range_m": "paper.task, paper.weapon, paper.situation",
    "launch_interval_s": "paper.weapon",
    "maximum_overload_g": "paper.missile",
    "maximum_aircraft_overload_g": "paper.task",
    "maximum_speed_mps": "paper.task, paper.situation",
    "navigation_gain_y": "paper.missile",
    "navigation_gain_z": "paper.missile",
    "minimum_safe_altitude_m": "paper.reward",
    "structural_limit_grace_s": "paper.task",
    "mav_detection_range_m": "core.aircraft_types",
    "uav_direct_detection_range_m": "core.aircraft_types",
}


def _rule_actions(env) -> dict[str, np.ndarray]:
    by_id = {agent.agent_id: agent for agent in env.task.agents}
    actions = {}
    for aid in env.agent_ids:
        agent = by_id[aid]
        target = by_id.get(env.task.current_targets.get(aid))
        incoming = [missile for missile in env.task.weapon.missiles
                    if missile.alive and missile.target_id == aid]
        actions[aid] = env.task.opponent.act(agent, target, incoming)[0]
    return actions


def rollout(config: Path, backend: str, steps: int, seed: int,
            rule_red: bool = False) -> dict:
    env = make_env(str(config), dynamics_backend=backend)
    rng = np.random.default_rng(seed)
    cumulative = {aid: 0.0 for aid in env.agent_ids}
    info = {}
    completed = 0
    finite = True
    try:
        _, info = env.reset(seed=seed)
        for _ in range(steps):
            if rule_red:
                actions = _rule_actions(env)
            else:
                actions = {aid: rng.integers(0, 40, size=4, dtype=np.int64)
                           for aid in env.agent_ids}
            _, rewards, terminated, truncated, info = env.step(actions)
            for aid, value in rewards.items():
                cumulative[aid] += float(value)
            finite = bool(finite and np.isfinite(list(rewards.values())).all()
                          and np.isfinite(env.get_state()).all())
            completed += 1
            if all(terminated.values()) or all(truncated.values()):
                break

        role_returns = {}
        for aid, value in cumulative.items():
            role = env.agent_roles[aid]
            role_returns[role] = role_returns.get(role, 0.0) + value
        metrics = info.get("aircraft_metrics", {})
        missile_types = sorted({type(missile).__name__
                                for missile in env.task.weapon.missiles})
        aircraft_backend_types = sorted({type(agent).__name__
                                         for agent in env.task.agents})
        return {
            "requested_backend": backend,
            "actual_backend": info.get("backend"),
            "aircraft_backend_types": aircraft_backend_types,
            "seed": seed,
            "steps": completed,
            "winner": info.get("winner"),
            "termination_reason": info.get("termination_reason"),
            "shotdown": info.get("shotdown", 0),
            "crash": info.get("crashes", 0),
            "out_of_zone": info.get("out_of_zone", 0),
            "structural_failure": info.get("structural_failures", 0),
            "missiles_fired": info.get("missiles_fired", 0),
            "hits": info.get("missile_hits", 0),
            "maximum_speed_mps": max(
                (item["max_speed_mps"] for item in metrics.values()), default=0.0),
            "maximum_abs_load_factor_g": max(
                (item["max_abs_load_factor_g"] for item in metrics.values()), default=0.0),
            "speed_violation_count": sum(
                item["speed_violation_count"] for item in metrics.values()),
            "overload_violation_count": sum(
                item["overload_violation_count"] for item in metrics.values()),
            "finite": finite,
            "cumulative_reward_by_agent": cumulative,
            "cumulative_reward_by_role": role_returns,
            "missile_object_types": missile_types,
        }
    finally:
        env.close()


def _run_pytest(paths: list[str]) -> dict:
    result = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q"],
                            cwd=ROOT, text=True, capture_output=True)
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _introspect(config: Path, backend: str) -> dict:
    env = make_env(str(config), dynamics_backend=backend)
    try:
        env.reset(seed=0)
        objects = {
            "environment": env,
            "reward": env.task.reward,
            "observation": env.task.observation,
            "weapon": env.task.weapon,
            "opponent": env.task.opponent,
        }
        legacy_found = {
            name: sorted(attr for attr in LEGACY_ATTRIBUTES if hasattr(obj, attr))
            for name, obj in objects.items()
        }
        class_checks = {
            "environment": isinstance(env, TAMPaperEnv),
            "reward": isinstance(env.task.reward, PaperReward),
            "observation": isinstance(env.task.observation, PaperObservation),
            "weapon": isinstance(env.task.weapon, PaperWeaponManager),
            "missile_class": PaperMissile.__module__ == "uav_env.JSBSim.paper.missile",
        }
        return {
            "class_names": {name: type(obj).__name__ for name, obj in objects.items()},
            "class_checks": class_checks,
            "legacy_attributes_found": legacy_found,
            "passed": all(class_checks.values())
                      and not any(legacy_found.values()),
        }
    finally:
        env.close()


def _config_consumption(config: dict) -> list[dict]:
    rows = []
    for source_key, published in (("published_parameters", True),
                                  ("inferred_parameters", False)):
        for name in sorted(config[source_key]):
            rows.append({
                "parameter": name,
                "source": source_key,
                "consumer": PARAMETER_CONSUMERS.get(name, "paper environment contract"),
                "published": published,
                "inferred": not published,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/tam_paper_env_v1_audit.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=("simple", "jsbsim"), default="jsbsim")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    config_dir = ROOT / "uav_env" / "JSBSim" / "configs"
    configs = [config_dir / name for name in CONFIG_NAMES]
    config = yaml.safe_load(configs[1].read_text(encoding="utf-8"))
    environment_tests = ({"passed": False, "skipped": True}
                         if args.skip_tests else _run_pytest([
                             "tests/test_tam_paper_env_v1_contract.py",
                             "tests/test_tam_paper_env_v1_phase2.py",
                         ]))
    happo_smoke = ({"passed": False, "skipped": True}
                   if args.skip_tests else _run_pytest([
                       "tests/test_tam_paper_env_v1_integration.py::"
                       "test_plain_happo_rollout_and_one_update_without_tam_network",
                   ]))
    introspection = _introspect(configs[1], args.backend)

    random_rollouts = {}
    rule_rollouts = {}
    for config_index, config_path in enumerate(configs):
        random_rollouts[config_path.stem] = [
            rollout(config_path, args.backend, args.steps,
                    args.seed + config_index * 100 + episode)
            for episode in range(args.episodes_per_scenario)
        ]
        rule_rollouts[config_path.stem] = [
            rollout(config_path, args.backend, args.steps,
                    args.seed + 10000 + config_index * 100 + episode, True)
            for episode in range(args.episodes_per_scenario)
        ]

    all_rollouts = [item for group in (*random_rollouts.values(), *rule_rollouts.values())
                    for item in group]
    real_jsbsim = (args.backend == "jsbsim"
                   and all(item["actual_backend"] == "jsbsim" for item in all_rollouts))
    stable = all(item["finite"] and item["steps"] > 0 for item in all_rollouts)
    rule_items = [item for group in rule_rollouts.values() for item in group]
    rule_combat = (all(any(item["missiles_fired"] > 0 for item in group)
                       for group in rule_rollouts.values())
                   and sum(item["hits"] for item in rule_items) > 0
                   and any(item["termination_reason"] for item in rule_items))
    environment_ready = bool(environment_tests["passed"] and introspection["passed"]
                             and real_jsbsim and stable and rule_combat)
    happo_ready = bool(happo_smoke["passed"])

    try:
        jsbsim_version = importlib.metadata.version("jsbsim")
    except importlib.metadata.PackageNotFoundError:
        jsbsim_version = None
    report = {
        "formal_environment_mode": "tam_paper_env_v1",
        "runtime": {
            "requested_backend": args.backend,
            "actual_backend": "jsbsim" if real_jsbsim else "not_all_jsbsim",
            "jsbsim_version": jsbsim_version,
            "python_version": platform.python_version(),
            "simple_fallback_used": args.backend != "jsbsim" or not real_jsbsim,
            "parameters": {
                "steps": args.steps,
                "episodes_per_scenario": args.episodes_per_scenario,
                "seed": args.seed,
            },
        },
        "config_consumption_table": _config_consumption(config),
        "runtime_introspection": introspection,
        "environment_contract_tests": environment_tests,
        "HAPPO_INTERFACE_SMOKE": happo_smoke,
        "random_rollout_statistics": random_rollouts,
        "rule_vs_rule_statistics": rule_rollouts,
        "acceptance_evidence": {
            "all_rollouts_finite": stable,
            "all_rollouts_real_jsbsim": real_jsbsim,
            "rule_combat_exercised": rule_combat,
        },
        "ENVIRONMENT_CONTRACT_READY": environment_ready,
        "HAPPO_INTERFACE_READY": happo_ready,
        "LEARNING_NOT_YET_VALIDATED": True,
        "READY_FOR_HAPPO": bool(environment_ready and happo_ready),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["READY_FOR_HAPPO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
