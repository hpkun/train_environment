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
from uav_env.JSBSim.core.aircraft import JSBSimAircraftPlatform
from uav_env.make_env import make_env
from scripts.vanilla_happo_runtime import stack_controlled_rule_actions
from uav_env.JSBSim.paper.protocol import (
    BLUE_POLICY_FIDELITY, ENVIRONMENT_FIDELITY_REVISION,
    GENERALIZATION_PERTURBATION_LEVELS,
    NOMINAL_PERTURBATION, derived_environment_values, validate_generalization_protocol,
    validate_nominal_protocol, REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED)
from uav_env.JSBSim.paper.missile import TIMEOUT_DERIVATION


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
    "missile_mass_kg": "paper.missile published metadata only; not used in point-mass dynamics",
    "missile_length_m": "paper.missile published metadata only; not used in point-mass dynamics",
    "missile_diameter_m": "paper.missile published metadata only; not used in point-mass dynamics",
    "minimum_safe_altitude_m": "paper.reward",
}


def _rule_actions(env) -> dict[str, np.ndarray]:
    env.prepare_decision_context()
    stacked = stack_controlled_rule_actions(env)
    return {aid: stacked[index] for index, aid in enumerate(env.agent_ids)}


def rollout(config: Path, backend: str, steps: int, seed: int,
            rule_red: bool = False) -> dict:
    env = make_env(str(config), dynamics_backend=backend)
    rng = np.random.default_rng(seed)
    cumulative = {aid: 0.0 for aid in env.agent_ids}
    info = {}
    completed = 0
    finite = True
    reward_components_finite = True
    target_consistency_violations = 0
    event_ordering_consistent = True
    event_metadata_complete = True
    simulation_time_consistent = True
    terminated_or_truncated = False
    try:
        _, info = env.reset(seed=seed)
        initial_control_status = {
            agent.agent_id: agent.control_initialization_status()
            for agent in env.task.agents
            if isinstance(agent, JSBSimAircraftPlatform)
        }
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
            reward_components_finite = bool(
                reward_components_finite and all(
                    np.isfinite(list(agent_components.values())).all()
                    for agent_components in info["reward_components"].values()))
            target_consistency_violations += len(
                info.get("target_consistency_violation", []))
            event_ordering_consistent = bool(
                event_ordering_consistent
                and info.get("event_ordering_consistent", False))
            ordered_events = [event for event in info.get("missile_events", [])
                              if event.get("reason") == "hit"
                              or event.get("event_type") == "aircraft_death"]
            event_metadata_complete = bool(
                event_metadata_complete and all(
                    all(key in event for key in (
                        "physics_frame_index", "simulation_time_s", "event_sequence_id"))
                    for event in ordered_events))
            completed += 1
            simulation_time_consistent = bool(
                simulation_time_consistent
                and np.isclose(info["simulation_time_s"], completed * 0.2))
            if all(terminated.values()) or all(truncated.values()):
                terminated_or_truncated = True
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
            "terminated_or_truncated": terminated_or_truncated,
            "within_episode_limit": completed <= env.task.episode_limit,
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
            "speed_limit_exceedance_count": sum(
                item["speed_limit_exceedance_count"] for item in metrics.values()),
            "overload_limit_exceedance_count": sum(
                item["overload_limit_exceedance_count"] for item in metrics.values()),
            "finite": finite,
            "reward_components_finite": reward_components_finite,
            "target_consistency_violations": target_consistency_violations,
            "event_ordering_consistent": event_ordering_consistent,
            "event_metadata_complete": event_metadata_complete,
            "simulation_time_consistent": simulation_time_consistent,
            "simulation_time_s": info.get("simulation_time_s"),
            "initial_control_status": initial_control_status,
            "throttle_adapter_status": sorted({
                status["throttle_adapter"] for status in initial_control_status.values()}),
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
            "all_aircraft_jsbsim": all(isinstance(agent, JSBSimAircraftPlatform)
                                        for agent in env.task.agents),
        }
        return {
            "class_names": {name: type(obj).__name__ for name, obj in objects.items()},
            "class_checks": class_checks,
            "legacy_attributes_found": legacy_found,
            "initial_control_status": {
                agent.agent_id: agent.control_initialization_status()
                for agent in env.task.agents
                if isinstance(agent, JSBSimAircraftPlatform)
            },
            "throttle_adapter_status": "command_only_no_adapter",
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
                "direct_dynamics_use": name not in {
                    "missile_mass_kg", "missile_length_m", "missile_diameter_m"},
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
    for scenario in ("2v2", "3v2", "5v4"):
        validate_nominal_protocol(scenario, NOMINAL_PERTURBATION)
    for level in GENERALIZATION_PERTURBATION_LEVELS:
        validate_generalization_protocol("5v4", level)
    environment_tests = ({"passed": False, "skipped": True}
                         if args.skip_tests else _run_pytest([
                             "tests/test_tam_paper_env_v1_contract.py",
                             "tests/test_tam_paper_env_v1_phase2.py",
                             "tests/test_tam_paper_env_v1_final_consistency.py",
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
    stable = all(item["finite"] and item["reward_components_finite"]
                 and item["steps"] > 0 for item in all_rollouts)
    complete_episodes = all(
        item["terminated_or_truncated"] and item["termination_reason"]
        and item["within_episode_limit"] for item in all_rollouts)
    target_consistent = all(
        item["target_consistency_violations"] == 0 for item in all_rollouts)
    event_ordering_consistent = all(
        item["event_ordering_consistent"] and item["event_metadata_complete"]
        for item in all_rollouts)
    simulation_time_consistent = all(
        item["simulation_time_consistent"] for item in all_rollouts)
    aircraft_classes_valid = all(
        item["aircraft_backend_types"] == ["JSBSimAircraftPlatform"]
        for item in all_rollouts)
    missile_classes_valid = all(
        item["missiles_fired"] == 0 or item["missile_object_types"] == ["PaperMissile"]
        for item in all_rollouts)
    initial_controls_valid = all(
        all(status["gear_command_norm"] == 0.0
            and status["gear_position_norm"] == 0.0
            and status["flap_command_norm"] == 0.0
            and status["flap_position_norm"] == 0.0
            and all(status["engine_running"])
            and status["throttle_adapter"] == "command_only_no_adapter"
            for status in item["initial_control_status"].values())
        for item in all_rollouts)
    rule_items = [item for group in rule_rollouts.values() for item in group]
    rule_combat = all(sum(item["hits"] for item in group) > 0
                      for group in rule_rollouts.values())
    environment_ready = bool(environment_tests["passed"] and introspection["passed"]
                             and real_jsbsim and stable and complete_episodes
                             and rule_combat and target_consistent
                             and event_ordering_consistent
                             and simulation_time_consistent
                             and aircraft_classes_valid and missile_classes_valid
                             and initial_controls_valid)
    happo_ready = bool(happo_smoke["passed"])

    try:
        jsbsim_version = importlib.metadata.version("jsbsim")
    except importlib.metadata.PackageNotFoundError:
        jsbsim_version = None
    report = {
        "formal_environment_mode": "tam_paper_env_v1",
        "environment_fidelity_revision": ENVIRONMENT_FIDELITY_REVISION,
        "blue_policy_fidelity": BLUE_POLICY_FIDELITY,
        "reference_8_exact_blue_fsm_reproduced": (
            REFERENCE_8_EXACT_BLUE_FSM_REPRODUCED),
        "experiment_protocol_validation": {
            "paper_nominal_none_only": True,
            "paper_5v4_generalization_levels": list(GENERALIZATION_PERTURBATION_LEVELS),
        },
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
        "derived_environment_parameters": derived_environment_values(
            config["published_parameters"]["maximum_attack_range_m"]),
        "missile_timeout": {
            "classification": "PAPER_SILENT_DERIVED_SIMPLIFICATION",
            "timeout_derivation": TIMEOUT_DERIVATION,
            "derived_timeout_s": (
                2.0 * config["published_parameters"]["maximum_attack_range_m"]
                / config["inferred_parameters"]["missile_initial_speed_mps"]),
        },
        "missile_published_metadata": {
            key: config["published_parameters"][key] for key in (
                "missile_mass_kg", "missile_length_m", "missile_diameter_m")
        },
        "runtime_introspection": introspection,
        "environment_contract_tests": environment_tests,
        "HAPPO_INTERFACE_SMOKE": happo_smoke,
        "random_rollout_statistics": random_rollouts,
        "rule_vs_rule_statistics": rule_rollouts,
        "acceptance_evidence": {
            "all_rollouts_finite": stable,
            "all_rollouts_real_jsbsim": real_jsbsim,
            "rule_combat_exercised": rule_combat,
            "all_episodes_terminated_or_truncated": complete_episodes,
            "target_consistency_violation_count": sum(
                item["target_consistency_violations"] for item in all_rollouts),
            "event_ordering_consistent": event_ordering_consistent,
            "event_metadata_complete": all(
                item["event_metadata_complete"] for item in all_rollouts),
            "simulation_time_consistent": simulation_time_consistent,
            "all_aircraft_objects_jsbsim": aircraft_classes_valid,
            "all_launched_missiles_paper_missile": missile_classes_valid,
            "all_reward_components_finite": all(
                item["reward_components_finite"] for item in all_rollouts),
            "initial_gear_flap_engine_valid": initial_controls_valid,
            "throttle_adapter_status": "command_only_no_adapter",
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
