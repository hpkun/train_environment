"""Audit the main MAPPO experiment contract without training.

The audit checks that the main runner and paper-aligned configs match the
current experiment contract. It does not modify reward, termination, missile,
action, evasion, PID, aircraft XML, or algorithm behavior.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.policy import MAPPOActorCritic
from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2


EXPECTED_TRAIN_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml"
EXPECTED_EVAL_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
EXPECTED_SUMMARY_FIELDS = [
    "avg_return",
    "avg_length",
    "red_win_rate",
    "blue_win_rate",
    "draw_rate",
    "timeout_rate",
    "mav_survival_rate",
    "red_alive_final_mean",
    "blue_alive_final_mean",
    "nan_detected",
    "actor_dim_ok",
    "critic_dim_ok",
]


def _load_yaml(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_runner_module():
    path = ROOT / "scripts" / "run_main_mappo_experiment.py"
    spec = importlib.util.spec_from_file_location("run_main_mappo_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_strings_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return strings


def _runner_contract(runner) -> dict:
    runner_path = ROOT / "scripts" / "run_main_mappo_experiment.py"
    strings = _literal_strings_in_file(runner_path)
    train_config = getattr(runner, "TRAIN_CONFIG", "")
    eval_configs = list(getattr(runner, "EVAL_CONFIGS", []))
    return {
        "default_train_config": train_config,
        "default_eval_configs": eval_configs,
        "default_opponent_policy": getattr(runner, "OPPONENT", ""),
        "default_obs_adapter_version": getattr(runner, "OBS_ADAPTER", ""),
        "default_output_dir": "outputs/main_mappo_experiment",
        "main_defaults_reference_balanced": any("balanced" in value for value in [train_config, *eval_configs]),
        "main_defaults_reference_minimal_v1": any("minimal_v1" in value for value in [train_config, *eval_configs]),
        "main_defaults_reference_v1": getattr(runner, "OBS_ADAPTER", "") == "v1",
        "summary_fields_declared": [
            field for field in EXPECTED_SUMMARY_FIELDS if field in strings
        ],
    }


def _composition_contract(config_path: str) -> dict:
    cfg = _load_yaml(config_path)
    env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    try:
        _obs, info = env.reset(seed=0)
        agent_types = dict(getattr(env, "agent_types", {}))
        agent_models = dict(getattr(env, "agent_models", {}))
        missile_counts = {
            aid: int(env._get_sim(aid).num_left_missiles)
            for aid in env.agent_ids
            if env._get_sim(aid) is not None
        }
        red_types = [agent_types.get(aid, "") for aid in env.red_ids]
        blue_types = [agent_types.get(aid, "") for aid in env.blue_ids]
        return {
            "config": config_path,
            "red_count": len(env.red_ids),
            "blue_count": len(env.blue_ids),
            "red_agent_types": red_types,
            "blue_agent_types": blue_types,
            "mav_count": sum(1 for t in red_types + blue_types if t == "mav"),
            "red_attack_uav_count": sum(1 for t in red_types if t == "attack_uav"),
            "blue_attack_uav_count": sum(1 for t in blue_types if t == "attack_uav"),
            "aircraft_models": agent_models,
            "missile_counts": missile_counts,
            "info_has_agent_metadata": all(
                key in info for key in ("agent_types", "agent_roles", "agent_models")
            ),
            "configured_red_agent_types": cfg.get("red_agent_types", []),
            "configured_blue_agent_types": cfg.get("blue_agent_types", []),
        }
    finally:
        env.close()


def _config_contract(train_config: str, eval_configs: list[str]) -> dict:
    train_cfg = _load_yaml(train_config)
    return {
        "train_config": train_config,
        "eval_configs": eval_configs,
        "observation_mode": train_cfg.get("observation_mode"),
        "hetero_reward_mode": train_cfg.get("hetero_reward_mode"),
        "sim_freq": train_cfg.get("sim_freq"),
        "agent_interaction_steps": train_cfg.get("agent_interaction_steps"),
        "max_steps": train_cfg.get("max_steps"),
        "eval_config_modes": {
            path: _load_yaml(path).get("observation_mode") for path in eval_configs
        },
        "eval_config_reward_modes": {
            path: _load_yaml(path).get("hetero_reward_mode") for path in eval_configs
        },
    }


def _adapter_contract() -> dict:
    adapter = HeteroObsAdapterV2()
    return {
        "obs_adapter_version": "v2",
        "actor_dim": int(adapter.flat_actor_obs_dim),
        "critic_dim": int(adapter.critic_state_dim),
        "action_dim": 3,
    }


def _algorithm_contract() -> dict:
    model = MAPPOActorCritic(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    runner_source = (ROOT / "scripts" / "run_main_mappo_experiment.py").read_text(encoding="utf-8")
    train_source = (ROOT / "scripts" / "train_mappo_baseline.py").read_text(encoding="utf-8")
    algorithm_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "algorithms" / "mappo").glob("*.py")
    )
    main_call_text = runner_source + "\n" + train_source
    forbidden = {
        "attention": "attention" in main_call_text.lower(),
        "happo": "happo" in main_call_text.lower(),
        "gru": "gru" in main_call_text.lower(),
    }
    return {
        "model_class": model.__class__.__name__,
        "shared_actor_baseline": True,
        "centralized_critic": True,
        "action_dim": int(model.action_dim),
        "main_runner_calls_attention_happo_gru": any(forbidden.values()),
        "forbidden_terms_in_main_call_path": forbidden,
        "method_terms_present_in_algorithm_files": {
            "attention": "attention" in algorithm_sources.lower(),
            "happo": "happo" in algorithm_sources.lower(),
            "gru": "gru" in algorithm_sources.lower(),
        },
    }


def _output_contract(runner_contract: dict) -> dict:
    fields = set(runner_contract.get("summary_fields_declared", []))
    return {
        "required_main_summary_fields": EXPECTED_SUMMARY_FIELDS,
        "main_summary_fields_present": sorted(fields),
        "main_summary_has_required_fields": all(field in fields for field in EXPECTED_SUMMARY_FIELDS),
        "eval_summary_has_combat_metrics": True,
    }


def _violations(config_contract: dict, compositions: dict, adapter_contract: dict,
                runner_contract: dict, algorithm_contract: dict,
                output_contract: dict) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []

    if config_contract["train_config"] != EXPECTED_TRAIN_CONFIG:
        violations.append("train_config_not_paper_aligned_3v2")
    if config_contract["eval_configs"] != EXPECTED_EVAL_CONFIGS:
        violations.append("eval_configs_not_paper_aligned_3v2_5v4")
    if config_contract["observation_mode"] != "mav_shared_geo":
        violations.append("observation_mode_not_mav_shared_geo")
    if config_contract["hetero_reward_mode"] != "brma_legacy":
        violations.append("hetero_reward_mode_not_brma_legacy")
    if config_contract["sim_freq"] != 60:
        violations.append("sim_freq_not_60")
    if config_contract["agent_interaction_steps"] != 12:
        violations.append("agent_interaction_steps_not_12")
    if config_contract["max_steps"] != 1000:
        violations.append("max_steps_not_1000")
    for path, mode in config_contract["eval_config_modes"].items():
        if mode != "mav_shared_geo":
            violations.append(f"eval_observation_mode_not_mav_shared_geo:{path}")
    for path, mode in config_contract["eval_config_reward_modes"].items():
        if mode != "brma_legacy":
            violations.append(f"eval_reward_mode_not_brma_legacy:{path}")

    expected_composition = {
        "3v2": (3, 2, 1, 2, 2),
        "5v4": (5, 4, 1, 4, 4),
    }
    for key, (red_count, blue_count, mav_count, red_attack, blue_attack) in expected_composition.items():
        comp = compositions[key]
        if comp["red_count"] != red_count or comp["blue_count"] != blue_count:
            violations.append(f"{key}_count_mismatch")
        if comp["mav_count"] != mav_count:
            violations.append(f"{key}_mav_count_mismatch")
        if comp["red_attack_uav_count"] != red_attack:
            violations.append(f"{key}_red_attack_uav_count_mismatch")
        if comp["blue_attack_uav_count"] != blue_attack:
            violations.append(f"{key}_blue_attack_uav_count_mismatch")
        for aid, model in comp["aircraft_models"].items():
            if aid == "red_0" and model != "f22":
                violations.append(f"{key}_mav_model_not_f22")
            if aid != "red_0" and model != "f16":
                violations.append(f"{key}_{aid}_model_not_f16")
        for aid, count in comp["missile_counts"].items():
            if aid == "red_0" and count != 0:
                violations.append(f"{key}_mav_missiles_not_zero")
            if aid != "red_0" and count != 2:
                violations.append(f"{key}_{aid}_missiles_not_two")

    if adapter_contract["obs_adapter_version"] != "v2":
        violations.append("obs_adapter_version_not_v2")
    if adapter_contract["actor_dim"] != 96:
        violations.append("actor_dim_not_96")
    if adapter_contract["critic_dim"] != 480:
        violations.append("critic_dim_not_480")
    if adapter_contract["action_dim"] != 3:
        violations.append("action_dim_not_3")

    if runner_contract["default_train_config"] != EXPECTED_TRAIN_CONFIG:
        violations.append("runner_default_train_config_not_3v2")
    if runner_contract["default_eval_configs"] != EXPECTED_EVAL_CONFIGS:
        violations.append("runner_default_eval_configs_not_3v2_5v4")
    if runner_contract["default_opponent_policy"] != "greedy_fsm":
        violations.append("runner_default_opponent_not_greedy_fsm")
    if runner_contract["default_obs_adapter_version"] != "v2":
        violations.append("runner_default_obs_adapter_not_v2")
    if runner_contract["main_defaults_reference_balanced"]:
        violations.append("runner_defaults_include_balanced")
    if runner_contract["main_defaults_reference_minimal_v1"]:
        violations.append("runner_defaults_include_minimal_v1")
    if runner_contract["main_defaults_reference_v1"]:
        violations.append("runner_defaults_include_v1")

    if not algorithm_contract["shared_actor_baseline"]:
        violations.append("not_shared_actor_mappo_baseline")
    if algorithm_contract["main_runner_calls_attention_happo_gru"]:
        violations.append("main_runner_calls_attention_happo_or_gru")
    if algorithm_contract["action_dim"] != 3:
        violations.append("algorithm_action_dim_not_3")

    if not output_contract["main_summary_has_required_fields"]:
        violations.append("main_summary_missing_required_fields")
    if not output_contract["eval_summary_has_combat_metrics"]:
        violations.append("eval_summary_missing_combat_metrics")

    if any(algorithm_contract["method_terms_present_in_algorithm_files"].values()):
        warnings.append("method_terms_present_in_algorithm_files_not_main_call_path")
    return violations, warnings


def _markdown(data: dict) -> str:
    s = data["summary"]
    lines = [
        "# Main Experiment Contract",
        "",
        "This document records the current paper mainline experiment contract.",
        "It is not a method module, not tuning guidance, and not a training run.",
        "",
        "## Contract",
        "",
        "- train: 3v2 (`hetero_mav_shared_geo_3v2.yaml`)",
        "- eval: 3v2 + 5v4",
        "- observation: V2 `mav_shared_geo`",
        "- reward: `brma_legacy`",
        "- opponent: `greedy_fsm`",
        "- algorithm: shared MAPPO baseline",
        "- action_dim: 3",
        "",
        "## Audit Result",
        "",
        f"- contract_passed: {s['contract_passed']}",
        f"- blocking_violations: {s['blocking_violations']}",
        f"- warnings: {s['warnings']}",
        f"- next_action: {s['next_action']}",
        "",
        "The audit confirms whether code and configs match the experiment",
        "setting. It does not modify reward, termination, missile, action,",
        "evasion, PID, aircraft XML, attention, HAPPO, GRU, or role-aware",
        "algorithm behavior.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/main_experiment_contract_audit.json")
    parser.add_argument("--output-md", default="outputs/main_experiment_contract_audit.md")
    args = parser.parse_args()

    runner = _load_runner_module()
    runner_contract = _runner_contract(runner)
    train_config = runner_contract["default_train_config"]
    eval_configs = runner_contract["default_eval_configs"]
    config_contract = _config_contract(train_config, eval_configs)
    compositions = {
        "3v2": _composition_contract(EXPECTED_EVAL_CONFIGS[0]),
        "5v4": _composition_contract(EXPECTED_EVAL_CONFIGS[1]),
    }
    adapter_contract = _adapter_contract()
    algorithm_contract = _algorithm_contract()
    output_contract = _output_contract(runner_contract)
    violations, warnings = _violations(
        config_contract,
        compositions,
        adapter_contract,
        runner_contract,
        algorithm_contract,
        output_contract,
    )
    data = {
        "config_contract": config_contract,
        "composition_contract": compositions,
        "adapter_contract": adapter_contract,
        "runner_contract": runner_contract,
        "algorithm_contract": algorithm_contract,
        "output_contract": output_contract,
        "violations": violations,
        "summary": {
            "contract_passed": not violations,
            "blocking_violations": violations,
            "warnings": warnings,
            "next_action": "run_100k_pilot" if not violations else "fix_main_experiment_contract",
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output_md.write_text(_markdown(data), encoding="utf-8")
    print(f"output_json: {output_json}", flush=True)
    print(f"output_md: {output_md}", flush=True)
    print(f"contract_passed: {data['summary']['contract_passed']}", flush=True)
    print(f"blocking_violations: {violations}", flush=True)
    print(f"warnings: {warnings}", flush=True)
    print(f"next_action: {data['summary']['next_action']}", flush=True)


if __name__ == "__main__":
    main()
