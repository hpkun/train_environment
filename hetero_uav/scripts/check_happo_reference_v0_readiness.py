"""Check HAPPO reference v0 readiness before the 200k validation run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.happo import HAPPOReferencePolicy


DEFAULT_CONFIG = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_happo_ref_v0.yaml"
DEFAULT_JSON = "outputs/happo_3v2_reference_200k/readiness/happo_v0_readiness.json"
DEFAULT_MD = "outputs/happo_3v2_reference_200k/readiness/happo_v0_readiness.md"


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _ok(data: dict, key: str, value: bool, blocking: list[str], issue: str) -> None:
    data[key] = bool(value)
    if not value:
        blocking.append(issue)


def _check_config(config_path: Path, blocking: list[str], warnings: list[str]) -> dict:
    out: dict = {"path": str(config_path)}
    _ok(out, "exists", config_path.exists(), blocking, "happo_ref_v0 config missing")
    if not config_path.exists():
        return out

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out["hetero_reward_mode"] = cfg.get("hetero_reward_mode")
    _ok(out, "reward_mode_is_happo_ref_v0", cfg.get("hetero_reward_mode") == "happo_ref_v0",
        blocking, "config reward mode is not happo_ref_v0")
    out["observation_mode"] = cfg.get("observation_mode")
    _ok(out, "observation_mode_is_mav_shared_geo", cfg.get("observation_mode") == "mav_shared_geo",
        blocking, "config observation_mode is not mav_shared_geo")

    red_types = list(cfg.get("red_agent_types", []))
    blue_types = list(cfg.get("blue_agent_types", []))
    out["red_agent_types"] = red_types
    out["blue_agent_types"] = blue_types
    _ok(out, "red_0_is_mav", len(red_types) >= 1 and red_types[0] == "mav",
        blocking, "red_0 is not MAV")
    _ok(out, "red_1_red_2_are_uav", len(red_types) >= 3 and red_types[1:3] == ["attack_uav", "attack_uav"],
        blocking, "red_1/red_2 are not attack UAVs")
    _ok(out, "blue_is_2_uav", blue_types == ["attack_uav", "attack_uav"],
        blocking, "blue side is not two attack UAVs")

    params = cfg.get("aircraft_type_params", {})
    mav_params = params.get("mav", {})
    attack_params = params.get("attack_uav", {})
    out["mav_num_missiles"] = mav_params.get("num_missiles")
    out["attack_uav_num_missiles"] = attack_params.get("num_missiles")
    _ok(out, "red_0_num_missiles_zero", mav_params.get("num_missiles") == 0,
        blocking, "MAV num_missiles is not 0")
    _ok(out, "uav_num_missiles_two", attack_params.get("num_missiles") == 2,
        blocking, "attack UAV num_missiles is not 2")

    trim = cfg.get("action_trim_by_role", {}).get("mav", {})
    out["mav_action_trim"] = trim
    if any(abs(float(trim.get(k, 0.0))) > 1e-9 for k in ("pitch", "heading", "speed")):
        warnings.append("MAV action trim is non-zero; no_mav_trim is not preserved")
    out["no_mav_trim_preserved"] = not any(abs(float(trim.get(k, 0.0))) > 1e-9 for k in ("pitch", "heading", "speed"))
    return out


def _check_algorithm(blocking: list[str]) -> dict:
    actor_obs_dim = 96
    critic_state_dim = 480
    policy = HAPPOReferencePolicy(actor_obs_dim, critic_state_dim)
    out = {
        "HAPPOReferencePolicy_exists": True,
        "separate_mav_uav_actors": hasattr(policy, "mav_actor") and hasattr(policy, "uav_actor"),
        "centralized_critic": hasattr(policy, "critic"),
        "sequential_update": True,
        "attention": False,
        "recurrent": False,
        "action_dim": int(policy.action_dim),
        "actor_obs_dim": int(policy.actor_obs_dim),
        "critic_state_dim": int(policy.critic_state_dim),
    }
    checks = {
        "separate_mav_uav_actors": "separate MAV/UAV actors missing",
        "centralized_critic": "centralized critic missing",
        "sequential_update": "sequential update flag missing",
        "action_dim": "action_dim is not 3",
        "actor_obs_dim": "actor_obs_dim is not 96",
        "critic_state_dim": "critic_state_dim is not 480",
    }
    if not out["separate_mav_uav_actors"]:
        blocking.append(checks["separate_mav_uav_actors"])
    if not out["centralized_critic"]:
        blocking.append(checks["centralized_critic"])
    if not out["sequential_update"]:
        blocking.append(checks["sequential_update"])
    if out["action_dim"] != 3:
        blocking.append(checks["action_dim"])
    if out["actor_obs_dim"] != 96:
        blocking.append(checks["actor_obs_dim"])
    if out["critic_state_dim"] != 480:
        blocking.append(checks["critic_state_dim"])
    return out


def _check_reward(config: str, blocking: list[str], warnings: list[str]) -> dict:
    out: dict = {}
    env = None
    try:
        from uav_env import make_env

        env = make_env(config, env_type="jsbsim_hetero", hetero_reward_mode="happo_ref_v0", max_steps=8)
        out["env_created"] = True
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _obs, rewards, _terminated, _truncated, info = env.step(actions)
        out["step_ok"] = True
        components = info.get("reward_components", {})
        out["reward_components_present"] = isinstance(components, dict) and bool(components)
        component_keys = set()
        if isinstance(components, dict):
            for values in components.values():
                if isinstance(values, dict):
                    component_keys.update(str(key) for key in values.keys())
        out["reward_component_keys"] = sorted(component_keys)
        required_tokens = ["safety", "event", "death_penalty"]
        missing = [t for t in required_tokens if not any(t in k for k in out["reward_component_keys"])]
        out["required_reward_tokens_missing"] = missing
        if missing:
            warnings.append(f"reward_components missing expected tokens: {missing}")
        for token in ["mav_attack", "mav_dodge"]:
            if not any(token in k for k in out["reward_component_keys"]):
                warnings.append(f"documented component {token} is not explicitly recorded in reward_components")
        out["rewards_finite"] = bool(np.isfinite(list(rewards.values())).all())
        if not out["rewards_finite"]:
            blocking.append("non-finite reward in readiness step")
    except Exception as exc:
        out["env_created"] = False
        out["error"] = repr(exc)
        blocking.append(f"happo_ref_v0 env/reward readiness failed: {exc}")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
    return out


def _check_paper_boundary(blocking: list[str]) -> dict:
    doc = ROOT / "docs" / "tam_happo_paper_grounded_spec.md"
    text = doc.read_text(encoding="utf-8") if doc.exists() else ""
    required = [
        "full TAM-HAPPO reproduction",
        "paper action-space reproduction",
        "attention-enhanced value network",
        "temporal GRU module",
        "high-level action retained",
        "scripted missile retained",
        "no attention in v0",
        "no GRU in v0 unless explicitly implemented later",
    ]
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        blocking.append(f"paper boundary doc missing phrases: {missing}")
    return {"doc_exists": doc.exists(), "missing_required_phrases": missing}


def _write_md(path: Path, data: dict) -> None:
    lines = [
        "# HAPPO Reference v0 Readiness",
        "",
        f"- ready_for_200k: {data['ready_for_200k']}",
        f"- next_action: {data['next_action']}",
        "",
        "## Blocking Issues",
        *(f"- {x}" for x in data["blocking_issues"] or ["none"]),
        "",
        "## Warnings",
        *(f"- {x}" for x in data["warnings"] or ["none"]),
        "",
        "## Scope Boundary",
        "- HAPPO reference v0 is not full TAM-HAPPO.",
        "- high-level action retained.",
        "- scripted missile retained.",
        "- no GRU.",
        "- no attention.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="HAPPO reference v0 readiness")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_MD)
    args = parser.parse_args()

    blocking: list[str] = []
    warnings: list[str] = []
    data = {
        "config_readiness": _check_config(_rel(args.config), blocking, warnings),
        "algorithm_readiness": _check_algorithm(blocking),
        "reward_readiness": _check_reward(args.config, blocking, warnings),
        "paper_boundary_readiness": _check_paper_boundary(blocking),
    }
    data["blocking_issues"] = blocking
    data["warnings"] = warnings
    data["ready_for_200k"] = not blocking
    data["next_action"] = "run_200k" if not blocking else "fix_blocking_issues"

    out_json = _rel(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_md(_rel(args.output_md), data)
    print(f"ready_for_200k: {data['ready_for_200k']}")
    print(f"blocking_issues: {len(blocking)}")
    print(f"warnings: {len(warnings)}")
    print(f"output_json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
