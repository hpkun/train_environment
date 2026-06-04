"""Audit heterogeneous environment protocol readiness without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.adapter_utils import make_obs_adapter
from uav_env import make_env


PAPER_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]
BALANCED_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_mav_shared_geo_4v4.yaml",
]
V1_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_3v3.yaml",
    "uav_env/JSBSim/configs/hetero_balanced_brma_sensor_4v4.yaml",
]
V2_REQUIRED_KEYS = {
    "ego_geo_state",
    "ally_geo_states",
    "enemy_geo_states",
    "enemy_observed_mask",
    "enemy_track_source",
    "ally_alive_mask",
    "enemy_alive_mask",
}


def _contains_nan(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    arr = np.asarray(value)
    return arr.dtype.kind in {"f", "c"} and bool(np.isnan(arr).any())


def _actions(env, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    if mode == "bounded_random":
        return {
            aid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
            for aid in env.agent_ids
        }
    raise ValueError(mode)


def _counts(agent_types: dict[str, str]) -> dict[str, int]:
    red_types = {k: v for k, v in agent_types.items() if k.startswith("red_")}
    blue_types = {k: v for k, v in agent_types.items() if k.startswith("blue_")}
    return {
        "red_attack_uav_count": sum(v == "attack_uav" for v in red_types.values()),
        "blue_attack_uav_count": sum(v == "attack_uav" for v in blue_types.values()),
        "mav_count": sum(v == "mav" for v in red_types.values()),
    }


def _missile_counts(env) -> dict[str, int]:
    out = {}
    for aid, sim in {**env.red_planes, **env.blue_planes}.items():
        out[aid] = int(getattr(sim, "num_left_missiles", -1))
    return out


def _adapter_version(obs_mode: str) -> str:
    return "v2" if obs_mode == "mav_shared_geo" else "v1"


def _expected_protocol_checks(record: dict) -> list[str]:
    warnings = []
    cfg = Path(record["config"]).name
    protocol = record["protocol_type"]
    missiles = record["missile_counts"]
    agent_types = {
        **{aid: t for aid, t in zip(record["red_ids"], record["red_agent_types"])},
        **{aid: t for aid, t in zip(record["blue_ids"], record["blue_agent_types"])},
    }

    def require(cond: bool, msg: str):
        if not cond:
            raise RuntimeError(f"{cfg}: {msg}")

    if protocol == "paper_aligned" and "3v2" in cfg:
        require(record["red_count"] == 3, "paper 3v2 red_count must be 3")
        require(record["blue_count"] == 2, "paper 3v2 blue_count must be 2")
        require(record["red_attack_uav_count"] == 2, "paper 3v2 red attacks must be 2")
        require(record["blue_attack_uav_count"] == 2, "paper 3v2 blue attacks must be 2")
        require(record["mav_count"] == 1, "paper 3v2 mav_count must be 1")
    elif protocol == "paper_aligned" and "5v4" in cfg:
        require(record["red_count"] == 5, "paper 5v4 red_count must be 5")
        require(record["blue_count"] == 4, "paper 5v4 blue_count must be 4")
        require(record["red_attack_uav_count"] == 4, "paper 5v4 red attacks must be 4")
        require(record["blue_attack_uav_count"] == 4, "paper 5v4 blue attacks must be 4")
        require(record["mav_count"] == 1, "paper 5v4 mav_count must be 1")
    elif protocol == "balanced" and "3v3" in cfg:
        require(record["red_count"] == 3 and record["blue_count"] == 3,
                "balanced 3v3 counts must be 3/3")
        require(record["red_attack_uav_count"] == 2, "balanced 3v3 red attacks must be 2")
        require(record["blue_attack_uav_count"] == 3, "balanced 3v3 blue attacks must be 3")
        require(record["mav_count"] == 1, "balanced 3v3 mav_count must be 1")
        warnings.append("balanced protocol has one fewer red attack UAV than blue")
    elif protocol == "balanced" and "4v4" in cfg:
        require(record["red_count"] == 4 and record["blue_count"] == 4,
                "balanced 4v4 counts must be 4/4")
        require(record["red_attack_uav_count"] == 3, "balanced 4v4 red attacks must be 3")
        require(record["blue_attack_uav_count"] == 4, "balanced 4v4 blue attacks must be 4")
        require(record["mav_count"] == 1, "balanced 4v4 mav_count must be 1")
        warnings.append("balanced protocol has one fewer red attack UAV than blue")

    for aid, type_name in agent_types.items():
        if type_name == "mav":
            require(missiles.get(aid) == 0, f"{aid} MAV missiles must be 0")
        if type_name == "attack_uav":
            require(missiles.get(aid) == 2, f"{aid} attack UAV missiles must be 2")

    if protocol == "paper_aligned" and record["max_steps"] < 1000:
        warnings.append("paper-aligned config max_steps < 1000")
    if protocol == "balanced" and record["max_steps"] < 500:
        warnings.append("balanced config max_steps < 500")
    return warnings


def _audit_config(config: str, protocol_type: str, steps: int) -> dict:
    env = None
    record = {
        "config": config,
        "protocol_type": protocol_type,
        "reset_ok": False,
        "zero_step_ok": False,
        "bounded_random_step_ok": False,
        "nan_detected": False,
        "warnings": [],
    }
    rng = np.random.default_rng(0)
    try:
        env = make_env(config, env_type="jsbsim_hetero")
        obs, info = env.reset(seed=0)
        record["reset_ok"] = True
        obs_mode = getattr(env, "observation_mode", "brma_sensor")
        adapter = make_obs_adapter(_adapter_version(obs_mode))
        adapted = adapter.adapt_all(
            obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)

        agent_types = info.get("agent_types", {})
        counts = _counts(agent_types)
        missile_counts = _missile_counts(env)
        record.update({
            "observation_mode": obs_mode,
            "red_ids": list(env.red_ids),
            "blue_ids": list(env.blue_ids),
            "red_count": len(env.red_ids),
            "blue_count": len(env.blue_ids),
            "red_attack_uav_count": counts["red_attack_uav_count"],
            "blue_attack_uav_count": counts["blue_attack_uav_count"],
            "mav_count": counts["mav_count"],
            "red_agent_types": [agent_types[aid] for aid in env.red_ids],
            "blue_agent_types": [agent_types[aid] for aid in env.blue_ids],
            "agent_models": info.get("agent_models", {}),
            "missile_counts": missile_counts,
            "max_steps": int(getattr(env, "max_steps", 0)),
            "sim_freq": int(getattr(env, "sim_freq", 0)),
            "agent_interaction_steps": int(getattr(env, "agent_interaction_steps", 0)),
            "decision_dt": (
                float(getattr(env, "agent_interaction_steps", 0))
                / float(getattr(env, "sim_freq", 1))
            ),
            "actor_dim": int(adapter.flat_actor_obs_dim),
            "critic_dim": int(adapter.critic_state_dim),
            "adapter_version": _adapter_version(obs_mode),
            "red_valid_mask": adapted["red_valid_mask"].astype(int).tolist(),
        })

        if obs_mode == "mav_shared_geo":
            missing_by_agent = {
                aid: sorted(V2_REQUIRED_KEYS.difference(obs[aid]))
                for aid in env.agent_ids
                if V2_REQUIRED_KEYS.difference(obs[aid])
            }
            if missing_by_agent:
                raise RuntimeError(f"{config}: missing V2 keys {missing_by_agent}")
            if adapter.flat_actor_obs_dim != 96 or adapter.critic_state_dim != 480:
                raise RuntimeError(f"{config}: V2 dimensions mismatch")
        else:
            if adapter.flat_actor_obs_dim != 140 or adapter.critic_state_dim != 700:
                raise RuntimeError(f"{config}: V1 dimensions mismatch")

        if _contains_nan(obs) or _contains_nan(adapted):
            record["nan_detected"] = True
            raise RuntimeError(f"{config}: NaN after reset/adapter")

        for mode in ("zero", "bounded_random"):
            for _ in range(steps):
                obs, rewards, terminated, truncated, info = env.step(
                    _actions(env, mode, rng))
                adapted = adapter.adapt_all(
                    obs, info=info, red_ids=env.red_ids, blue_ids=env.blue_ids)
                if _contains_nan(obs) or _contains_nan(adapted) or _contains_nan(rewards):
                    record["nan_detected"] = True
                    raise RuntimeError(f"{config}: NaN during {mode} step")
            if mode == "zero":
                record["zero_step_ok"] = True
            else:
                record["bounded_random_step_ok"] = True

        record["warnings"].extend(_expected_protocol_checks(record))
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        if env is not None:
            env.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json",
                        default="outputs/environment_audit/hetero_environment_readiness.json")
    parser.add_argument("--include-v1", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    configs: list[tuple[str, str]] = []
    configs.extend((cfg, "paper_aligned") for cfg in PAPER_CONFIGS)
    configs.extend((cfg, "balanced") for cfg in BALANCED_CONFIGS)
    if args.include_v1:
        configs.extend((cfg, "v1_balanced") for cfg in V1_CONFIGS)

    records = [_audit_config(cfg, protocol, args.steps) for cfg, protocol in configs]
    failed = [r for r in records if r.get("error")]
    warning_count = sum(len(r.get("warnings", [])) for r in records)
    passed = [r for r in records if not r.get("error")]
    output = {
        "records": records,
        "summary": {
            "configs_checked": len(records),
            "passed_configs": len(passed),
            "failed_configs": len(failed),
            "warnings": warning_count,
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"configs_checked: {len(records)}")
    print(f"passed_configs: {len(passed)}")
    print(f"warnings: {warning_count}")
    print(f"failed_configs: {len(failed)}")
    print(f"output_json: {out_path}")
    if failed:
        for record in failed:
            print(f"FAILED {record['config']}: {record.get('error')}")
        raise RuntimeError("hetero environment readiness audit failed")


if __name__ == "__main__":
    main()
