"""Audit F-22 MAV model resources and mainline hetero env smoke behavior.

This script does not train and does not modify reward, termination, missile,
action, evasion, PID, aircraft XML, or MAPPO code.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env
from uav_env.JSBSim.adapters.hetero_obs_adapter_v2 import HeteroObsAdapterV2
from uav_env.JSBSim.pid_controller import PIDController
from uav_env.JSBSim.simulator import AircraftSimulator, jsbsim
from uav_env.JSBSim.utils import get_package_data_dir


MAIN_CONFIGS = [
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml",
    "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml",
]


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


def _aircraft_refs(xml_path: Path) -> dict:
    refs = {"engine": [], "thruster": [], "system": [], "include": []}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for tag in refs:
        for elem in root.iter(tag):
            file_ref = elem.attrib.get("file")
            if file_ref:
                refs[tag].append(file_ref)
    return refs


def _resolve_ref(kind: str, ref: str, aircraft_dir: Path) -> Path:
    name = ref if ref.endswith(".xml") else f"{ref}.xml"
    if kind in {"engine", "thruster"}:
        return ROOT / "uav_env" / "JSBSim" / "data" / "engine" / name
    if kind == "system":
        return aircraft_dir / "Systems" / name
    return aircraft_dir / name


def audit_resources() -> dict:
    aircraft_dir = ROOT / "uav_env" / "JSBSim" / "data" / "aircraft" / "f22"
    xml_path = aircraft_dir / "f22.xml"
    refs = _aircraft_refs(xml_path) if xml_path.exists() else {
        "engine": [], "thruster": [], "system": [], "include": []
    }
    missing: list[str] = []
    for kind, values in refs.items():
        for value in values:
            path = _resolve_ref(kind, value, aircraft_dir)
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
    return {
        "f22_folder_exists": aircraft_dir.exists(),
        "f22_model_exists": xml_path.exists(),
        "f22_xml": str(xml_path.relative_to(ROOT)),
        "has_systems_dir": (aircraft_dir / "Systems").exists(),
        "engine_refs": refs["engine"],
        "thruster_refs": refs["thruster"],
        "system_refs": refs["system"],
        "include_refs": refs["include"],
        "missing_dependencies": missing,
        "f119_exists": (ROOT / "uav_env/JSBSim/data/engine/F119-PW-1.xml").exists(),
        "direct_thruster_exists": (ROOT / "uav_env/JSBSim/data/engine/direct.xml").exists(),
    }


def audit_jsbsim_load() -> dict:
    out = {"f22_load_model_ok": False, "f22_run_ic_ok": False, "nan_detected": False, "error": None}
    try:
        fdm = jsbsim.FGFDMExec(get_package_data_dir())
        fdm.set_debug_level(0)
        out["f22_load_model_ok"] = bool(fdm.load_model("f22"))
        fdm.set_property_value("ic/long-gc-deg", 120.0)
        fdm.set_property_value("ic/lat-geod-deg", 60.0)
        fdm.set_property_value("ic/h-sl-ft", 20000.0)
        fdm.set_property_value("ic/psi-true-deg", 0.0)
        fdm.set_property_value("ic/u-fps", 800.0)
        out["f22_run_ic_ok"] = bool(fdm.run_ic())
        values = [
            fdm.get_property_value("position/h-sl-ft"),
            fdm.get_property_value("velocities/vt-fps"),
            fdm.get_property_value("attitude/theta-rad"),
        ]
        out["nan_detected"] = any(not np.isfinite(v) for v in values)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _config_contract(config_path: str) -> dict:
    cfg = _read_yaml(config_path)
    env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
    try:
        _obs, info = env.reset(seed=0)
        models = dict(getattr(env, "agent_models", {}))
        types = dict(getattr(env, "agent_types", {}))
        roles = dict(getattr(env, "agent_roles", {}))
        missiles = {
            aid: int(env._get_sim(aid).num_left_missiles)
            for aid in env.agent_ids
            if env._get_sim(aid) is not None
        }
        return {
            "config": config_path,
            "main_red0_model": models.get("red_0"),
            "red0_type": types.get("red_0"),
            "red0_role": roles.get("red_0"),
            "mav_missiles": missiles.get("red_0"),
            "attack_uav_models": sorted({
                model for aid, model in models.items() if aid != "red_0"
            }),
            "attack_uav_missiles": sorted({
                count for aid, count in missiles.items() if aid != "red_0"
            }),
            "observation_mode": cfg.get("observation_mode"),
            "hetero_reward_mode": cfg.get("hetero_reward_mode"),
            "sim_freq": cfg.get("sim_freq"),
            "agent_interaction_steps": cfg.get("agent_interaction_steps"),
            "max_steps": cfg.get("max_steps"),
            "info_has_models": "agent_models" in info,
        }
    finally:
        env.close()


def _actions(env, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if mode == "zero":
        return {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
    return {
        aid: rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
        for aid in env.agent_ids
    }


def _env_smoke(config_path: str) -> dict:
    adapter = HeteroObsAdapterV2()
    result = {
        "config": config_path,
        "reset_ok": False,
        "zero_step_ok": False,
        "bounded_random_step_ok": False,
        "nan_detected": False,
        "actor_dim": int(adapter.flat_actor_obs_dim),
        "critic_dim": int(adapter.critic_state_dim),
        "error": None,
    }
    rng = np.random.default_rng(0)
    try:
        for mode in ("zero", "bounded_random"):
            env = make_env(config_path, env_type="jsbsim_hetero", suppress_jsbsim_output=False)
            try:
                obs, info = env.reset(seed=0)
                result["reset_ok"] = True
                result["nan_detected"] = result["nan_detected"] or _contains_nan(obs)
                for _ in range(3):
                    obs, rewards, terminated, truncated, info = env.step(_actions(env, mode, rng))
                    result["nan_detected"] = (
                        result["nan_detected"]
                        or _contains_nan(obs)
                        or _contains_nan(rewards)
                    )
                    if all(terminated.values()) or all(truncated.values()):
                        break
                result[f"{mode}_step_ok"] = True
            finally:
                env.close()
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def _single_f22_scenario(name: str, action: np.ndarray, duration: float = 30.0) -> dict:
    sim = AircraftSimulator(
        uid=f"f22_{name}",
        color="Red",
        model="f22",
        init_state={
            "ic/long-gc-deg": 120.0,
            "ic/lat-geod-deg": 60.0,
            "ic/h-sl-ft": 20000.0,
            "ic/psi-true-deg": 0.0,
            "ic/u-fps": 820.0,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
        },
        sim_freq=60,
        num_missiles=0,
        suppress_jsbsim_output=False,
    )
    pid = PIDController(dt=1.0 / 60.0)
    try:
        target_pitch = float(action[0]) * math.radians(90.0)
        target_heading = float(action[1]) * math.pi
        target_velocity = 102.0 + (float(action[2]) + 1.0) / 2.0 * (408.0 - 102.0)
        nan_detected = False
        crashed = False
        steps = int(duration * 60)
        for _ in range(steps):
            rpy = sim.get_rpy()
            speed = float(np.linalg.norm(sim.get_velocity()))
            vel_ned = np.array([sim.get_velocity()[0], sim.get_velocity()[1], -sim.get_velocity()[2]])
            aileron, elevator, rudder, throttle = pid.compute_control(
                rpy, speed, target_pitch, target_heading, target_velocity,
                ned_velocity=vel_ned,
            )
            sim.set_property_value("fcs/aileron-cmd-norm", aileron)
            sim.set_property_value("fcs/elevator-cmd-norm", elevator)
            sim.set_property_value("fcs/rudder-cmd-norm", rudder)
            sim.set_property_value("fcs/throttle-cmd-norm", throttle)
            sim.run()
            alt = float(sim.get_geodetic()[2])
            spd = float(np.linalg.norm(sim.get_velocity()))
            rpy_now = sim.get_rpy()
            nan_detected = nan_detected or not np.isfinite([alt, spd, *rpy_now]).all()
            crashed = crashed or (not sim.is_alive) or alt < 2500.0
            if crashed:
                break
        return {
            "scenario": name,
            "final_altitude_m": float(sim.get_geodetic()[2]),
            "final_speed_mps": float(np.linalg.norm(sim.get_velocity())),
            "crashed": bool(crashed),
            "nan_detected": bool(nan_detected),
        }
    finally:
        sim.close()


def audit_single_aircraft() -> list[dict]:
    scenarios = {
        "level": np.array([0.0, 0.0, 0.5], dtype=np.float32),
        "climb": np.array([0.2, 0.0, 0.5], dtype=np.float32),
        "turn_left": np.array([0.0, -0.5, 0.5], dtype=np.float32),
        "speed_up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    return [_single_f22_scenario(name, action) for name, action in scenarios.items()]


def _markdown(data: dict) -> str:
    lines = [
        "# F-22 MAV Model Audit",
        "",
        "This audit validates the F-22 MAV resource migration without changing",
        "missile, reward, termination, action, evasion, PID, aircraft XML, or MAPPO.",
        "",
        "## Summary",
        "",
        f"- f22_model_exists: {data['resources']['f22_model_exists']}",
        f"- f22_load_model_ok: {data['jsbsim_load']['f22_load_model_ok']}",
        f"- f22_run_ic_ok: {data['jsbsim_load']['f22_run_ic_ok']}",
        f"- nan_detected: {data['summary']['nan_detected']}",
        "",
        "## Mainline Configs",
    ]
    for record in data["main_config_contract"].values():
        lines.extend([
            "",
            f"### {Path(record['config']).name}",
            f"- red_0 model: {record['main_red0_model']}",
            f"- MAV missiles: {record['mav_missiles']}",
            f"- attack UAV models: {record['attack_uav_models']}",
            f"- attack UAV missiles: {record['attack_uav_missiles']}",
        ])
    lines.extend(["", "## Single-Aircraft 30s Diagnostics"])
    for record in data["single_aircraft_diagnostics"]:
        lines.append(
            f"- {record['scenario']}: alt={record['final_altitude_m']:.1f} m, "
            f"speed={record['final_speed_mps']:.1f} m/s, "
            f"crashed={record['crashed']}, nan={record['nan_detected']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/environment_audit/f22_mav_model_audit.json")
    parser.add_argument("--output-md", default="outputs/environment_audit/f22_mav_model_audit.md")
    args = parser.parse_args()

    resources = audit_resources()
    jsbsim_load = audit_jsbsim_load()
    contracts = {
        "3v2": _config_contract(MAIN_CONFIGS[0]),
        "5v4": _config_contract(MAIN_CONFIGS[1]),
    }
    smokes = {
        "3v2": _env_smoke(MAIN_CONFIGS[0]),
        "5v4": _env_smoke(MAIN_CONFIGS[1]),
    }
    single = audit_single_aircraft()
    nan_detected = (
        bool(jsbsim_load.get("nan_detected"))
        or any(record.get("nan_detected") for record in smokes.values())
        or any(record.get("nan_detected") for record in single)
    )
    data = {
        "resources": resources,
        "jsbsim_load": jsbsim_load,
        "main_config_contract": contracts,
        "env_smoke": smokes,
        "single_aircraft_diagnostics": single,
        "summary": {
            "f22_model_exists": resources["f22_model_exists"],
            "f22_load_model_ok": jsbsim_load["f22_load_model_ok"],
            "f22_run_ic_ok": jsbsim_load["f22_run_ic_ok"],
            "main_3v2_red0_model": contracts["3v2"]["main_red0_model"],
            "main_5v4_red0_model": contracts["5v4"]["main_red0_model"],
            "mav_missiles": contracts["3v2"]["mav_missiles"],
            "attack_uav_model": contracts["3v2"]["attack_uav_models"][0],
            "attack_uav_missiles": contracts["3v2"]["attack_uav_missiles"][0],
            "actor_dim": smokes["3v2"]["actor_dim"],
            "critic_dim": smokes["3v2"]["critic_dim"],
            "nan_detected": nan_detected,
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
    print(json.dumps(data["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
