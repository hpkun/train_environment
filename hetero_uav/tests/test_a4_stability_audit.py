from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_jsbsim_aircraft_models import AIRCRAFT, audit_aircraft
from scripts.diagnose_a4_pid_mismatch import diagnose_scenario
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_script_functions_run_without_modifying_xml():
    before = {name: _sha256(path) for name, path in AIRCRAFT.items()}
    reports = {name: audit_aircraft(path) for name, path in AIRCRAFT.items()}
    after = {name: _sha256(path) for name, path in AIRCRAFT.items()}
    assert before == after
    assert reports["A-4"]["aircraft_name"] == "A-4"
    assert reports["f16"]["aircraft_name"] == "General Dynamics F-16A"


def test_diagnose_a4_pid_mismatch_runs_without_modifying_xml():
    before = {name: _sha256(path) for name, path in AIRCRAFT.items()}
    a4 = diagnose_scenario("a4_level", duration=2.0)
    f16 = diagnose_scenario("f16_level", duration=2.0)
    after = {name: _sha256(path) for name, path in AIRCRAFT.items()}
    assert before == after
    assert a4["model"] == "A-4"
    assert f16["model"] == "f16"
    assert not a4["nan_detected"]
    assert not f16["nan_detected"]


def test_no_mav_gcas_field_added():
    source = Path("uav_env/JSBSim/envs/hetero_uav_combat_env.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "mav_gcas" not in lowered
    assert "enable_gcas_for_mav" not in lowered
    assert "enable_mav_gcas" not in lowered


def test_hetero_env_reset_and_five_step_smoke():
    env = HeteroUavCombatEnv(
        max_num_red=2,
        max_num_blue=2,
        sim_freq=60,
        agent_interaction_steps=2,
        max_steps=10,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == set(env.agent_ids)
        assert info["agent_models"]["red_0"] == "A-4"
        for _ in range(5):
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
        assert set(obs.keys()) == set(env.agent_ids)
    finally:
        env.close()
