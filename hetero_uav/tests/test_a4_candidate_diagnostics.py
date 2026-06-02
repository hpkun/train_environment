from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_jsbsim_aircraft_models import AIRCRAFT
from scripts.diagnose_a4_candidate_fixes import run_all as run_candidate_fixes
from scripts.diagnose_a4_control_paths import diagnose_model
from scripts.diagnose_hetero_a4_init_options import run_rollout
from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jsbsim") is None,
    reason="jsbsim is not installed",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xml_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in AIRCRAFT.items()}


def test_control_path_diagnostic_runs_without_xml_changes():
    before = _xml_hashes()
    a4 = diagnose_model("A-4", duration=1.0)
    f16 = diagnose_model("f16", duration=1.0)
    after = _xml_hashes()
    assert before == after
    assert a4["conclusion"] in {"active", "inactive", "inconclusive"}
    assert f16["conclusion"] in {"active", "inactive", "inconclusive"}


def test_candidate_fix_diagnostic_runs_without_xml_changes():
    before = _xml_hashes()
    rows = run_candidate_fixes(duration=1.0, seed=0)
    after = _xml_hashes()
    assert before == after
    names = {row["name"] for row in rows}
    assert "baseline" in names
    assert "pitch_bias_0.10" in names
    assert all(not row["nan_detected"] for row in rows)


def test_hetero_init_options_diagnostic_runs_without_xml_changes():
    before = _xml_hashes()
    row = run_rollout({"name": "test", "red0_altitude_delta_m": 100.0}, "zero", steps=2, seed=0)
    after = _xml_hashes()
    assert before == after
    assert row["steps_executed"] == 2
    assert not row["nan_detected"]


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
        for _ in range(5):
            actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
        assert set(obs.keys()) == set(env.agent_ids)
    finally:
        env.close()
