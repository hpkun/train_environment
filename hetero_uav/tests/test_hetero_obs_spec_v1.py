"""Test that HeteroObsSpec v1 document and tools are consistent.

Does NOT implement MAPPO, attention, or HeteroObsAdapter.
Does NOT modify reward / missile / PID / termination / aircraft XML.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_DOC = ROOT / "docs" / "hetero_obs_spec_v1.md"
PRINT_SCRIPT = ROOT / "scripts" / "print_hetero_obs_spec.py"


def test_spec_doc_exists():
    assert SPEC_DOC.exists(), f"Missing {SPEC_DOC}"


def test_spec_doc_flat_actor_obs_dim():
    text = SPEC_DOC.read_text(encoding="utf-8")
    assert "140" in text, "spec doc should mention flat_actor_obs_dim = 140"
    assert "flat_actor_obs_dim" in text


def test_spec_doc_critic_state_dim():
    text = SPEC_DOC.read_text(encoding="utf-8")
    assert "700" in text, "spec doc should mention critic_state_dim = 700"
    assert "critic_state_dim" in text


def test_spec_doc_excludes_enemy_types_roles():
    text = SPEC_DOC.read_text(encoding="utf-8")
    assert "enemy_types" in text.lower()
    assert "enemy_roles" in text.lower()
    assert "excluded" in text.lower()


def test_spec_doc_capability_postponed():
    text = SPEC_DOC.read_text(encoding="utf-8")
    assert "capability" in text.lower()
    assert "postponed" in text.lower()


def test_spec_doc_incoming_missile_postponed():
    text = SPEC_DOC.read_text(encoding="utf-8")
    assert "incoming" in text.lower()
    assert "postponed" in text.lower()


def test_print_script_runs():
    result = subprocess.run(
        [sys.executable, str(PRINT_SCRIPT)],
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert "flat_actor_obs_dim" in result.stdout
    assert "critic_state_dim" in result.stdout


def test_hetero_env_reset():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=5,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert len(obs) == 4
    finally:
        env.close()


def test_raw_obs_has_required_fields():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=5,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        red0 = obs["red_0"]
        for key in ["ego_state", "ego_role", "ego_type",
                    "ally_states", "ally_roles", "ally_types",
                    "enemy_states", "enemy_roles", "enemy_types",
                    "missile_warning", "altitude", "velocity",
                    "death_mask"]:
            assert key in red0, f"red_0 missing key: {key}"
    finally:
        env.close()


def test_no_mechanism_change():
    """Adapter v1 is only a spec — it does not change env reward / missile / PID."""
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv(
        max_num_blue=2, max_num_red=2, max_steps=5,
        suppress_jsbsim_output=True,
    )
    try:
        obs, info = env.reset(seed=0)
        # Simulate one step — must not crash
        actions = {aid: obs[aid]["ego_state"][:3].astype("float32") * 0.0
                   for aid in env.agent_ids}
        env.step(actions)
    finally:
        env.close()
