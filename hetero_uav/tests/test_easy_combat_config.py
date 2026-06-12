from __future__ import annotations

from pathlib import Path
import importlib.util

import yaml

from algorithms.happo import HAPPOReferencePolicy


ROOT = Path(__file__).resolve().parents[1]
EASY = ROOT / "uav_env" / "JSBSim" / "configs" / "hetero_mav_shared_geo_3v2_easy_combat_f16_mav_surrogate.yaml"
BASE = ROOT / "uav_env" / "JSBSim" / "configs" / "hetero_mav_shared_geo_3v2_happo_ref_v0_f16_mav_surrogate.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _adapter_class():
    path = ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py"
    spec = importlib.util.spec_from_file_location("hetero_obs_adapter_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HeteroObsAdapterV2


def test_easy_combat_config_contract():
    assert EASY.exists()
    easy = _load(EASY)
    base = _load(BASE)

    assert easy["hetero_reward_mode"] == "happo_ref_v0"
    assert easy["observation_mode"] == "mav_shared_geo"
    assert easy["red_agent_types"] == ["mav", "attack_uav", "attack_uav"]
    assert easy["blue_agent_types"] == ["attack_uav", "attack_uav"]
    assert easy["aircraft_type_params"]["mav"]["aircraft_model"] == "f16"
    assert easy["aircraft_type_params"]["mav"]["num_missiles"] == 0
    assert easy["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2

    adapter = _adapter_class()()
    policy = HAPPOReferencePolicy(adapter.flat_actor_obs_dim, adapter.critic_state_dim)
    assert adapter.flat_actor_obs_dim == 96
    assert adapter.critic_state_dim == 480
    assert policy.action_dim == 3

    assert base["initial_states"]["blue_0"]["lat"] == 60.2
    assert easy["initial_states"]["blue_0"]["lat"] == 60.07


def test_easy_combat_only_changes_initial_geometry_and_keeps_models():
    easy = _load(EASY)
    base = _load(BASE)
    for key in [
        "hetero_reward_mode",
        "observation_mode",
        "aircraft_type_params",
        "red_agent_types",
        "blue_agent_types",
        "sim_freq",
        "agent_interaction_steps",
    ]:
        assert easy[key] == base[key]
    assert easy["initial_states"] != base["initial_states"]
