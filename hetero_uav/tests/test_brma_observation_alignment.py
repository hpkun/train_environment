from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "uav_env" / "JSBSim" / "adapters" / "hetero_obs_adapter_v2.py"


def _load_adapter_class():
    spec = importlib.util.spec_from_file_location("hetero_obs_adapter_v2", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.HeteroObsAdapterV2


HeteroObsAdapterV2 = _load_adapter_class()


def _load_config(config: str) -> dict:
    with (ROOT / config).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _synthetic_obs(agent_id: str, red_ids: list[str], blue_ids: list[str]) -> dict:
    is_red = agent_id.startswith("red_")
    allies = [aid for aid in (red_ids if is_red else blue_ids) if aid != agent_id]
    enemies = blue_ids if is_red else red_ids
    role = np.array([1, 0, 0, 0], dtype=np.float32) if agent_id == "red_0" else np.array([0, 1, 0, 0], dtype=np.float32)
    return {
        "ego_geo_state": np.zeros(7, dtype=np.float32),
        "ego_role": role,
        "missile_warning": np.zeros(1, dtype=np.float32),
        "ally_geo_states": np.zeros((len(allies), 5), dtype=np.float32),
        "ally_roles": np.zeros((len(allies), 4), dtype=np.float32),
        "ally_alive_mask": np.ones(len(allies), dtype=np.float32),
        "enemy_geo_states": np.zeros((len(enemies), 5), dtype=np.float32),
        "enemy_alive_mask": np.ones(len(enemies), dtype=np.float32),
        "enemy_observed_mask": np.ones(len(enemies), dtype=np.float32),
        "enemy_track_source": np.zeros((len(enemies), 2), dtype=np.float32),
    }


def _reset_adapt(config: str):
    cfg = _load_config(config)
    assert cfg["observation_mode"] == "mav_shared_geo"
    red_ids = [f"red_{i}" for i in range(len(cfg["red_agent_types"]))]
    blue_ids = [f"blue_{i}" for i in range(len(cfg["blue_agent_types"]))]
    obs = {rid: _synthetic_obs(rid, red_ids, blue_ids) for rid in red_ids}
    adapter = HeteroObsAdapterV2()
    adapted = adapter.adapt_all(obs, red_ids=red_ids, blue_ids=blue_ids)
    return red_ids, blue_ids, adapter, adapted


def test_v2_actor_and_critic_dims_match_between_3v2_and_5v4():
    red3, _blue3, adapter3, adapted3 = _reset_adapt(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml")
    red5, _blue5, adapter5, adapted5 = _reset_adapt(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml")
    assert adapter3.flat_actor_obs_dim == 96
    assert adapter5.flat_actor_obs_dim == 96
    assert adapter3.critic_state_dim == 480
    assert adapter5.critic_state_dim == 480
    assert adapted3["critic_state"].shape == adapted5["critic_state"].shape == (480,)

    for rid in red3:
        assert adapted3["actor_obs"][rid].shape == (96,)
    for rid in red5:
        assert adapted5["actor_obs"][rid].shape == (96,)


def test_v2_padding_and_masks_support_3v2_and_5v4_zero_shot_contract():
    _red3, _blue3, _adapter3, adapted3 = _reset_adapt(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2.yaml")
    _red5, _blue5, _adapter5, adapted5 = _reset_adapt(
        "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4.yaml")
    assert adapted3["red_valid_mask"].shape == (5,)
    assert adapted5["red_valid_mask"].shape == (5,)
    np.testing.assert_array_equal(
        adapted3["red_valid_mask"],
        np.array([1, 1, 1, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        adapted5["red_valid_mask"],
        np.array([1, 1, 1, 1, 1], dtype=np.float32),
    )

    padded_tail = adapted3["critic_state"][3 * 96:]
    assert padded_tail.shape == (2 * 96,)
    assert np.allclose(padded_tail, 0.0)

    assert "red_3" in adapted5["actor_obs"]
    assert "red_4" in adapted5["actor_obs"]
    assert adapted5["actor_obs"]["red_3"].shape == (96,)
    assert adapted5["actor_obs"]["red_4"].shape == (96,)


def test_brma_observation_alignment_docs_state_boundaries():
    doc = ROOT / "docs" / "brma_observation_alignment.md"
    protocol = ROOT / "docs" / "main_experiment_protocol.md"
    text = doc.read_text(encoding="utf-8")
    protocol_text = protocol.read_text(encoding="utf-8")

    for token in [
        "BRMA-MAPPO paper observation design",
        "Current HeteroObsAdapterV2 design",
        "Alignment table",
        "paper-aligned",
        "partially aligned",
        "not implemented",
        "biased random mask",
        "fixed-capacity 3v2 to 5v4",
    ]:
        assert token in text

    assert "BRMA-inspired entity/mask observation" in protocol_text
    assert "not the full BRMA attention encoder" in protocol_text
    assert "fixed-capacity 3v2-to-5v4" in protocol_text
