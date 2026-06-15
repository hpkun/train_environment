from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "uav_env" / "JSBSim" / "adapters"
ENTITY_ADAPTER_PATH = ADAPTER_DIR / "entity_set_adapter.py"


def _load_entity_adapter_class():
    sys.path.insert(0, str(ADAPTER_DIR))
    try:
        spec = importlib.util.spec_from_file_location("entity_set_adapter", ENTITY_ADAPTER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.EntitySetAdapter
    finally:
        if sys.path[0] == str(ADAPTER_DIR):
            sys.path.pop(0)


def _synthetic_obs(agent_id: str, red_ids: list[str], blue_ids: list[str]) -> dict:
    is_red = agent_id.startswith("red_")
    allies = [aid for aid in (red_ids if is_red else blue_ids) if aid != agent_id]
    enemies = blue_ids if is_red else red_ids
    role = (
        np.array([1, 0, 0, 0], dtype=np.float32)
        if agent_id == "red_0"
        else np.array([0, 1, 0, 0], dtype=np.float32)
    )
    return {
        "ego_geo_state": np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2], dtype=np.float32),
        "ego_role": role,
        "missile_warning": np.array([0.0], dtype=np.float32),
        "ally_geo_states": np.ones((len(allies), 5), dtype=np.float32) * 0.2,
        "ally_roles": np.tile(np.array([[0, 1, 0, 0]], dtype=np.float32), (len(allies), 1)),
        "ally_alive_mask": np.ones(len(allies), dtype=np.float32),
        "enemy_geo_states": np.ones((len(enemies), 5), dtype=np.float32) * -0.3,
        "enemy_alive_mask": np.ones(len(enemies), dtype=np.float32),
        "enemy_observed_mask": np.ones(len(enemies), dtype=np.float32),
        "enemy_track_source": np.tile(np.array([[1, 0]], dtype=np.float32), (len(enemies), 1)),
    }


def _entity_adapt(red_count: int, blue_count: int):
    EntitySetAdapter = _load_entity_adapter_class()
    red_ids = [f"red_{i}" for i in range(red_count)]
    blue_ids = [f"blue_{i}" for i in range(blue_count)]
    obs = {rid: _synthetic_obs(rid, red_ids, blue_ids) for rid in red_ids}
    adapter = EntitySetAdapter(max_red=5, max_blue=4)
    return adapter, adapter.adapt_all(obs, red_ids=red_ids, blue_ids=blue_ids)


def test_entity_adapter_converts_3v2_flat_contract_to_entity_set():
    adapter, adapted = _entity_adapt(red_count=3, blue_count=2)
    red0 = adapted["entity_actor_obs"]["red_0"]

    assert adapter.entity_dim > 0
    assert red0["entities"].shape == (9, adapter.entity_dim)
    assert red0["self_entity"].shape == (adapter.entity_dim,)
    assert red0["ally_entities"].shape == (4, adapter.entity_dim)
    assert red0["enemy_entities"].shape == (4, adapter.entity_dim)
    assert adapted["critic_state"].shape == (480,)
    np.testing.assert_array_equal(adapted["red_valid_mask"], np.array([1, 1, 1, 0, 0], dtype=np.float32))


def test_entity_adapter_supports_5v4_with_same_entity_dim_and_more_valid_slots():
    adapter3, adapted3 = _entity_adapt(red_count=3, blue_count=2)
    adapter5, adapted5 = _entity_adapt(red_count=5, blue_count=4)

    assert adapter3.entity_dim == adapter5.entity_dim
    assert adapted3["entity_actor_obs"]["red_1"]["entities"].shape == adapted5["entity_actor_obs"]["red_4"]["entities"].shape
    np.testing.assert_array_equal(adapted5["red_valid_mask"], np.ones(5, dtype=np.float32))
    assert adapted5["entity_actor_obs"]["red_4"]["role_name"] == "uav"


def test_attention_mask_excludes_padding_dead_and_unobserved_entities():
    adapter, adapted = _entity_adapt(red_count=3, blue_count=2)
    red0 = adapted["entity_actor_obs"]["red_0"]

    assert red0["attention_mask"].dtype == np.float32
    # self + 2 valid living allies + 2 valid observed enemies are attendable.
    assert int(red0["attention_mask"].sum()) == 5
    assert np.all(red0["attention_mask"][3:5] == 0.0)
    assert np.all(red0["attention_mask"][7:] == 0.0)


def test_entity_policy_forward_for_mav_and_shared_uav_heads():
    from algorithms.happo.entity_policy import EntityHAPPOReferencePolicy

    adapter, adapted = _entity_adapt(red_count=5, blue_count=4)
    batch = [adapted["entity_actor_obs"][f"red_{i}"] for i in range(5)]
    entities = torch.as_tensor(np.stack([item["entities"] for item in batch]), dtype=torch.float32)
    masks = torch.as_tensor(np.stack([item["attention_mask"] for item in batch]), dtype=torch.float32)
    roles = [item["role_id"] for item in batch]
    critic = torch.as_tensor(adapted["critic_state"], dtype=torch.float32).unsqueeze(0)

    policy = EntityHAPPOReferencePolicy(entity_dim=adapter.entity_dim, critic_state_dim=480, action_dim=3)
    out = policy.act({"entities": entities, "attention_mask": masks}, roles=roles, critic_state=critic, deterministic=True)

    assert out["action"].shape == (5, 3)
    assert out["mean"].shape == (5, 3)
    assert out["value"].shape == (1,)
    assert torch.isfinite(out["action"]).all()
    assert torch.all(out["action"] <= 1.0)
    assert torch.all(out["action"] >= -1.0)
    assert int(out["role_mask"][0].item()) == 0
    assert set(out["role_mask"][1:].tolist()) == {1}


def test_old_happo_reference_policy_import_forward_and_checkpoint_shape_unchanged():
    from algorithms.happo.happo_policy import HAPPOReferencePolicy

    policy = HAPPOReferencePolicy(actor_obs_dim=96, critic_state_dim=480, action_dim=3)
    obs = torch.zeros((3, 96), dtype=torch.float32)
    critic = torch.zeros((1, 480), dtype=torch.float32)
    out = policy.act(obs, roles=[0, 1, 1], critic_state=critic, deterministic=True)

    assert out["action"].shape == (3, 3)
    assert out["value"].shape == (1,)
    checkpoint = ROOT / "outputs" / "happo_geometry_curriculum_100k" / "normal_50k" / "best" / "model.pt"
    if checkpoint.exists():
        policy.load(checkpoint, map_location="cpu")
