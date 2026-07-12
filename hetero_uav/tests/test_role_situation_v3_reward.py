from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env import make_env
from uav_env.JSBSim.envs.role_situation_v3 import (
    V3_REWARD_COMPONENT_FIELDS,
    _compute_pair_quality,
    _softmax_agg,
    validate_v3_reward_components,
)


ROOT = Path(__file__).resolve().parents[1]
CFG3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_role_situation_v3_tam_paper_protocol_v1.yaml"
CFG5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_role_situation_v3_tam_paper_protocol_v1.yaml"
V2_3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v2_tam_paper_protocol_v1.yaml"
V2_5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scale_aligned_v2_tam_paper_protocol_v1.yaml"


class _Sim:
    def __init__(self, speed):
        self.speed = speed

    def get_velocity(self):
        return np.array([self.speed, 0.0, 0.0])


class _GeomEnv:
    def __init__(self, ata, aa, distance):
        self.geom = {"tam_ata_rad": ata, "tam_aa_rad": aa, "target_distance_m": distance}

    def _brma_tam_3d_geometry(self, _attacker, _target):
        return self.geom

    @staticmethod
    def _brma_tam_safe_vec(sim, name):
        return getattr(sim, name)()


@pytest.mark.parametrize("path", [CFG3, CFG5])
def test_v3_config_contract_and_real_reward_fields(path):
    env = make_env(path)
    try:
        obs, _ = env.reset(seed=13)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        _, _, _, _, info = env.step(actions)
        components = info["reward_components"]
        for rid in env.red_ids:
            values = validate_v3_reward_components(components[rid], agent_id=rid, step=1)
            assert set(V3_REWARD_COMPONENT_FIELDS) <= set(values)
            assert abs(values["role_situation_v3_identity_error"]) <= 1e-6
    finally:
        env.close()


def test_v3_pair_quality_ordering_and_attacker_speed_direction():
    attacker = _Sim(300.0)
    target = _Sim(150.0)
    args = (500.0, 4900.0, 10500.0, 14000.0, 0.75, 1.0)
    tail = _compute_pair_quality(attacker, target, _GeomEnv(0.0, 0.0, 7000.0), *args)
    side = _compute_pair_quality(attacker, target, _GeomEnv(np.pi / 2, 0.0, 7000.0), *args)
    head = _compute_pair_quality(attacker, target, _GeomEnv(np.pi, 0.0, 7000.0), *args)
    far = _compute_pair_quality(attacker, target, _GeomEnv(0.0, 0.0, 30000.0), *args)
    assert tail["combined"] > side["combined"] > head["combined"]
    assert tail["combined"] > far["combined"]
    reverse = _compute_pair_quality(target, attacker, _GeomEnv(0.0, 0.0, 7000.0), *args)
    assert tail["speed_q"] > reverse["speed_q"]


def test_v3_softmax_and_negative_threat_signal():
    offense = _softmax_agg([0.2, 0.3], 0.2)
    threat = _softmax_agg([0.8, 0.9], 0.2)
    assert offense - threat < 0.0
    with pytest.raises(ValueError, match="must be finite"):
        _softmax_agg([0.2, float("nan")], 0.2)


def test_v3_missing_required_contract_key_fails(tmp_path):
    data = yaml.safe_load(CFG3.read_text(encoding="utf-8"))
    del data["brma_tam_role_situation_v3"]["situation"]["softmax_temperature"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="situation.softmax_temperature"):
        make_env(path)


def test_v3_bad_protocol_fails(tmp_path):
    data = yaml.safe_load(CFG3.read_text(encoding="utf-8"))
    data["missile_guidance"]["max_overload_g"] = 29.0
    path = tmp_path / "bad_protocol.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="max_overload_g=30"):
        make_env(path)


def test_v3_episode_state_resets():
    env = make_env(CFG3)
    try:
        env.reset(seed=1)
        env._v3_episode_state["prev_blue_loss"] = 0.5
        env._v3_episode_state["terminal_applied"] = True
        env.reset(seed=2)
        assert env._v3_episode_state["prev_blue_loss"] == 0.0
        assert env._v3_episode_state["terminal_applied"] is False
    finally:
        env.close()


@pytest.mark.parametrize("v3_path,v2_path", [(CFG3, V2_3), (CFG5, V2_5)])
def test_v3_only_differs_from_v2_protocol_by_reward_contract(v3_path, v2_path):
    v3 = yaml.safe_load(v3_path.read_text(encoding="utf-8"))
    v2 = yaml.safe_load(v2_path.read_text(encoding="utf-8"))
    for data in (v3, v2):
        data.pop("hetero_reward_mode", None)
        data.pop("brma_tam_role_situation_v3", None)
        data.pop("brma_tam_scale_aligned_v2", None)
    assert v3 == v2
