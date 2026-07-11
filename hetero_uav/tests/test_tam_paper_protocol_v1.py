from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.train_happo_reference import _experiment_base_v2_meta
from uav_env import make_env


ROOT = Path(__file__).resolve().parents[1]
CFG3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v2_tam_paper_protocol_v1.yaml"
CFG5 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scale_aligned_v2_tam_paper_protocol_v1.yaml"
OLD3 = ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scale_aligned_v2.yaml"


@pytest.mark.parametrize("path", [CFG3, CFG5])
def test_tam_paper_protocol_config_applies_active_and_metadata_values(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["missile_launch_range_m"] == 14000.0
    assert data["missile_attack_interval_sec"] == 25.0
    assert data["missile_guidance"] == {"mode": "pn", "navigation_gain": 3.0, "max_overload_g": 30.0}
    protocol = data["missile_protocol"]
    assert protocol["missile_mass_kg"] == 84.0
    assert set(protocol["missile_unmodeled_parameters"]) == {
        "missile_mass_kg", "missile_length_m", "missile_diameter_m"
    }
    env = make_env(str(path))
    try:
        assert env._missile_launch_range_m_effective == pytest.approx(14000.0)
        assert env._missile_attack_interval_sec_effective == pytest.approx(25.0)
        assert env.missile_guidance_config["mode"] == "pn"
        assert env.missile_guidance_config["navigation_gain"] == pytest.approx(3.0)
        assert env.missile_guidance_config["max_overload_g"] == pytest.approx(30.0)
        assert env.missile_protocol_meta == protocol
    finally:
        env.close()


def test_old_v2_config_does_not_silently_enable_paper_protocol():
    data = yaml.safe_load(OLD3.read_text(encoding="utf-8"))
    assert "missile_protocol" not in data
    assert "missile_launch_range_m" not in data
    assert "missile_attack_interval_sec" not in data
    assert "missile_guidance" not in data


def test_training_meta_records_paper_protocol_without_changing_reward_meta():
    meta = _experiment_base_v2_meta(
        actual_reward_mode="brma_tam_scale_aligned_v2",
        rich_logging_enabled=False,
        rich_log_mode="summary",
        config_path=str(CFG3),
    )
    assert meta["reward_contract_revision"] == 4
    assert meta["missile_protocol_version"] == "tam_paper_protocol_v1"
    assert meta["missile_attack_range_m"] == 14000.0
    assert meta["missile_attack_interval_sec"] == 25.0
    assert meta["missile_navigation_gain"] == 3.0
    assert meta["missile_max_overload_g"] == 30.0
