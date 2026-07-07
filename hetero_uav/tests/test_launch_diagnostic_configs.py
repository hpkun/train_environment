from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env import make_env  # noqa: E402

CONFIG_DIR = ROOT / "uav_env" / "JSBSim" / "configs"

BASE_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"
BASE_5V4 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"

DIAGNOSTIC_CONFIGS = {
    "range15": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15.yaml",
        ),
        "range": 15000.0,
        "ao_deg": 45.0,
        "ta_deg": 90.0,
        "mode": "closest",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
    "ao60": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ao60.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ao60.yaml",
        ),
        "range": 10000.0,
        "ao_deg": 60.0,
        "ta_deg": 90.0,
        "mode": "closest",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
    "range15_ao60": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60.yaml",
        ),
        "range": 15000.0,
        "ao_deg": 60.0,
        "ta_deg": 90.0,
        "mode": "closest",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
    "mav_rank": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_mav_rank.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_mav_rank.yaml",
        ),
        "range": 10000.0,
        "ao_deg": 45.0,
        "ta_deg": 90.0,
        "mode": "mav_threat_rank",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
    "range15_ao60_mav_rank": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60_mav_rank.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_range15_ao60_mav_rank.yaml",
        ),
        "range": 15000.0,
        "ao_deg": 60.0,
        "ta_deg": 90.0,
        "mode": "mav_threat_rank",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
    "tam_interval25": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_tam_interval25.yaml",
        ),
        "range": 10000.0,
        "ao_deg": 45.0,
        "ta_deg": 90.0,
        "mode": "closest",
        "interval_sec": 25.0,
        "min_range": 500.0,
    },
    "ta60": {
        "files": (
            "hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ta60.yaml",
            "hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_paper_aligned_v1_diagnostic_ta60.yaml",
        ),
        "range": 10000.0,
        "ao_deg": 45.0,
        "ta_deg": 60.0,
        "mode": "closest",
        "interval_sec": 0.5,
        "min_range": 500.0,
    },
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _make(config_name: str):
    return make_env(f"uav_env/JSBSim/configs/{config_name}", max_steps=5)


def _assert_launch_contract(env, *, range_m: float, ao_deg: float, ta_deg: float, mode: str,
                            interval_sec: float, min_range: float) -> None:
    assert env._missile_launch_range_m_effective == pytest.approx(range_m)
    assert env.MISSILE_LAUNCH_RANGE_THRESH == pytest.approx(range_m)
    assert math.degrees(env.MISSILE_LAUNCH_AO_THRESH) == pytest.approx(ao_deg)
    assert math.degrees(env.MISSILE_LAUNCH_TA_THRESH) == pytest.approx(ta_deg)
    assert env.MISSILE_LAUNCH_MIN_RANGE == pytest.approx(min_range)
    assert env.red_target_selection_mode == mode
    assert env._missile_attack_interval_sec_effective == pytest.approx(interval_sec)
    assert env.missile_cooldown_frames == int(round(interval_sec * env.sim_freq))


def test_original_paper_aligned_defaults_are_unchanged():
    for config_path in (BASE_3V2, BASE_5V4):
        env = make_env(config_path, max_steps=5)
        try:
            _assert_launch_contract(
                env,
                range_m=10000.0,
                ao_deg=45.0,
                ta_deg=90.0,
                mode="closest",
                interval_sec=0.5,
                min_range=500.0,
            )
        finally:
            env.close()


@pytest.mark.parametrize("name,spec", DIAGNOSTIC_CONFIGS.items())
def test_diagnostic_configs_load_and_set_expected_launch_contract(name: str, spec: dict):
    for filename in spec["files"]:
        path = CONFIG_DIR / filename
        assert path.exists(), filename
        text = path.read_text(encoding="utf-8").lower()
        assert "diagnostic" in text
        cfg = _load_yaml(path)
        assert cfg["hetero_reward_mode"] == "tam_brma_paper_aligned_v1"
        assert cfg["observation_mode"] == "mav_shared_geo"
        assert cfg["aircraft_type_params"]["mav"]["num_missiles"] == 0
        assert cfg["aircraft_type_params"]["attack_uav"]["num_missiles"] == 2
        assert cfg["action_trim_by_role"]["mav"]["pitch"] == 0.0
        assert cfg["sim_freq"] == 60
        assert cfg["agent_interaction_steps"] == 12

        env = _make(filename)
        base_env = make_env(
            BASE_3V2 if "3v2" in filename else BASE_5V4,
            max_steps=5,
        )
        try:
            expected = {k: v for k, v in spec.items() if k != "files"}
            expected["range_m"] = expected.pop("range")
            _assert_launch_contract(env, **expected)
            assert env.action_space["red_0"].shape == (3,)
            assert env.observation_space["red_0"] == base_env.observation_space["red_0"]
        finally:
            env.close()
            base_env.close()


def test_ta60_configs_are_marked_diagnostic_only():
    for filename in DIAGNOSTIC_CONFIGS["ta60"]["files"]:
        text = (CONFIG_DIR / filename).read_text(encoding="utf-8").lower()
        assert "diagnostic-only" in text
        assert "rear-hemisphere" in text
