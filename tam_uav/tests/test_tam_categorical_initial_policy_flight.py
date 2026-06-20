from __future__ import annotations

from scripts.validate_tam_categorical_initial_policy_flight import run_validation


CONFIG = "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"


def test_deterministic_neutral_policy_survives_and_env_advances(tmp_path):
    result = run_validation(
        CONFIG, output_dir=tmp_path, episodes=1, steps=300,
        device="cpu", modes=("deterministic",), seed=23,
    )
    deterministic = result["modes"]["deterministic"]
    assert deterministic["mav_survival_rate"] == 1.0
    assert deterministic["episodes"][0]["env_steps"] == 300
    assert deterministic["episodes"][0]["red_action_override_detected"] is False
    assert deterministic["throttle_high_rate"] > 0.95
    assert deterministic["surface_middle_rate"] > 0.95
