from scripts.diagnose_tam_fcs_authority import summarize_authority
from scripts.sweep_tam_f22_neutral_stability import select_stable_candidate


def test_authority_summary_detects_throttle_response_from_thrust_or_speed():
    summary = summarize_authority({
        "throttle_high_neutral": {
            "red_0": {"final_speed_mps": 220, "min_speed_mps": 180, "mean_thrust_lbs": 9000},
            "red_1": {"mean_thrust_lbs": 8000}, "red_2": {"mean_thrust_lbs": 8000},
            "blue_0": {"mean_thrust_lbs": 8000},
        },
        "throttle_low_neutral": {
            "red_0": {"final_speed_mps": 150, "min_speed_mps": 120, "mean_thrust_lbs": 3000},
            "red_1": {"mean_thrust_lbs": 2000}, "red_2": {"mean_thrust_lbs": 2000},
            "blue_0": {"mean_thrust_lbs": 2000},
        },
    })
    assert summary["f22_throttle_authority_passed"] is True
    assert summary["f22_final_speed_delta_mps"] == 70
    assert summary["f22_mean_thrust_delta_lbs"] == 6000


def test_sweep_prefers_primary_pass_then_trim_quality():
    candidates = [
        {"action": [39, 20, 6, 20], "primary_pass": False,
         "final_speed_mps": 170, "min_altitude_m": 4400,
         "mean_vertical_speed_mps": -5, "max_abs_roll_rad": 0.1,
         "max_abs_pitch_rad": 0.2, "death_step": 800},
        {"action": [39, 20, 8, 20], "primary_pass": True,
         "final_speed_mps": 190, "min_altitude_m": 4700,
         "mean_vertical_speed_mps": 2, "max_abs_roll_rad": 0.2,
         "max_abs_pitch_rad": 0.3, "death_step": -1},
        {"action": [39, 19, 8, 20], "primary_pass": True,
         "final_speed_mps": 200, "min_altitude_m": 4800,
         "mean_vertical_speed_mps": 0.5, "max_abs_roll_rad": 0.1,
         "max_abs_pitch_rad": 0.2, "death_step": -1},
    ]
    selected = select_stable_candidate(candidates)
    assert selected["action"] == [39, 19, 8, 20]


def test_sweep_returns_pareto_fallback_when_no_primary_candidate_passes():
    candidates = [
        {"action": [39, 20, 6, 20], "primary_pass": False,
         "final_speed_mps": 160, "min_altitude_m": 4300,
         "mean_vertical_speed_mps": -8, "max_abs_roll_rad": 0.3,
         "max_abs_pitch_rad": 0.4, "death_step": 700},
        {"action": [39, 20, 4, 20], "primary_pass": False,
         "final_speed_mps": 175, "min_altitude_m": 4400,
         "mean_vertical_speed_mps": -3, "max_abs_roll_rad": 0.2,
         "max_abs_pitch_rad": 0.3, "death_step": 900},
    ]
    assert select_stable_candidate(candidates)["action"] == [39, 20, 4, 20]
