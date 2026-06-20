from scripts.validate_tam_missile_threat import summarize_missile_threat


def test_missile_threat_summary_counts_hits_warnings_and_zero_hit_reasons():
    episodes = [{
        "death_step": 50,
        "death_reason": "Missile_Kill",
        "trace": [
            {"step": step, "missile_warning": step >= 30, "nearest_blue_range_m": 8000}
            for step in range(1, 51)
        ],
        "launch_quality": [
            {
                "missile_id": "r0", "shooter_team": "red", "launch_step": 10,
                "range_m": 4000, "AO_deg": 20, "TA_deg": 100,
                "shooter_speed_mps": 250,
            },
            {
                "missile_id": "r0", "shooter_team": "red", "launch_step": 10,
                "termination_reason": "low_speed", "is_success": False,
            },
            {
                "missile_id": "b0", "shooter_team": "blue", "launch_step": 20,
                "range_m": 7000, "AO_deg": 10, "TA_deg": 80,
                "shooter_speed_mps": 300,
            },
            {
                "missile_id": "b0", "shooter_team": "blue", "launch_step": 20,
                "termination_reason": "hit", "is_success": True,
            },
        ],
    }]
    summary = summarize_missile_threat(episodes)
    assert summary["red_launch_count"] == 1
    assert summary["red_hit_count"] == 0
    assert summary["red_hit_rate"] == 0.0
    assert summary["blue_launch_count"] == 1
    assert summary["blue_hit_count"] == 1
    assert summary["missile_warning_count"] == 21
    assert summary["warning_to_hit_rate"]["5s"] == 1.0
    assert summary["blue_launch_to_red_death_time_sec_mean"] == 6.0
    assert summary["red_zero_hit_termination_reasons"] == {"low_speed": 1}
    assert summary["red_launch_opportunity_without_launch_episodes"] == 0
    assert summary["geometry_hit_rate"]["range_m"]["3000-6000"]["launches"] == 1


def test_close_range_without_red_launch_is_reported_as_opportunity():
    summary = summarize_missile_threat([{
        "death_step": -1,
        "death_reason": "alive",
        "trace": [{"step": 1, "missile_warning": False, "nearest_blue_range_m": 9000}],
        "launch_quality": [],
    }])
    assert summary["red_launch_opportunity_without_launch_episodes"] == 1
