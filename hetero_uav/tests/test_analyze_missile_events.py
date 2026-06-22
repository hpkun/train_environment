"""Tests for analyze_missile_events.py: launch vs termination row separation."""
import csv
import json
import os
import tempfile

import pytest


MISSILE_CSV_CONTENT = """run_id,scenario,episode_id,step,sim_time,event_type,missile_id,owner_id,owner_team,target_id,target_team,lon,lat,altitude,distance_to_target,hit_success,death_caused,raw_termination_reason,AO_rad,AO_deg,TA_rad,TA_deg,flight_time_sec,launch_step,termination_step,step_delta,target_alive_at_launch,target_alive_at_termination,shooter_speed_mps,target_speed_mps,closing_speed_mps,shooter_alt_m,target_alt_m
test,cfg,0,0,0.0,launch,m0,red_1,red,blue_0,blue,,,,,,,,,,,,,,,,,,250.0,,,,
test,cfg,0,0,0.0,launch,m1,red_2,red,blue_1,blue,,,,,,,,,,,,,,,,,,200.0,,,,
test,cfg,0,0,0.0,launch,m2,blue_0,blue,red_0,red,,,,,,,,,,,,,,,,,,300.0,,,,
test,cfg,0,10,2.0,termination,m0,red_1,red,blue_0,blue,,,,,,,,hit,,,,,,,,,180.0,,,,
test,cfg,0,10,2.0,termination,m1,red_2,red,blue_1,blue,,,,,,,,low_speed,,,,,,,,,95.0,,,,
test,cfg,0,10,2.0,termination,m2,blue_0,blue,red_0,red,,,,,,,,hit,,,,,,,,,310.0,,,,
"""


def test_launch_vs_termination_row_counts(tmp_path):
    csv_path = tmp_path / "missile_events.csv"
    csv_path.write_text(MISSILE_CSV_CONTENT)
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()

    from scripts.analyze_missile_events import analyse
    summary = analyse(str(csv_path), str(out_dir))

    # Row counts
    assert summary["launch_rows"] == 3
    assert summary["termination_rows"] == 3

    # Red stats (from term rows only)
    assert summary["red"]["launch_count"] == 2
    assert summary["red"]["term_count"] == 2
    assert summary["red"]["hit_count"] == 1
    assert summary["red"]["low_speed_count"] == 1
    assert summary["red"]["hit_rate_by_launch"] == pytest.approx(0.5)
    assert summary["red"]["hit_rate_by_terminated"] == pytest.approx(0.5)

    # Blue stats
    assert summary["blue"]["launch_count"] == 1
    assert summary["blue"]["term_count"] == 1
    assert summary["blue"]["hit_count"] == 1
    assert summary["blue"]["low_speed_count"] == 0

    # Red hit speed = 180 (red only, NOT including blue's 310)
    assert summary["red"]["hit_shooter_speed_mean"] == pytest.approx(180.0)
    # Red low_speed speed = 95
    assert summary["red"]["low_speed_shooter_speed_mean"] == pytest.approx(95.0)
    # All-team hit speed = (180 + 310) / 2 = 245
    assert summary["all_team_hit_shooter_speed_mean"] == pytest.approx(245.0)

    # Red hit speed must NOT be contaminated by blue hit
    assert abs(summary["red"]["hit_shooter_speed_mean"] - 310.0) > 1.0, \
        "red hit speed should NOT include blue's 310 m/s hit"

    # Owner IDs from term rows
    assert "red_1" in summary["red_owner_reason_from_term_rows"]
    assert "red_2" in summary["red_owner_reason_from_term_rows"]
    assert summary["red_owner_reason_from_term_rows"]["red_1"]["hit"] == 1
    assert summary["red_owner_reason_from_term_rows"]["red_2"]["low_speed"] == 1

    # JSON saved
    assert (out_dir / "missile_analysis_summary.json").exists()
    assert (out_dir / "missile_analysis_summary.csv").exists()

    # CSV contains correct values
    with open(out_dir / "missile_analysis_summary.csv") as f:
        reader = dict(csv.reader(f))
    assert int(reader["red_low_speed_count"]) == 1
    assert float(reader["red_hit_shooter_speed_mean"]) == pytest.approx(180.0)


def test_no_missiles(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "run_id,scenario,episode_id,step,sim_time,event_type,missile_id,owner_id,"
        "owner_team,target_id,target_team,lon,lat,altitude,distance_to_target,"
        "hit_success,death_caused,raw_termination_reason,AO_rad,AO_deg,TA_rad,"
        "TA_deg,flight_time_sec,launch_step,termination_step,step_delta,"
        "target_alive_at_launch,target_alive_at_termination,shooter_speed_mps,"
        "target_speed_mps,closing_speed_mps,shooter_alt_m,target_alt_m\n"
    )
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()

    from scripts.analyze_missile_events import analyse
    summary = analyse(str(csv_path), str(out_dir))

    assert summary["launch_rows"] == 0
    assert summary["termination_rows"] == 0
    assert summary["red"]["hit_count"] == 0
    assert summary["red"]["hit_rate_by_launch"] == 0.0


def test_unresolved_missiles(tmp_path):
    """Launch without matching termination -> unresolved."""
    csv_path = tmp_path / "unresolved.csv"
    csv_path.write_text(
        "run_id,scenario,episode_id,step,sim_time,event_type,missile_id,owner_id,"
        "owner_team,target_id,target_team,lon,lat,altitude,distance_to_target,"
        "hit_success,death_caused,raw_termination_reason,AO_rad,AO_deg,TA_rad,"
        "TA_deg,flight_time_sec,launch_step,termination_step,step_delta,"
        "target_alive_at_launch,target_alive_at_termination,shooter_speed_mps,"
        "target_speed_mps,closing_speed_mps,shooter_alt_m,target_alt_m\n"
        "test,cfg,0,0,0.0,launch,m0,red_1,red,blue_0,blue,,,,,,,,,,,,,,,,,,250.0,,,,\n"
        "test,cfg,0,0,0.0,launch,m1,red_1,red,blue_0,blue,,,,,,,,,,,,,,,,,,250.0,,,,\n"
        "test,cfg,0,10,2.0,termination,m0,red_1,red,blue_0,blue,,,,,,,,hit,,,,,,,,,180.0,,,,\n"
    )
    out_dir = tmp_path / "analysis"
    out_dir.mkdir()

    from scripts.analyze_missile_events import analyse
    summary = analyse(str(csv_path), str(out_dir))
    assert summary["unresolved_missiles"] == 1  # m1 launched but never terminated
