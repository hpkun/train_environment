from __future__ import annotations

import csv
import math

from scripts.audit_reward_outcome_alignment import analyze


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mock_output(tmp_path):
    rich = tmp_path / "rich_logs"
    component_rows = [
        {
            "episode_id": "0", "agent_id": "red_0", "role": "mav", "team": "red",
            "episode_length": 50, "episode_return": 30.0, "outcome": "blue",
            "end_reason": "red_eliminated", "red_alive_final": 0,
            "blue_alive_final": 2, "mav_alive_final": 0,
            "red_launch_count": 0, "red_hit_count": 0,
            "blue_launch_count": 1, "blue_hit_count": 1,
            "paper_v1_mav_safety_sum": 50.0,
            "paper_v1_mav_support_sum": 20.0,
            "paper_v1_mav_event_raw_sum": -200.0,
            "paper_v1_mav_scaled_tam_sum": -10.0,
            "paper_v1_mav_total_sum": 60.0,
        },
        {
            "episode_id": "0", "agent_id": "red_1", "role": "attack_uav",
            "team": "red", "episode_length": 50, "episode_return": 10.0,
            "paper_v1_uav_flight_sum": 5.0,
            "paper_v1_uav_adv_sum": 50.0,
            "paper_v1_uav_end_sum": -5.0,
            "paper_v1_uav_total_sum": 50.0,
        },
        {
            "episode_id": "0", "agent_id": "red_2", "role": "attack_uav",
            "team": "red", "episode_length": 50, "episode_return": 20.0,
            "paper_v1_uav_flight_sum": 5.0,
            "paper_v1_uav_adv_sum": 60.0,
            "paper_v1_uav_end_sum": -5.0,
            "paper_v1_uav_total_sum": 60.0,
        },
        {
            "episode_id": "1", "agent_id": "red_0", "role": "mav", "team": "red",
            "episode_length": 100, "episode_return": 20.0, "outcome": "red",
            "end_reason": "timeout", "red_alive_final": 3,
            "blue_alive_final": 2, "mav_alive_final": 1,
            "red_launch_count": 1, "red_hit_count": 0,
            "blue_launch_count": 0, "blue_hit_count": 0,
            "paper_v1_mav_safety_sum": 10.0,
            "paper_v1_mav_support_sum": 30.0,
            "paper_v1_mav_event_raw_sum": 0.0,
            "paper_v1_mav_scaled_tam_sum": 0.0,
            "paper_v1_mav_total_sum": 40.0,
        },
        {
            "episode_id": "1", "agent_id": "red_1", "role": "attack_uav",
            "team": "red", "episode_length": 100, "episode_return": -2.0,
            "paper_v1_uav_flight_sum": 1.0,
            "paper_v1_uav_adv_sum": 20.0,
            "paper_v1_uav_end_sum": 0.0,
            "paper_v1_uav_total_sum": 21.0,
        },
        {
            "episode_id": "1", "agent_id": "red_2", "role": "attack_uav",
            "team": "red", "episode_length": 100, "episode_return": -2.0,
            "paper_v1_uav_flight_sum": 1.0,
            "paper_v1_uav_adv_sum": 25.0,
            "paper_v1_uav_end_sum": 0.0,
            "paper_v1_uav_total_sum": 26.0,
        },
    ]
    _write_csv(rich / "episode_reward_components.csv", component_rows)
    _write_csv(tmp_path / "terminal_episode_audit.csv", [
        {
            "episode_id": "0", "winner": "blue", "end_reason": "red_eliminated",
            "red_alive_info": 0, "blue_alive_info": 2, "mav_alive_info": 0,
            "red_launch_count": 0, "red_hit_count": 0,
            "blue_launch_count": 1, "blue_hit_count": 1,
        },
        {
            "episode_id": "1", "winner": "red", "end_reason": "timeout",
            "red_alive_info": 3, "blue_alive_info": 2, "mav_alive_info": 1,
            "red_launch_count": 1, "red_hit_count": 0,
            "blue_launch_count": 0, "blue_hit_count": 0,
        },
    ])
    _write_csv(tmp_path / "train_log.csv", [
        {"iteration": 1, "total_steps": 100, "avg_return": 1.0},
    ])


def test_reward_outcome_alignment_outputs_and_flags_without_missile_events(tmp_path):
    _mock_output(tmp_path)

    payload = analyze(tmp_path, phase_bins="0,50,100")

    assert payload["episode_count"] == 2
    assert payload["missile_event_rows"] == 0
    assert any("missile_events" in item for item in payload["limitations"])
    assert payload["misalignment_flags"]["blue_win_red_eliminated_positive_return"] is True
    assert payload["misalignment_flags"]["no_kill_high_return"] is True
    assert payload["misalignment_flags"]["mav_survival_overdominance"] is True
    assert (tmp_path / "reward_outcome_alignment_report.md").exists()
    assert (tmp_path / "reward_outcome_alignment_summary.csv").exists()
    assert (tmp_path / "reward_outcome_by_episode.csv").exists()


def test_reward_outcome_alignment_groups_and_proxies(tmp_path):
    _mock_output(tmp_path)
    analyze(tmp_path)

    with (tmp_path / "reward_outcome_alignment_summary.csv").open(newline="", encoding="utf-8") as f:
        summary = {row["outcome_group"]: row for row in csv.DictReader(f)}
    assert float(summary["red_win_timeout"]["episode_count"]) == 1
    assert float(summary["blue_win_red_eliminated"]["episode_count"]) == 1

    with (tmp_path / "reward_outcome_by_episode.csv").open(newline="", encoding="utf-8") as f:
        episodes = {row["episode_id"]: row for row in csv.DictReader(f)}
    assert float(episodes["0"]["tam_proxy_mav_event"]) == -200.0
    assert float(episodes["0"]["tam_proxy_uav_event"]) == -400.0
    assert float(episodes["0"]["brma_proxy_terminal_team"]) == -60.0
    assert float(episodes["1"]["brma_proxy_terminal_team"]) == 30.0


def test_reward_outcome_alignment_csv_has_no_nan_or_inf(tmp_path):
    _mock_output(tmp_path)
    analyze(tmp_path)

    for name in ("reward_outcome_alignment_summary.csv", "reward_outcome_by_episode.csv"):
        with (tmp_path / name).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for value in row.values():
                    if value == "":
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    assert math.isfinite(number)
