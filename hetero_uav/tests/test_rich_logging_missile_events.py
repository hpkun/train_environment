from __future__ import annotations

import csv

from scripts.rich_logging import RichExperimentLogger


def test_rich_logger_writes_missile_events(tmp_path):
    logger = RichExperimentLogger(
        tmp_path,
        run_id="run",
        method_name="method",
        scenario_name="scenario",
        device="cpu",
        num_envs=1,
        rollout_length_per_env=4,
        transitions_per_rollout=4,
    )
    try:
        logger.write_missile_events(
            {
                "__launch_quality_step__": [
                    {
                        "missile_id": "m0",
                        "shooter_id": "red_1",
                        "shooter_team": "red",
                        "target_id": "blue_0",
                        "target_team": "blue",
                        "range_m": 4500.0,
                        "shooter_alt_m": 6000.0,
                    }
                ],
                "__launch_quality_done__": [
                    {
                        "missile_id": "m0",
                        "shooter_id": "red_1",
                        "shooter_team": "red",
                        "target_id": "blue_0",
                        "target_team": "blue",
                        "range_m": 4500.0,
                        "shooter_alt_m": 6000.0,
                        "termination_reason": "hit",
                        "is_success": True,
                    }
                ],
            },
            scenario="test_scenario",
            episode_id=7,
            step=12,
            sim_time=2.4,
        )
    finally:
        logger.close()

    rows = list(csv.DictReader((tmp_path / "missile_events.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["event_type"] == "launch"
    assert rows[0]["owner_id"] == "red_1"
    assert rows[0]["target_id"] == "blue_0"
    assert rows[1]["event_type"] == "hit"
    assert rows[1]["hit_success"] == "1"
    assert rows[1]["death_caused"] == "1"
