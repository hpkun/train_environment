"""Test that RichExperimentLogger de-duplicates missile events."""
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path


def _make_logger():
    from scripts.rich_logging import RichExperimentLogger
    tmp = Path(tempfile.mkdtemp(prefix="dedup_test_"))
    logger = RichExperimentLogger(
        tmp, run_id="test", method_name="test",
        scenario_name="test", device="cpu",
        num_envs=1, rollout_length_per_env=256, transitions_per_rollout=256,
    )
    return logger, tmp


def _base_record():
    return {"missile_id": "m0", "shooter_id": "red_1", "shooter_team": "red",
            "target_id": "blue_0", "target_team": "blue",
            "current_step": 100, "physics_frame": 6000,
            "launch_step": 100, "range_m": 5000.0,
            "AO_rad": 0.5, "TA_rad": 0.3,
            "shooter_role": "attack_uav", "target_role": "attack_uav",
            "shooter_alt_m": 6000, "target_alt_m": 6000,
            "shooter_speed_mps": 300, "target_speed_mps": 280,
            "closing_speed_mps": 200}


def test_launch_dedup():
    logger, tmp = _make_logger()
    try:
        rec = _base_record()
        info = {"__launch_quality_step__": [dict(rec)]}
        logger.write_missile_events(info, scenario="test", episode_id="0", step=100)
        logger.write_missile_events(info, scenario="test", episode_id="0", step=100)
        logger.close()
        with open(tmp / "missile_events.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        launches = [r for r in rows if r["event_type"] == "launch"]
        assert len(launches) == 1, f"expected 1 launch, got {len(launches)}"
    finally:
        logger.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_missile_ids_not_deduped():
    logger, tmp = _make_logger()
    try:
        rec1 = _base_record()
        rec2 = dict(_base_record(), missile_id="m1", physics_frame=6120)
        info = {"__launch_quality_step__": [dict(rec1), dict(rec2)]}
        logger.write_missile_events(info, scenario="test", episode_id="0", step=100)
        logger.close()
        with open(tmp / "missile_events.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        launches = [r for r in rows if r["event_type"] == "launch"]
        assert len(launches) == 2, f"expected 2 launches, got {len(launches)}"
    finally:
        logger.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_different_episodes_not_deduped():
    logger, tmp = _make_logger()
    try:
        rec = _base_record()
        logger.write_missile_events({"__launch_quality_step__": [dict(rec)]}, scenario="test", episode_id="0", step=100)
        logger.write_missile_events({"__launch_quality_step__": [dict(rec)]}, scenario="test", episode_id="1", step=50)
        logger.close()
        with open(tmp / "missile_events.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        launches = [r for r in rows if r["event_type"] == "launch"]
        assert len(launches) == 2, f"expected 2 launches, got {len(launches)}"
    finally:
        logger.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_termination_dedup():
    logger, tmp = _make_logger()
    try:
        rec = {"missile_id": "m0", "shooter_id": "red_1", "shooter_team": "red",
               "target_id": "blue_0", "target_team": "blue",
               "termination_reason": "hit", "is_success": True,
               "termination_step": 200, "range_m": 100.0}
        info = {"__launch_quality_done__": [dict(rec)]}
        logger.write_missile_events(info, scenario="test", episode_id="0", step=200)
        logger.write_missile_events(info, scenario="test", episode_id="0", step=200)
        logger.close()
        with open(tmp / "missile_events.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        terms = [r for r in rows if r["event_type"] == "hit"]
        assert len(terms) == 1, f"expected 1 hit, got {len(terms)}"
    finally:
        logger.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_evasion_dedup():
    logger, tmp = _make_logger()
    try:
        rec = {"evasion_triggered": 1, "evasion_team": "red",
               "evasion_agent_id": "red_1", "incoming_missile_id": "m0",
               "incoming_range_m": 5000, "incoming_closing_speed_mps": 400,
               "incoming_t_go_sec": 12.5, "evasion_mode": "brma_scripted"}
        info = {"__evasion_events__": [dict(rec)]}
        logger.write_missile_events(info, scenario="test", episode_id="0", step=100)
        logger.write_missile_events(info, scenario="test", episode_id="0", step=100)
        logger.close()
        with open(tmp / "missile_events.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ev = [r for r in rows if r["event_type"] == "evasion"]
        assert len(ev) == 1, f"expected 1 evasion, got {len(ev)}"
    finally:
        logger.close()
        shutil.rmtree(tmp, ignore_errors=True)
