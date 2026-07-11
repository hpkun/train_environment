from __future__ import annotations

from scripts.compare_pure_happo_checkpoints import (
    _blue_death_record, _event_key, _gate_flags,
)


def test_gate_flags_use_formal_launch_gate_records():
    info = {"__launch_gate_diagnostics__": [{
        "agent_id": "red_1", "role": "attack_uav",
        "any_range_pass": 1, "any_ata_pass": 1,
        "any_ta_pass": 1, "any_geometry_pass": 1,
    }]}
    assert _gate_flags(info) == (True, True, True, True)


def test_launch_event_keys_deduplicate_real_missile_records():
    record = {"missile_id": "red_1-m1", "shooter_id": "red_1", "target_id": "blue_0"}
    assert _event_key(record, 10) == _event_key(dict(record), 11)


def test_blue_missile_death_record_uses_real_hit_record_without_guessing():
    event = {"step": 42, "agent_id": "blue_0", "death_reason": "missile_hit", "missile_owner": "red_1"}
    done = [{"missile_id": "m7", "shooter_id": "red_1", "target_id": "blue_0", "termination_reason": "hit"}]
    record = _blue_death_record(event, done)
    assert record == {
        "step": 42, "blue_agent_id": "blue_0", "death_reason": "missile_hit",
        "killer_id": "red_1", "shooter_id": "red_1", "missile_id": "m7",
    }


def test_blue_crash_record_leaves_unavailable_missile_fields_empty():
    record = _blue_death_record(
        {"step": 9, "agent_id": "blue_1", "death_reason": "Crash_LowAlt"}, []
    )
    assert record["shooter_id"] == ""
    assert record["missile_id"] == ""
