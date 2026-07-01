"""Test that per-step event buffers are drained after _get_info()."""
from __future__ import annotations


def test_event_buffers_drained_after_get_info():
    """_get_info must copy records into info then clear the step buffers."""
    from uav_env.JSBSim.env import UavCombatEnv as _Env

    env = object.__new__(_Env)
    env.agent_ids = []
    env._missile_term_reasons = {"red": {}, "blue": {}}
    env._launch_diag_step = {"red": {}, "blue": {}}
    env._launch_quality_step_records = [{"missile_id": "m1"}]
    env._launch_quality_done_step_records = [{"missile_id": "m1", "termination_reason": "hit"}]
    env._evasion_step_records = [{"evasion_triggered": 1, "incoming_missile_id": "m1"}]
    env._death_events_step = []
    env._missile_launch_range_m_effective = 8000.0
    env._missile_attack_interval_sec_effective = 0.5
    env.use_boresight_launch_gate = False
    env._launch_quality_records = {"m1": {"missile_id": "m1"}}

    info1 = env._get_info()
    assert len(info1["__launch_quality_step__"]) == 1
    assert len(info1["__launch_quality_done__"]) == 1
    assert len(info1["__evasion_events__"]) == 1
    # Buffers should be empty now
    assert len(env._launch_quality_step_records) == 0
    assert len(env._launch_quality_done_step_records) == 0
    assert len(env._evasion_step_records) == 0
    # _launch_quality_records should NOT be cleared
    assert len(env._launch_quality_records) == 1

    info2 = env._get_info()
    assert len(info2["__launch_quality_step__"]) == 0
    assert len(info2["__launch_quality_done__"]) == 0
    assert len(info2["__evasion_events__"]) == 0
