from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uav_env import make_env  # noqa: E402
from uav_env.JSBSim.env import UavCombatEnv  # noqa: E402

CFG_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml"
CFG_5V4 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_tam_brma_mav_guided_v1.yaml"
BASE_3V2 = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_tam_brma_paper_aligned_v1.yaml"


def _dummy_env(policy: str):
    env = UavCombatEnv.__new__(UavCombatEnv)
    env.red_uav_track_policy = policy
    env.blue_ids = ["blue_0"]
    env.red_ids = ["red_0", "red_1"]
    env.agent_roles = {"red_0": "mav", "red_1": "attack_uav"}
    env._last_step_obs = {}
    env.red_planes = {"red_0": SimpleNamespace(is_alive=True)}
    env.blue_planes = {"blue_0": SimpleNamespace(uid="blue_0", is_alive=True)}
    env.use_boresight_launch_gate = False
    return env


def _set_red_obs(env, *, direct: float, shared: float, observed: float = 1.0):
    env._last_step_obs = {
        "red_1": {
            "enemy_track_source": np.array([[direct, shared]], dtype=np.float32),
            "enemy_observed_mask": np.array([observed], dtype=np.float32),
        }
    }


def test_default_red_uav_track_policy_preserves_legacy_behavior():
    env = make_env(BASE_3V2, max_steps=5)
    try:
        assert env.red_uav_track_policy == "direct_or_mav_shared"
        assert env.red_target_selection_mode == "closest"
        assert math.degrees(env.MISSILE_LAUNCH_TA_THRESH) == pytest.approx(90.0)
    finally:
        env.close()


@pytest.mark.parametrize("config", [CFG_3V2, CFG_5V4])
def test_mav_guided_v1_configs_load_and_apply_contract(config: str):
    env = make_env(config, max_steps=5)
    try:
        assert env.hetero_reward_mode == "tam_brma_paper_aligned_v1"
        assert env.observation_mode == "mav_shared_geo"
        assert env.red_uav_track_policy == "mav_required_when_alive"
        assert env.red_target_selection_mode == "mav_threat_rank"
        assert math.degrees(env.MISSILE_LAUNCH_AO_THRESH) == pytest.approx(60.0)
        assert math.degrees(env.MISSILE_LAUNCH_TA_THRESH) == pytest.approx(90.0)
        assert env.MISSILE_LAUNCH_RANGE_THRESH == pytest.approx(10000.0)
        assert env.MISSILE_LAUNCH_MIN_RANGE == pytest.approx(500.0)
        assert env._missile_attack_interval_sec_effective == pytest.approx(0.5)
        assert env._num_missiles_for("red_0") == 0
        for rid in env.red_ids[1:]:
            assert env._num_missiles_for(rid) == 2
        assert env.action_space["red_0"].shape == (3,)
        assert env.observation_space["red_0"].contains(env.observation_space["red_0"].sample())
    finally:
        env.close()


def test_direct_or_mav_shared_allows_direct_track():
    env = _dummy_env("direct_or_mav_shared")
    _set_red_obs(env, direct=1.0, shared=0.0)
    assert env._has_launch_track("red_1", "blue_0") == (True, "direct")


def test_mav_preferred_when_alive_prefers_shared_track():
    env = _dummy_env("mav_preferred_when_alive")
    _set_red_obs(env, direct=1.0, shared=1.0)
    assert env._has_launch_track("red_1", "blue_0") == (True, "mav_shared")


def test_mav_required_when_alive_blocks_direct_only_if_mav_observes(monkeypatch):
    env = _dummy_env("mav_required_when_alive")
    monkeypatch.setattr(env, "_red_mav_observes_target", lambda _target: True)
    _set_red_obs(env, direct=1.0, shared=0.0)
    assert env._has_launch_track("red_1", "blue_0") == (False, "mav_required_missing_shared")


def test_mav_required_when_mav_dead_allows_direct_fallback():
    env = _dummy_env("mav_required_when_alive")
    env.red_planes["red_0"].is_alive = False
    _set_red_obs(env, direct=1.0, shared=0.0)
    assert env._has_launch_track("red_1", "blue_0") == (True, "direct")


def test_mav_role_still_cannot_launch():
    env = _dummy_env("mav_required_when_alive")
    _set_red_obs(env, direct=1.0, shared=1.0)
    assert env._has_launch_track("red_0", "blue_0") == (False, "role_blocked_mav")


def test_mav_threat_rank_does_not_bypass_geometry_gate(monkeypatch):
    env = _dummy_env("direct_or_mav_shared")
    env.red_target_selection_mode = "mav_threat_rank"
    env.MISSILE_LAUNCH_AO_THRESH = np.deg2rad(60.0)
    env.MISSILE_LAUNCH_TA_THRESH = np.deg2rad(90.0)
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 10000.0
    env._engaged_targets = set()
    shooter = SimpleNamespace(uid="red_1", is_alive=True)
    target = SimpleNamespace(uid="blue_0", is_alive=True)
    enemies = {"blue_0": target}
    monkeypatch.setattr(env, "_has_launch_track", lambda aid, tid: (True, "mav_shared"))
    monkeypatch.setattr(
        env,
        "_missile_candidate_metrics",
        lambda _s, _t: {
            "range_m": 9000.0,
            "AO_rad": np.deg2rad(80.0),
            "TA_rad": np.deg2rad(120.0),
            "range_ok": True,
            "ao_ok": False,
            "ta_ok": True,
            "boresight_ok_3d": True,
        },
    )
    target_out, *_ = env._select_missile_target(
        "red_1", shooter, enemies, {"alive_enemy_pairs": 0, "unengaged_enemy_pairs": 0,
                                    "track_unobserved_blocked": 0, "range_ok_pairs": 0,
                                    "ao_ok_pairs": 0, "ta_ok_pairs": 0, "geometry_ok_pairs": 0}
    )
    assert target_out is None


def test_launch_diagnostics_summary_counts_mav_shared_usage():
    import eval_policy_launch_diagnostics as script

    rows = [
        {
            "episode_id": 0,
            "step": 10,
            "launch_track_source": "mav_shared",
            "actual_missiles_fired_this_step": 1,
            "actual_red_hit_delta_this_step": 0,
            "missile_hits": 0,
            "blue_dead": 0,
            "range_ok": True,
            "ao_ok": True,
            "ta_ok": True,
            "lock_ready": True,
            "cooldown_ready": True,
            "deconflict_ok": True,
            "track_available": True,
            "direct_track_available": False,
            "mav_shared_track_available": True,
            "final_launch_allowed": True,
            "launch_allowed": True,
            "predicted_allowed_but_not_fired": 0,
            "fired_without_predicted_allowed": 0,
            "predicted_vs_final_mismatch": 0,
            "launch_block_reason_primary": "allowed",
            "action_pitch": 0.0,
            "action_heading": 0.0,
            "action_speed": 0.0,
        },
        {
            "episode_id": 0,
            "step": 20,
            "launch_track_source": "direct",
            "actual_missiles_fired_this_step": 1,
            "actual_red_hit_delta_this_step": 1,
            "missile_hits": 1,
            "blue_dead": 1,
            "range_ok": True,
            "ao_ok": True,
            "ta_ok": True,
            "lock_ready": True,
            "cooldown_ready": True,
            "deconflict_ok": True,
            "track_available": True,
            "direct_track_available": True,
            "mav_shared_track_available": False,
            "final_launch_allowed": True,
            "launch_allowed": True,
            "predicted_allowed_but_not_fired": 0,
            "fired_without_predicted_allowed": 0,
            "predicted_vs_final_mismatch": 0,
            "launch_block_reason_primary": "allowed",
            "action_pitch": 0.0,
            "action_heading": 0.0,
            "action_speed": 0.0,
        },
    ]
    summary = script._summarize(rows, episodes=1, label="m", scenario="3v2", arch="brma_recurrent_masked")
    assert summary["red_launch_mav_shared_count"] == 1
    assert summary["red_launch_direct_count"] == 1
    assert summary["red_hit_direct_count"] == 1
    assert summary["red_hit_mav_shared_count"] == 0
    assert summary["first_red_launch_step"] == 10
    assert summary["first_red_mav_shared_launch_step"] == 10


def _minimal_diag_row(**overrides):
    row = {
        "episode_id": 0,
        "step": 10,
        "launch_track_source": "none",
        "actual_missiles_fired_this_step": 0,
        "actual_red_hit_delta_this_step": 0,
        "missile_hits": 0,
        "blue_dead": 0,
        "range_ok": True,
        "ao_ok": True,
        "ta_ok": True,
        "lock_ready": True,
        "cooldown_ready": True,
        "deconflict_ok": True,
        "track_available": True,
        "direct_track_available": True,
        "mav_shared_track_available": True,
        "final_launch_allowed": True,
        "launch_allowed": True,
        "predicted_allowed_but_not_fired": 0,
        "fired_without_predicted_allowed": 0,
        "predicted_vs_final_mismatch": 0,
        "launch_block_reason_primary": "allowed",
        "action_pitch": 0.0,
        "action_heading": 0.0,
        "action_speed": 0.0,
    }
    row.update(overrides)
    return row


def test_launch_diagnostics_summary_uses_actual_launch_records_over_mixed_rows():
    import eval_policy_launch_diagnostics as script

    rows = [
        _minimal_diag_row(
            step=12,
            launch_track_source="mixed",
            actual_missiles_fired_this_step=1,
        )
    ]
    actual_launch_events = {
        ("missile", "m1"): {
            "step": 12,
            "missile_id": "m1",
            "source": "mav_shared",
        }
    }

    summary = script._summarize(
        rows,
        episodes=1,
        label="m",
        scenario="3v2",
        arch="brma_recurrent_masked",
        actual_launch_events=actual_launch_events,
        actual_hit_events={},
    )

    assert summary["red_launch_mav_shared_count"] == 1
    assert summary["red_launch_direct_count"] == 0
    assert summary["red_launch_unknown_source_count"] == 0
    assert summary["first_red_mav_shared_launch_step"] == 12


def test_launch_diagnostics_records_actual_direct_and_shared_launch_events():
    import eval_policy_launch_diagnostics as script

    events = {}
    script._record_actual_launch_event(
        events,
        {
            "team": "red",
            "shooter_id": "red_1",
            "target_id": "blue_0",
            "missile_id": "m1",
            "launch_track_source": "mav_shared",
        },
        episode_id=0,
        step=11,
    )
    script._record_actual_launch_event(
        events,
        {
            "shooter_team": "red",
            "shooter_id": "red_2",
            "target_id": "blue_1",
            "missile_id": "m2",
            "launch_track_source": "direct",
        },
        episode_id=0,
        step=17,
    )
    script._record_actual_launch_event(
        events,
        {
            "team": "blue",
            "shooter_id": "blue_0",
            "target_id": "red_1",
            "missile_id": "b1",
            "launch_track_source": "direct",
        },
        episode_id=0,
        step=18,
    )

    summary = script._summarize(
        [_minimal_diag_row(step=11), _minimal_diag_row(step=17)],
        episodes=1,
        label="m",
        scenario="3v2",
        arch="brma_recurrent_masked",
        actual_launch_events=events,
        actual_hit_events={},
    )

    assert summary["red_launch_mav_shared_count"] == 1
    assert summary["red_launch_direct_count"] == 1
    assert summary["red_launch_unknown_source_count"] == 0
    assert summary["first_red_launch_step"] == 11
    assert summary["first_red_mav_shared_launch_step"] == 11


def test_launch_diagnostics_summary_deduplicates_hits_by_missile_id():
    import eval_policy_launch_diagnostics as script

    rows = [
        _minimal_diag_row(step=30, actual_red_hit_delta_this_step=1),
        _minimal_diag_row(step=30, red_id="red_2", actual_red_hit_delta_this_step=1),
    ]
    actual_hit_events = {}
    hit_record = {
        "team": "red",
        "shooter_id": "red_1",
        "target_id": "blue_0",
        "missile_id": "m2",
        "launch_track_source": "mav_shared",
        "raw_termination_reason": "hit",
    }
    script._record_actual_hit_event(actual_hit_events, hit_record, episode_id=0, step=30)
    script._record_actual_hit_event(actual_hit_events, hit_record, episode_id=0, step=30)

    summary = script._summarize(
        rows,
        episodes=1,
        label="m",
        scenario="3v2",
        arch="brma_recurrent_masked",
        actual_launch_events={},
        actual_hit_events=actual_hit_events,
    )

    assert summary["red_hit_mav_shared_count"] == 1
    assert summary["red_hit_direct_count"] == 0
    assert summary["red_hit_unknown_source_count"] == 0
    assert summary["first_red_mav_shared_hit_step"] == 30


def test_launch_diagnostics_summary_counts_unknown_actual_source_separately():
    import eval_policy_launch_diagnostics as script

    rows = [_minimal_diag_row(step=15)]
    actual_launch_events = {
        ("missile", "m3"): {
            "step": 15,
            "missile_id": "m3",
            "source": "mixed",
        }
    }
    actual_hit_events = {
        ("missile", "m3"): {
            "step": 22,
            "missile_id": "m3",
            "source": "mixed",
        }
    }

    summary = script._summarize(
        rows,
        episodes=1,
        label="m",
        scenario="3v2",
        arch="brma_recurrent_masked",
        actual_launch_events=actual_launch_events,
        actual_hit_events=actual_hit_events,
    )

    assert summary["red_launch_direct_count"] == 0
    assert summary["red_launch_mav_shared_count"] == 0
    assert summary["red_launch_unknown_source_count"] == 1
    assert summary["red_hit_direct_count"] == 0
    assert summary["red_hit_mav_shared_count"] == 0
    assert summary["red_hit_unknown_source_count"] == 1


@pytest.mark.parametrize("arch", ["pure_happo", "pure_happo_tanh"])
def test_launch_diagnostics_builds_pure_happo_policy_arches(arch: str):
    import eval_policy_launch_diagnostics as script
    from algorithms.pure_happo import PureHAPPOPolicy

    policy = script._build_policy(
        {
            "policy_arch": arch,
            "actor_obs_dim": 96,
            "critic_state_dim": 480,
            "num_agents": 3,
        },
        torch.device("cpu"),
    )

    assert isinstance(policy, PureHAPPOPolicy)
    assert policy.actor_obs_dim == 96
    assert policy.critic_state_dim == 480
    assert policy.action_dim == 3
    assert policy.num_agents == 3


def test_launch_diagnostics_build_policy_existing_arches_still_work():
    import eval_policy_launch_diagnostics as script
    from algorithms.happo import BRMARecurrentMaskedHAPPOReferencePolicy, HAPPOReferencePolicy

    flat = script._build_policy(
        {"policy_arch": "flat", "actor_obs_dim": 96, "critic_state_dim": 480},
        torch.device("cpu"),
    )
    recurrent = script._build_policy(
        {"policy_arch": "brma_recurrent_masked", "entity_dim": 19, "critic_state_dim": 480},
        torch.device("cpu"),
    )

    assert isinstance(flat, HAPPOReferencePolicy)
    assert isinstance(recurrent, BRMARecurrentMaskedHAPPOReferencePolicy)
