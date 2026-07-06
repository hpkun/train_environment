from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_eval_policy_launch_diagnostics_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_policy_launch_diagnostics.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Evaluate learned policy launch-envelope diagnostics" in result.stdout
    assert "--scenario" in result.stdout
    assert "--diagnostic-output-dir" in result.stdout


def test_eval_policy_launch_diagnostics_summary():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    rows = [
        {
            "episode_id": 0,
            "range_ok": True,
            "ao_ok": False,
            "ta_ok": True,
            "lock_ready": False,
            "launch_allowed": False,
            "launch_block_reason": "ao_blocked",
            "action_pitch": 0.1,
            "action_heading": 0.2,
            "action_speed": 0.3,
            "missiles_fired": 0,
            "missile_hits": 0,
            "blue_dead": 0,
        },
        {
            "episode_id": 0,
            "range_ok": True,
            "ao_ok": True,
            "ta_ok": True,
            "lock_ready": True,
            "launch_allowed": True,
            "launch_block_reason": "allowed",
            "action_pitch": 1.0,
            "action_heading": 0.0,
            "action_speed": 0.0,
            "missiles_fired": 1,
            "missile_hits": 1,
            "blue_dead": 1,
        },
    ]
    summary = script._summarize(rows, 1, "model", "3v2", "flat")
    assert summary["range_ok_rate"] == 1.0
    assert summary["ao_ok_rate"] == 0.5
    assert summary["lock_ready_rate"] == 0.5
    assert summary["red_missiles_fired"] == 1
    assert summary["missile_hits"] == 1
    assert summary["dominant_block_reason"] in {"ao_blocked", "allowed"}


class _Sim:
    def __init__(self, uid, alive=True, pos=(0.0, 0.0, 0.0), missiles=1):
        self.uid = uid
        self.is_alive = alive
        self.num_left_missiles = missiles
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_position(self):
        return self._pos


class _Env:
    def __init__(self):
        self.blue_ids = ["blue_0", "blue_1", "blue_2"]
        self.red_ids = ["red_0", "red_1"]
        self.agent_roles = {"red_0": "mav", "red_1": "attack_uav"}
        self.red_planes = {"red_1": _Sim("red_1", True, (0.0, 0.0, 0.0), missiles=2)}
        self.blue_planes = {
            "blue_0": _Sim("blue_0", False, (1000.0, 0.0, 0.0)),
            "blue_1": _Sim("blue_1", True, (2000.0, 0.0, 0.0)),
            "blue_2": _Sim("blue_2", True, (5000.0, 0.0, 0.0)),
        }
        self._engaged_targets = {"blue_2"}
        self._agents_deny_kill = set()
        self.use_boresight_launch_gate = False


def test_pre_step_track_snapshot_uses_pre_step_obs_not_post_step():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    env = _Env()
    diag = {
        "target_id": "blue_1",
        "has_missile": True,
        "target_alive": True,
        "range_ok": True,
        "ao_ok": True,
        "ta_ok": True,
        "lock_ready": True,
        "cooldown_ready": True,
        "deconflict_ok": True,
        "launch_allowed_predicted": True,
    }
    pre_obs = {"red_1": {"enemy_track_source": np.asarray([[0, 0], [0, 1], [0, 0]], dtype=np.float32)}}
    post_obs = {"red_1": {"enemy_track_source": np.zeros((3, 2), dtype=np.float32)}}

    pre = script._pre_step_launch_snapshot(env, pre_obs, "red_1", diag)
    post = script._pre_step_launch_snapshot(env, post_obs, "red_1", diag)

    assert pre["mav_shared_track_available"] is True
    assert pre["launch_track_source"] == "mav_shared"
    assert pre["final_launch_allowed"] is True
    assert post["mav_shared_track_available"] is False
    assert post["final_launch_allowed"] is False


def test_candidate_counts_use_alive_and_unengaged_blue_counts():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    env = _Env()
    stats = script._target_candidate_stats(env, "red_1", "blue_1")

    assert stats["alive_blue_count"] == 2
    assert stats["unengaged_alive_blue_count"] == 1
    assert stats["candidate_count"] == 2
    assert stats["nearest_blue_id"] == "blue_1"


def test_summary_hits_sum_unique_step_deltas_not_max_cumulative():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    rows = [
        {"episode_id": 0, "step": 1, "actual_red_hit_delta_this_step": 1,
         "actual_missiles_fired_this_step": 0, "range_ok": True, "ao_ok": True,
         "ta_ok": True, "lock_ready": True, "cooldown_ready": True, "deconflict_ok": True,
         "track_available": True, "direct_track_available": True, "mav_shared_track_available": False,
         "final_launch_allowed": True, "launch_allowed": True, "action_pitch": 0,
         "action_heading": 0, "action_speed": 0, "blue_dead": 1,
         "launch_block_reason_primary": "allowed", "missile_hits": 1},
        {"episode_id": 0, "step": 1, "actual_red_hit_delta_this_step": 1,
         "actual_missiles_fired_this_step": 0, "range_ok": True, "ao_ok": True,
         "ta_ok": True, "lock_ready": True, "cooldown_ready": True, "deconflict_ok": True,
         "track_available": True, "direct_track_available": True, "mav_shared_track_available": False,
         "final_launch_allowed": True, "launch_allowed": True, "action_pitch": 0,
         "action_heading": 0, "action_speed": 0, "blue_dead": 1,
         "launch_block_reason_primary": "allowed", "missile_hits": 1},
        {"episode_id": 1, "step": 2, "actual_red_hit_delta_this_step": 1,
         "actual_missiles_fired_this_step": 1, "range_ok": True, "ao_ok": True,
         "ta_ok": True, "lock_ready": True, "cooldown_ready": True, "deconflict_ok": True,
         "track_available": True, "direct_track_available": True, "mav_shared_track_available": False,
         "final_launch_allowed": True, "launch_allowed": True, "action_pitch": 0,
         "action_heading": 0, "action_speed": 0, "blue_dead": 1,
         "launch_block_reason_primary": "allowed", "missile_hits": 1},
    ]
    summary = script._summarize(rows, 2, "m", "3v2", "flat")
    assert summary["missile_hits"] == 2
    assert summary["red_missiles_fired"] == 1


def test_block_reason_priority_delimiter_and_boresight_not_enabled():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    reasons = script._block_reasons(
        {"shooter_alive": True, "has_missile": True, "target_alive": True,
         "range_ok": False, "ao_ok": False, "ta_ok": True, "lock_ready": True,
         "cooldown_ready": True, "deconflict_ok": True, "kill_cooldown": False},
        track_available=False,
        boresight_ok=True,
    )
    assert reasons[0] == "no_track"
    assert ";".join(reasons).startswith("no_track;out_of_range")

    allowed = script._block_reasons(
        {"shooter_alive": True, "has_missile": True, "target_alive": True,
         "range_ok": True, "ao_ok": True, "ta_ok": True, "lock_ready": True,
         "cooldown_ready": True, "deconflict_ok": True, "kill_cooldown": False},
        track_available=True,
        boresight_ok=True,
    )
    assert allowed == ["allowed"]


def test_zero_dead_recurrent_hidden_clears_only_dead_red_agents():
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_policy_launch_diagnostics as script

    env = type("E", (), {"red_ids": ["red_0", "red_1", "red_2"]})()
    hidden = np.ones((3, 4), dtype=np.float32)
    info = {"red_0": {"alive": True}, "red_1": {"alive": False}, "red_2": {"alive": True}}

    out = script._zero_dead_recurrent_hidden(hidden, env, info)

    assert np.all(out[0] == 1.0)
    assert np.all(out[1] == 0.0)
    assert np.all(out[2] == 1.0)
