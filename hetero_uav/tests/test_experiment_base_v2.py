from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_reward_mode_kwargs_do_not_override_yaml_when_omitted():
    from scripts.train_happo_reference import _make_env_kwargs_for_reward_mode

    assert _make_env_kwargs_for_reward_mode(None) == {}
    assert _make_env_kwargs_for_reward_mode("happo_ref_v0") == {
        "hetero_reward_mode": "happo_ref_v0",
    }


def test_common_meta_records_actual_reward_and_rich_logging_mode():
    from scripts.train_happo_reference import _experiment_base_v2_meta

    meta = _experiment_base_v2_meta(
        actual_reward_mode="happo_ref_v1_mav_support",
        rich_logging_enabled=True,
        rich_log_mode="summary",
    )
    assert meta["actual_reward_mode"] == "happo_ref_v1_mav_support"
    assert meta["reward_mode"] == "happo_ref_v1_mav_support"
    assert meta["rich_logging_enabled"] is True
    assert meta["rich_log_mode"] == "summary"
    assert meta["per_step_reward_components"] is False
    assert meta["aircraft_timeseries"] is False


def test_rich_logger_summary_mode_skips_per_step_files(tmp_path):
    from scripts.rich_logging import RichExperimentLogger

    logger = RichExperimentLogger(
        tmp_path,
        run_id="run",
        method_name="method",
        scenario_name="scenario",
        device="cpu",
        num_envs=1,
        rollout_length_per_env=1,
        transitions_per_rollout=1,
        mode="summary",
    )
    logger.write_reward_components(
        {"reward_components": {"red_0": {"total": 1.0}}},
        scenario="s",
        episode_id=0,
        step=1,
    )
    logger.write_aircraft_timeseries(
        object(),
        scenario="s",
        episode_id=0,
        step=1,
    )
    logger.write_train_metrics({"train_steps": 1, "total_env_steps_actual": 1})
    logger.write_missile_events({}, scenario="s", episode_id=0, step=1)
    logger.write_episode_reward_components(
        scenario="s",
        episode_id=0,
        agent_id="red_0",
        role="mav",
        team="red",
        episode_length=1,
        episode_return=0.0,
        component_sums={},
    )
    logger.close()

    for filename in ("reward_components.csv", "aircraft_timeseries.csv"):
        rows = list(csv.reader((tmp_path / filename).open(newline="", encoding="utf-8")))
        assert len(rows) == 1
    assert sum(1 for _ in (tmp_path / "train_metrics.csv").open(encoding="utf-8")) == 2
    assert (tmp_path / "missile_events.csv").exists()
    assert (tmp_path / "episode_reward_components.csv").exists()


def test_launch_diagnostics_summary_fields_include_track_and_fire_accounting():
    import eval_policy_launch_diagnostics as script

    rows = [
        {
            "launch_block_reason_primary": "no_track",
            "launch_block_reason": "no_track",
            "range_ok": True,
            "ao_ok": False,
            "ta_ok": True,
            "lock_ready": True,
            "cooldown_ready": True,
            "deconflict_ok": True,
            "track_available": False,
            "direct_track_available": False,
            "mav_shared_track_available": False,
            "final_launch_allowed": False,
            "launch_allowed": False,
            "actual_missiles_fired_this_step": 0,
            "predicted_allowed_but_not_fired": 0,
            "fired_without_predicted_allowed": 0,
            "action_pitch": 0.0,
            "action_heading": 0.5,
            "action_speed": 1.0,
            "missiles_fired": 0,
            "missile_hits": 0,
            "blue_dead": 0,
            "episode_id": 0,
        },
        {
            "launch_block_reason_primary": "allowed",
            "launch_block_reason": "allowed",
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
            "actual_missiles_fired_this_step": 1,
            "predicted_allowed_but_not_fired": 0,
            "fired_without_predicted_allowed": 0,
            "action_pitch": 0.0,
            "action_heading": 0.0,
            "action_speed": 0.0,
            "missiles_fired": 1,
            "missile_hits": 1,
            "blue_dead": 1,
            "episode_id": 0,
        },
    ]
    summary = script._summarize(rows, episodes=1, label="m", scenario="3v2", arch="brma_recurrent_masked")
    for key in (
        "cooldown_ready_rate",
        "deconflict_ok_rate",
        "track_available_rate",
        "direct_track_available_rate",
        "mav_shared_track_available_rate",
        "final_launch_allowed_rate",
        "actual_fire_rate",
        "predicted_allowed_but_not_fired_count",
        "fired_without_predicted_allowed_count",
        "recurrent_eval_used",
    ):
        assert key in summary
    assert summary["dominant_block_reason"] == "no_track"
    assert summary["actual_fire_rate"] == pytest.approx(0.5)
    assert summary["mav_shared_track_available_rate"] == pytest.approx(0.5)
    assert summary["recurrent_eval_used"] is True


def test_v1_configs_keep_static_environment_contracts():
    cfgs = [
        ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
        ROOT / "uav_env/JSBSim/configs/hetero_mav_shared_geo_5v4_f16_mav_surrogate_happo_ref_v1_mav_support.yaml",
    ]
    for path in cfgs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["red_target_selection_mode"] == "closest"
        assert cfg["aircraft_type_params"]["mav"]["num_missiles"] == 0
        assert cfg["observation_mode"] == "mav_shared_geo"
        assert "missile_launch_range_m" not in cfg
        assert "missile_launch_ao_thresh" not in cfg
        assert "missile_launch_ta_thresh" not in cfg
