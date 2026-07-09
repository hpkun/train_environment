from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_tam_happo_direct.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_tam_happo_direct", TRAIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_action_dim4_train_log_header_has_maneuver_fcs_four_axis_fields():
    module = _load_train_module()

    header = module._train_log_header(
        action_dim=4, tam_control_mode="maneuver_fcs")

    required = {
        "mav_action_mean_throttle",
        "mav_action_mean_roll_cmd",
        "mav_action_mean_pitch_or_load_cmd",
        "mav_action_mean_yaw_cmd",
        "uav_action_mean_throttle",
        "uav_action_mean_roll_cmd",
        "uav_action_mean_pitch_or_load_cmd",
        "uav_action_mean_yaw_cmd",
        "mav_action_std_yaw_cmd",
        "uav_action_mean_abs_yaw_cmd",
        "mav_action_saturation_yaw_cmd",
        "uav_action_active_yaw_cmd",
    }
    assert required.issubset(set(header))
    assert "mav_action_mean_pitch" not in header
    assert "mav_action_mean_heading" not in header
    assert "mav_action_mean_speed" not in header


def test_action_dim3_train_log_header_keeps_legacy_action_fields_only():
    module = _load_train_module()

    header = module._train_log_header(action_dim=3, tam_control_mode="legacy_pid_3d")

    assert "mav_action_saturation_rate" in header
    assert "uav_action_saturation_rate" in header
    assert "mav_action_mean_yaw_cmd" not in header


def test_maneuver_action_stats_include_yaw_cmd_for_action_dim4():
    module = _load_train_module()
    actions = np.array(
        [
            [[0.0, 0.1, -0.2, 0.3], [0.2, -0.4, 0.6, -0.8]],
            [[1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    active = np.ones((2, 2), dtype=np.float32)
    role_ids = [0, 1]

    stats = module._maneuver_action_stats(
        actions, active, role_ids, action_dim=4, tam_control_mode="maneuver_fcs")

    assert abs(stats["mav_action_mean_yaw_cmd"] - 0.65) < 1e-6
    assert stats["uav_action_active_yaw_cmd"] == 2
    assert "uav_action_mean_pitch_or_load_cmd" in stats


def test_checkpoint_meta_records_tam_action_order_for_action_dim4():
    module = _load_train_module()

    meta = module._action_semantics_meta(action_dim=4, tam_control_mode="maneuver_fcs")

    assert meta["tam_action_order"] == [
        "throttle", "roll_cmd", "pitch_or_load_cmd", "yaw_cmd"
    ]
    assert meta["action_semantics"] == meta["tam_action_order"]


def test_final_eval_call_exists_after_latest_checkpoint_save():
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    source = TRAIN.read_text(encoding="utf-8")

    assert "def _maybe_run_final_eval" in source
    assert "_maybe_run_final_eval(" in source
    assert "final_eval" in source
    assert "latest_model = out_dir / \"latest\" / \"model.pt\"" in source


def test_episode_termination_summary_writer_creates_required_columns(tmp_path):
    module = _load_train_module()
    path = tmp_path / "episode_termination_summary.csv"
    writer, handle = module._open_episode_termination_summary(path)
    try:
        module._write_episode_termination_summary(
            writer,
            {
                "episode_id": 3,
                "episode_length": 12,
                "winner": "blue",
                "end_reason": "red_eliminated",
                "red_alive_final": 0,
                "blue_alive_final": 2,
                "mav_alive_final": 0,
                "red_dead_count": 3,
                "blue_dead_count": 0,
                "red_missiles_fired": 1,
                "blue_missiles_fired": 0,
                "red_missile_hits": 0,
                "blue_missile_hits": 0,
                "red_crash_count": 2,
            },
        )
    finally:
        handle.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["episode_id"] == "3"
    for field in module.EPISODE_TERMINATION_FIELDS:
        assert field in rows[0]


def test_tam_control_summary_writer_creates_required_columns(tmp_path):
    module = _load_train_module()
    path = tmp_path / "tam_control_summary.csv"
    writer, handle = module._open_tam_control_summary(path)
    try:
        module._write_tam_control_summary_row(
            writer,
            {
                "total_steps": 1,
                "episode_id": 0,
                "agent_id": "red_0",
                "alive": 1,
                "raw_action_0": 0.1,
                "raw_action_1": 0.2,
                "raw_action_2": 0.3,
                "raw_action_3": 0.4,
            },
        )
    finally:
        handle.close()

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["agent_id"] == "red_0"
    for field in module.TAM_CONTROL_SUMMARY_FIELDS:
        assert field in rows[0]
