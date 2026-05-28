from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_uav_env.alignment.launch_quality import (
    LAUNCH_QUALITY_FIELDS,
    make_launch_quality_record,
)
from scripts.summarize_launch_quality import (
    build_markdown,
    percentile_summary,
    summarize_group,
    write_summary,
)
from train_vanilla_mappo import (
    _accumulate_action_clip_totals,
    _action_clip_metrics,
    _empty_action_clip_totals,
)


def test_launch_quality_record_creation() -> None:
    record = make_launch_quality_record(
        team="red",
        shooter_id="red_0",
        target_id="blue_0",
        current_step=7,
        physics_frame=84,
        range_m=1500.0,
        AO_rad=0.2,
        TA_rad=2.4,
        shooter_pos=np.array([0.0, 0.0, 3000.0]),
        shooter_vel=np.array([250.0, 0.0, 0.0]),
        target_pos=np.array([1000.0, 0.0, 3100.0]),
        target_vel=np.array([100.0, 0.0, 0.0]),
        target_alive_at_launch=True,
    )
    assert set(LAUNCH_QUALITY_FIELDS).issubset(record.keys())
    assert record["team"] == "red"
    assert record["AO_deg"] > 0.0
    assert record["relative_distance_3d_m"] > 0.0


def test_empty_percentiles() -> None:
    summary = percentile_summary([])
    assert summary["mean"] == 0.0
    assert summary["p50"] == 0.0


def test_hit_miss_grouping() -> None:
    records = [
        {"team": "red", "is_success": "True", "range_m": "1000", "AO_deg": "5",
         "TA_deg": "170", "closing_speed_mps": "300", "altitude_diff_m": "50"},
        {"team": "red", "is_success": "False", "range_m": "2000", "AO_deg": "40",
         "TA_deg": "95", "closing_speed_mps": "10", "altitude_diff_m": "-100"},
    ]
    all_summary = summarize_group(records)
    miss_summary = summarize_group([r for r in records if r["is_success"] == "False"])
    assert all_summary["launch_count"] == 2
    assert all_summary["hit_count"] == 1
    assert all_summary["hit_rate"] == 0.5
    assert miss_summary["range"]["mean"] == 2000.0


def test_action_clip_fraction() -> None:
    totals = _empty_action_clip_totals()
    raw = np.array([[0.0, 0.5, 1.2], [-1.5, 0.1, 0.2]], dtype=np.float32)
    clamped = np.clip(raw, -0.999, 0.999)
    _accumulate_action_clip_totals(totals, raw, clamped)
    metrics = _action_clip_metrics(totals)
    assert metrics["RedActionRawClipFrac"] > 0.0
    assert metrics["RedActionClampedFrac"] > 0.0
    assert metrics["RedActionRawClipFracPitch"] > 0.0
    assert metrics["RedActionRawClipFracVelocity"] > 0.0


def test_summary_writer_fake_csv() -> None:
    rows = [
        {"team": "red", "is_success": "True", "termination_reason": "hit",
         "range_m": "1000", "AO_deg": "5", "TA_deg": "170",
         "closing_speed_mps": "300", "altitude_diff_m": "50"},
        {"team": "red", "is_success": "False", "termination_reason": "overshoot",
         "range_m": "2500", "AO_deg": "43", "TA_deg": "92",
         "closing_speed_mps": "-20", "altitude_diff_m": "-150"},
        {"team": "blue", "is_success": "True", "termination_reason": "hit",
         "range_m": "900", "AO_deg": "3", "TA_deg": "178",
         "closing_speed_mps": "350", "altitude_diff_m": "20"},
    ]
    markdown = build_markdown(rows, "fake.csv")
    assert "Red all" in markdown
    assert "Blue all" in markdown

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "launch_quality.csv"
        out_path = Path(td) / "summary.md"
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_summary(csv_path, out_path)
        text = out_path.read_text(encoding="utf-8")
        assert "Launch quality summary" in text


def main() -> None:
    test_launch_quality_record_creation()
    test_empty_percentiles()
    test_hit_miss_grouping()
    test_action_clip_fraction()
    test_summary_writer_fake_csv()
    print("launch quality diagnostics static smoke test passed")


if __name__ == "__main__":
    main()
