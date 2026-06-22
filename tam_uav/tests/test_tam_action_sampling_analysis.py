import csv

from scripts.analyze_tam_action_sampling import analyze_action_sampling


def test_action_sampling_analysis_handles_legacy_missing_fields(tmp_path):
    run_dir = tmp_path / "run"
    rich_dir = run_dir / "rich_logs"
    rich_dir.mkdir(parents=True)
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["total_steps", "dominant_bin_mav_elevator"])
        writer.writeheader()
        writer.writerow({"total_steps": 10, "dominant_bin_mav_elevator": 20})
    with (rich_dir / "tam_action_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "action_index_2"])
        writer.writeheader()
        writer.writerow({"agent_id": "red_0", "action_index_2": 18})
    result = analyze_action_sampling(run_dir)
    assert result["sampled_bins"]["elevator"]["count"] == 1
    assert result["argmax_comparison"]["available"] is False
    assert result["pre_death_50_steps"]["available"] is False
