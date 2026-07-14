from __future__ import annotations

import torch

from scripts.run_tam_v5_fast_learnability import METRICS, _state_dict_equal, classify


def _row(seed: int, *, launch_delta=0.0, geometry_delta=0.0,
         hit_delta=0.0, kill_delta=0.0, blue_alive_delta=0.0):
    row = {"seed": seed}
    for metric in METRICS:
        row[f"init_{metric}"] = 0.0
        row[f"final_{metric}"] = 0.0
        row[f"delta_{metric}"] = 0.0
    row["final_finite"] = 1.0
    row["final_nan_detected"] = 0.0
    row["delta_red_launch_per_episode"] = launch_delta
    row["delta_geometry_rate"] = geometry_delta
    row["delta_red_hit_total"] = hit_delta
    row["delta_red_kill_total"] = kill_delta
    row["delta_blue_alive_final_mean"] = blue_alive_delta
    return row


def test_classification_pass_requires_two_behavior_and_outcome_seeds():
    rows = [
        _row(0, launch_delta=0.2, hit_delta=1),
        _row(1, geometry_delta=0.02, blue_alive_delta=-0.5),
        _row(2),
    ]
    status, evidence = classify(rows, [True, True, True])
    assert status == "PURE_HAPPO_LEARNABILITY_PASS"
    assert evidence["behavior_improved_seed_count"] == 2
    assert evidence["outcome_improved_seed_count"] == 2


def test_classification_weak_and_fail_paths():
    weak, _ = classify([_row(0, launch_delta=0.2), _row(1), _row(2)], [True] * 3)
    failed, _ = classify([_row(0), _row(1), _row(2)], [True] * 3)
    assert weak == "PURE_HAPPO_LEARNABILITY_WEAK_SIGNAL"
    assert failed == "PURE_HAPPO_LEARNABILITY_FAIL"


def test_state_dict_comparison_is_exact(tmp_path):
    first = tmp_path / "first.pt"
    same = tmp_path / "same.pt"
    different = tmp_path / "different.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, first)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, same)
    torch.save({"weight": torch.tensor([1.0, 3.0])}, different)
    assert _state_dict_equal(first, same)
    assert not _state_dict_equal(first, different)
