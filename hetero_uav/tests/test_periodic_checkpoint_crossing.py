from __future__ import annotations

import json

import torch

from scripts.train_happo_reference import (
    _crossed_checkpoint_thresholds,
    _save_periodic_checkpoint,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save(self, path):
        torch.save(self.state_dict(), path)


def test_non_divisible_rollout_crosses_50k_100k_and_150k_once():
    next_step = 50_000
    crossed, next_step = _crossed_checkpoint_thresholds(next_step, 49_920, 50_000)
    assert crossed == [] and next_step == 50_000

    crossed, next_step = _crossed_checkpoint_thresholds(next_step, 50_176, 50_000)
    assert crossed == [50_000] and next_step == 100_000
    repeated, next_step = _crossed_checkpoint_thresholds(next_step, 50_176, 50_000)
    assert repeated == [] and next_step == 100_000

    crossed, next_step = _crossed_checkpoint_thresholds(next_step, 100_096, 50_000)
    assert crossed == [100_000] and next_step == 150_000
    crossed, next_step = _crossed_checkpoint_thresholds(next_step, 150_016, 50_000)
    assert crossed == [150_000] and next_step == 200_000


def test_disabled_interval_has_no_periodic_threshold():
    assert _crossed_checkpoint_thresholds(None, 1024, 0) == ([], None)
    assert _crossed_checkpoint_thresholds(None, 1024, -1) == ([], None)


def test_single_update_crossing_multiple_thresholds_advances_once():
    crossed, next_step = _crossed_checkpoint_thresholds(100, 256, 100)
    assert crossed == [100, 200]
    assert next_step == 300


def test_periodic_checkpoint_meta_records_actual_and_nominal_steps(tmp_path):
    target = _save_periodic_checkpoint(
        TinyPolicy(), tmp_path / "checkpoints",
        total_steps=50_176, iteration=196,
        checkpoint_interval_steps=50_000, keep_checkpoints=5,
        crossed_thresholds=[50_000], meta={"policy_arch": "pure_happo"},
    )
    assert target.name == "step_000050176"
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    assert meta["total_env_steps_actual"] == 50_176
    assert meta["requested_checkpoint_step"] == 50_000
    assert meta["checkpoint_threshold_step"] == 50_000
    assert meta["checkpoint_stage"] == "periodic"
    assert meta["checkpoint_interval_steps"] == 50_000


def test_periodic_checkpoint_pruning_keeps_newest_actual_steps(tmp_path):
    root = tmp_path / "checkpoints"
    policy = TinyPolicy()
    for actual, threshold in ((512, 500), (1024, 1000), (1536, 1500)):
        _save_periodic_checkpoint(
            policy, root, total_steps=actual, iteration=actual // 256,
            checkpoint_interval_steps=500, keep_checkpoints=2,
            crossed_thresholds=[threshold], meta={"policy_arch": "pure_happo"},
        )
    assert [path.name for path in sorted(root.glob("step_*"))] == [
        "step_000001024", "step_000001536"
    ]


def test_periodic_save_is_independent_of_eval_flag_contract():
    # The scheduler has no eval argument or state: a crossed training threshold
    # is sufficient to trigger a periodic save.
    crossed, _ = _crossed_checkpoint_thresholds(500, 512, 500)
    assert crossed == [500]

