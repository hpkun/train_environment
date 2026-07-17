from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.happo.vanilla_happo import VanillaHAPPOTrainer
from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, read_vanilla_happo_checkpoint_metadata,
    save_vanilla_happo_checkpoint)
from scripts.analyze_tam_paper_learnability import determine_robust_verdict
from scripts.analyze_tam_paper_multiseed_learnability import multiseed_verdict
from scripts.audit_tam_paper_2v2_symmetry import mirrored_config
from scripts.evaluate_tam_paper_checkpoint_panel import (
    paired_bootstrap, paired_delta_summary)
from scripts.tam_learnability_metrics import (
    flatten_evaluation, start_episode, summarize_baseline)
from scripts.vanilla_happo_runtime import (
    active_action_statistics, evaluate_policy_panel, infer_policy)
from uav_env.JSBSim.paper.protocol import PAPER_NOMINAL_PROTOCOL, protocol_metadata
from uav_env.make_env import make_env


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("active", [
    np.array([[1, 0], [0, 0]], dtype=bool),
    np.ones((2, 2), dtype=bool),
    np.zeros((2, 2), dtype=bool),
])
def test_active_action_statistics_excludes_inactive_placeholder(active):
    actions = np.array([
        [[1, 2, 3, 4], [20, 20, 20, 20]],
        [[20, 20, 20, 20], [20, 20, 20, 20]],
    ])
    result = active_action_statistics(actions, active)
    assert result["active_action_sample_count"] == int(active.sum())
    if not active.any():
        assert result["active_action_head_0_distribution"] is None
        assert result["active_action_head_0_entropy"] is None
    else:
        expected = np.bincount(actions[active][:, 0], minlength=40) / active.sum()
        assert np.allclose(result["active_action_head_0_distribution"], expected)
    if active.sum() == 1:
        assert result["active_action_head_0_distribution"][20] == 0.0


@pytest.mark.parametrize(("scenario", "red", "blue", "red_combat", "blue_combat"), [
    ("2v2", 2, 2, 2, 2), ("3v2", 3, 2, 2, 2), ("5v4", 5, 4, 4, 4),
])
def test_initial_side_counts_are_scenario_driven(
        scenario, red, blue, red_combat, blue_combat):
    path = ROOT / "uav_env/JSBSim/configs" / f"tam_paper_env_v1_{scenario}.yaml"
    env = make_env(str(path), dynamics_backend="simple")
    try:
        env.reset(seed=1)
        accumulator = start_episode(env, 0, 1, 0, 0)
        assert accumulator["initial_side_count"] == {"red": red, "blue": blue}
        assert accumulator["initial_combat_count"] == {
            "red": red_combat, "blue": blue_combat}
    finally:
        env.close()


def test_baseline_timing_summary_uses_observed_values_and_missing_count():
    base = {
        "winner": "red", "red_episode_return": 1.0, "episode_steps": 2,
        "red_survival_rate": 1.0, "blue_survival_rate": 0.0,
        "red_missiles_fired": 1, "blue_missiles_fired": 1,
        "red_hits": 1, "blue_hits": 0, "red_boundary": 0,
        "blue_boundary": 0, "red_crashes": 0, "blue_crashes": 0,
        "target_consistency_violations": 0, "finite": True,
    }
    for side in ("red", "blue"):
        for name in ("first_detection_time_s", "first_attack_range_entry_s",
                     "first_launch_time_s", "first_hit_time_s", "target_switches"):
            base[f"{side}_{name}"] = 2.0
    missing = copy.deepcopy(base); missing["red_first_hit_time_s"] = None
    result = summarize_baseline({
        "baseline": "random", "episode_seeds": [1, 2],
        "episodes_detail": [base, missing]})
    assert result["first_hit_time_s/red"] == {
        "mean": 2.0, "std": 0.0, "sample_count": 1, "missing_count": 1}
    assert result["first_detection_time_s/red"]["sample_count"] == 2


def test_paired_delta_and_bootstrap_are_reproducible():
    def row(value):
        return {"red_episode_return": value, "winner": "red",
                "red_survival_rate": value, "red_hit_rate": value,
                "red_crash_rate": 0.0, "red_boundary_rate": 0.0,
                "episode_steps": value}
    candidate, reference = [row(2), row(4)], [row(1), row(2)]
    paired = paired_delta_summary(candidate, reference, bootstrap_samples=1000)
    assert paired["return"]["mean"] == 1.5
    assert paired["return"]["positive_fraction"] == 1.0
    assert paired_bootstrap([1, 2, 3], seed=9, samples=1000) == paired_bootstrap(
        [1, 2, 3], seed=9, samples=1000)


def test_robust_and_multiseed_verdict_guards():
    assert determine_robust_verdict(
        True, True, 2, 1, 2, True, True) == "ROBUST_EARLY_LEARNING_SIGNAL"
    assert determine_robust_verdict(
        True, True, 2, -1, 2, True, True) == "MIXED_OR_WEAK_EARLY_SIGNAL"
    assert determine_robust_verdict(
        True, True, 2, 1, 2, False, True) != "ROBUST_EARLY_LEARNING_SIGNAL"
    assert determine_robust_verdict(
        True, True, 2, 1, 2, True, False) != "ROBUST_EARLY_LEARNING_SIGNAL"
    assert determine_robust_verdict(
        False, True, 2, 1, 2, True, True) == "RUNTIME_INVALID"
    assert multiseed_verdict(True, 2, 2, 2, 1) == (
        "ROBUST_MULTI_SEED_EARLY_LEARNING_SIGNAL")
    assert multiseed_verdict(True, 1, 2, 2, 1) == "MIXED_MULTI_SEED_EARLY_SIGNAL"
    assert multiseed_verdict(False, 3, 3, 3, 1) == "MULTISEED_RUNTIME_INVALID"


def _formal_config():
    return {"scenario": "2v2", **protocol_metadata(
        "2v2", "none", "jsbsim", PAPER_NOMINAL_PROTOCOL)}


def test_checkpoint_restart_and_exact_metadata(tmp_path):
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml"),
                   dynamics_backend="simple")
    try:
        policy, _, _ = infer_policy(env, hidden_dim=16)
        trainer = VanillaHAPPOTrainer(policy)
        restart = tmp_path / "restart.pt"
        save_vanilla_happo_checkpoint(
            restart, policy, trainer, environment_steps=7, episodes=1,
            config=_formal_config(), numpy_rng=np.random.default_rng(1),
            checkpoint_type="episode_boundary_restart", at_episode_boundary=True,
            discarded_partial_rollout_steps=7)
        metadata = read_vanilla_happo_checkpoint_metadata(restart)
        assert metadata["resume_semantics"] == "episode_boundary_restart"
        assert metadata["discarded_partial_rollout_steps"] == 7
        with pytest.raises(ValueError, match="strict resume"):
            load_vanilla_happo_checkpoint(restart, policy, for_resume=True)
        loaded = load_vanilla_happo_checkpoint(
            restart, policy, for_resume=True, allow_episode_restart=True,
            restore_rng=False)
        assert loaded["resume_semantics"] == "episode_boundary_restart"
        exact = tmp_path / "exact.pt"
        save_vanilla_happo_checkpoint(
            exact, policy, trainer, environment_steps=128, episodes=2,
            config=_formal_config(), numpy_rng=np.random.default_rng(1),
            checkpoint_type="exact_update_and_episode_boundary",
            at_episode_boundary=True, saved_at_update_boundary=True)
        exact_meta = read_vanilla_happo_checkpoint_metadata(exact)
        assert exact_meta["exact_training_continuation"] is True
    finally:
        env.close()


def test_stochastic_panel_reproducibility_rng_restore_and_direct_kills():
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml"),
                   dynamics_backend="simple")
    try:
        torch.manual_seed(77)
        policy, _, _ = infer_policy(env, hidden_dim=16)
        before = torch.get_rng_state().clone()
        a = evaluate_policy_panel(
            env, policy, 1, 11, environment_seeds=[11],
            policy_action_seeds=[21], deterministic=False)
        assert torch.equal(before, torch.get_rng_state())
        b = evaluate_policy_panel(
            env, policy, 1, 11, environment_seeds=[11],
            policy_action_seeds=[21], deterministic=False)
        c = evaluate_policy_panel(
            env, policy, 1, 11, environment_seeds=[11],
            policy_action_seeds=[22], deterministic=False)
        assert a["episodes_detail"] == b["episodes_detail"]
        assert (a["summary"]["active_action_head_0_distribution"] !=
                c["summary"]["active_action_head_0_distribution"])
        episode = copy.deepcopy(a["episodes_detail"][0])
        episode["red_kills"] = 7
        result = {**a, "episodes_detail": [episode]}
        flat = flatten_evaluation(
            result, environment_steps=0, trainer_update_count=0,
            policy_version=0, actor_sharing="independent",
            algorithm_label="test", evaluation_stage="test")
        assert flat["red_kills"] == 7
    finally:
        env.close()


def test_symmetry_config_is_in_memory_and_does_not_change_yaml():
    path = ROOT / "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml"
    before = path.read_bytes()
    import yaml
    config = yaml.safe_load(before.decode("utf-8"))
    mirrored = mirrored_config(config)
    assert mirrored["red_agents"][0]["lat_deg"] == config["blue_agents"][0]["lat_deg"]
    assert mirrored["red_agents"][0]["id"] == "red_0"
    assert path.read_bytes() == before
