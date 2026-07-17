from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from algorithms.happo.vanilla_happo import VanillaHAPPOTrainer
from algorithms.happo.vanilla_happo_checkpoint import (
    read_vanilla_happo_checkpoint_metadata, save_vanilla_happo_checkpoint)
from scripts.analyze_tam_paper_learnability import (
    analyze_run, determine_verdict, split_episode_windows, trend_statistics)
from scripts.tam_learnability_metrics import (
    RecordWriter, finish_episode, start_episode, strictly_better_evaluation)
from scripts.train_tam_paper_vanilla_happo import aggregate_episode_rows
from scripts.vanilla_happo_runtime import infer_policy
from uav_env.JSBSim.paper.protocol import PAPER_NOMINAL_PROTOCOL, protocol_metadata
from uav_env.make_env import make_env


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "uav_env/JSBSim/configs/tam_paper_env_v1_2v2.yaml"


def env_2v2():
    return make_env(str(CONFIG), dynamics_backend="simple")


def test_2v2_role_and_side_metrics_do_not_invent_mav():
    env = env_2v2(); env.reset(seed=1)
    accumulator = start_episode(env, 0, 1, 0, 0)
    accumulator["returns"] = {"red_0": 10.0, "red_1": 30.0}
    blue_0 = next(agent for agent in env.task.agents if agent.agent_id == "blue_0")
    blue_0.kill("shotdown")
    info = {
        "episode_step": 20, "winner": "red", "termination_reason": "test",
        "kills": {"red": 1, "blue": 0},
        "aircraft_metrics": {agent.agent_id: {
            "death_reason": agent.death_reason, "max_speed_mps": 300.0,
            "max_abs_load_factor_g": 2.0} for agent in env.task.agents},
    }
    record = finish_episode(
        accumulator, env, info, 20, "vanilla_happo",
        "published_rules_simplified_v4", "paper_nominal")
    assert record["red_team_episode_return"] == 40.0
    assert record["role_mean_return/mav"] is None
    assert record["role_total_return/mav"] is None
    assert record["role_mean_return/attack_uav"] == 20.0
    assert record["red_survival_rate"] == 1.0
    assert record["blue_survival_rate"] == 0.5
    assert max(record["red_survival_rate"], record["blue_survival_rate"]) <= 1.0
    aggregate = aggregate_episode_rows([record], env.agent_ids, env.agent_roles)
    assert aggregate["mav_return"] is None
    assert aggregate["uav_return"] == 20.0
    assert aggregate["survival_rate"] == aggregate["red_survival_rate"] == 1.0
    env.close()


def test_controlled_reward_components_and_side_events_are_separate():
    env = env_2v2(); env.reset(seed=2)
    accumulator = start_episode(env, 0, 2, 0, 0)
    accumulator["reward_components"]["red_0"] = {"r_event": 1.0}
    accumulator["reward_components"]["red_1"] = {"r_event": 2.0}
    accumulator["side_tracker"]["red"].update({"missiles_fired": 3, "hits": 2})
    accumulator["side_tracker"]["blue"].update({"missiles_fired": 5, "hits": 1})
    info = {
        "episode_step": 1, "winner": "draw", "termination_reason": "test",
        "kills": {"red": 0, "blue": 0},
        "aircraft_metrics": {agent.agent_id: {
            "death_reason": None, "max_speed_mps": 250.0,
            "max_abs_load_factor_g": 1.0} for agent in env.task.agents},
    }
    record = finish_episode(
        accumulator, env, info, 1, "vanilla_happo",
        "published_rules_simplified_v4", "paper_nominal")
    assert record["reward_component/controlled_total/r_event"] == 3.0
    assert not any("blue_" in key for key in record if key.startswith("reward_component/agent"))
    assert (record["red_missiles_fired"], record["red_hits"]) == (3, 2)
    assert (record["blue_missiles_fired"], record["blue_hits"]) == (5, 1)
    env.close()


def test_record_writer_writes_complete_csv_and_jsonl(tmp_path):
    writer = RecordWriter(tmp_path / "episodes.csv", tmp_path / "episodes.jsonl")
    record = {"episode_index": 0, "episode_seed": 7,
              "red_team_episode_return": 1.5, "finite": True}
    writer.append(record)
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert set(row) == set(record)
    assert json.loads((tmp_path / "episodes.jsonl").read_text())["episode_seed"] == 7


def test_window_split_single_episode_and_trend_are_safe():
    records = [{"x": value} for value in range(10)]
    early, middle, late = split_episode_windows(records)
    assert len(early) == 2 and len(middle) == 6 and len(late) == 2
    single = trend_statistics([{"x": 3.0}], "x")
    assert single["early_mean"] == single["late_mean"] == 3.0
    assert single["relative_change"] == 0.0
    assert single["linear_slope"] == 0.0


@pytest.mark.parametrize(("runtime", "training", "combat", "evaluation", "expected"), [
    (True, True, True, False, "CLEAR_EARLY_LEARNING_SIGNAL"),
    (True, True, False, False, "MIXED_OR_WEAK_EARLY_SIGNAL"),
    (True, False, False, False, "NO_EARLY_SIGNAL_AT_102400_STEPS"),
    (False, True, True, True, "RUNTIME_INVALID"),
])
def test_verdict_rule(runtime, training, combat, evaluation, expected):
    assert determine_verdict(runtime, training, combat, evaluation) == expected


def test_analyzer_reports_insufficient_data_without_episodes(tmp_path):
    (tmp_path / "config_snapshot.json").write_text(json.dumps({
        "total_environment_steps": 128}), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({
        "actual_final_environment_steps": 128,
        "checkpoint_environment_fidelity_revision": "published_rules_simplified_v4"}),
        encoding="utf-8")
    (tmp_path / "baseline_reference.json").write_text("{}", encoding="utf-8")
    for name in ("episodes.csv", "training.csv", "evaluation_history.csv"):
        (tmp_path / name).write_text("", encoding="utf-8")
    report = analyze_run(tmp_path)
    assert report["analysis_status"] == "INSUFFICIENT_DATA"
    assert report["learnability_verdict"] == "INSUFFICIENT_DATA"
    assert "0 episodes" in report["insufficient_data_reason"]


def test_best_evaluation_is_strictly_better_and_extra_metadata_roundtrips(tmp_path):
    assert strictly_better_evaluation(None, 1.0)
    assert strictly_better_evaluation(1.0, 2.0)
    assert not strictly_better_evaluation(1.0, 1.0)
    env = env_2v2()
    policy, _, _ = infer_policy(env, "independent", 16, "cpu")
    trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "best.pt"
    config = {"scenario": "2v2", **protocol_metadata(
        "2v2", "none", "jsbsim", PAPER_NOMINAL_PROTOCOL)}
    import numpy as np
    save_vanilla_happo_checkpoint(
        path, policy, trainer, environment_steps=10, episodes=1,
        config=config, numpy_rng=np.random.default_rng(),
        extra_metadata={"selected_by": "best_deterministic_red_team_return"})
    assert read_vanilla_happo_checkpoint_metadata(path)["selected_by"] == (
        "best_deterministic_red_team_return")
    env.close()


def test_training_source_reuses_step_zero_policy_and_explicit_random_seeds():
    source = (ROOT / "scripts/train_tam_paper_vanilla_happo.py").read_text(
        encoding="utf-8")
    assert '"untrained_happo": summarize_baseline(evaluation_0)' in source
    assert 'range(10)' in source
    assert '"untrained_policy_is_training_policy_object": True' in source
