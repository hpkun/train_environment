import inspect
from collections import Counter

import numpy as np

from aircombat_env_v1.env import event_semantics
from aircombat_env_v1.evaluation import record_event
from aircombat_env_v1.opponent import (
    MANOEUVRE_NAMES, candidate_actions, paper_greedy_action)
from aircombat_env_v1.reward import step_reward, terminal_reward
from aircombat_env_v1.scenario import curriculum_scenario
from aircombat_env_v1.training import (
    best_fixed_key, best_joint_key, best_randomized_key)


def state(altitude=6000.0, speed=250.0):
    return {
        "longitude": 120.0, "latitude": 60.0, "altitude": altitude,
        "roll": 0.0, "pitch": 0.0, "heading": 0.0,
        "true_airspeed": speed, "v_north": speed, "v_east": 0.0,
        "v_down": 0.0, "load_factor": 1.0,
    }


def result(hit, ret, eligible=True):
    return {
        "red_hit_rate": hit, "mean_return": ret,
        "numerical_invalid": 0, "best_eligible": eligible,
    }


def test_blue_crash_and_invalid_events_are_not_wins():
    blue_crash = event_semantics("blue_crash")
    assert terminal_reward("blue_crash") == 0.0
    assert blue_crash["winner"] == "red"
    assert blue_crash["opponent_failure"]
    assert blue_crash["valid_combat_outcome"]
    for event in (
            "red_numerical_invalid", "blue_numerical_invalid",
            "draw_both_numerical_invalid", "physics_exception"):
        semantics = event_semantics(event)
        assert semantics["winner"] is None
        assert semantics["invalid_episode"]


def test_blue_crash_total_step_reward_is_zero():
    red, blue = state(), state()
    geometry = {"ata_rad": 0.0, "aa_rad": 0.0, "distance_m": 1000.0}
    reward, _ = step_reward(red, blue, geometry, "blue_crash")
    assert reward == 0.0


def test_baseline_statistics_do_not_count_blue_crash_as_red_hit():
    counts = Counter()
    record_event(counts, "blue_crash")
    assert counts["blue_crashes"] == 1
    assert counts["red_hits"] == 0


def test_action_change_penalty_is_absent():
    assert "action" not in inspect.signature(step_reward).parameters
    assert "previous_action" not in inspect.signature(step_reward).parameters


def test_paper_greedy_candidates_are_finite_and_deterministic():
    own, target = state(), state(speed=270.0)
    target["latitude"] = 60.04
    actions = candidate_actions(own, target)
    assert tuple(actions) == MANOEUVRE_NAMES
    assert len(actions) == 8
    assert all(np.isfinite(action).all() for action in actions.values())
    np.testing.assert_array_equal(
        paper_greedy_action(own, target),
        paper_greedy_action(own, target))


def test_stage_two_sampling_is_seeded_and_twenty_eighty():
    first_rng, second_rng = np.random.default_rng(7), np.random.default_rng(7)
    first = [curriculum_scenario(first_rng) for _ in range(10000)]
    second = [curriculum_scenario(second_rng) for _ in range(10000)]
    assert first == second
    fixed_fraction = first.count("fixed_tail_chase") / len(first)
    assert 0.18 < fixed_fraction < 0.22


def test_three_best_rankings_are_independent_and_reject_invalid():
    fixed, randomized = result(0.9, 1.0), result(0.7, 2.0)
    assert best_fixed_key(fixed) == best_randomized_key(fixed)
    assert best_joint_key(fixed, randomized)[:2] == (0.7, 1.6)
    assert best_fixed_key(result(1.0, 100.0, eligible=False)) is None
    assert best_joint_key(
        fixed, result(1.0, 100.0, eligible=False)) is None
