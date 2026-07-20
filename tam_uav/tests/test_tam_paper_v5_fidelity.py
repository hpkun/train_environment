from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_env.make_env import make_env
from uav_env.JSBSim.core.aircraft import JSBSimAircraftPlatform
from uav_env.JSBSim.paper.missile import PaperMissile
from uav_env.JSBSim.paper.protocol import (
    ENVIRONMENT_FIDELITY_REVISION, validate_parameter_provenance)
from uav_env.JSBSim.paper.reward import unpublished_height_reward_approximation
from uav_env.JSBSim.paper.task import TAMPaperTask


CONFIG_DIR = Path(__file__).parents[1] / "uav_env" / "JSBSim" / "configs"


def config(scenario="2v2"):
    path = CONFIG_DIR / f"tam_paper_env_v1_{scenario}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def env(scenario="2v2", backend="simple"):
    return make_env(str(CONFIG_DIR / f"tam_paper_env_v1_{scenario}.yaml"),
                    dynamics_backend=backend)


@pytest.mark.parametrize("scenario", ["2v2", "3v2", "5v4"])
def test_v5_parameter_provenance_and_published_values(scenario):
    cfg = config(scenario)
    validate_parameter_provenance(cfg)
    assert "inferred_parameters" not in cfg
    assert cfg["environment_fidelity_revision"] == ENVIRONMENT_FIDELITY_REVISION
    published = cfg["published_parameters"]
    assert (published["simulation_frequency_hz"],
            published["decision_frequency_hz"],
            published["physics_frames_per_action"]) == (60, 5, 12)
    assert published["uav_reward_weights"] == {
        "height": 10.0, "speed": 10.0, "angle": 15.0,
        "distance": 10.0, "dodge": 30.0}
    assert published["event_rewards"] == {
        "kill": 200.0, "death_or_crash": -200.0, "out_of_zone": -100.0}
    assert set(cfg["derived_parameters"]) == {"decision_dt_s"}


def test_no_range_gate_dead_mask_and_targeted_missile_slots():
    e = env("2v2")
    e.reset(seed=3)
    by_id = {a.agent_id: a for a in e.task.agents}
    ego, enemy = by_id["red_0"], by_id["blue_0"]
    enemy.position = ego.position + np.array([100000.0, 0.0, 0.0])
    missile_cfg = e.task.published | e.task.unpublished
    targeted = PaperMissile("targeted", "blue_0", ego.agent_id,
                            ego.position + np.array([1000.0, 0.0, 0.0]),
                            np.array([-1.0, 0.0, 0.0]), missile_cfg)
    other = PaperMissile("other", "blue_0", "red_1",
                         ego.position + np.array([500.0, 0.0, 0.0]),
                         np.array([-1.0, 0.0, 0.0]), missile_cfg)
    obs = e.task.observation.build(e.task.agents, [targeted, other])
    assert obs[ego.agent_id]["enemy_mask"].sum() == 2
    assert obs[ego.agent_id]["incoming_missile_mask"].sum() == 1
    enemy.kill("shotdown")
    obs = e.task.observation.build(e.task.agents, [targeted, other])
    assert obs[ego.agent_id]["enemy_mask"].sum() == 1
    e.close()


def test_observation_shape_is_stable_across_scenarios():
    shapes = []
    for scenario in ("2v2", "3v2", "5v4"):
        e = env(scenario)
        obs, _ = e.reset(seed=4)
        shapes.append(e.flatten_observation(obs[e.agent_ids[0]]).shape)
        e.close()
    assert shapes == [(103,), (103,), (103,)]


def test_target_snapshot_is_frozen_and_all_consumers_match():
    e = env("2v2")
    e.reset(seed=5)
    context = e.prepare_decision_context()
    frozen = dict(context["targets"])
    _, rewards, _, _, info = e.step({aid: np.array([20, 20, 20, 20])
                                     for aid in e.agent_ids})
    assert info["target_used_by_weapon"] == frozen
    assert info["target_used_by_reward"] == frozen
    assert info["target_consistency_violation"] == []
    assert all(np.isfinite(list(rewards.values())))
    e.close()


def test_target_mismatch_is_a_hard_invariant():
    context = {"targets": {"red_0": "blue_0"},
               "target_used_by_rule_action": {"red_0": "blue_1"}}
    violations = TAMPaperTask._target_consistency_violations(
        context, {"red_0": "blue_0"}, {"red_0": "blue_0"})
    assert violations == [{"agent_id": "red_0", "consumer": "rule_action",
                           "expected_target": "blue_0", "used_target": "blue_1"}]


def test_roles_and_2v2_profiles_are_not_paper_aircraft_claims():
    two = config("2v2")
    roles = two["role_definitions"]
    agents = (two["scenario_initial_conditions"]["red_agents"]
              + two["scenario_initial_conditions"]["blue_agents"])
    assert {item["type"] for item in agents} == {"attack_uav"}
    assert roles["mav"]["missile_num"] == 0
    assert roles["attack_uav"]["missile_num"] == 2
    assert "unpublished_execution_aircraft_model" in roles["attack_uav"]


def test_launch_conditions_and_explicit_unpublished_timeout():
    e = env("2v2")
    e.reset(seed=6)
    by_id = {a.agent_id: a for a in e.task.agents}
    shooter, target = by_id["red_0"], by_id["blue_0"]
    target.position = shooter.position + np.array([1000.0, 0.0, 0.0])
    first = e.task.weapon.try_launch(shooter, target, 0.0)
    assert first is not None
    assert e.task.weapon.try_launch(shooter, target, 24.999) is None
    assert e.task.weapon.missiles[0].timeout_s == 56.0
    target.kill("shotdown")
    assert e.task.weapon.try_launch(shooter, target, 25.0) is None
    e.close()


def test_height_reward_is_explicit_approximation_without_new_band():
    assert unpublished_height_reward_approximation(750.0, 750.0, 6000.0) == 0.0
    assert unpublished_height_reward_approximation(6000.0, 750.0, 6000.0) == 1.0
    assert config()["unpublished_parameters"]["height_reward_approximation"][
        "exact_formula_available"] is False


def test_jsbsim_candidate_clone_is_repeatable_and_side_effect_free():
    e = env("2v2", "jsbsim")
    e.reset(seed=7)
    by_id = {a.agent_id: a for a in e.task.agents}
    agent, target = by_id["blue_0"], by_id["red_0"]
    assert isinstance(agent, JSBSimAircraftPlatform)
    before = (agent.position.copy(), agent.velocity.copy(), agent.roll,
              agent.pitch, agent.heading,
              agent._get_property("simulation/sim-time-sec"))
    action_a, diag_a = e.task.opponent.act(agent, target, [])
    action_b, diag_b = e.task.opponent.act(agent, target, [])
    after = (agent.position.copy(), agent.velocity.copy(), agent.roll,
             agent.pitch, agent.heading,
             agent._get_property("simulation/sim-time-sec"))
    assert np.array_equal(action_a, action_b)
    assert diag_a["manoeuvre"] == diag_b["manoeuvre"]
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert before[2:] == pytest.approx(after[2:])
    assert all(row["candidate_backend"] == "independent_jsbsim_clone_12_frames"
               for row in diag_a["candidates"].values())
    totals = {name: row["total_dense_reward"]
              for name, row in diag_a["candidates"].items()}
    assert totals[diag_a["manoeuvre"]] == max(totals.values())
    e.close()
