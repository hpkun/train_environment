from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algorithms.happo.vanilla_happo import (
    ParameterSharingPPOTrainer, VanillaHAPPOPolicy,
    VanillaHAPPORolloutBuffer, VanillaHAPPOTrainer)
from algorithms.happo.vanilla_happo_checkpoint import (
    load_vanilla_happo_checkpoint, read_vanilla_happo_checkpoint_metadata,
    save_vanilla_happo_checkpoint)
from scripts.audit_tam_paper_vanilla_happo import audit, environment_contract_unchanged
from scripts.tam_output_paths import resolve_tam_output
from scripts.vanilla_happo_runtime import (
    stack_controlled_rule_actions, update_side_timing)
from scripts.vanilla_happo_runtime import seed_all
from scripts.train_tam_paper_vanilla_happo import (
    agent_update_contract, is_episode_boundary, resume_seed_state,
    should_run_evaluation)
from scripts.eval_tam_paper_vanilla_happo import parse_args as parse_eval_args


def make_policy(n=3, hidden=16, sharing="independent"):
    ids = [f"red_{i}" for i in range(n)]
    roles = {aid: "mav" if i == 0 else "uav" for i, aid in enumerate(ids)}
    return VanillaHAPPOPolicy(ids, roles, 5, 8, hidden_dim=hidden,
                              actor_sharing=sharing)


def make_buffer(policy, length=8, inactive_agent=None):
    rng = np.random.default_rng(4)
    buffer = VanillaHAPPORolloutBuffer(
        length, len(policy.agent_ids), policy.actor_obs_dim, policy.critic_state_dim)
    available = np.ones((len(policy.agent_ids), 4, 40), np.float32)
    for step in range(length):
        obs = rng.normal(size=(len(policy.agent_ids), 5)).astype(np.float32)
        state = rng.normal(size=8).astype(np.float32)
        with torch.no_grad():
            out = policy.act(obs, state, available)
            next_value = policy.value(torch.as_tensor(state)).numpy()
        active = np.ones(len(policy.agent_ids), np.float32)
        if inactive_agent is not None:
            active[inactive_agent] = 0
        rewards = np.asarray([10 * (i + 1) + step for i in range(len(policy.agent_ids))],
                             np.float32)
        buffer.add(obs=obs, state=state, actions=out["actions"].numpy(),
                   log_probs=out["log_probs"].numpy(), rewards=rewards,
                   value=out["value"].numpy(), next_value=next_value,
                   terminated=np.zeros(len(policy.agent_ids)),
                   truncated=np.zeros(len(policy.agent_ids)), active_masks=active,
                   available_actions=available, agent_alive=active,
                   episode_id=0, decision_step=step, policy_version=0)
    return buffer


def test_fixed_horizon_collects_across_episode_boundaries_before_one_update():
    policy = make_policy(1)
    buffer = VanillaHAPPORolloutBuffer(8, 1, 5, 8)
    available = np.ones((1, 4, 40), np.float32)
    episode_id = 0
    for step in range(8):
        obs = np.zeros((1, 5), np.float32)
        state = np.zeros(8, np.float32)
        with torch.no_grad():
            out = policy.act(obs, state, available)
        episode_end = step % 3 == 2
        buffer.add(
            obs=obs, state=state, actions=out["actions"].numpy(),
            log_probs=out["log_probs"].numpy(), rewards=np.ones(1),
            value=out["value"].numpy(), next_value=out["value"].numpy(),
            terminated=np.array([episode_end]), truncated=np.zeros(1),
            active_masks=np.ones(1), available_actions=available,
            agent_alive=np.ones(1), episode_id=episode_id,
            decision_step=step % 3, policy_version=0)
        if episode_end:
            episode_id += 1
    trainer = VanillaHAPPOTrainer(policy, ppo_epochs=1)
    trainer.update(buffer)
    assert buffer.pos == 8
    assert buffer.episode_ids[:buffer.pos].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert trainer.update_count == 1


def test_gae_does_not_propagate_across_episode_ids_and_keeps_bootstrap_semantics():
    buffer = VanillaHAPPORolloutBuffer(4, 1, 2, 3)
    available = np.ones((1, 4, 40), np.float32)
    for step, episode_id in enumerate((0, 0, 1, 1)):
        truncated = float(step == 1)
        terminated = float(step == 3)
        next_value = 5.0 if truncated else 100.0 if terminated else 0.0
        buffer.add(
            obs=np.zeros((1, 2)), state=np.zeros(3), actions=np.zeros((1, 4)),
            log_probs=np.zeros(1), rewards=np.array([1.0 if episode_id == 0 else 10.0]),
            value=np.zeros(1), next_value=np.array([next_value]),
            terminated=np.array([terminated]), truncated=np.array([truncated]),
            active_masks=np.ones(1), available_actions=available,
            agent_alive=np.ones(1), episode_id=episode_id,
            decision_step=step % 2, policy_version=0)
    advantages, _ = buffer.compute_gae(gamma=0.9, gae_lambda=1.0)
    assert advantages[1, 0] == pytest.approx(1.0 + 0.9 * 5.0)
    assert advantages[0, 0] == pytest.approx(1.0 + 0.9 * advantages[1, 0])
    assert advantages[3, 0] == pytest.approx(10.0)
    assert advantages[1, 0] != pytest.approx(1.0 + 0.9 * 5.0 + 0.9 * 10.0)


def test_critic_has_shared_backbone_independent_heads_and_vector_shapes():
    policy = make_policy(3)
    assert list(policy.critic.heads) == policy.agent_ids
    assert policy.value(torch.zeros(8)).shape == (3,)
    assert policy.value(torch.zeros(7, 8)).shape == (7, 3)


def test_buffer_rejects_scalar_values_instead_of_broadcasting():
    buffer = VanillaHAPPORolloutBuffer(1, 2, 5, 8)
    with pytest.raises(ValueError, match="shape"):
        buffer.add(obs=np.zeros((2, 5)), state=np.zeros(8), actions=np.zeros((2, 4)),
                   log_probs=np.zeros(2), rewards=np.zeros(2), value=0.0, next_value=0.0,
                   terminated=np.zeros(2), truncated=np.zeros(2), active_masks=np.ones(2),
                   available_actions=np.ones((2, 4, 40)), agent_alive=np.ones(2),
                   episode_id=0, decision_step=0, policy_version=0)


def test_one_order_per_update_with_multiple_epochs_and_one_factor_initialization():
    policy = make_policy(3)
    result = VanillaHAPPOTrainer(policy, ppo_epochs=3, minibatch_size=2, seed=7).update(
        make_buffer(policy))
    assert len(result.update_orders) == 1
    assert sorted(result.update_orders[0]) == sorted(policy.agent_ids)
    assert result.metrics["agent_order_count"] == 1
    assert result.metrics["factor_initialization_count"] == 1
    assert result.metrics["policy_version"] == 0


@pytest.mark.parametrize(("terminated", "truncated", "expected"), [
    ([1, 1, 1], [0, 0, 0], True),
    ([0, 0, 0], [1, 1, 1], True),
    ([1, 0, 0], [0, 1, 1], True),
    ([0, 0, 0], [0, 0, 0], False),
])
def test_episode_boundary_handles_termination_truncation_mixtures(
        terminated, truncated, expected):
    assert is_episode_boundary(terminated, truncated) is expected


def test_mixed_policy_versions_are_rejected_but_unused_tail_is_ignored():
    policy = make_policy(2)
    buffer = make_buffer(policy, length=4)
    buffer.policy_versions[1] = 7
    with pytest.raises(ValueError, match="mixes policy versions"):
        VanillaHAPPOTrainer(policy).update(buffer)
    tail_buffer = VanillaHAPPORolloutBuffer(6, 2, 5, 8)
    source = make_buffer(policy, length=3)
    for name in vars(source):
        value = getattr(source, name)
        target = getattr(tail_buffer, name, None)
        if isinstance(value, np.ndarray) and isinstance(target, np.ndarray):
            target[:3] = value[:3]
    tail_buffer.pos = 3
    tail_buffer.policy_versions[3:] = 99
    assert tail_buffer.validated_policy_version() == 0


def test_factor_after_each_agent_uses_final_policy_on_full_rollout():
    policy = make_policy(3)
    buffer = make_buffer(policy)
    trainer = VanillaHAPPOTrainer(policy, ppo_epochs=2, minibatch_size=3, seed=2)
    result = trainer.update(buffer)
    data = buffer.tensors("cpu")
    log_factor = torch.zeros(buffer.pos)
    with torch.no_grad():
        for aid in result.update_orders[0]:
            index = policy.agent_ids.index(aid)
            logp, _, _ = policy.evaluate_agent(
                aid, data["obs"][:, index], data["actions"][:, index],
                data["available_actions"][:, index])
            log_factor = trainer.update_log_factor(
                log_factor, logp, data["old_log_probs"][:, index],
                data["active_masks"][:, index])
            assert result.metrics[f"factor_after_mean/{aid}"] == pytest.approx(
                float(torch.exp(log_factor).mean()), rel=1e-6)


def test_inactive_ratio_is_one_and_log_factor_is_detached():
    old = torch.zeros(3)
    new = torch.log(torch.tensor([2.0, 3.0, 5.0], requires_grad=True))
    log_factor = VanillaHAPPOTrainer.update_log_factor(
        torch.zeros(3), new, old, torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(torch.exp(log_factor), torch.tensor([2.0, 1.0, 5.0]))
    assert not log_factor.requires_grad and log_factor.grad_fn is None


def test_log_space_factor_matches_manual_three_agent_product():
    log_factor = torch.zeros(2)
    old = torch.zeros(2)
    for ratio in ([2.0, 3.0], [5.0, 7.0], [11.0, 13.0]):
        log_factor = VanillaHAPPOTrainer.update_log_factor(
            log_factor, torch.log(torch.tensor(ratio)), old, torch.ones(2))
    torch.testing.assert_close(torch.exp(log_factor), torch.tensor([110.0, 273.0]))


def test_nonfinite_log_factor_fails_explicitly():
    with pytest.raises(FloatingPointError, match="non-finite"):
        VanillaHAPPOTrainer.update_log_factor(
            torch.zeros(1), torch.tensor([float("inf")]), torch.zeros(1), torch.ones(1))


def test_advantage_normalization_is_per_agent_and_respects_masks():
    advantages = torch.tensor([[1., 100.], [2., 200.], [3., 999.]])
    active = torch.tensor([[1., 1.], [1., 1.], [1., 0.]])
    normalized = VanillaHAPPOTrainer._normalize_advantages(advantages, active)
    assert float(normalized[:, 0].mean()) == pytest.approx(0, abs=1e-6)
    assert float(normalized[:2, 1].mean()) == pytest.approx(0, abs=1e-6)
    assert normalized[2, 1] == 0


def test_per_agent_gae_keeps_heterogeneous_reward_scales_separate():
    policy = make_policy(2)
    buffer = make_buffer(policy, length=2)
    advantages, _ = buffer.compute_gae(gamma=0, gae_lambda=0)
    np.testing.assert_allclose(advantages, buffer.rewards[:2] - buffer.values[:2])
    assert not np.allclose(advantages[:, 0], advantages[:, 1])


def test_fully_inactive_agent_head_receives_no_critic_gradient():
    policy = make_policy(2)
    buffer = make_buffer(policy, inactive_agent=1)
    before = copy.deepcopy(policy.critic.heads["red_1"].state_dict())
    VanillaHAPPOTrainer(policy, ppo_epochs=2).update(buffer)
    for key, value in policy.critic.heads["red_1"].state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_adam_state_does_not_move_inactive_critic_head_on_second_update():
    policy = make_policy(2)
    trainer = VanillaHAPPOTrainer(policy, ppo_epochs=2, minibatch_size=4)
    trainer.update(make_buffer(policy, length=8))
    for parameter in policy.critic.heads["red_1"].parameters():
        assert parameter in trainer.critic_optimizer.state
        assert trainer.critic_optimizer.state[parameter]
    inactive_buffer = make_buffer(policy, length=8, inactive_agent=1)
    red_1_before = copy.deepcopy(policy.critic.heads["red_1"].state_dict())
    red_0_before = copy.deepcopy(policy.critic.heads["red_0"].state_dict())
    backbone_before = copy.deepcopy(policy.critic.backbone.state_dict())
    result = trainer.update(inactive_buffer)
    for key, value in policy.critic.heads["red_1"].state_dict().items():
        torch.testing.assert_close(value, red_1_before[key], rtol=0, atol=0)
    red_0_changed = any(not torch.equal(value, red_0_before[key])
                        for key, value in policy.critic.heads["red_0"].state_dict().items())
    backbone_changed = any(not torch.equal(value, backbone_before[key])
                           for key, value in policy.critic.backbone.state_dict().items())
    assert red_0_changed or backbone_changed
    assert result.metrics["active_sample_count/red_1"] == 0
    assert "critic_head_momentum_isolation_valid/red_1" not in result.metrics


def test_inactive_agent_is_not_expected_to_change_and_optimization_contract_passes():
    policy = make_policy(2)
    buffer = make_buffer(policy, inactive_agent=1)
    actor_before = {aid: copy.deepcopy(policy.actors[aid].state_dict())
                    for aid in policy.agent_ids}
    head_before = {aid: copy.deepcopy(policy.critic.heads[aid].state_dict())
                   for aid in policy.agent_ids}
    result = VanillaHAPPOTrainer(policy, ppo_epochs=2).update(buffer)
    actor_changed = {aid: any(not torch.equal(value, actor_before[aid][key])
                              for key, value in policy.actors[aid].state_dict().items())
                     for aid in policy.agent_ids}
    head_changed = {aid: any(not torch.equal(value, head_before[aid][key])
                             for key, value in policy.critic.heads[aid].state_dict().items())
                    for aid in policy.agent_ids}
    assert result.metrics["active_sample_count/red_0"] == buffer.pos
    assert result.metrics["active_sample_count/red_1"] == 0
    assert actor_changed == {"red_0": True, "red_1": False}
    assert head_changed == {"red_0": True, "red_1": False}
    contracts = [agent_update_contract(
        result.metrics[f"active_sample_count/{aid}"], actor_changed[aid], head_changed[aid])
        for aid in policy.agent_ids]
    assert contracts == [(True, True), (False, True)]
    assert agent_update_contract(0, True, False) == (False, False)


def test_approximate_kl_uses_active_joint_log_probability_and_is_finite():
    policy = make_policy(2)
    buffer = make_buffer(policy)
    result = VanillaHAPPOTrainer(policy, ppo_epochs=2).update(buffer)
    data = buffer.tensors("cpu")
    with torch.no_grad():
        for index, aid in enumerate(policy.agent_ids):
            active = data["active_masks"][:, index] > 0.5
            final_logp, _, _ = policy.evaluate_agent(
                aid, data["obs"][:, index], data["actions"][:, index],
                data["available_actions"][:, index])
            expected = (data["old_log_probs"][active, index]
                        - final_logp[active]).mean().item()
            assert np.isfinite(result.metrics[f"approx_kl/{aid}"])
            assert result.metrics[f"approx_kl/{aid}"] == pytest.approx(expected)


def test_formal_happo_rejects_shared_actor_and_ablation_has_no_factor():
    policy = make_policy(3, sharing="parameter_sharing_ppo_ablation")
    with pytest.raises(ValueError, match="independent"):
        VanillaHAPPOTrainer(policy)
    trainer = ParameterSharingPPOTrainer(policy)
    assert trainer.algorithm_mode == "parameter_sharing_ppo_ablation"
    assert trainer.uses_sequential_factor is False
    result = trainer.update(make_buffer(policy, length=3))
    assert result.update_orders == []
    assert "importance_factor_mean" not in result.metrics


def test_checkpoint_requires_episode_boundary_for_resumable(tmp_path):
    policy = make_policy(2); trainer = VanillaHAPPOTrainer(policy)
    with pytest.raises(ValueError, match="episode_boundary"):
        save_vanilla_happo_checkpoint(
            tmp_path / "x.pt", policy, trainer, environment_steps=2, episodes=0,
            config={"scenario": "2v2"}, numpy_rng=np.random.default_rng(),
            checkpoint_type="resumable", at_episode_boundary=False)


def test_mid_episode_weights_rejected_for_strict_resume_and_explicit_restart_marked(tmp_path):
    policy = make_policy(2); trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "x.pt"
    save_vanilla_happo_checkpoint(path, policy, trainer, environment_steps=2, episodes=0,
                                  config={"scenario": "2v2"},
                                  numpy_rng=np.random.default_rng())
    with pytest.raises(ValueError, match="strict resume"):
        load_vanilla_happo_checkpoint(path, policy, trainer, for_resume=True)
    loaded = load_vanilla_happo_checkpoint(
        path, policy, trainer, for_resume=True, allow_episode_restart=True)
    assert loaded["resume_semantics"] == "episode_restart"


def test_checkpoint_restores_next_order_permutations_optimizer_and_update(tmp_path):
    torch.manual_seed(8)
    policy = make_policy(3); trainer = VanillaHAPPOTrainer(
        policy, ppo_epochs=2, minibatch_size=3, seed=19)
    buffer = make_buffer(policy)
    trainer.update(buffer)
    path = tmp_path / "resume.pt"
    save_vanilla_happo_checkpoint(
        path, policy, trainer, environment_steps=8, episodes=1,
        config={"scenario": "3v2"}, numpy_rng=np.random.default_rng(19),
        checkpoint_type="resumable", at_episode_boundary=True, policy_version=1,
        seed_schedule={"next_episode": 1})
    restored_policy = make_policy(3); restored = VanillaHAPPOTrainer(
        restored_policy, ppo_epochs=2, minibatch_size=3, seed=999)
    load_vanilla_happo_checkpoint(
        path, restored_policy, restored, for_resume=True, restore_rng=False,
        expected_scenario="3v2")
    result_a = trainer.update(buffer)
    result_b = restored.update(buffer)
    assert result_a.update_orders == result_b.update_orders
    assert result_a.minibatch_permutations == result_b.minibatch_permutations
    assert trainer.update_count == restored.update_count == 2
    for (name_a, value_a), (name_b, value_b) in zip(
            policy.state_dict().items(), restored_policy.state_dict().items()):
        assert name_a == name_b
        torch.testing.assert_close(value_a, value_b, rtol=1e-6, atol=1e-7)


def test_strict_resume_uses_checkpoint_seed_schedule_despite_different_requested_seed(tmp_path):
    policy = make_policy(2); trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "seed_resume.pt"
    save_vanilla_happo_checkpoint(
        path, policy, trainer, environment_steps=10, episodes=4,
        config={"scenario": "2v2"}, numpy_rng=np.random.default_rng(11),
        checkpoint_type="resumable", at_episode_boundary=True,
        seed_schedule={"episode_seed_base": 11, "next_episode": 4,
                       "next_episode_seed": 15})
    loaded = load_vanilla_happo_checkpoint(
        path, policy, trainer, for_resume=True, restore_rng=False,
        expected_scenario="2v2")
    restored_seed, next_seed = resume_seed_state(999, loaded["episodes"], loaded)
    assert restored_seed == 11
    assert next_seed == 15
    restart = dict(loaded, resume_semantics="episode_restart")
    assert resume_seed_state(999, loaded["episodes"], restart) == (999, 1003)


def test_checkpoint_metadata_reconstructs_hidden_mode_heads_and_mappings(tmp_path):
    policy = make_policy(3, hidden=23); trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "weights.pt"
    save_vanilla_happo_checkpoint(path, policy, trainer, environment_steps=0, episodes=0,
                                  config={"scenario": "3v2"}, numpy_rng=np.random.default_rng())
    metadata = read_vanilla_happo_checkpoint_metadata(path)
    assert metadata["hidden_dim"] == 23
    assert metadata["actor_sharing"] == "independent"
    assert metadata["critic_head_ids"] == policy.agent_ids
    assert metadata["agent_critic_mapping"] == {aid: aid for aid in policy.agent_ids}


def test_five_agent_checkpoint_construction_has_five_actor_and_critic_mappings(tmp_path):
    policy = make_policy(5); trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "5v4.pt"
    save_vanilla_happo_checkpoint(path, policy, trainer, environment_steps=0, episodes=0,
                                  config={"scenario": "5v4"}, numpy_rng=np.random.default_rng())
    metadata = read_vanilla_happo_checkpoint_metadata(path)
    assert len(metadata["agent_actor_mapping"]) == 5
    assert len(metadata["agent_critic_mapping"]) == 5
    assert metadata["critic_head_ids"] == policy.agent_ids


def test_untrained_policy_initialization_is_reproducible_with_explicit_seed():
    seed_all(44); first = make_policy(2).state_dict()
    seed_all(44); second = make_policy(2).state_dict()
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_wrong_scenario_is_rejected(tmp_path):
    policy = make_policy(2); trainer = VanillaHAPPOTrainer(policy)
    path = tmp_path / "weights.pt"
    save_vanilla_happo_checkpoint(path, policy, trainer, environment_steps=0, episodes=0,
                                  config={"scenario": "2v2"}, numpy_rng=np.random.default_rng())
    with pytest.raises(ValueError, match="scenario mismatch"):
        load_vanilla_happo_checkpoint(
            path, policy, restore_rng=False, expected_scenario="3v2")


def test_red_and_blue_detection_and_attack_times_are_separate():
    red = SimpleNamespace(agent_id="red_0", side="red",
                          position=np.array([20000., 0., 0.]))
    blue = SimpleNamespace(agent_id="blue_0", side="blue",
                           position=np.array([0., 0., 0.]))
    tracker = {side: {"first_detection_time_s": None,
                      "first_attack_range_entry_s": None,
                      "first_launch_time_s": None, "first_hit_time_s": None,
                      "target_switches": 0, "missiles_fired": 0, "hits": 0}
               for side in ("red", "blue")}
    info = {"simulation_time_s": 1.0,
            "current_targets": {"red_0": None, "blue_0": "red_0"},
            "target_selection": {}, "missile_events": []}
    update_side_timing(tracker, info, [red, blue])
    assert tracker["blue"]["first_detection_time_s"] == 1.0
    assert tracker["red"]["first_detection_time_s"] is None
    red.position[:] = [10000, 0, 0]
    info.update(simulation_time_s=2.0,
                current_targets={"red_0": "blue_0", "blue_0": "red_0"})
    update_side_timing(tracker, info, [red, blue])
    assert tracker["red"]["first_detection_time_s"] == 2.0
    assert tracker["red"]["first_attack_range_entry_s"] == 2.0


def test_rule_action_stack_fills_only_dead_controlled_agent():
    agents = [
        SimpleNamespace(agent_id="red_0", alive=False),
        SimpleNamespace(agent_id="red_1", alive=True),
    ]
    env = SimpleNamespace(
        agent_ids=["red_0", "red_1"],
        task=SimpleNamespace(agents=agents),
        build_rule_actions=lambda _: {"red_1": np.array([24, 20, 20, 20])},
    )
    actions = stack_controlled_rule_actions(env)
    assert actions.shape == (2, 4)
    assert np.issubdtype(actions.dtype, np.integer)
    assert actions.tolist() == [[20, 20, 20, 20], [24, 20, 20, 20]]


def test_rule_action_stack_rejects_missing_alive_agent_action():
    env = SimpleNamespace(
        agent_ids=["red_0"],
        task=SimpleNamespace(
            agents=[SimpleNamespace(agent_id="red_0", alive=True)]),
        build_rule_actions=lambda _: {},
    )
    with pytest.raises(RuntimeError, match="red_0"):
        stack_controlled_rule_actions(env)


def test_rule_action_stack_rejects_wrong_action_shape():
    env = SimpleNamespace(
        agent_ids=["red_0"],
        task=SimpleNamespace(
            agents=[SimpleNamespace(agent_id="red_0", alive=True)]),
        build_rule_actions=lambda _: {"red_0": np.array([20, 20, 20])},
    )
    with pytest.raises(RuntimeError, match=r"shape \(4,\)"):
        stack_controlled_rule_actions(env)


def test_output_path_guard_rejects_parent_and_absolute_escape(tmp_path):
    root = tmp_path / "tam_uav"; root.mkdir()
    assert resolve_tam_output(root, "outputs/x.json") == root / "outputs/x.json"
    with pytest.raises(ValueError, match="inside tam_uav"):
        resolve_tam_output(root, "../outside.json")
    with pytest.raises(ValueError, match="inside tam_uav"):
        resolve_tam_output(root, tmp_path / "outside.json")


def test_output_path_guard_rejects_symlink_escape_when_supported(tmp_path):
    root = tmp_path / "tam_uav"; root.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    with pytest.raises(ValueError, match="inside tam_uav"):
        resolve_tam_output(root, "linked/result.json")


def test_independent_eval_perturbation_argument_and_zero_episode_skip_contract():
    assert parse_eval_args([]).perturbation == "low"
    assert parse_eval_args(["--perturbation", "medium"]).perturbation == "medium"
    source = (Path(__file__).parents[1] /
              "scripts/eval_tam_paper_vanilla_happo.py").read_text()
    assert "initial_perturbation=args.perturbation" in source
    assert should_run_evaluation(1024, 4, 1024, 2048)
    assert not should_run_evaluation(1024, 0, 1024, 2048)
    assert not should_run_evaluation(0, 4, 2048, 2048)
    assert not should_run_evaluation(1024, 4, 512, 2048)


def test_readiness_audit_uses_evidence_and_environment_hashes():
    unchanged, _ = environment_contract_unchanged()
    assert unchanged
    result = audit(tests_passed=True, smoke_summaries=[])
    assert result["ENVIRONMENT_CONTRACT_UNCHANGED"] is True
    assert result["HAPPO_ALGORITHM_CONTRACT_READY"] is False
    assert result["EARLY_PERFORMANCE_SIGNAL_OBSERVED"] is False
    assert result["LEARNING_CONVERGENCE_NOT_VALIDATED"] is True
    train_source = (Path(__file__).parents[1] /
                    "scripts/train_tam_paper_vanilla_happo.py").read_text()
    assert "HAPPO_ALGORITHM_CONTRACT_READY" not in train_source
    assert "ENVIRONMENT_CONTRACT_UNCHANGED" not in train_source


def test_no_forbidden_recurrent_or_attention_networks():
    source = (Path(__file__).parents[1] / "algorithms/happo/vanilla_happo.py").read_text()
    assert all(token not in source for token in (
        "nn.GRU", "nn.LSTM", "Transformer", "MultiheadAttention"))
