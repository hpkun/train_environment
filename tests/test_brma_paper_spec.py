import math

import numpy as np
import torch

from attention_models import AttentionActor, CentralizedAttentionCritic
from configs.brma_mappo_paper_spec import PAPER_EXPLICIT, PAPER_SPEC
from configs.experiment_presets import EXPERIMENT_PRESETS
from my_uav_env.alignment.reward_utils import (
    altitude_reward_pairwise_sum_eq17,
)


def test_every_paper_spec_value_has_supported_provenance():
    supported = {
        "paper_explicit", "paper_equation", "paper_inferred",
        "paper_unspecified_engineering",
    }
    assert PAPER_SPEC
    assert all(item.source in supported for item in PAPER_SPEC.values())
    assert PAPER_SPEC["episode_steps"].source == PAPER_EXPLICIT


def test_required_presets_keep_canonical_and_main_scale_separate():
    required = {
        "main_3v3_mappo_mlp", "main_3v3_mappo_attention",
        "main_3v3_brma_mappo", "paper_6v6_mappo_mlp",
        "paper_6v6_mappo_attention", "paper_6v6_mappo_dropout",
        "paper_6v6_brma_mappo", "paper_eval_6v6",
        "paper_eval_8v8_zero_shot", "paper_eval_10v10_zero_shot",
    }
    assert required <= EXPERIMENT_PRESETS.keys()
    canonical = EXPERIMENT_PRESETS["paper_6v6_brma_mappo"]
    assert canonical["num_red"] == canonical["num_blue"] == 6
    assert canonical["num_envs"] == 32
    assert canonical["total_env_steps"] == 10_000_000
    assert canonical["brma_max_mask_allies"] == 2
    assert canonical["brma_max_mask_enemies"] == 2
    assert canonical["brma_lr"] == 5e-4
    main = EXPERIMENT_PRESETS["main_3v3_brma_mappo"]
    assert main["num_red"] == main["num_blue"] == 3
    assert main["reward_mode"] == canonical["reward_mode"]


def test_eq17_multi_enemy_interpretation_is_pairwise_sum():
    one = altitude_reward_pairwise_sum_eq17(5000.0, [3000.0])
    two = altitude_reward_pairwise_sum_eq17(5000.0, [3000.0, 3000.0])
    assert math.isclose(two, 2.0 * one)


def test_attention_critic_outputs_one_team_value_per_env_step():
    critic = CentralizedAttentionCritic(
        entity_dim=10, hidden_size=16, num_heads=4,
        num_agents=3, encoder_mode="paper_eq33",
    )
    entities = torch.zeros(5, 3, 6, 10)
    masks = torch.zeros(5, 3, 6, dtype=torch.long)
    assert critic(entities, masks).shape == (5, 1)


def test_attention_squashed_distribution_is_finite_at_extremes():
    actor = AttentionActor(
        entity_dim=10, hidden_size=16, rnn_hidden=8,
        num_heads=4, encoder_mode="paper_eq33",
    )
    with torch.no_grad():
        actor.action_log_std.fill_(100.0)
        actor.action_head[-1].bias.fill_(100.0)
    entities = torch.zeros(2, 6, 10)
    masks = torch.zeros(2, 6, dtype=torch.long)
    dist, _hidden, _weights = actor(entities, masks, torch.zeros(2, 8))
    action = dist.mode
    assert torch.isfinite(action).all()
    assert torch.max(torch.abs(action)) <= 1.0
    assert torch.isfinite(dist.log_prob(action)).all()
