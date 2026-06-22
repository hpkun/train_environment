"""Test death-causing transition active mask semantics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def test_death_transition_has_active_mask_1():
    """Agent alive before step, dead after step: actor_active_mask must be 1."""
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    num_red, actor_dim, critic_dim, action_dim = 3, 96, 480, 4
    buf = HAPPORolloutBuffer(
        max_len=4, num_red=num_red, actor_dim=actor_dim, critic_dim=critic_dim,
        action_dim=action_dim, role_ids=[0, 1, 1],
        rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )

    # Transition 0: all alive
    obs = np.zeros((num_red, actor_dim), dtype=np.float32)
    critic = np.zeros(critic_dim, dtype=np.float32)
    actions = np.zeros((num_red, action_dim), dtype=np.int64)
    log_probs = np.zeros(num_red, dtype=np.float32)

    # alive_before = [1, 1, 1], alive_after = [0, 1, 1] (MAV died)
    active = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    next_active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    death_mask = active * (1.0 - next_active)  # [1, 0, 0]

    buf.store(
        obs, critic, actions, log_probs,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        0.0, active, next_value=0.0, env_id=0,
        death_transition_masks=death_mask,
        episode_start_masks=np.ones(num_red, dtype=np.float32),
        rnn_hidden=np.zeros((num_red, 128), dtype=np.float32),
    )

    # Verify death transition has active=1 for MAV
    stored_active = buf.active_masks[0, 0]
    stored_death = buf.death_transition_masks[0, 0]
    stored_alive = buf.agent_alive_masks[0, 0]

    assert stored_active == 1.0, f"Death transition active_mask should be 1, got {stored_active}"
    assert stored_death == 1.0, f"Death transition death_mask should be 1, got {stored_death}"
    assert stored_alive == 1.0, f"Death transition agent_alive should be 1, got {stored_alive}"


def test_post_death_transition_has_active_mask_0():
    """Agent dead before step: actor_active_mask must be 0."""
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    num_red, actor_dim, critic_dim, action_dim = 3, 96, 480, 4
    buf = HAPPORolloutBuffer(
        max_len=4, num_red=num_red, actor_dim=actor_dim, critic_dim=critic_dim,
        action_dim=action_dim, role_ids=[0, 1, 1],
        rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )

    obs = np.zeros((num_red, actor_dim), dtype=np.float32)
    critic = np.zeros(critic_dim, dtype=np.float32)
    actions = np.zeros((num_red, action_dim), dtype=np.int64)
    log_probs = np.zeros(num_red, dtype=np.float32)

    # Post-death: MAV dead before action
    active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    next_active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    death_mask = active * (1.0 - next_active)

    buf.store(
        obs, critic, actions, log_probs,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        0.0, active, next_value=0.0, env_id=0,
        death_transition_masks=death_mask,
        episode_start_masks=np.ones(num_red, dtype=np.float32),
        rnn_hidden=np.zeros((num_red, 128), dtype=np.float32),
    )

    assert buf.active_masks[0, 0] == 0.0, "Post-death active_mask should be 0"
    assert buf.death_transition_masks[0, 0] == 0.0, "Post-death death_mask should be 0"
    assert buf.agent_alive_masks[0, 0] == 0.0, "Post-death agent_alive should be 0"


def test_death_transition_reward_in_buffer():
    """Death-causing transition reward must be stored in buffer."""
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    num_red, actor_dim, critic_dim, action_dim = 3, 96, 480, 4
    buf = HAPPORolloutBuffer(
        max_len=4, num_red=num_red, actor_dim=actor_dim, critic_dim=critic_dim,
        action_dim=action_dim, role_ids=[0, 1, 1],
        rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )

    obs = np.zeros((num_red, actor_dim), dtype=np.float32)
    critic = np.zeros(critic_dim, dtype=np.float32)
    actions = np.zeros((num_red, action_dim), dtype=np.int64)
    log_probs = np.zeros(num_red, dtype=np.float32)
    active = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    next_active = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    death_mask = active * (1.0 - next_active)
    reward = np.array([-4.0, 0.0, 0.0], dtype=np.float32)

    buf.store(
        obs, critic, actions, log_probs,
        reward, np.array([1.0, 1.0, 1.0], dtype=np.float32),
        0.0, active, next_value=0.0, env_id=0,
        death_transition_masks=death_mask,
        episode_start_masks=np.ones(num_red, dtype=np.float32),
        rnn_hidden=np.zeros((num_red, 128), dtype=np.float32),
    )

    assert buf.rewards[0, 0] == -4.0, f"Death reward should be -4.0, got {buf.rewards[0,0]}"


def test_death_mask_stored_in_buffer():
    """Buffer must have death_transition_masks stored and retrievable."""
    from algorithms.happo.happo_buffer import HAPPORolloutBuffer

    num_red, actor_dim, critic_dim, action_dim = 3, 96, 480, 4
    buf = HAPPORolloutBuffer(
        max_len=4, num_red=num_red, actor_dim=actor_dim, critic_dim=critic_dim,
        action_dim=action_dim, role_ids=[0, 1, 1],
        rnn_hidden_size=128, action_dtype=np.int64, num_envs=1,
    )

    obs = np.zeros((num_red, actor_dim), dtype=np.float32)
    critic = np.zeros(critic_dim, dtype=np.float32)
    actions = np.zeros((num_red, action_dim), dtype=np.int64)
    log_probs = np.zeros(num_red, dtype=np.float32)

    for i, (alive_before, alive_after) in enumerate([
        ([1, 1, 1], [1, 1, 1]),  # t-1: all alive
        ([1, 1, 1], [0, 1, 1]),  # t: MAV dies
        ([0, 1, 1], [0, 1, 1]),  # t+1: MAV dead
        ([0, 1, 1], [0, 1, 1]),  # t+2: MAV dead
    ]):
        active_arr = np.array(alive_before, dtype=np.float32)
        next_arr = np.array(alive_after, dtype=np.float32)
        dmask = active_arr * (1.0 - next_arr)
        buf.store(
            obs, critic, actions, log_probs,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            0.0, active_arr, next_value=0.0, env_id=0,
            death_transition_masks=dmask,
            episode_start_masks=np.ones(num_red, dtype=np.float32),
            rnn_hidden=np.zeros((num_red, 128), dtype=np.float32),
        )

    data = buf.get("cpu")
    assert "death_transition_masks" in data, "death_transition_masks not in buffer data"
    dm = data["death_transition_masks"].numpy()
    assert dm[1, 0] == 1.0, f"Death transition should have death_mask=1, got {dm[1,0]}"
    assert dm[0, 0] == 0.0, "t-1 should have death_mask=0"
    assert dm[2, 0] == 0.0, "t+1 should have death_mask=0"
    assert dm[3, 0] == 0.0, "t+2 should have death_mask=0"

    # t=1: MAV alive before, dead after → active=1, death=1
    assert data["active_masks"][1, 0] == 1.0
    assert data["agent_alive_masks"][1, 0] == 1.0
