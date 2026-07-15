from __future__ import annotations

import numpy as np
import pytest

from configs.experiment_presets import get_preset
from configs.paper_learnable_3v3_spec import (
    LEARNABLE_MISSILE_LAUNCH_SPEED_MPS,
    LEARNABLE_PAPER_ENVIRONMENT_CONFIG,
    PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
    learnable_environment_snapshot,
)
from my_uav_env.blue_policy_profiles import BluePolicyController
from my_uav_env.env import UavCombatEnv, make_empty_launch_diag
from my_uav_env.fire_control import FireControlState
from my_uav_env.simulator import MissileSimulator
import torch
from train_vanilla_mappo import (
    Config,
    VanillaActor,
    CentralizedCritic,
    _atomic_torch_save,
    _build_training_state,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _run_periodic_evaluation,
    _training_log_fields,
    _validate_training_state,
)


def _strict_obs() -> dict:
    enemies = np.zeros((3, 10), dtype=np.float32)
    enemies[:, 0] = [3000.0, 4000.0, 5000.0]
    enemies[:, 9] = [3000.0, 4000.0, 5000.0]
    ego = np.zeros(10, dtype=np.float32)
    ego[2:4] = [6000.0, 300.0]
    return {
        "ego_state": ego,
        "ally_states": np.zeros((2, 10), dtype=np.float32),
        "enemy_states": enemies,
        "alive_mask": np.ones(6, dtype=np.float32),
        "missile_warning": np.zeros(1, dtype=np.float32),
        "altitude": np.array([6000.0], dtype=np.float32),
        "velocity": np.array([300.0, 0.0, 0.0], dtype=np.float32),
    }


def test_learnable_profile_contract_and_dimensions():
    snapshot = learnable_environment_snapshot(
        num_red=3, num_blue=3, sim_freq=60, agent_interaction_steps=12,
        max_episode_length=1400,
        blue_policy_profile="paper_learnable_fixed_pair_v1", seed=3,
        initial_condition_randomization_mode="deterministic_v1")
    assert snapshot["profile_provenance"]["value"] == "learnability_adaptation"
    assert snapshot["launch_range_m"]["value"] == (1000.0, 8000.0)
    assert snapshot["initial_missile_direction_mode"]["value"] == (
        "aircraft_body_x_v1")
    assert _compute_obs_dim(3, 3, True, "paper_strict") == 60
    assert _compute_global_state_dim(3, "paper_strict") == 30
    preset = get_preset("vanilla_3v3_paper_learnable_500k")
    assert preset["environment_profile"] == PAPER_LEARNABLE_ENVIRONMENT_PROFILE
    assert preset["total_env_steps"] == 500_000
    assert "MissilePNNonzeroCommandFrames" in _training_log_fields()
    assert "TargetSwitchesWhileAlive" in _training_log_fields()


def test_learnable_blue_fixed_pair_reallocates_only_after_death():
    obs = {f"blue_{i}": _strict_obs() for i in range(3)}
    controller = BluePolicyController("paper_learnable_fixed_pair_v1")
    controller.reset(
        [f"blue_{i}" for i in range(3)], [f"red_{i}" for i in range(3)],
        {f"blue_{i}": 0.0 for i in range(3)},
        {f"blue_{i}": 6000.0 for i in range(3)})
    controller.act(
        obs, 3, 3, {"red_0"}, {}, {f"blue_{i}": 0.0 for i in range(3)}, 1)
    assert controller.current_targets["blue_0"] == "red_0"
    assert controller.target_switches_while_alive == 0

    obs["blue_0"]["alive_mask"][3] = 0.0
    controller.act(
        obs, 3, 3, {"red_1"}, {}, {f"blue_{i}": 0.2 for i in range(3)}, 2)
    assert controller.current_targets["blue_0"] == "red_2"
    diag = controller.snapshot_episode_diagnostics()
    assert diag["blue_target_reallocations_after_death"] == 1
    assert diag["blue_target_switches_while_alive"] == 0


class _Aircraft:
    dt = 1 / 60
    lon0, lat0, alt0 = 120.0, 60.0, 0.0
    is_alive = True

    def __init__(self, uid, color, position, velocity, rpy):
        self.uid = uid
        self.color = color
        self.position = np.asarray(position, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.rpy = np.asarray(rpy, dtype=np.float64)
        self.launch_missiles = []
        self.under_missiles = []

    def get_geodetic(self):
        return np.array([120.0, 60.0, self.position[2]])

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_rpy(self):
        return self.rpy

    def shotdown(self):
        self.is_alive = False


def test_learnable_missile_uses_body_x_and_respects_arming_time():
    parent = _Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [0.0, 300.0, 0.0],
        [0.4, np.deg2rad(10.0), np.deg2rad(90.0)])
    target = _Aircraft(
        "blue_0", "Blue", [0.0, 80.0, 6000.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    missile = MissileSimulator.create(
        parent, target, "m0",
        guidance_mode="paper_learnable_point_mass_v1",
        config=LEARNABLE_PAPER_ENVIRONMENT_CONFIG.missile,
        launch_speed_mps=LEARNABLE_MISSILE_LAUNCH_SPEED_MPS,
        overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    velocity = missile.get_velocity()
    assert np.linalg.norm(velocity) == pytest.approx(800.0)
    assert velocity[1] > 700.0
    missile._roll_hit_probability = lambda: True
    for _ in range(10):
        missile.run()
    assert not missile.is_success


def test_learnable_point_mass_is_finite_constant_speed_and_multiframe():
    parent = _Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    target = _Aircraft(
        "blue_0", "Blue", [5000.0, 1000.0, 6000.0], [250.0, 0.0, 0.0],
        [0.0, 0.0, np.pi])
    missile = MissileSimulator.create(
        parent, target, "m0",
        guidance_mode="paper_learnable_point_mass_v1",
        config=LEARNABLE_PAPER_ENVIRONMENT_CONFIG.missile,
        launch_speed_mps=800.0, overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    for _ in range(20):
        missile.run()
        assert np.all(np.isfinite(missile.get_velocity()))
        assert np.linalg.norm(missile.get_velocity()) == pytest.approx(800.0)
    assert missile._pn_guidance_frames == 20
    assert missile._pn_nonzero_command_frames > 0
    assert missile._maximum_command_g <= 30.0 + 1e-6


@pytest.mark.parametrize("reason", [
    "hit", "p_hit_fail", "timeout", "target_dead", "overshoot"])
def test_learnable_missile_terminal_reasons_are_distinct(reason):
    parent = _Aircraft(
        "red_0", "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    target = _Aircraft(
        "blue_0", "Blue", [1000.0, 0.0, 6000.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    missile = MissileSimulator.create(
        parent, target, "m0",
        guidance_mode="paper_learnable_point_mass_v1",
        config=LEARNABLE_PAPER_ENVIRONMENT_CONFIG.missile,
        launch_speed_mps=800.0, overshoot_window_s=0.5,
        overshoot_distance_hysteresis_m=50.0,
        positive_closing_threshold_mps=1.0)
    missile._state_trans = lambda _action: None
    missile._t = 0.3
    if reason in ("hit", "p_hit_fail"):
        missile._guidance = lambda: (np.zeros(2), 50.0)
        missile._roll_hit_probability = lambda: reason == "hit"
        missile.run()
    elif reason == "timeout":
        missile._t = missile._t_max
        missile._guidance = lambda: (np.zeros(2), 500.0)
        missile.run()
    elif reason == "target_dead":
        target.is_alive = False
        missile._guidance = lambda: (np.zeros(2), 500.0)
        missile.run()
    else:
        missile._velocity[:] = [-100.0, 0.0, 0.0]
        missile._historical_min_range_m = 100.0
        missile._has_ever_positive_closing = True
        missile._guidance = lambda: (np.zeros(2), 200.0)
        for _ in range(missile._distance_increment.maxlen):
            missile.run()
    assert missile.is_done
    assert missile._termination_reason == reason


def test_learnable_jitter_is_reproducible_and_mirrored():
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
        initial_condition_randomization_mode="small_symmetric_jitter_v1")
    env._seed = 17
    env.np_random = np.random.default_rng(17)
    first = env._make_initial_jitter()
    env.np_random = np.random.default_rng(17)
    second = env._make_initial_jitter()
    assert first == second
    env._initial_jitter_by_index = first
    blue = env._make_init_state("Blue", 0)
    red = env._make_init_state("Red", 0)
    assert blue["ic/psi-true-deg"] == pytest.approx(
        -red["ic/psi-true-deg"])
    assert blue["ic/h-sl-ft"] == pytest.approx(red["ic/h-sl-ft"])
    assert blue["ic/u-fps"] == pytest.approx(red["ic/u-fps"])


@pytest.mark.parametrize("range_m,expected_launch", [
    (999.0, False), (1000.0, True), (3000.0, True),
    (5000.0, True), (8000.0, True), (8001.0, False),
])
def test_learnable_fire_control_range_boundaries(
        monkeypatch, range_m, expected_launch):
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    shooter = _Aircraft(
        "blue_0", "Blue", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    shooter.num_left_missiles = 999
    target = _Aircraft(
        "red_0", "Red", [range_m, 0.0, 6000.0], [-300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    dead = _Aircraft(
        "red_1", "Red", [range_m, 0.0, 6000.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    dead.is_alive = False
    env.blue_planes = {"blue_0": shooter}
    env.red_planes = {"red_0": target, "red_1": dead, "red_2": dead}
    env.agent_ids = ["blue_0"]
    env._lock_timer = {"blue_0": 0}
    env._lock_target = {"blue_0": None}
    env._missile_cooldown = {"blue_0": 0}
    env._fire_control_states = {"blue_0": FireControlState()}
    env._fire_control_assignments = {"blue_0": "red_0"}
    env._target_assignment_diagnostics = {
        "target_reallocations": 0, "target_reallocations_after_death": 0,
        "target_switches_while_alive": 0, "engaged_wait_frames": 0,
        "no_alive_target_frames": 0}
    env._engaged_targets = set()
    env._launch_diag_step = make_empty_launch_diag()
    env._agents_deny_kill = set()
    monkeypatch.setattr(env, "_is_detected_by_electro_optical", lambda *_: True)
    monkeypatch.setattr("my_uav_env.env.compute_3d_range", lambda *_: range_m)
    monkeypatch.setattr("my_uav_env.env.compute_body_x_q_los", lambda *_: 0.0)
    monkeypatch.setattr(
        "my_uav_env.env.get2d_heading_AO_TA_R",
        lambda *_: (0.0, np.pi, range_m))
    launches = []
    monkeypatch.setattr(
        env, "_launch_missile",
        lambda parent, enemy, quality: launches.append((parent.uid, enemy.uid)))
    for _ in range(env.missile_lock_delay_frames):
        env._check_missile_launch()
    assert bool(launches) is expected_launch
    diag = env._launch_diag_step["blue"]
    assert diag["range_low_blocked"] > 0 if range_m < 1000.0 else True
    assert diag["range_high_blocked"] > 0 if range_m > 8000.0 else True


def _learnable_training_config(tmp_path) -> Config:
    config = Config()
    config.environment_profile = PAPER_LEARNABLE_ENVIRONMENT_PROFILE
    config.environment_version = PAPER_LEARNABLE_ENVIRONMENT_PROFILE
    config.blue_policy_profile = "paper_learnable_fixed_pair_v1"
    config.reward_mode = "paper_minimal_joint_v1"
    config.pid_profile = "paper_minimal_shared_v1"
    config.missile_guidance_mode = "paper_learnable_point_mass_v1"
    config.total_env_steps = 500_000
    config.seed = 3
    config.eval_episodes = 2
    config.eval_log_file = str(tmp_path / "eval.csv")
    return config


def test_full_training_state_round_trip_and_total_step_extension(tmp_path):
    config = _learnable_training_config(tmp_path)
    metadata = _checkpoint_metadata(config, 60, 30)
    actor = VanillaActor(obs_dim=60, hidden=8, rnn_hidden=4)
    critic = CentralizedCritic(global_obs_dim=30, hidden=8)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    runtime = {
        "run_id": "test", "total_steps": 10_000, "iteration": 6,
        "total_episodes": 4, "red_wins": 1, "blue_wins": 2, "draws": 1,
    }
    payload = _build_training_state(
        actor, critic, actor_opt, critic_opt, config, metadata, runtime)
    path = tmp_path / "latest_training_state.pt"
    _atomic_torch_save(payload, str(path))
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _validate_training_state(loaded, config, metadata)
    config.total_env_steps = 1_000_000
    _validate_training_state(loaded, config, metadata)
    config.total_env_steps = 9_999
    with pytest.raises(ValueError, match="only stay equal or increase"):
        _validate_training_state(loaded, config, metadata)


def test_periodic_eval_marks_final_checkpoint_and_restores_rng(
        tmp_path, monkeypatch):
    import evaluate_vanilla_mappo

    config = _learnable_training_config(tmp_path)
    metadata = _checkpoint_metadata(config, 60, 30)
    actor = VanillaActor(obs_dim=60, hidden=8, rnn_hidden=4)
    monkeypatch.setattr(
        evaluate_vanilla_mappo, "run_one_episode",
        lambda **kwargs: {
            "RedWin": 1, "BlueWin": 0, "Draw": 0,
            "EpisodeRewardRed": 5.0})
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    result = _run_periodic_evaluation(
        actor, config, torch.device("cpu"), metadata,
        iteration=7, total_steps=10_000, final_checkpoint=True)
    assert result["FinalCheckpoint"] == 1
    assert result["Step"] == 10_000
    assert torch.equal(before, torch.get_rng_state())
    header = (tmp_path / "eval.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "FinalCheckpoint" in header
