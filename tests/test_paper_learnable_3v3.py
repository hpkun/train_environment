from __future__ import annotations

from collections import Counter
import json
import numpy as np
import pytest

from configs.experiment_presets import get_preset
from configs.paper_learnable_3v3_spec import (
    LEARNABILITY_ADAPTATION,
    LEARNABLE_MISSILE_LAUNCH_SPEED_MPS,
    LEARNABLE_PAPER_ENVIRONMENT_CONFIG,
    PAPER_LEARNABLE_ENVIRONMENT_PROFILE,
    learnable_environment_snapshot,
)
from my_uav_env.blue_policy_profiles import BluePolicyController
from my_uav_env.env import UavCombatEnv, make_empty_launch_diag
from my_uav_env.fire_control import FireControlState
from my_uav_env.pid_controller import PIDController
from my_uav_env.simulator import MissileSimulator
import torch
from train_vanilla_mappo import (
    ACTION_DISTRIBUTION_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    ENTROPY_ESTIMATOR_VERSION,
    Config,
    RolloutBuffer,
    SubprocVecEnv,
    VanillaActor,
    CentralizedCritic,
    _atomic_torch_save,
    _build_training_state,
    _checkpoint_metadata,
    _compute_global_state_dim,
    _compute_obs_dim,
    _episode_is_invalid,
    _default_eval_log_file,
    _default_extreme_load_trace_file,
    _default_launch_quality_file,
    _append_invalid_trace_jsonl,
    _learnability_iteration_metrics,
    _run_periodic_evaluation,
    _training_log_fields,
    _validate_training_state,
    _validate_preset_resume_semantics,
)
from scripts.audit_paper_learnable_3v3 import _circular_error


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
    assert snapshot["launch_range_m"]["value"] == (0.0, 10_000.0)
    assert snapshot["initial_missile_direction_mode"]["value"] == (
        "aircraft_body_x_v1")
    for key in (
            "environment_profile", "profile_provenance", "missile_profile",
            "initial_missile_direction_mode", "initial_missile_speed_mps",
            "missile_hit_radius_m", "missile_arming_time_s"):
        assert snapshot[key]["source"] == LEARNABILITY_ADAPTATION
    missile = snapshot["environment_config"]["missile"]
    assert missile["maximum_flight_time_s"]["source"] == LEARNABILITY_ADAPTATION
    assert snapshot["setpoint_rate_limiter"]["value"] == (
        "disabled_for_paper_eq12_14")
    assert snapshot["load_command_scaling"]["value"] == (
        "disabled_for_paper_eq12_14")
    assert snapshot["pid_error_definition"]["value"] == (
        "paper_eq13_quadrant_preserving_operational_v1")
    assert snapshot["environment_config"]["pid"]["throttle_base"]["value"] == 0.8
    assert _compute_obs_dim(3, 3, True, "paper_strict") == 60
    assert _compute_global_state_dim(3, "paper_strict") == 30
    preset = get_preset("vanilla_3v3_paper_learnable_500k")
    assert preset["environment_profile"] == PAPER_LEARNABLE_ENVIRONMENT_PROFILE
    assert preset["total_env_steps"] == 500_000
    assert "MissilePNNonzeroCommandFrames" in _training_log_fields()
    assert "TargetSwitchesWhileAlive" in _training_log_fields()
    for field in (
            "NonFiniteLoadInvalidEpisodes",
            "CatastrophicFiniteLoadInvalidEpisodes",
            "PersistentExtremeFiniteLoadInvalidEpisodes",
            "RedMWSWarningGenerations",
            "RedSetpointRateLimitActivations",
            "RedMaximumAbsoluteEPhi",
            "BlueDegenerateArctanRatioCount"):
        assert field in _training_log_fields()


def test_1m_preset_has_isolated_paths_and_explicit_resume_only():
    from types import SimpleNamespace
    preset_500k = get_preset("vanilla_3v3_paper_learnable_500k")
    preset_1m = get_preset("vanilla_3v3_paper_learnable_1m")
    for field in ("checkpoint_dir", "log_file", "results_file"):
        assert preset_500k[field] != preset_1m[field]
        assert "1m" in preset_1m[field]
    paths_500k = {
        preset_500k["log_file"], preset_500k["results_file"],
        _default_eval_log_file(preset_500k["results_file"]),
        _default_extreme_load_trace_file(preset_500k["results_file"]),
        _default_launch_quality_file(preset_500k["results_file"]),
        *(f"{preset_500k['checkpoint_dir']}/{name}" for name in (
            "run_manifest.json", "latest_training_state.pt",
            "vanilla_actor_best.pt", "centralized_critic_best.pt")),
    }
    paths_1m = {
        preset_1m["log_file"], preset_1m["results_file"],
        _default_eval_log_file(preset_1m["results_file"]),
        _default_extreme_load_trace_file(preset_1m["results_file"]),
        _default_launch_quality_file(preset_1m["results_file"]),
        *(f"{preset_1m['checkpoint_dir']}/{name}" for name in (
            "run_manifest.json", "latest_training_state.pt",
            "vanilla_actor_best.pt", "centralized_critic_best.pt")),
    }
    assert paths_500k.isdisjoint(paths_1m)
    assert not preset_1m.get("resume_latest", False)
    assert not preset_1m.get("resume_from_best", False)
    _validate_preset_resume_semantics(SimpleNamespace(
        preset="vanilla_3v3_paper_learnable_1m", resume_latest=False,
        resume_from_best=False, resume_state=None))
    with pytest.raises(ValueError, match="explicit --resume-state"):
        _validate_preset_resume_semantics(SimpleNamespace(
            preset="vanilla_3v3_paper_learnable_1m", resume_latest=True,
            resume_from_best=False, resume_state=None))


def test_learnable_blue_policy_consumes_environment_assignment_only():
    obs = {f"blue_{i}": _strict_obs() for i in range(3)}
    controller = BluePolicyController("paper_learnable_fixed_pair_v1")
    controller.reset(
        [f"blue_{i}" for i in range(3)], [f"red_{i}" for i in range(3)],
        {f"blue_{i}": 0.0 for i in range(3)},
        {f"blue_{i}": 6000.0 for i in range(3)})
    controller.act(
        obs, 3, 3, {"red_0"}, {}, {f"blue_{i}": 0.0 for i in range(3)}, 1,
        assigned_targets={f"blue_{i}": f"red_{i}" for i in range(3)})
    assert controller.current_targets["blue_0"] == "red_0"
    assert controller.target_switches_while_alive == 0

    obs["blue_0"]["alive_mask"][3] = 0.0
    controller.act(
        obs, 3, 3, {"red_1"}, {}, {f"blue_{i}": 0.2 for i in range(3)}, 2,
        assigned_targets={"blue_0": "red_2", "blue_1": "red_1",
                          "blue_2": "red_2"})
    assert controller.current_targets["blue_0"] == "red_2"
    diag = controller.snapshot_episode_diagnostics()
    assert diag["blue_target_reallocations_after_death"] == 0
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


class _Incoming:
    def __init__(self, uid):
        self.uid = uid


def test_learnable_red_mws_direction_and_absolute_target_are_stable():
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    env._learnable_mws_state = {"red_0": env._empty_learnable_mws_state()}
    env._mws_decision_diagnostics = Counter()
    first = env._learnable_red_mws_target(
        "red_0", _Incoming("m0"), 0.2, 1.0)
    second = env._learnable_red_mws_target(
        "red_0", _Incoming("m0"), 1.1, -1.0)
    assert first == second
    assert first[1] == pytest.approx(env._wrap_angle_rad(0.2 + np.deg2rad(60)))
    assert env._mws_decision_diagnostics["red_warning_generations"] == 1
    assert env._mws_decision_diagnostics[
        "red_direction_changes_within_same_missile"] == 0
    assert env._mws_decision_diagnostics[
        "red_suppressed_direction_flip_attempts"] == 1
    env._deactivate_learnable_mws_state("red_0")
    env._learnable_red_mws_target("red_0", _Incoming("m1"), 0.4, -1.0)
    assert env._mws_decision_diagnostics["red_warning_generations"] == 2


def test_invalid_trace_jsonl_preserves_twelve_frames_and_context(tmp_path):
    path = tmp_path / "run_extreme_load_traces.jsonl"
    frames = [{
        "physics_frame": index,
        "d_B_des": [1.0, 0.0, 0.0],
        "e_phi": 0.0,
        "e_theta": 0.0,
        "roll_pid": {"p": 0.0, "i": 0.0, "d": 0.0},
        "pitch_pid": {"p": 0.0, "i": 0.0, "d": 0.0},
        "velocity_pid": {"p": 0.0, "i": 0.0, "d": 0.0},
        "unsaturated_commands": [0.0, 0.0, 0.0, 0.5],
        "saturated_commands": [0.0, 0.0, 0.0, 0.5],
        "g_load_components": [0.0, 0.0, float("nan")],
        "g_load_total": float("nan"),
        "rpy_rad": [0.0, 0.0, 0.0],
        "body_rates_rad_s": [0.0, 0.0, 0.0],
        "airspeed_mps": 300.0,
        "alpha_rad": 0.0,
        "beta_rad": 0.0,
        "mws_active": False,
        "incoming_missile_uid": None,
    } for index in range(12)]
    written = _append_invalid_trace_jsonl(
        str(path), run_id="audit-run", seed=7, total_step=123,
        env_index=2,
        episode_info={
            "EpisodeLength": 17,
            "invalid_numerical_reasons": ["red_0:NonFiniteLoad"],
        },
        traces=[{
            "trigger_agent_id": "red_0",
            "trigger_g": float("nan"),
            "trigger_level": "numerical_invalid",
            "frames": frames,
        }])
    assert written == 1
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "audit-run"
    assert record["seed"] == 7
    assert record["total_step"] == 123
    assert record["env_index"] == 2
    assert record["episode_step"] == 17
    assert record["agent_id"] == "red_0"
    assert record["team"] == "red"
    assert record["invalid_reason"] == "NonFiniteLoad"
    assert len(record["physics_frames"]) == 12
    assert record["physics_frames"][-1]["g_load_total"] is None


def test_invalid_trace_without_matching_agent_never_uses_unknown(tmp_path):
    path = tmp_path / "fallback_trace.jsonl"
    _append_invalid_trace_jsonl(
        str(path), run_id="audit-run", seed=3, total_step=1, env_index=0,
        episode_info={
            "EpisodeLength": 1,
            "invalid_numerical_reasons": ["red_1:NonFiniteState"],
        },
        traces=[{
            "trigger_agent_id": "red_0",
            "trigger_level": "numerical_invalid",
            "frames": [],
        }])
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["invalid_reason"] == "NonFiniteState"
    assert record["invalid_reason"] != "unknown"


@pytest.mark.parametrize("aid", ["red_0", "blue_0"])
def test_learnable_setpoints_pass_through_without_rate_limiting(aid):
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    aircraft = _Aircraft(
        aid, "Red" if aid.startswith("red") else "Blue",
        [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, np.deg2rad(170.0)])
    if aid.startswith("red"):
        env.red_planes = {aid: aircraft}
    else:
        env.blue_planes = {aid: aircraft}
    env._aircraft_diagnostics = {aid: {
        "setpoint_rate_limit_activations": 0,
        "heading_rate_limit_activations": 0,
        "pitch_rate_limit_activations": 0,
        "velocity_rate_limit_activations": 0,
        "requested_heading_jump_max_rad": 0.0,
        "applied_heading_jump_max_rad": 0.0,
        "requested_pitch_jump_max_rad": 0.0,
        "applied_pitch_jump_max_rad": 0.0,
        "requested_velocity_jump_max_mps": 0.0,
        "applied_velocity_jump_max_mps": 0.0}}
    env._learnable_setpoint_state = {}
    env._learnable_requested_setpoints = {}
    env._learnable_command_sources = {}
    applied = env._finalize_learnable_target(
        aid, (np.deg2rad(40), np.deg2rad(-170), 400.0), "base_policy")
    assert applied == pytest.approx(
        (np.deg2rad(40), np.deg2rad(-170), 400.0))
    assert env._aircraft_diagnostics[aid][
        "setpoint_rate_limit_activations"] == 0


def test_learnable_load_monitor_does_not_compute_command_scale():
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    aid = "red_0"
    env._aircraft_diagnostics = {aid: {
        "maximum_load_g_seen": 0.0, "frames_above_9g": 0,
        "consecutive_above_9g_frames": 0,
        "maximum_consecutive_above_9g_frames": 0,
        "episode_ever_exceeded_9g": False,
        "consecutive_above_30g_frames": 0,
        "maximum_consecutive_above_30g_frames": 0,
        "transient_above_30g_events": 0,
        "load_limiter_activations": 0,
        "load_protection_active_frames": 0,
        "load_protection_minimum_scale": 1.0}}
    assert env._update_paper_learnable_load_diagnostics(aid, 9.0) is None
    env._update_paper_learnable_load_diagnostics(aid, 12.0)
    env._update_paper_learnable_load_diagnostics(aid, 15.0)
    env._update_paper_learnable_load_diagnostics(aid, 31.0)
    assert env._paper_learnable_load_invalid_reason(31.0, 1) is None
    env._update_paper_learnable_load_diagnostics(aid, 8.0)
    assert env._aircraft_diagnostics[aid]["transient_above_30g_events"] == 1
    assert env._paper_learnable_load_invalid_reason(31.0, 3) == (
        "PersistentExtremeFiniteLoad")
    assert env._paper_learnable_load_invalid_reason(101.0, 1) == (
        "CatastrophicFiniteLoad")
    assert env._paper_learnable_load_invalid_reason(float("nan"), 0) == (
        "NonFiniteLoad")
    assert env._aircraft_diagnostics[aid]["load_limiter_activations"] == 0
    assert env._aircraft_diagnostics[aid]["load_protection_active_frames"] == 0


def test_extreme_load_trace_retains_last_twelve_frames_and_causal_fields():
    from collections import deque
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    aid = "red_0"
    sim = _Aircraft(
        aid, "Red", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    env._load_frame_history = {aid: deque(maxlen=12)}
    env._last_load_trace_level = {aid: 0}
    env._retained_extreme_load_traces = []
    env._aircraft_diagnostics = {aid: {
        "consecutive_above_30g_frames": 0}}
    env._learnable_mws_state = {aid: env._empty_learnable_mws_state()}
    env._learnable_requested_setpoints = {aid: (0.1, 0.2, 300.0)}
    env._learnable_previous_setpoints = {aid: (0.0, 0.0, 300.0)}
    env._learnable_setpoint_state = {aid: (0.1, 0.2, 300.0)}
    env._learnable_command_sources = {aid: "mws_override"}
    for frame in range(12):
        env._physics_frame = frame
        env._retain_load_frame(
            aid, sim, (0.0, 0.0, 1.0), 1.0,
            env._learnable_requested_setpoints[aid],
            env._learnable_setpoint_state[aid], (0.2, 0.3, 0.0, 0.5), 1.0)
    env._physics_frame = 12
    env._retain_load_frame(
        aid, sim, (0.0, 0.0, 21.0), 21.0,
        env._learnable_requested_setpoints[aid],
        env._learnable_setpoint_state[aid], (1.0, 1.0, 0.0, 0.5), 1.0)
    trace = env._retained_extreme_load_traces[-1]
    assert len(trace["frames"]) == 12
    assert trace["trigger_level"] == 2
    final = trace["frames"][-1]
    for field in (
            "g_components", "requested_setpoints", "applied_setpoints",
            "previous_applied_setpoints", "pid_commands_before_load_protection",
            "mws_state", "pid_saturation", "load_protection_scale",
            "action_update"):
        assert field in final


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
        target.position[:] = [50.0, 0.0, 6000.0]
        missile._guidance = lambda: pytest.fail(
            "contact frame must terminate before PN guidance")
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
    (0.0, False), (100.0, True), (999.0, True), (1000.0, True),
    (8000.0, True), (8001.0, True), (10_000.0, True),
    (10_000.1, False),
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
    assert diag["range_low_blocked"] > 0 if range_m <= 0.0 else True
    assert diag["range_high_blocked"] > 0 if range_m > 10_000.0 else True


def test_learnable_fire_control_tracks_before_ta_and_enforces_cooldown(
        monkeypatch):
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    shooter = _Aircraft(
        "blue_0", "Blue", [0.0, 0.0, 6000.0], [300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    target = _Aircraft(
        "red_0", "Red", [5000.0, 0.0, 6000.0], [-300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    shooter.num_left_missiles = 999
    env.blue_planes = {"blue_0": shooter}
    env.red_planes = {"red_0": target}
    env.agent_ids = ["blue_0"]
    env._lock_timer = {"blue_0": 0}
    env._lock_target = {"blue_0": None}
    env._missile_cooldown = {"blue_0": 0}
    env._fire_control_states = {"blue_0": FireControlState()}
    env._fire_control_assignments = {"blue_0": "red_0"}
    env._target_assignment_diagnostics = Counter()
    env._engaged_targets = set()
    env._launch_diag_step = make_empty_launch_diag()
    env._agents_deny_kill = set()
    visible = {"value": True}
    ta = {"value": 0.0}
    monkeypatch.setattr(
        env, "_is_detected_by_electro_optical",
        lambda *_: visible["value"])
    monkeypatch.setattr("my_uav_env.env.compute_3d_range", lambda *_: 5000.0)
    monkeypatch.setattr("my_uav_env.env.compute_body_x_q_los", lambda *_: 0.0)
    monkeypatch.setattr(
        "my_uav_env.env.get2d_heading_AO_TA_R",
        lambda *_: (0.0, ta["value"], 5000.0))
    launches = []
    def launch(parent, enemy, _quality):
        launches.append((parent.uid, enemy.uid))
        env._missile_cooldown[parent.uid] = env.missile_cooldown_frames
    monkeypatch.setattr(env, "_launch_missile", launch)

    for _ in range(14):
        env._check_missile_launch()
    assert launches == []
    assert env._lock_timer["blue_0"] == 14
    env._check_missile_launch()
    assert launches == []
    assert env._lock_timer["blue_0"] == 15
    assert env._fire_control_states["blue_0"].lock_mature is True

    ta["value"] = np.pi
    env._check_missile_launch()
    assert len(launches) == 1
    assert env._lock_timer["blue_0"] == 16
    env._engaged_targets.clear()
    for _ in range(env.missile_cooldown_frames - 1):
        env._check_missile_launch()
    assert len(launches) == 1
    env._check_missile_launch()
    assert len(launches) == 2

    visible["value"] = False
    env._check_missile_launch()
    assert env._lock_target["blue_0"] is None
    assert env._lock_timer["blue_0"] == 0


def test_learnable_same_frame_live_missile_deconfliction(monkeypatch):
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    shooters = {
        f"blue_{i}": _Aircraft(
            f"blue_{i}", "Blue", [0.0, float(i), 6000.0],
            [300.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for i in range(2)}
    target = _Aircraft(
        "red_0", "Red", [5000.0, 0.0, 6000.0], [-300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    for shooter in shooters.values():
        shooter.num_left_missiles = 999
    env.blue_planes = shooters
    env.red_planes = {"red_0": target}
    env.agent_ids = list(shooters)
    env._lock_timer = {aid: 14 for aid in shooters}
    env._lock_target = {aid: "red_0" for aid in shooters}
    env._missile_cooldown = {aid: 0 for aid in shooters}
    env._fire_control_states = {aid: FireControlState() for aid in shooters}
    env._fire_control_assignments = {aid: "red_0" for aid in shooters}
    env._target_assignment_diagnostics = Counter()
    env._engaged_targets = {"blue_0"}  # A red missile must not block blue fire.
    env._launch_diag_step = make_empty_launch_diag()
    env._agents_deny_kill = set()
    monkeypatch.setattr(env, "_is_detected_by_electro_optical", lambda *_: True)
    monkeypatch.setattr("my_uav_env.env.compute_3d_range", lambda *_: 5000.0)
    monkeypatch.setattr("my_uav_env.env.compute_body_x_q_los", lambda *_: 0.0)
    monkeypatch.setattr(
        "my_uav_env.env.get2d_heading_AO_TA_R",
        lambda *_: (0.0, np.pi, 5000.0))
    launches = []
    monkeypatch.setattr(
        env, "_launch_missile",
        lambda parent, enemy, _quality: launches.append(
            (parent.uid, enemy.uid)))
    env._check_missile_launch()
    assert len(launches) == 1
    assert launches[0][1] == "red_0"
    assert env._engaged_targets == {"blue_0", "red_0"}

    second_target = _Aircraft(
        "red_1", "Red", [5000.0, 10.0, 6000.0], [-300.0, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    env.red_planes["red_1"] = second_target
    env._fire_control_assignments = {
        "blue_0": "red_0", "blue_1": "red_1"}
    env._lock_timer = {aid: 14 for aid in shooters}
    env._lock_target = {
        "blue_0": "red_0", "blue_1": "red_1"}
    env._missile_cooldown = {aid: 0 for aid in shooters}
    env._engaged_targets.clear()
    launches.clear()
    env._check_missile_launch()
    assert sorted(target_id for _, target_id in launches) == ["red_0", "red_1"]


def test_environment_assignment_waits_when_all_live_targets_engaged():
    env = UavCombatEnv(
        max_num_red=3, max_num_blue=3, max_steps=1400,
        environment_profile=PAPER_LEARNABLE_ENVIRONMENT_PROFILE)
    targets = {
        f"red_{i}": _Aircraft(
            f"red_{i}", "Red", [3000.0 + i, 0.0, 6000.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for i in range(3)}
    targets["red_0"].is_alive = False
    env._fire_control_assignments = {"blue_0": "red_0"}
    env._fire_control_pending_reallocation_after_death = set()
    env._target_assignment_diagnostics = Counter()
    env._engaged_targets = {"red_1", "red_2"}
    assert env._learnable_fire_control_target("blue_0", targets) is None
    assert env._fire_control_assignments["blue_0"] == "red_0"
    assert env._target_assignment_diagnostics["engaged_wait_frames"] == 1
    env._engaged_targets = {"red_2"}
    replacement = env._learnable_fire_control_target("blue_0", targets)
    assert replacement.uid == "red_1"
    assert env._fire_control_assignments["blue_0"] == "red_1"
    assert env._target_assignment_diagnostics[
        "target_reallocations_after_death"] == 1


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


def test_squashed_actor_probability_semantics_and_finite_gradients():
    torch.manual_seed(7)
    actor = VanillaActor(obs_dim=60, hidden=16, rnn_hidden=8)
    obs = torch.randn(4096, 60)
    hidden = torch.zeros(4096, 8)
    distribution, _ = actor(obs, hidden)
    actions = distribution.rsample()
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    old_log_prob = distribution.log_prob(actions).sum(dim=-1)
    recomputed, _ = actor(obs, hidden)
    new_log_prob = recomputed.log_prob(actions).sum(dim=-1)
    assert torch.allclose(old_log_prob, new_log_prob, atol=2e-5, rtol=2e-5)
    ratio = torch.exp(new_log_prob - old_log_prob)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=2e-5, rtol=2e-5)
    assert torch.allclose(distribution.mode, torch.tanh(distribution.loc))
    boundary_distribution, _ = actor(obs[:4], hidden[:4])
    boundary = torch.full_like(actions[:4], 1.0 - 1e-9)
    assert torch.all(torch.isfinite(boundary_distribution.log_prob(boundary)))
    loss = -(old_log_prob.mean() + 0.05 * distribution.base_entropy().mean())
    loss.backward()
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in actor.parameters())


def test_rollout_buffer_stores_executed_squashed_action_exactly():
    actor = VanillaActor(obs_dim=60, hidden=8, rnn_hidden=4)
    obs = torch.zeros(1, 60)
    hidden = torch.zeros(1, 4)
    distribution, _ = actor(obs, hidden)
    action = distribution.sample()[0].detach().numpy()
    buffer = RolloutBuffer(1, 1, 1, 3, 4)
    buffer.store_step(
        0, 0, 0, np.zeros(60, dtype=np.float32), action,
        float(distribution.log_prob(
            torch.as_tensor(action)[None]).sum().detach()),
        True, np.zeros(4, dtype=np.float32), True,
        policy_mean_action=distribution.mode[0].detach().numpy())
    env_action = action
    assert np.array_equal(buffer.actions[0, 0, 0], env_action)


def test_first_event_and_configured_hit_radius_metrics():
    events = [
        {"red_first_launch_step": 4, "blue_first_launch_step": None,
         "red_first_hit_step": 9, "blue_first_hit_step": None},
        {"red_first_launch_step": 8, "blue_first_launch_step": 6,
         "red_first_hit_step": None, "blue_first_hit_step": 12},
    ]
    metrics = _learnability_iteration_metrics(
        [{"team": "red", "range_m": 200.0},
         {"team": "red", "range_m": 300.0}], [],
        Counter(), [20, 30], events, 250.0, 2, 1.0, 10, 100, 7, 0)
    assert metrics["RedLaunchInsideHitRadiusCount"] == 1
    assert metrics["RedFirstLaunchStepMean"] == pytest.approx(6.0)
    assert metrics["BlueFirstLaunchStepMean"] == pytest.approx(6.0)
    assert metrics["RedFirstHitStepMean"] == pytest.approx(9.0)
    assert metrics["BlueFirstHitStepMean"] == pytest.approx(12.0)
    assert metrics["WorkerRestartCount"] == 7


def test_worker_restart_transition_is_explicit_and_counted(monkeypatch):
    class _BrokenRemote:
        def send(self, _message):
            raise BrokenPipeError

    vec = SubprocVecEnv.__new__(SubprocVecEnv)
    vec.remotes = [_BrokenRemote()]
    vec.processes = []
    vec._dead_workers = set()
    vec._env_kwargs = {}
    vec.worker_restart_count = 4
    new_obs = {"red_0": _strict_obs(), "blue_0": _strict_obs()}
    monkeypatch.setattr(vec, "_restart_worker", lambda *_: new_obs)
    obs, rewards, dones, infos = vec.step([{}])
    assert vec.worker_restart_count == 5
    assert all(dones[0].values())
    assert _episode_is_invalid(infos[0])
    assert infos[0]["__episode__"]["worker_restart_episode"] is True
    assert infos[0]["__episode__"]["invalid_numerical_reasons"] == [
        "WorkerRestart"]


def test_checkpoint_schema_rejects_v5_action_distribution(tmp_path):
    config = _learnable_training_config(tmp_path)
    metadata = _checkpoint_metadata(config, 60, 30)
    assert metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert metadata["action_distribution"] == ACTION_DISTRIBUTION_VERSION
    assert metadata["entropy_estimator"] == ENTROPY_ESTIMATOR_VERSION
    actor = VanillaActor(obs_dim=60, hidden=8, rnn_hidden=4)
    critic = CentralizedCritic(global_obs_dim=30, hidden=8)
    actor_opt = torch.optim.Adam(actor.parameters())
    critic_opt = torch.optim.Adam(critic.parameters())
    payload = _build_training_state(
        actor, critic, actor_opt, critic_opt, config, metadata,
        {"run_id": "test", "total_steps": 0})
    payload["checkpoint_metadata"]["schema_version"] = (
        "vanilla_mappo_paper_env_v5")
    payload["checkpoint_metadata"]["action_distribution"] = (
        "state_dependent_diag_gaussian_env_clip_v2")
    with pytest.raises(ValueError, match="checkpoint schema mismatch"):
        _validate_training_state(payload, config, metadata)


def test_mirror_audit_heading_error_is_circular_and_has_no_launch_gate():
    assert np.rad2deg(_circular_error(
        np.deg2rad(179.0), np.deg2rad(-179.0))) == pytest.approx(2.0)
    source = open(
        "scripts/audit_paper_learnable_3v3.py", encoding="utf-8").read()
    assert "_mirror_launch_health" not in source
def test_fire_control_state_uses_paper_diagnostic_field_names():
    state = FireControlState()
    state.current_target_id = "blue_1"
    state.continuous_detection_frames = 15
    state.cooldown_frames_remaining = 30
    snapshot = state.snapshot()
    assert snapshot["tracked_target_id"] == "blue_1"
    assert snapshot["continuous_eo_detection_frames"] == 15
    assert snapshot["launch_cooldown_frames"] == 30

