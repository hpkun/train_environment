import copy

from my_uav_env import UavCombatEnv
from my_uav_env.fire_control import FireControlState


class CleanupMissile:
    def __init__(self, uid="m0", parent_id="red_0", target_id="blue_0"):
        self.uid = uid
        self._parent_id = parent_id
        self._target_id = target_id
        self._t = 2.5
        self.is_alive = True
        self.detached = False
        self.closed = False

    def detach_references(self):
        self.detached = True

    def close(self):
        self.closed = True


def prepared_env():
    env = UavCombatEnv(max_num_red=3, max_num_blue=3)
    env._lock_timer = {aid: 0 for aid in env.agent_ids}
    env._lock_target = {aid: None for aid in env.agent_ids}
    env._fire_control_states = {aid: FireControlState() for aid in env.agent_ids}
    env._missile_term_reasons = {"red": {}, "blue": {}}
    env._launch_quality_records = {}
    env._launch_quality_step_records = []
    env._launch_quality_done_step_records = []
    env._engaged_targets = set()
    env._terminal_cleanup_done = False
    return env


def test_live_missile_is_censored_without_physical_termination():
    env = prepared_env()
    missile = CleanupMissile()
    env._missiles_in_flight = {missile.uid: missile}
    env._engaged_targets = {"blue_0"}
    env._launch_quality_records[missile.uid] = {"team": "red"}
    before_terms = copy.deepcopy(env._missile_term_reasons)
    frozen = env._freeze_terminal_info({"__missile_term__": before_terms})
    cleanup = env._clear_terminal_combat_state()
    assert frozen["__terminal_cleanup__"]["missiles_in_flight_at_episode_end"] == 1
    assert len(frozen["__terminal_cleanup__"]["censored_launch_records"]) == 1
    assert env._missile_term_reasons == before_terms
    assert cleanup["red_episode_end_cleanup_count"] == 1
    assert cleanup["missiles_remaining_after_cleanup"] == 0
    assert missile.detached and missile.closed


def test_cleanup_does_not_duplicate_natural_target_dead_or_hit():
    env = prepared_env()
    env._missile_term_reasons = {
        "red": {"hit": 1, "target_dead": 1}, "blue": {"overshoot": 1}}
    before = copy.deepcopy(env._missile_term_reasons)
    first = env._clear_terminal_combat_state()
    second = env._clear_terminal_combat_state()
    assert env._missile_term_reasons == before
    assert first["red_episode_end_cleanup_count"] == 0
    assert second["red_episode_end_cleanup_count"] == 0


def test_final_fire_control_lock_and_engaged_snapshot_survives_cleanup():
    env = prepared_env()
    env._fire_control_states["red_0"] = FireControlState(
        current_target_id="blue_0", detection_state="tracking",
        continuous_detection_frames=14, lock_mature=False,
        cooldown_frames_remaining=3, blocked_reason="cooldown",
        transition_reason="continuous_detection")
    env._lock_target["red_0"] = "blue_0"
    env._lock_timer["red_0"] = 14
    env._engaged_targets = {"blue_1"}
    frozen = env._freeze_terminal_info({})
    env._clear_terminal_combat_state()
    terminal = frozen["__terminal_cleanup__"]
    assert terminal["final_fire_control_snapshot"]["red_0"][
        "continuous_detection_frames"] == 14
    assert terminal["final_lock_snapshot"]["red_0"]["target_id"] == "blue_0"
    assert terminal["engaged_targets_at_episode_end"] == ["blue_1"]
    assert env._fire_control_states["red_0"].continuous_detection_frames == 0
    assert env._lock_target["red_0"] is None
    assert not env._engaged_targets


def test_cleanup_is_isolated_between_environment_instances():
    env_a = prepared_env()
    env_b = prepared_env()
    missile_a = CleanupMissile("a")
    missile_b = CleanupMissile("b", "blue_0", "red_0")
    env_a._missiles_in_flight = {"a": missile_a}
    env_b._missiles_in_flight = {"b": missile_b}
    env_b._engaged_targets = {"red_0"}
    env_a._clear_terminal_combat_state()
    assert "b" in env_b._missiles_in_flight
    assert env_b._engaged_targets == {"red_0"}
    assert not missile_b.detached


def test_empty_timeout_cleanup_has_zero_counts():
    env = prepared_env()
    frozen = env._freeze_terminal_info({"__missile_term__": {"red": {}, "blue": {}}})
    cleanup = env._clear_terminal_combat_state()
    assert frozen["__terminal_cleanup__"]["missiles_in_flight_at_episode_end"] == 0
    assert cleanup["red_episode_end_cleanup_count"] == 0
    assert cleanup["blue_episode_end_cleanup_count"] == 0
    assert cleanup["missiles_remaining_after_cleanup"] == 0
