from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv  # noqa: E402


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_brma_tam_scripted_composite_v1.yaml"
)


class FakeSim:
    def __init__(self, uid, *, pos, vel, alive=True, warning=False):
        self.uid = uid
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)
        self.is_alive = bool(alive)
        self._warning = bool(warning)
        self.under_missiles = []

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return np.zeros(3, dtype=np.float64)

    def get_geodetic(self):
        return np.asarray([0.0, 0.0, self._pos[2]], dtype=np.float64)

    def check_missile_warning(self):
        return object() if self._warning else None


class FakeMissile(FakeSim):
    def __init__(self, uid, *, pos, vel, target, alive=True):
        super().__init__(uid, pos=pos, vel=vel, alive=alive)
        self.target_aircraft = target
        self._target_id = target.uid if target is not None else ""


def _cfg():
    data = yaml.safe_load((ROOT / CFG_3V2).read_text(encoding="utf-8"))
    return data["brma_tam_scripted_composite_v1"]


def _bare_env():
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = ["blue_0", "blue_1"]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {
        "red_0": "mav",
        "red_1": "attack_uav",
        "red_2": "attack_uav",
        "blue_0": "attack_uav",
        "blue_1": "attack_uav",
    }
    env.red_planes = {
        "red_0": FakeSim("red_0", pos=(0.0, 0.0, 6500.0), vel=(250.0, 0.0, 0.0)),
        "red_1": FakeSim("red_1", pos=(0.0, 0.0, 6000.0), vel=(300.0, 0.0, 0.0)),
        "red_2": FakeSim("red_2", pos=(0.0, 4000.0, 6000.0), vel=(300.0, 0.0, 0.0)),
    }
    env.blue_planes = {
        "blue_0": FakeSim("blue_0", pos=(4000.0, 0.0, 6000.0), vel=(230.0, 0.0, 0.0)),
        "blue_1": FakeSim("blue_1", pos=(12000.0, 0.0, 7000.0), vel=(230.0, 0.0, 0.0)),
    }
    env._last_step_obs = {
        "red_0": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
        "red_1": {"enemy_track_source": np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)},
        "red_2": {"enemy_track_source": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
    }
    env._step_kill_count = {aid: 0 for aid in env.agent_ids}
    env._death_reasons = {}
    env._evasion_step_records = []
    env._launch_quality_step_records = []
    env._reward_target_diagnostic_records = []
    env.brma_tam_scripted_composite_v1_config = _cfg()
    env.BATTLEFIELD_HALF_SIZE = 40000.0
    env.BATTLEFIELD_ALTITUDE_MAX = 10000.0
    env.MISSILE_LAUNCH_MIN_RANGE = 500.0
    env.MISSILE_LAUNCH_RANGE_THRESH = 14000.0
    env.mav_observation_range_m = 80000.0
    env.uav_direct_observation_range_m = 10000.0
    env._missiles_in_flight = {}
    env._brma_tam_scripted_reset_episode_state()
    env._brma_tam_alive_before_step = {aid: True for aid in env.agent_ids}
    return env


def _base_components():
    rewards = {"red_0": 999.0, "red_1": 999.0, "red_2": 999.0}
    components = {
        rid: {
            "r_pitch": 1.0,
            "r_roll": 2.0,
            "r_vel": 3.0,
            "r_alt": 100.0,
            "r_bound": 200.0,
            "r_adv": 300.0,
            "r_end": 400.0,
            "r_death": -500.0,
            "total": 999.0,
        }
        for rid in rewards
    }
    return rewards, components


def test_composite_mode_configs_load_and_contract():
    from uav_env import make_env

    for cfg in [CFG_3V2, CFG_5V4]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "brma_tam_scripted_composite_v1"
            assert env.observation_mode == "mav_shared_geo"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.red_target_selection_mode == "closest"
            assert env.missile_evasion_mode == "brma_scripted"
            assert env.missile_evasion_teams in {"red_only", "both"}
        finally:
            env.close()


def test_config_contract_rejects_disabled_red_scripted_evasion():
    with pytest.raises(ValueError, match="red evasion is scripted"):
        HeteroUavCombatEnv(
            hetero_reward_mode="brma_tam_scripted_composite_v1",
            brma_tam_scripted_composite_v1=_cfg(),
            observation_mode="mav_shared_geo",
            missile_evasion={"mode": "none", "teams": "none"},
            max_num_red=3,
            max_num_blue=2,
            max_steps=5,
        )


def test_3d_angle_contract_cases():
    h = HeteroUavCombatEnv
    red = FakeSim("r", pos=(0.0, 0.0, 0.0), vel=(300.0, 0.0, 0.0))
    blue = FakeSim("b", pos=(1000.0, 0.0, 0.0), vel=(300.0, 0.0, 0.0))
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_ata_rad"] == pytest.approx(0.0, abs=1e-6)
    assert g["tam_aa_rad"] == pytest.approx(0.0, abs=1e-6)
    assert g["tam_angle_raw"] == pytest.approx(1.0, abs=1e-6)

    blue._vel = np.asarray([-300.0, 0.0, 0.0])
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_angle_raw"] == pytest.approx(0.0, abs=1e-6)

    blue._pos = np.asarray([-1000.0, 0.0, 0.0])
    blue._vel = np.asarray([300.0, 0.0, 0.0])
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_angle_raw"] == pytest.approx(-1.0, abs=1e-6)

    blue._pos = np.asarray([1000.0, 0.0, 1000.0])
    blue._vel = np.asarray([300.0, 0.0, 0.0])
    g3 = h._brma_tam_3d_geometry(red, blue)
    assert g3["target_distance_m"] == pytest.approx(math.sqrt(2) * 1000.0)
    assert g3["tam_ata_rad"] > 0.0

    red._vel = np.zeros(3)
    g = h._brma_tam_3d_geometry(red, blue)
    assert g["tam_geometry_valid"] == 0.0
    assert np.isfinite(g["tam_angle_raw"])


def test_speed_and_distance_formulas():
    h = HeteroUavCombatEnv
    assert h._brma_tam_speed_raw(300.0, 300.0)["tam_speed_raw"] == pytest.approx(0.0)
    assert h._brma_tam_speed_raw(300.0, 120.0)["tam_speed_raw"] == pytest.approx(1.0)
    assert h._brma_tam_speed_raw(300.0, 450.0)["tam_speed_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_speed_raw(300.0, 600.0)["tam_speed_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_speed_raw(0.0, 200.0)["speed_ratio_valid"] == pytest.approx(0.0)
    assert h._brma_tam_speed_raw(0.0, 200.0)["tam_speed_raw"] == pytest.approx(0.0)

    assert h._brma_tam_distance_raw(5000.0)["tam_distance_raw"] == pytest.approx(1.0)
    assert 0.0 < h._brma_tam_distance_raw(7500.0)["tam_distance_raw"] < 1.0
    assert h._brma_tam_distance_raw(10000.0)["tam_distance_raw"] == pytest.approx(-1.0)
    assert h._brma_tam_distance_raw(float("nan"))["reward_distance_zone_code"] == pytest.approx(-1.0)


def test_uav_total_whitelists_terms_and_excludes_dodge():
    env = _bare_env()
    env._step_kill_count["red_1"] = 1
    rewards, components = _base_components()

    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    comp = components["red_1"]

    expected = (
        1.0 + 2.0 + 3.0
        + 10.0 * comp["tam_speed_raw"]
        + 15.0 * comp["tam_angle_raw"]
        + 10.0 * comp["tam_distance_raw"]
        + comp["uav_event_kill"]
        + comp["uav_event_loss"]
    )
    assert rewards["red_1"] == pytest.approx(expected)
    assert comp["brma_alt_log_only"] == pytest.approx(100.0)
    assert comp["brma_adv_log_only"] == pytest.approx(300.0)
    assert comp["tam_dodge_raw_log"] == pytest.approx(0.0)
    assert comp["uav_total"] == pytest.approx(expected)


def test_dead_before_step_gets_zero_but_same_step_death_gets_loss():
    env = _bare_env()
    env.red_planes["red_1"].is_alive = False
    env._brma_tam_alive_before_step["red_1"] = False
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert rewards["red_1"] == pytest.approx(0.0)

    env = _bare_env()
    env.red_planes["red_1"].is_alive = False
    env._brma_tam_alive_before_step["red_1"] = True
    env._death_reasons["red_1"] = "Missile_Kill"
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert components["red_1"]["uav_event_loss"] == pytest.approx(-200.0)


def test_mav_team_credit_not_awarded_after_previous_death():
    env = _bare_env()
    env.red_planes["red_0"].is_alive = False
    env._brma_tam_alive_before_step["red_0"] = False
    env._step_kill_count["red_1"] = 1
    rewards, components = _base_components()
    rewards, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert rewards["red_0"] == pytest.approx(0.0)
    assert components["red_0"]["mav_team_credit_delta"] == pytest.approx(0.0)


def test_mav_threat_uses_actual_incoming_missile():
    env = _bare_env()
    missile = FakeMissile(
        "m1", pos=(1000.0, 0.0, 6500.0), vel=(-600.0, 0.0, 0.0),
        target=env.red_planes["red_0"],
    )
    env.red_planes["red_0"].under_missiles = [missile]
    rewards, components = _base_components()
    _, components = env._compute_brma_tam_scripted_composite_v1(rewards, components)
    assert components["red_0"]["mav_threat_raw"] == pytest.approx(-1.0)
    assert components["red_0"]["mav_actual_incoming_missile_count"] == pytest.approx(1.0)


def test_contract_accepts_red_only_and_both_and_rejects_invalid_variants():
    env = _bare_env()
    env.observation_mode = "mav_shared_geo"
    env.red_target_selection_mode = "closest"
    env.aircraft_type_params = {"mav": {"num_missiles": 0}}
    for teams in ("red_only", "both"):
        env.missile_evasion_config = {"mode": "brma_scripted", "teams": teams}
        env._validate_brma_tam_scripted_composite_v1_contract()
    for mode, teams in (("none", "red_only"), ("brma_scripted", "none"), ("brma_scripted", "blue_only")):
        env.missile_evasion_config = {"mode": mode, "teams": teams}
        with pytest.raises(ValueError, match="red evasion is scripted"):
            env._validate_brma_tam_scripted_composite_v1_contract()
    env.missile_evasion_config = {"mode": "brma_scripted", "teams": "red_only"}
    env.observation_mode = "brma_sensor"
    with pytest.raises(ValueError, match="mav_shared_geo"):
        env._validate_brma_tam_scripted_composite_v1_contract()
    env.observation_mode = "mav_shared_geo"
    env.red_target_selection_mode = "mav_threat_rank"
    with pytest.raises(ValueError, match="red_target_selection_mode"):
        env._validate_brma_tam_scripted_composite_v1_contract()
    env.red_target_selection_mode = "closest"
    env.aircraft_type_params["mav"]["num_missiles"] = 1
    with pytest.raises(ValueError, match="num_missiles"):
        env._validate_brma_tam_scripted_composite_v1_contract()


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, -1.0), (4000.0, -0.5), (7999.999, -0.000000125),
     (8000.0, 1.0), (16500.0, 0.5), (25000.0, -0.5)],
)
def test_mav_position_piecewise_exact(distance, expected):
    assert HeteroUavCombatEnv._brma_tam_mav_pos_raw(distance, 8000.0, 25000.0) == pytest.approx(expected)


@pytest.mark.parametrize("distance", [7999.0, 8000.0, 14999.0, 15000.0])
def test_mav_distance_piecewise_boundaries(distance):
    expected = (
        -(1.0 - distance / 8000.0) if distance < 8000.0
        else -0.5 * (1.0 - (distance - 8000.0) / 7000.0) if distance < 15000.0
        else 0.2
    )
    assert HeteroUavCombatEnv._brma_tam_mav_dist_raw(distance, 8000.0, 15000.0) == pytest.approx(expected)


def test_mav_distance_is_zero_without_alive_blue():
    env = _bare_env()
    for blue in env.blue_planes.values():
        blue.is_alive = False
    _, logs = env._brma_tam_mav_safety(env.red_planes["red_0"], _cfg())
    assert logs["mav_dist_raw"] == 0.0


def test_mav_position_center_requires_alive_blue_and_uav_and_excludes_mav_altitude():
    env = _bare_env()
    mav = env.red_planes["red_0"]
    _, logs = env._brma_tam_mav_support(mav, _cfg())
    expected_center = np.mean([
        env.red_planes["red_1"].get_position()[:2],
        env.red_planes["red_2"].get_position()[:2],
        env.blue_planes["blue_0"].get_position()[:2],
        env.blue_planes["blue_1"].get_position()[:2],
    ], axis=0)
    assert logs["battlefield_center_valid"] == 1.0
    assert logs["battlefield_center_x"] == pytest.approx(expected_center[0])
    assert logs["battlefield_center_y"] == pytest.approx(expected_center[1])
    mav._pos[2] = 50000.0
    _, high_logs = env._brma_tam_mav_support(mav, _cfg())
    assert high_logs["mav_pos_raw"] == pytest.approx(logs["mav_pos_raw"])
    env.blue_planes["blue_0"].is_alive = env.blue_planes["blue_1"].is_alive = False
    _, no_blue = env._brma_tam_mav_support(mav, _cfg())
    assert no_blue["battlefield_center_valid"] == 0.0
    assert no_blue["mav_pos_raw"] == 0.0
    env = _bare_env()
    env.red_planes["red_1"].is_alive = env.red_planes["red_2"].is_alive = False
    _, no_uav = env._brma_tam_mav_support(env.red_planes["red_0"], _cfg())
    assert no_uav["battlefield_center_valid"] == 0.0
    assert no_uav["mav_pos_raw"] == 0.0


def test_current_visibility_helper_direct_shared_and_mav_death():
    env = _bare_env()
    direct = env._mav_shared_track_state("red_1", "blue_0")
    assert direct["direct_visible"] and direct["mav_shared_visible"]
    env.red_planes["red_1"]._pos = np.asarray((-30000.0, 0.0, 6000.0))
    shared = env._mav_shared_track_state("red_1", "blue_0")
    assert not shared["direct_visible"] and shared["mav_shared_visible"]
    env.red_planes["red_0"].is_alive = False
    hidden = env._mav_shared_track_state("red_1", "blue_0")
    assert not hidden["observed"]
    env = _bare_env()
    env.blue_planes["blue_0"]._pos = np.asarray((79900.0, 0.0, 6000.0))
    assert env._mav_shared_track_state("red_0", "blue_0")["observed"]
    env.blue_planes["blue_0"]._pos = np.asarray((80100.0, 0.0, 6000.0))
    assert not env._mav_shared_track_state("red_0", "blue_0")["observed"]


def test_mav_awareness_is_3d_sum_with_per_blue_mean_log():
    env = _bare_env()
    mav = env.red_planes["red_0"]
    env.blue_planes["blue_0"]._pos = np.asarray((1000.0, 0.0, 6500.0))
    env.blue_planes["blue_1"]._pos = np.asarray((2000.0, 0.0, 6500.0))
    _, logs = env._brma_tam_mav_support(mav, _cfg())
    assert logs["mav_aware_raw_sum"] == pytest.approx(0.6)
    assert logs["mav_aware_per_blue_mean"] == pytest.approx(0.3)
    env.blue_planes["blue_0"]._pos = np.asarray((0.0, 1000.0, 6500.0))
    env.blue_planes["blue_1"]._pos = np.asarray((-1000.0, 0.0, 6500.0))
    _, side_back = env._brma_tam_mav_support(mav, _cfg())
    assert side_back["mav_aware_raw_sum"] == pytest.approx(0.0, abs=1e-8)


def test_threat_filters_unrelated_and_dead_missiles_and_prelaunch_is_log_only(monkeypatch):
    env = _bare_env()
    mav = env.red_planes["red_0"]
    other = env.red_planes["red_1"]
    unrelated = FakeMissile("m-other", pos=(1, 0, 0), vel=(600, 0, 0), target=other)
    dead = FakeMissile("m-dead", pos=(1, 0, 0), vel=(600, 0, 0), target=mav, alive=False)
    mav.under_missiles = [unrelated, dead]
    monkeypatch.setattr(env, "_missile_candidate_metrics", lambda *_: {"launch_geometry_ok_3d": True})
    _, logs = env._brma_tam_mav_safety(mav, _cfg())
    assert logs["mav_threat_raw"] == 0.0
    assert logs["mav_prelaunch_geometry_threat_count_log"] == 2.0
    assert logs["mav_prelaunch_geometry_threat_log"] == -2.0


def test_dodge_diagnostic_real_geometry_cache_and_reset():
    env = _bare_env()
    aircraft = env.red_planes["red_1"]
    missile = FakeMissile("m1", pos=(-1000.0, 0.0, 6000.0), vel=(600.0, 0.0, 0.0), target=aircraft)
    env._missiles_in_flight = {"m1": missile}
    env._evasion_step_records = [{"evasion_agent_id": "red_1", "incoming_missile_id": "m1"}]
    first, selected = env._brma_tam_dodge_diagnostic("red_1", aircraft)
    assert selected == "m1"
    assert first["tam_dodge_geometry_valid"] == 1.0
    assert first["tam_dodge_angle_log"] == pytest.approx(-1.0)
    assert first["tam_dodge_speed_log"] == 0.0
    missile._vel = np.asarray((500.0, 0.0, 0.0))
    second, _ = env._brma_tam_dodge_diagnostic("red_1", aircraft)
    assert second["tam_dodge_speed_log"] == pytest.approx(0.1)
    assert second["tam_dodge_raw_log"] == pytest.approx(
        second["tam_dodge_angle_log"] + second["tam_dodge_speed_log"]
    )
    env._brma_tam_scripted_reset_episode_state()
    assert env._brma_tam_missile_speed_cache == {}
    env._evasion_step_records = [{"evasion_agent_id": "red_1", "incoming_missile_id": "missing"}]
    missing, _ = env._brma_tam_dodge_diagnostic("red_1", aircraft)
    assert missing["tam_dodge_geometry_valid"] == 0.0
    assert missing["tam_dodge_missing_reason"] == "missile_not_found"


def test_reward_target_tie_break_switch_and_current_step_diagnostics():
    env = _bare_env()
    sim = env.red_planes["red_1"]
    env.blue_planes["blue_0"]._pos = np.asarray((1000.0, 0.0, 6000.0))
    env.blue_planes["blue_1"]._pos = np.asarray((-1000.0, 0.0, 6000.0))
    assert env._brma_tam_closest_alive_blue(sim)[0] == "blue_0"
    env._last_step_obs = {"red_1": {"enemy_track_source": np.zeros((2, 2))}}
    env._launch_quality_step_records = [{
        "shooter_id": "red_1", "target_id": "blue_0",
        "lock_target_id_at_launch": "blue_0", "lock_timer_frames_at_launch": 15,
    }]
    rewards, components = _base_components()
    env._compute_brma_tam_scripted_composite_v1(rewards, components)
    diag = next(row for row in env._reward_target_diagnostic_records if row["agent_id"] == "red_1")
    assert diag["launch_target_id"] == "blue_0"
    assert diag["launch_target_ids"] == "blue_0"
    assert diag["lock_target_id"] == "blue_0"
    assert diag["reward_target_matches_lock"] == 1.0
    assert diag["reward_target_matches_launch"] == 1.0
    assert diag["reward_target_observed"] == 1.0


def test_lifecycle_oob_is_once_and_death_takes_precedence():
    env = _bare_env()
    sim = env.red_planes["red_1"]
    sim._pos[0] = 50000.0
    value, logs = env._brma_tam_uav_event("red_1", sim, _cfg())
    assert value == pytest.approx(-100.0)
    assert env._brma_tam_uav_event("red_1", sim, _cfg())[0] == 0.0
    sim._pos[0] = 0.0
    env._brma_tam_uav_event("red_1", sim, _cfg())
    sim._pos[0] = 50000.0
    assert env._brma_tam_uav_event("red_1", sim, _cfg())[0] == 0.0

    env = _bare_env()
    sim = env.red_planes["red_1"]
    sim._pos[0] = 50000.0
    sim.is_alive = False
    value, logs = env._brma_tam_uav_event("red_1", sim, _cfg())
    assert value == pytest.approx(-200.0)
    assert logs["uav_event_first_horizontal_out_of_zone"] == 0.0


@pytest.mark.parametrize("config", [CFG_3V2, CFG_5V4])
def test_real_env_short_rollout_is_finite(config):
    from uav_env import make_env

    env = make_env(config, max_steps=20)
    try:
        obs, _ = env.reset(seed=17)
        for _ in range(20):
            obs, rewards, terminated, truncated, info = env.step(
                {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
            )
            assert all(np.isfinite(float(value)) for value in rewards.values())
            for comp in info.get("reward_components", {}).values():
                assert all(
                    not isinstance(value, (int, float, np.floating)) or np.isfinite(float(value))
                    for value in comp.values()
                )
            if all(terminated.values()) or all(truncated.values()):
                break
    finally:
        env.close()


def test_revision2_logging_schema_keeps_string_diagnostics_and_height_aggregates(tmp_path):
    import csv
    from scripts.experiment_logging_schema import FILE_SCHEMAS
    from scripts.rich_logging import RichExperimentLogger

    episode_schema = FILE_SCHEMAS["episode_reward_components.csv"]
    assert "max_altitude_m" in episode_schema
    assert "above_altitude_max_episode_flag" in episode_schema
    assert "max_altitude_m_sum" not in episode_schema
    assert "above_altitude_max_episode_flag_sum" not in episode_schema
    assert "lock_target_id_at_launch" in FILE_SCHEMAS["missile_events.csv"]

    logger = RichExperimentLogger(
        tmp_path, run_id="test", method_name="test", scenario_name="test",
        device="cpu", num_envs=1, rollout_length_per_env=1,
        transitions_per_rollout=1, mode="full",
    )
    logger.write_reward_target_diagnostics(
        {"__reward_target_diagnostics__": [{
            "agent_id": "red_1", "reward_target_id": "blue_0",
            "launch_target_ids": "blue_0|blue_1", "action_source": "scripted_evasion",
            "tam_dodge_missing_reason": "missile_not_found",
        }]},
        scenario="test", episode_id=2, step=3,
    )
    logger.close()
    rows = list(csv.DictReader((tmp_path / "reward_target_diagnostics.csv").open(encoding="utf-8")))
    assert rows[0]["launch_target_ids"] == "blue_0|blue_1"
    assert rows[0]["action_source"] == "scripted_evasion"
    assert rows[0]["tam_dodge_missing_reason"] == "missile_not_found"
