from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
import uav_env.JSBSim.envs.hetero_uav_combat_env as hetero_env_mod


CFG_3V2 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_3v2_f16_mav_surrogate_happo_ref_v1_mav_support.yaml"
)
CFG_5V4 = (
    "uav_env/JSBSim/configs/"
    "hetero_mav_shared_geo_5v4_f16_mav_surrogate_happo_ref_v1_mav_support.yaml"
)


class _Sim:
    def __init__(self, alive=True, pos=(0.0, 0.0, 6000.0), vel=(250.0, 0.0, 0.0)):
        self.is_alive = alive
        self._pos = np.asarray(pos, dtype=np.float64)
        self._vel = np.asarray(vel, dtype=np.float64)

    def get_position(self):
        return self._pos

    def get_velocity(self):
        return self._vel

    def get_rpy(self):
        return np.zeros(3, dtype=np.float64)

    def get_geodetic(self):
        return np.asarray([0.0, 0.0, self._pos[2]], dtype=np.float64)

    def check_missile_warning(self):
        return None


def _bare_env():
    env = object.__new__(HeteroUavCombatEnv)
    env.red_ids = ["red_0", "red_1", "red_2"]
    env.blue_ids = ["blue_0", "blue_1"]
    env.agent_ids = env.red_ids + env.blue_ids
    env.agent_roles = {"red_0": "mav", "red_1": "attack_uav", "red_2": "attack_uav"}
    env.hetero_reward_mode = "happo_ref_v1_mav_support"
    env.happo_ref_v1_mav_support_config = {
        "scale": 0.1,
        "mav_safety": {
            "dist_weight": 0.5,
            "threat_weight": 0.3,
            "aspect_weight": 0.2,
            "d_danger_m": 8000.0,
            "d_safe_m": 15000.0,
        },
        "mav_support": {
            "pos_weight": 0.6,
            "aware_weight": 0.4,
            "pos_active": False,
            "d_opt_m": 8000.0,
            "d_max_m": 25000.0,
        },
        "mav_event": {
            "death_penalty": -4.0,
            "team_credit_per_kill": 0.5,
            "team_credit_cap": 1.0,
        },
    }
    env.red_planes = {
        "red_0": _Sim(True, (0.0, 0.0, 6500.0)),
        "red_1": _Sim(True, (1000.0, 0.0, 6000.0)),
        "red_2": _Sim(True, (2000.0, 0.0, 6000.0)),
    }
    env.blue_planes = {
        "blue_0": _Sim(True, (12000.0, 0.0, 6500.0), (-250.0, 0.0, 0.0)),
        "blue_1": _Sim(True, (20000.0, 0.0, 6500.0), (-250.0, 0.0, 0.0)),
    }
    env._last_step_obs = {
        "red_0": {
            "enemy_observed_mask": np.asarray([1.0, 1.0], dtype=np.float32),
        },
        "red_1": {
            "enemy_track_source": np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            "enemy_geo_states": np.zeros((2, 5), dtype=np.float32),
            "enemy_alive_mask": np.asarray([1.0, 1.0], dtype=np.float32),
            "altitude": np.asarray([6000.0], dtype=np.float32),
            "velocity": np.asarray([250.0, 0.0, 0.0], dtype=np.float32),
        },
        "red_2": {
            "enemy_track_source": np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            "enemy_geo_states": np.zeros((2, 5), dtype=np.float32),
            "enemy_alive_mask": np.asarray([1.0, 1.0], dtype=np.float32),
            "altitude": np.asarray([6000.0], dtype=np.float32),
            "velocity": np.asarray([250.0, 0.0, 0.0], dtype=np.float32),
        },
    }
    env._mav_death_penalized = False
    env._uav_death_penalized = set()
    env._missile_launch_counts = {}
    env._step_kill_count = {"red_1": 1, "red_2": 0}
    env._launch_quality_step_records = [
        {"shooter_id": "red_1", "shooter_role": "attack_uav", "launch_track_source": "mav_shared"},
    ]
    env._launch_quality_done_step_records = [
        {
            "shooter_id": "red_1",
            "shooter_role": "attack_uav",
            "launch_track_source": "mav_shared",
            "raw_termination_reason": "hit",
        },
    ]
    env._happo_v1_mav_death_penalized = False
    env._happo_v1_mav_team_credit_used = 0.0
    return env


def _base_rewards_components():
    rewards = {"red_0": 10.0, "red_1": 11.0, "red_2": 12.0}
    components = {}
    for idx, rid in enumerate(["red_0", "red_1", "red_2"]):
        components[rid] = {
            "r_pitch": 0.10 + idx,
            "r_roll": 0.20 + idx,
            "r_alt": 0.30 + idx,
            "r_bound": 0.40 + idx,
            "r_vel": 0.50 + idx,
            "r_adv": 6.0 + idx,
            "r_end": 1.0,
            "total": rewards[rid],
        }
    return rewards, components


def test_v1_configs_load_and_use_f16_mav_surrogate_with_f22_visual():
    from uav_env import make_env

    for cfg, max_red, max_blue in [(CFG_3V2, 3, 2), (CFG_5V4, 5, 4)]:
        env = make_env(cfg, max_steps=5)
        try:
            assert env.hetero_reward_mode == "happo_ref_v1_mav_support"
            assert env.observation_mode == "mav_shared_geo"
            assert env.max_num_red == max_red
            assert env.max_num_blue == max_blue
            assert env.red_agent_types[0] == "mav"
            assert env.aircraft_type_params["mav"]["aircraft_model"] == "f16"
            assert env.aircraft_type_params["mav"]["role"] == "mav"
            assert env.aircraft_type_params["mav"]["num_missiles"] == 0
            assert env.aircraft_type_params["attack_uav"]["aircraft_model"] == "f16"
            assert env.aircraft_type_params["attack_uav"]["num_missiles"] == 2
            assert env.red_target_selection_mode == "closest"
        finally:
            env.close()


def test_v1_config_has_required_mav_support_block_and_no_mav_threat_rank():
    for cfg in [CFG_3V2, CFG_5V4]:
        data = yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))
        assert data["hetero_reward_mode"] == "happo_ref_v1_mav_support"
        assert data["red_target_selection_mode"] == "closest"
        assert data["aircraft_type_params"]["mav"]["aircraft_model"] == "f16"
        assert data["acmi_visual_by_role"]["mav"] == "f22"
        block = data["happo_ref_v1_mav_support"]
        assert "scale" in block
        assert block["mav_support"]["pos_active"] is False
        assert set(block) >= {"mav_safety", "mav_support", "mav_event"}


def test_v1_mav_reward_removes_v0_overlay_r_adv_and_r_end():
    env = _bare_env()
    rewards, components = _base_rewards_components()
    rewards, components = env._compute_happo_ref_v1_mav_support(rewards, components)
    comp = components["red_0"]

    assert comp["v1_mav_removed_r_adv"] == pytest.approx(6.0)
    assert comp["v1_mav_removed_r_end"] == pytest.approx(1.0)
    assert comp["v1_mav_removed_v0_overlay"] == pytest.approx(1.0)
    assert comp["r_adv"] == pytest.approx(0.0)
    assert comp["r_end"] == pytest.approx(0.0)
    for forbidden in ("mav_survival", "mav_support", "mav_attack", "mav_dodge", "death_penalty"):
        assert forbidden not in comp
    assert "v1_mav_safety" in comp
    assert "v1_mav_support" in comp
    assert "v1_mav_event" in comp
    assert rewards["red_0"] == pytest.approx(comp["v1_mav_total"])


def test_v1_mav_support_pos_is_log_only_when_disabled():
    env = _bare_env()
    rewards, components = _base_rewards_components()
    _, components = env._compute_happo_ref_v1_mav_support(rewards, components)
    comp = components["red_0"]

    assert comp["v1_mav_support_pos_active"] == pytest.approx(0.0)
    assert comp["v1_mav_support"] == pytest.approx(0.4 * comp["v1_mav_support_aware"])
    assert comp["v1_mav_support_aware_raw"] > 0.0


def test_v1_mav_support_pos_active_true_raises_until_paper_grounded():
    env = _bare_env()
    env.happo_ref_v1_mav_support_config["mav_support"]["pos_active"] = True
    with pytest.raises(ValueError, match="R_pos is not implemented"):
        env._happo_v1_mav_support("red_0", env.red_planes["red_0"], env.happo_ref_v1_mav_support_config)


def test_v1_mav_safety_dist_matches_tam_v2_formula():
    env = _bare_env()
    cfg = env.happo_ref_v1_mav_support_config
    cfg["mav_safety"]["threat_weight"] = 0.0
    cfg["mav_safety"]["aspect_weight"] = 0.0
    cfg["mav_safety"]["dist_weight"] = 1.0
    d_danger = cfg["mav_safety"]["d_danger_m"]
    d_safe = cfg["mav_safety"]["d_safe_m"]

    for distance, expected in [
        (0.0, -1.0),
        (d_danger, 0.0),
        ((d_danger + d_safe) / 2.0, -0.25),
        (d_safe, 0.2),
        (d_safe + 1000.0, 0.2),
    ]:
        env.blue_planes = {"blue_0": _Sim(True, (distance, 0.0, 6500.0))}
        _total, logs = env._happo_v1_mav_safety(env.red_planes["red_0"], cfg)
        assert logs["v1_mav_safety_dist"] == pytest.approx(expected)


def test_v1_mav_safety_aspect_uses_ta_not_ao(monkeypatch):
    env = _bare_env()
    env.blue_planes = {"blue_0": env.blue_planes["blue_0"]}
    cfg = env.happo_ref_v1_mav_support_config
    cfg["mav_safety"]["dist_weight"] = 0.0
    cfg["mav_safety"]["threat_weight"] = 0.0
    cfg["mav_safety"]["aspect_weight"] = 1.0

    monkeypatch.setattr(
        hetero_env_mod,
        "get2d_AO_TA_R",
        lambda _mav_feat, _blue_feat: (0.0, np.pi / 8.0, 10000.0),
    )
    _total, logs = env._happo_v1_mav_safety(env.red_planes["red_0"], cfg)
    assert logs["v1_mav_safety_aspect"] == pytest.approx(-0.5)

    monkeypatch.setattr(
        hetero_env_mod,
        "get2d_AO_TA_R",
        lambda _mav_feat, _blue_feat: (0.0, np.pi / 3.0, 10000.0),
    )
    _total, logs = env._happo_v1_mav_safety(env.red_planes["red_0"], cfg)
    assert logs["v1_mav_safety_aspect"] == pytest.approx(0.0)


def test_v1_mav_event_team_credit_only_when_mav_alive_and_capped():
    env = _bare_env()
    rewards, components = _base_rewards_components()
    _, components = env._compute_happo_ref_v1_mav_support(rewards, components)
    comp = components["red_0"]

    assert comp["v1_mav_event_team_credit_delta"] == pytest.approx(0.5)
    assert comp["v1_mav_event_team_credit_used"] == pytest.approx(0.5)
    assert comp["v1_mav_event_team_credit_cap"] == pytest.approx(1.0)


def test_v1_uav_side_keeps_happo_ref_v0_style_components():
    env = _bare_env()
    rewards, components = _base_rewards_components()
    rewards, components = env._compute_happo_ref_v1_mav_support(rewards, components)

    for rid in ["red_1", "red_2"]:
        comp = components[rid]
        assert "uav_attack_window" in comp
        assert "uav_fire" in comp
        assert "uav_hit" in comp
        assert "uav_dodge" in comp
        assert comp["r_adv"] != 0.0
        assert rewards[rid] != 0.0


def test_v1_summary_fields_exist_on_mav_components():
    env = _bare_env()
    rewards, components = _base_rewards_components()
    _, components = env._compute_happo_ref_v1_mav_support(rewards, components)
    comp = components["red_0"]
    required = [
        "mav_observed_ratio",
        "mav_shared_track_ratio",
        "red_launch_with_mav_shared_track",
        "red_hit_with_mav_shared_track",
        "team_kill_while_mav_alive",
        "team_kill_after_mav_death",
        "red_launch_rate_before_mav_death",
        "red_launch_rate_after_mav_death",
        "mav_reward_safety_sum",
        "mav_reward_support_sum",
        "mav_reward_event_sum",
        "mav_reward_total_sum",
        "mav_removed_r_adv_sum",
        "mav_removed_r_end_sum",
    ]
    for key in required:
        assert key in comp


def test_v1_mav_observed_ratio_uses_full_blue_index_alignment():
    env = _bare_env()
    env.blue_planes["blue_0"].is_alive = False
    env.blue_planes["blue_1"].is_alive = True
    env._last_step_obs["red_0"]["enemy_observed_mask"] = np.asarray([0.0, 1.0], dtype=np.float32)
    logs = env._happo_v1_episode_support_logs("red_0", env.red_planes["red_0"])
    assert logs["mav_observed_ratio"] == pytest.approx(1.0)


def test_v1_episode_ratio_aggregation_uses_mean_not_linear_sum():
    from scripts.train_happo_reference import _finalize_happo_v1_episode_reward_components

    comp = {
        "mav_observed_ratio": 10.0,
        "mav_shared_track_ratio": 5.0,
        "red_launch_before_mav_death": 2.0,
        "red_uav_alive_steps_before_mav_death": 4.0,
        "red_launch_after_mav_death": 1.0,
        "red_uav_alive_steps_after_mav_death": 2.0,
    }
    out = _finalize_happo_v1_episode_reward_components(comp, episode_length=10)
    assert out["mav_observed_ratio"] == pytest.approx(1.0)
    assert out["mav_shared_track_ratio"] == pytest.approx(0.5)
    assert out["red_launch_rate_before_mav_death"] == pytest.approx(0.5)
    assert out["red_launch_rate_after_mav_death"] == pytest.approx(0.5)


def test_rich_logging_v1_mav_fields_in_schema_and_unknown_skipped():
    """write_episode_reward_components must not crash on v1_mav fields or unknowns."""
    import tempfile, shutil
    from scripts.rich_logging import RichExperimentLogger
    tmp = Path(tempfile.mkdtemp(prefix="v1_log_"))
    try:
        logger = RichExperimentLogger(
            tmp, run_id="t", method_name="t", scenario_name="t",
            device="cpu", num_envs=1, rollout_length_per_env=256, transitions_per_rollout=256,
        )
        v1_fields = {
            "v1_mav_safety_sum": 1.0, "v1_mav_support_sum": 2.0,
            "v1_mav_event_sum": 3.0, "v1_mav_total_sum": 6.0,
            "v1_mav_safety_dist_sum": 0.5, "v1_mav_safety_threat_sum": 0.3,
            "v1_mav_safety_aspect_sum": 0.2, "v1_mav_safety_danger_m_sum": 1,
            "v1_mav_safety_safe_m_sum": 2, "v1_mav_support_pos_sum": 0.6,
            "v1_mav_support_pos_active_sum": 0.0, "v1_mav_support_aware_sum": 0.4,
            "v1_mav_support_observed_count_sum": 3, "v1_mav_support_aware_raw_sum": 0.4,
            "v1_mav_event_death_sum": -4.0, "v1_mav_event_team_credit_delta_sum": 0.5,
            "v1_mav_event_team_credit_used_sum": 0.5, "v1_mav_event_team_credit_cap_sum": 1.0,
            "v1_mav_removed_v0_overlay_sum": 1.0, "v1_mav_total_pre_clip_sum": 7.0,
            "v1_mav_blue_launch_window_on_mav_log_sum": 0,
            "unknown_component_sum": 999.0,  # must not crash
        }
        logger.write_episode_reward_components(
            scenario="test", episode_id="0", agent_id="red_0", role="mav", team="red",
            episode_length=100, episode_return=10.0, component_sums=v1_fields,
            outcome="timeout", end_reason="timeout",
        )
        logger.close()
        with open(tmp / "episode_reward_components.csv", newline="", encoding="utf-8") as f:
            import csv
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        header = list(rows[0].keys())
        for field in v1_fields:
            if field == "unknown_component_sum":
                assert field not in header
            else:
                assert field in header, f"missing {field}"
        assert rows[0]["v1_mav_safety_sum"] == "1.0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_v1_needs_last_step_obs_cache_and_mav_never_launches():
    from uav_env import make_env

    env = make_env(CFG_3V2, max_steps=5)
    try:
        assert env._needs_last_step_obs_cache() is True
        obs, _ = env.reset(seed=1)
        assert env._last_step_obs
        assert env._num_missiles_for("red_0") == 0
        assert env.agent_roles["red_0"] == "mav"
        zero_actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        for _ in range(2):
            obs, _rewards, _terminated, _truncated, info = env.step(zero_actions)
            assert env._last_step_obs
            for rec in info.get("__launch_quality_step__", []) or []:
                assert rec.get("shooter_role") != "mav"
    finally:
        env.close()
