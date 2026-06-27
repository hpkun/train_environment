"""Tests for tam_paper_reward_v4 — BRMA flight status + situation + terminal outcome."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_v4_env(**overrides):
    from uav_env import make_env
    config_path = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml"
    kwargs = {}
    kwargs.update(overrides)
    return make_env(config_path, **kwargs)


class TestV4EnvCreation:
    def test_v4_mode_registered(self):
        env = _make_v4_env()
        assert env.hetero_reward_mode == "tam_paper_reward_v4"
        env.close()

    def test_missing_config_raises(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        with pytest.raises(ValueError, match="tam_paper_reward_v4"):
            HeteroUavCombatEnv(
                hetero_reward_mode="tam_paper_reward_v4",
                max_num_red=3, max_num_blue=2, max_steps=100,
            )

    def test_v2_unaffected(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml")
        assert env.hetero_reward_mode == "tam_paper_reward_v2"
        env.close()

    def test_v3_unaffected(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v3.yaml")
        assert env.hetero_reward_mode == "tam_paper_reward_v3"
        env.close()


class TestV4FlightStatus:
    def test_pitch_function_works(self):
        """BRMA pitch penalty function exists and returns finite values."""
        env = _make_v4_env()
        env.reset(seed=0)
        sim = env.red_planes["red_1"]
        raw = env._pitch_penalty(sim)
        # Normal flight should have near-zero pitch penalty
        assert abs(raw) < 0.5, f"normal pitch penalty should be small, got {raw}"
        assert np.isfinite(raw)
        env.close()

    def test_pitch_normal_near_zero(self):
        env = _make_v4_env()
        env.reset(seed=0)
        sim = env.red_planes["red_1"]
        raw = env._pitch_penalty(sim)
        assert abs(raw) < 0.01, f"normal pitch should be ~0, got {raw}"
        env.close()

    def test_v4_uav_has_flight_status_keys(self):
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        c1 = rc.get("red_1", {})
        for k in ("tam_v4_uav_pitch", "tam_v4_uav_roll", "tam_v4_uav_flight_status"):
            assert k in c1, f"UAV missing {k}"
        env.close()

    def test_v4_mav_has_flight_status_keys(self):
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        c0 = rc.get("red_0", {})
        for k in ("tam_v4_mav_pitch", "tam_v4_mav_roll", "tam_v4_mav_flight_status"):
            assert k in c0, f"MAV missing {k}"
        env.close()


class TestV4Situation:
    def test_situation_has_own_adv_and_threat(self):
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        c1 = rc.get("red_1", {})
        for k in ("tam_v4_uav_situation", "tam_v4_uav_situation_raw",
                  "tam_v4_uav_own_adv_log", "tam_v4_uav_enemy_threat_log"):
            assert k in c1, f"UAV missing {k}"
        env.close()

    def test_situation_replaces_angle_distance(self):
        """v4 uses situation as active; angle/distance are diagnostic only."""
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        c1 = rc.get("red_1", {})
        # Diagnostic angle/distance should exist but not be in total computation name
        assert "tam_v2_uav_angle_diag" in c1, "should have angle diagnostic"
        assert "tam_v2_uav_distance_diag" in c1, "should have distance diagnostic"
        # Situation should be the active reward, not angle+distance
        assert "tam_v4_uav_situation" in c1
        total = c1.get("tam_v4_total", 0)
        assert isinstance(total, (int, float))
        env.close()


class TestV4TerminalOutcome:
    def test_terminal_outcome_field_present(self):
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(5):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
            if all(terminated.values()) or all(truncated.values()):
                break
        rc = info.get("reward_components", {})
        c0 = rc.get("red_0", {})
        assert "tam_v4_team_outcome" in c0, "MAV team outcome missing"
        env.close()

    def test_terminal_outcome_values(self):
        """Verify config has team_win=+200, team_loss=-200, team_draw=0."""
        import yaml
        with open("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        ev = c["tam_paper_reward_v4"]["uav"]["event"]
        assert ev["team_win"] == 200.0
        assert ev["team_loss"] == -200.0
        assert ev["team_draw"] == 0.0


class TestV4DeadGuardAndNoForbidden:
    def test_v4_has_no_fire_launch_guided(self):
        env = _make_v4_env()
        env.reset(seed=0)
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        for uid in ("red_0", "red_1"):
            comp = rc.get(uid, {})
            for bad in ("uav_fire", "guided_fire", "launch_window_bonus", "fire_reward"):
                assert comp.get(bad, 0.0) == 0.0, f"{uid} has forbidden {bad}"
        env.close()

    def test_v4_config_keys(self):
        import yaml
        with open("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        v4 = c["tam_paper_reward_v4"]
        assert "flight_status" in v4
        assert "situation" in v4
        assert v4["uav"]["event"]["team_win"] == 200.0
        assert v4["uav"]["event"]["team_loss"] == -200.0
        assert v4["uav"]["reward_weights"]["situation"] == 25.0
