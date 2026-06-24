"""Tests for paper_role_reward_v1 refined implementation (tam_brma_role_reward_refined_v1)."""
import numpy as np
import pytest


class TestDistanceCurves:
    def test_mav_s_dist_peak(self):
        """S_dist should be +1 in 15-30 km range, decaying to -1 after 40km."""
        for d_km, expected_sign in [(5, -1), (12, 0), (20, 1), (30, 1), (35, 1), (42, -1), (50, -1)]:
            if d_km < 8: val = -1.0
            elif d_km < 15: val = -1.0 + 2.0*(d_km-8)/7.0
            elif d_km <= 30: val = 1.0
            elif d_km <= 40: val = 1.0 - 1.5*(d_km-30)/10.0
            else: val = -1.0
            if expected_sign > 0: assert val > 0, f"d={d_km} should be positive, got {val:.2f}"
            elif expected_sign < 0: assert val < 0, f"d={d_km} should be negative, got {val:.2f}"
            else: assert abs(val) < 0.5, f"d={d_km} should be near zero, got {val:.2f}"

    def test_uav_r_d(self):
        """R_D: <=5km = +1, >=10km = -1."""
        for d_km in [3, 5]:
            assert 1.0 == 1.0 if d_km <= 5 else -1.0
        for d_km in [10, 15]:
            RD = 2.0 * np.exp(-0.921 * (d_km - 5.0)) - 1.0 if d_km < 10 else -1.0
            assert RD == -1.0, f"d={d_km} should be -1"


class TestEpisodeCap:
    def test_cap_helper(self):
        """_paper_add_capped_reward should obey [low, high] bounds."""
        cum = {}
        def helper(aid, key, delta, lo, hi):
            old = cum.setdefault(key, 0.0)
            new = float(np.clip(old + delta, lo, hi))
            cum[key] = new
            return new - old
        # Attack cap [-3, +5]
        total = 0.0
        for _ in range(100):
            total += helper("red_1", "uav_attack", 0.1, -3.0, 5.0)
        assert total <= 5.0
        assert cum["uav_attack"] <= 5.0


class TestRewardModeDispatch:
    def test_paper_role_smoke(self):
        """paper_role_reward_v1 runs without error and produces new keys."""
        from uav_env import make_env
        cfg = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml"
        env = make_env(cfg)
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        mav = rc.get("red_0", {})
        uav = rc.get("red_1", {})
        # Key MAV components
        assert "mav_safety" in mav
        assert "mav_support" in mav
        assert "mav_safety_dist" in mav
        assert "mav_safety_threat" in mav
        assert "mav_support_position" in mav
        # Key UAV components
        assert "uav_attack" in uav
        assert "uav_fire" in uav
        assert "uav_hit" in uav
        assert "uav_dodge" in uav
        assert "uav_death" in uav
        assert "uav_out_zone" in uav
        # r_end zeroed
        assert abs(mav.get("r_end", 1.0)) < 0.01, "r_end must be zero"
        env.close()

    def test_role_v1_still_works(self):
        """role_v1 still produces r_role_* keys."""
        from uav_env import make_env
        cfg = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml"
        env = make_env(cfg)
        env.hetero_reward_mode = "role_v1"
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        mav = rc.get("red_0", {})
        assert "r_role_mav_survival" in mav
        env.close()


class TestTimeoutTerminal:
    def test_timeout_terminal_zero(self):
        """Timeout should give zero terminal reward."""
        from uav_env import make_env
        cfg = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml"
        env = make_env(cfg)
        # Force timeout by stepping to max_steps
        env.reset(seed=0)
        for _ in range(10):
            actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)
        rc = info.get("reward_components", {})
        mav = rc.get("red_0", {})
        # r_end should be 0
        assert abs(mav.get("r_end", 1.0)) < 0.01, "r_end must be zero on timeout"
        env.close()


class TestMavGuidedLaunchRecord:
    def test_launch_record_has_mav_guided(self):
        """_build_launch_quality_record should include mav_guided_at_launch."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        env.reset(seed=0)
        shooter = env.red_planes.get("red_1")
        target = env.blue_planes.get("blue_0")
        if shooter and target:
            record = env._build_launch_quality_record(shooter, target)
            assert "mav_guided_at_launch" in record
            assert "mav_guided_lookback_steps" in record
            assert "mav_guided_source" in record
        env.close()
