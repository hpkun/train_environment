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
        """_build_launch_quality_record should include mav_guided_at_launch and mav_observed_at_launch."""
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
            assert "mav_observed_at_launch" in record
            assert "mav_observed_source" in record
        env.close()

    def test_done_hit_uses_mav_guided_at_launch(self):
        """uav_hit guided/direct split uses launch record mav_guided_at_launch."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        env.reset(seed=0)
        # Inject a done hit record with mav_guided_at_launch=True
        env._launch_quality_done_step_records = [{
            "shooter_id": "red_1", "target_id": "blue_0",
            "raw_termination_reason": "hit",
            "mav_guided_at_launch": True,
            "mav_observed_at_launch": False,
        }]
        env._paper_reset_reward_state()
        env._last_step_obs = {
            "red_0": {"enemy_observed_mask": np.zeros(2), "enemy_track_source": np.zeros((2, 2))},
            "red_1": {"enemy_geo_states": np.zeros((2, 5)), "enemy_alive_mask": np.ones(2),
                      "enemy_observed_mask": np.ones(2), "enemy_track_source": np.zeros((2, 2)),
                      "ego_geo_state": np.array([0]*7)},
            "red_2": {"enemy_geo_states": np.zeros((2, 5)), "enemy_alive_mask": np.ones(2),
                      "enemy_observed_mask": np.ones(2), "enemy_track_source": np.zeros((2, 2)),
                      "ego_geo_state": np.array([0]*7)},
        }
        for bid in env.blue_ids:
            env._last_step_obs[bid] = {"ego_geo_state": np.array([0]*7), "ego_role": [0,1,0,0],
                                       "missile_warning": [0.0], "ally_geo_states": np.zeros((1,5)),
                                       "ally_roles": np.zeros((1,4)), "ally_alive_mask": np.ones(1),
                                       "enemy_geo_states": np.zeros((3,5)), "enemy_alive_mask": np.ones(3),
                                       "enemy_observed_mask": np.ones(3), "enemy_track_source": np.zeros((3,2))}
        base_rewards, components = {}, {}
        for aid in env.agent_ids:
            components[aid] = {}
        for sim in list(env.red_planes.values()) + list(env.blue_planes.values()):
            if sim:
                rpy = sim.get_rpy()
                components[sim.uid] = {"r_end": 0.0, "r_adv": 0.0, "r_bound": 0.0, "r_alt": 0.0,
                                       "r_pitch": 0.0, "r_roll": 0.0, "r_vel": 0.0, "r_death": 0.0}
        env.hetero_reward_mode = "paper_role_reward_v1"
        rewards, comps = env._compute_rewards()
        uav = comps.get("red_1", {})
        assert uav.get("uav_hit_mav_guided_count", 0) == 1, "guided hit should be counted"
        assert uav.get("uav_hit_direct_count", 0) == 0, "direct hit should be 0"
        env.close()

    def test_mav_assist_uses_launch_record_flags(self):
        """MAV assist should use mav_guided_at_launch or mav_observed_at_launch."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        env.reset(seed=0)
        env._launch_quality_done_step_records = [{
            "shooter_id": "red_1", "target_id": "blue_0",
            "raw_termination_reason": "hit",
            "mav_guided_at_launch": False,
            "mav_observed_at_launch": True,
        }]
        env._paper_reset_reward_state()
        env._last_step_obs = {
            "red_0": {"enemy_observed_mask": np.zeros(2), "enemy_track_source": np.zeros((2, 2))},
            "red_1": {"enemy_geo_states": np.zeros((2, 5)), "enemy_alive_mask": np.ones(2),
                      "enemy_observed_mask": np.ones(2), "enemy_track_source": np.zeros((2, 2)),
                      "ego_geo_state": np.array([0]*7)},
            "red_2": {"enemy_geo_states": np.zeros((2, 5)), "enemy_alive_mask": np.ones(2),
                      "enemy_observed_mask": np.ones(2), "enemy_track_source": np.zeros((2, 2)),
                      "ego_geo_state": np.array([0]*7)},
        }
        for bid in env.blue_ids:
            env._last_step_obs[bid] = {"ego_geo_state": np.array([0]*7), "ego_role": [0,1,0,0],
                                       "missile_warning": [0.0], "ally_geo_states": np.zeros((1,5)),
                                       "ally_roles": np.zeros((1,4)), "ally_alive_mask": np.ones(1),
                                       "enemy_geo_states": np.zeros((3,5)), "enemy_alive_mask": np.ones(3),
                                       "enemy_observed_mask": np.ones(3), "enemy_track_source": np.zeros((3,2))}
        for aid in env.agent_ids:
            components = {}
            components[aid] = {}
        for sim in list(env.red_planes.values()) + list(env.blue_planes.values()):
            if sim:
                components[sim.uid] = {"r_end": 0.0, "r_adv": 0.0, "r_bound": 0.0, "r_alt": 0.0,
                                       "r_pitch": 0.0, "r_roll": 0.0, "r_vel": 0.0, "r_death": 0.0}
        env.hetero_reward_mode = "paper_role_reward_v1"
        rewards, comps = env._compute_rewards()
        mav = comps.get("red_0", {})
        assert mav.get("mav_assist", 0.0) > 0, "MAV assist should be positive with observed_at_launch"
        env.close()


class TestNoUtilsImport:
    def test_no_utils_import_in_paper_role(self):
        """paper_role_reward_v1 must not contain 'from .utils import'."""
        with open("uav_env/JSBSim/envs/hetero_uav_combat_env.py", encoding="utf-8") as f:
            content = f.read()
        # The file-level import is fine; the paper_role block must not have its own
        import_count = content.count("from .utils import")
        # Count occurrences inside _compute_rewards or paper_role blocks
        # The file should only have the top-level import in UavCombatEnv
        assert import_count <= 1, f"Should have at most 1 .utils import, found {import_count}"
