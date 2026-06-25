"""Tests for brma_uav_tam_mav_event_v1 reward."""
import numpy as np
import pytest


class TestRewardProfile:
    def test_reward_profile_name(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        assert HeteroUavCombatEnv.PAPER_ROLE_REWARD_PROFILE == "brma_uav_tam_mav_event_v1"

    def test_paper_role_only_calls_helper(self):
        """paper_role_reward_v1 branch must only call the helper and return."""
        import inspect
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        src = inspect.getsource(HeteroUavCombatEnv._compute_rewards)
        # After paper_role check, the next non-comment line must be role_v1
        lines = src.split('\n')
        paper_idx = next(i for i, l in enumerate(lines) if 'hetero_reward_mode == "paper_role_reward_v1"' in l)
        next_lines = [l.strip() for l in lines[paper_idx:paper_idx+5] if l.strip() and not l.strip().startswith('#')]
        assert len(next_lines) >= 2
        assert 'return self._compute_brma_uav_tam_mav_event_v1' in next_lines[1], \
            "paper_role must return helper immediately"


class TestUAVReward:
    def test_uav_keeps_r_adv(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, t, tr, info = env.step(actions)
        rc = info.get("reward_components", {})
        assert rc["red_1"].get("r_adv", 0.0) != 0.0, "UAV must keep BRMA r_adv"
        env.close()

    def test_mav_removes_r_adv(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, t, tr, info = env.step(actions)
        rc = info.get("reward_components", {})
        assert rc["red_0"].get("r_adv", 1.0) == 0.0, "MAV must remove r_adv"
        env.close()

    def test_all_red_remove_r_end(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, t, tr, info = env.step(actions)
        rc = info.get("reward_components", {})
        for aid in ["red_0", "red_1", "red_2"]:
            assert rc[aid].get("r_end", 1.0) == 0.0, f"{aid} r_end must be 0"
        env.close()

    def test_uav_fire_uav_hit_uav_attack_mav_assist_all_zero(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, t, tr, info = env.step(actions)
        rc = info.get("reward_components", {})
        for k in ("uav_fire", "uav_hit", "uav_attack", "mav_assist"):
            assert rc["red_1"].get(k, -1.0) == 0.0, f"{k} must be 0"
        env.close()


class TestEventRewards:
    def test_uav_kill_is_4(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1", observation_mode="mav_shared_geo",
            red_agent_types=["mav","attack_uav","attack_uav"], blue_agent_types=["attack_uav","attack_uav"])
        env.reset(seed=0)
        env._launch_quality_done_step_records = [{"shooter_id":"red_1","target_id":"blue_0","raw_termination_reason":"hit","mav_guided_at_launch":False}]
        env._paper_reset_reward_state()
        for aid in env.agent_ids:
            env._last_step_obs = {aid: {"enemy_geo_states":np.zeros((2,5)),"enemy_alive_mask":np.ones(2),"enemy_observed_mask":np.ones(2),"enemy_track_source":np.zeros((2,2)),"ego_geo_state":np.zeros(7),"ego_role":[0,1,0,0],"missile_warning":[0.0],"ally_geo_states":np.zeros((2,5)),"ally_roles":np.zeros((2,4)),"ally_alive_mask":np.ones(2),"ego_geo_state":np.zeros(7)}}
        env._missile_candidate_metrics = lambda s,t: {"AO_rad":0.1,"TA_rad":2.0,"range_m":5000,"range_ok":True,"ao_ok":True,"ta_ok":True}
        rewards, comps = env._compute_rewards()
        assert comps["red_1"]["event_uav_kill"] == pytest.approx(4.0)
        env.close()

    def test_team_kill_is_0_5(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1", observation_mode="mav_shared_geo",
            red_agent_types=["mav","attack_uav","attack_uav"], blue_agent_types=["attack_uav","attack_uav"])
        env.reset(seed=0)
        env._launch_quality_done_step_records = [{"shooter_id":"red_1","target_id":"blue_0","raw_termination_reason":"hit","mav_guided_at_launch":False}]
        env._paper_reset_reward_state()
        for aid in env.agent_ids:
            env._last_step_obs = {aid: {"enemy_geo_states":np.zeros((2,5)),"enemy_alive_mask":np.ones(2),"enemy_observed_mask":np.ones(2),"enemy_track_source":np.zeros((2,2)),"ego_geo_state":np.zeros(7),"ego_role":[0,1,0,0],"missile_warning":[0.0],"ally_geo_states":np.zeros((2,5)),"ally_roles":np.zeros((2,4)),"ally_alive_mask":np.ones(2)}}
        env._missile_candidate_metrics = lambda s,t: {"AO_rad":0.1,"TA_rad":2.0,"range_m":5000,"range_ok":True,"ao_ok":True,"ta_ok":True}
        rewards, comps = env._compute_rewards()
        assert comps["red_0"]["event_team_kill"] == pytest.approx(0.5)
        env.close()


class TestLaunchGate:
    def test_use_boresight_default_false(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        assert info.get("use_boresight_launch_gate") == False
        env.close()

    def test_get_info_fields(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")
        obs, info = env.reset(seed=0)
        assert "effective_missile_launch_range_m" in info
        assert "effective_missile_attack_interval_sec" in info
        assert "use_boresight_launch_gate" in info
        env.close()
