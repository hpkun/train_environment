"""Tests for brma_uav_tam_mav_event_v1 — active reward + static audit."""
import inspect
import numpy as np
import pytest


def _make_env():
    from uav_env import make_env
    return make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml")


def _full_obs(env):
    obs_dict = {}
    for aid in env.agent_ids:
        obs_dict[aid] = {
            "ego_geo_state": np.zeros(7, dtype=np.float32),
            "ego_role": np.array([1, 0, 0, 0], dtype=np.float32) if aid == "red_0" else np.array([0, 1, 0, 0], dtype=np.float32),
            "missile_warning": np.array([0.0], dtype=np.float32),
            "ally_geo_states": np.zeros((2, 5), dtype=np.float32),
            "ally_roles": np.zeros((2, 4), dtype=np.float32),
            "ally_alive_mask": np.ones(2, dtype=np.float32),
            "enemy_geo_states": np.zeros((2, 5), dtype=np.float32),
            "enemy_alive_mask": np.ones(2, dtype=np.float32),
            "enemy_observed_mask": np.ones(2, dtype=np.float32),
            "enemy_track_source": np.zeros((2, 2), dtype=np.float32),
        }
    return obs_dict


class TestRewardProfile:
    def test_profile_name(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        assert HeteroUavCombatEnv.PAPER_ROLE_REWARD_PROFILE == "brma_uav_tam_mav_event_v1"

    def test_dispatch_single_helper(self):
        src = inspect.getsource(
            __import__("uav_env.JSBSim.envs.hetero_uav_combat_env",
                       fromlist=["HeteroUavCombatEnv"]).HeteroUavCombatEnv._compute_rewards)
        lines = src.split("\n")
        in_paper = False
        for line in lines:
            if 'hetero_reward_mode == "paper_role_reward_v1"' in line:
                in_paper = True
            if in_paper:
                assert "uav_fire" not in line or "0.0" in line, f"uav_fire active in {line.strip()}"
                assert "uav_attack" not in line or "0.0" in line, f"uav_attack active in {line.strip()}"
            if in_paper and "return self._compute_brma" in line:
                break


class TestSmokeReward:
    def test_components_clean(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        actions = {aid: [0, 0, 0.3] for aid in env.agent_ids}
        obs, rewards, t, tr, info = env.step(actions)
        rc = info.get("reward_components", {})
        assert rc["red_1"]["r_adv"] != 0.0, "UAV keeps r_adv"
        assert rc["red_0"]["r_adv"] == 0.0, "MAV removes r_adv"
        for aid in ("red_0", "red_1", "red_2"):
            assert rc[aid]["r_end"] == 0.0, f"{aid} r_end must be 0"
        assert rc["red_1"]["uav_attack"] == 0.0
        assert rc["red_1"]["uav_fire"] == 0.0
        assert rc["red_1"]["uav_hit"] == 0.0
        assert rc["red_0"]["mav_assist"] == 0.0
        assert "tam_mav_safety_raw" in rc["red_0"]
        assert "tam_mav_support_raw" in rc["red_0"]
        assert "tam_mav_dense_reward" in rc["red_0"]
        env.close()


class TestEventRewards:
    def _setup_env_with_hit(self, guided=False, mav_dead=False, uav_crashed=False):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1", observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"])
        env.reset(seed=0)
        env._launch_quality_done_step_records = [{
            "shooter_id": "red_1", "target_id": "blue_0",
            "raw_termination_reason": "hit",
            "mav_guided_at_launch": guided,
        }]
        if mav_dead:
            env.red_planes["red_0"].crash()
        if uav_crashed:
            env.red_planes["red_1"].crash()
            env._crashed_this_step.add("red_1")
        env._paper_reset_reward_state()
        return env

    def _setup_obs(self, env):
        obs = _full_obs(env)
        env._last_step_obs = obs
        orig = env._missile_candidate_metrics
        env._missile_candidate_metrics = lambda s, t: {"AO_rad": 0.1, "TA_rad": 2.0, "range_m": 5000, "range_ok": True, "ao_ok": True, "ta_ok": True}
        return orig

    def test_uav_kill_direct_vs_guided_same(self):
        for guided in (False, True):
            env = self._setup_env_with_hit(guided=guided)
            self._setup_obs(env)
            rewards, comps = env._compute_rewards()
            assert comps["red_1"]["event_uav_kill"] == pytest.approx(4.0)
            assert comps["red_1"]["uav_hit_direct_count"] + comps["red_1"]["uav_hit_mav_guided_count"] == 1
            env.close()

    def test_team_kill(self):
        env = self._setup_env_with_hit()
        self._setup_obs(env)
        rewards, comps = env._compute_rewards()
        assert comps["red_0"]["event_team_kill"] == pytest.approx(0.5)
        env.close()

    def test_uav_death(self):
        env = self._setup_env_with_hit(uav_crashed=False)
        env.red_planes["red_1"].crash()
        self._setup_obs(env)
        rewards, comps = env._compute_rewards()
        assert comps["red_1"]["event_uav_death"] == pytest.approx(-4.0)
        env.close()

    def test_uav_crash(self):
        env = self._setup_env_with_hit(uav_crashed=True)
        self._setup_obs(env)
        rewards, comps = env._compute_rewards()
        assert comps["red_1"]["event_uav_crash"] == pytest.approx(-5.0)
        env.close()

    def test_mav_death(self):
        env = self._setup_env_with_hit(mav_dead=True)
        self._setup_obs(env)
        rewards, comps = env._compute_rewards()
        assert comps["red_0"]["event_mav_death"] == pytest.approx(-6.0)
        assert comps["red_1"]["event_mav_loss_team"] == pytest.approx(-1.0)
        env.close()

    def test_out_zone_once(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1", observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"])
        env.reset(seed=0)
        self._setup_obs(env)
        env._paper_out_zone_penalized = set()
        # Force UAV out of zone
        pos = env.red_planes["red_1"].get_position()
        env.red_planes["red_1"]._position = np.array([50000.0, 0.0, 5000.0])
        rewards, comps = env._compute_rewards()
        assert comps["red_1"]["event_out_zone"] == pytest.approx(-2.0)
        # Second call should not penalize again
        rewards2, comps2 = env._compute_rewards()
        assert comps2["red_1"].get("event_out_zone", 0.0) == 0.0
        env.close()

    def test_fire_log_only(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1", observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"])
        env.reset(seed=0)
        self._setup_obs(env)
        env._launch_quality_step_records = [
            {"shooter_id": "red_1", "target_id": "blue_0", "mav_guided_at_launch": False},
            {"shooter_id": "red_1", "target_id": "blue_1", "mav_guided_at_launch": True},
        ]
        rewards, comps = env._compute_rewards()
        assert comps["red_1"]["uav_fire"] == 0.0
        assert comps["red_1"]["uav_fire_log"] == 0.0
        assert comps["red_1"]["uav_fire_direct_count"] == 1
        assert comps["red_1"]["uav_fire_mav_guided_count"] == 1
        env.close()


class TestLaunchGate:
    def test_default_boresight_off(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        assert info.get("use_boresight_launch_gate") == False
        env.close()

    def test_info_fields(self):
        env = _make_env()
        obs, info = env.reset(seed=0)
        for k in ("effective_missile_launch_range_m", "effective_missile_attack_interval_sec",
                  "use_boresight_launch_gate"):
            assert k in info, f"missing {k}"
        env.close()


class TestStaticAudit:
    def test_no_old_active_reward_in_helper(self):
        src = inspect.getsource(
            __import__("uav_env.JSBSim.envs.hetero_uav_combat_env",
                       fromlist=["HeteroUavCombatEnv"]).HeteroUavCombatEnv._compute_brma_uav_tam_mav_event_v1)
        forbidden = [
            "12.0 * direct_hit", "15.0 * guided_hit",
            "base_rewards[rid] = base_rewards.get(rid, 0.0) + comp[\"uav_fire\"]",
            "base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + comp[\"uav_attack\"]",
            "base_rewards[mav_id] = base_rewards.get(mav_id, 0.0) + r_hit",
            "mav_assist" + "base_rewards",  # rough check
        ]
        for f in forbidden:
            assert f not in src, f"forbidden pattern found: {f}"
