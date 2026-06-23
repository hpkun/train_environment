"""Tests for paper_role_reward_v1 isolation from other reward modes."""
import numpy as np
import pytest


class TestRewardModeIsolation:
    """Verify paper_role_reward_v1 does NOT stack with role_v1."""

    def test_paper_role_reward_v1_accepted(self):
        """paper_role_reward_v1 is a valid reward mode."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        # This should not raise
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        env.close()

    def test_role_v1_accepted(self):
        """role_v1 still works."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="role_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        env.close()

    def test_paper_role_reward_components_exist(self):
        """paper_role_reward_v1 produces its own component keys."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        components = info.get("reward_components", {})
        mav_comp = components.get("red_0", {})
        uav_comp = components.get("red_1", {})
        # paper_role keys must be present
        assert "mav_safety" in mav_comp
        assert "mav_support" in mav_comp
        assert "mav_event" in mav_comp
        assert "mav_death" in mav_comp
        assert "uav_attack_window" in uav_comp
        assert "uav_fire" in uav_comp
        assert "uav_hit" in uav_comp
        assert "uav_dodge" in uav_comp
        assert "uav_death" in uav_comp
        env.close()

    def test_paper_role_does_not_contain_role_v1_keys(self):
        """paper_role_reward_v1 must NOT have role_v1 overlay keys."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="paper_role_reward_v1",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        components = info.get("reward_components", {})
        mav_comp = components.get("red_0", {})
        uav_comp = components.get("red_1", {})
        # role_v1 keys must NOT be present
        assert "r_role_mav_survival" not in mav_comp
        assert "r_role_mav_death" not in mav_comp
        assert "r_role_mav_support" not in mav_comp
        assert "r_role_mav_team_contribution" not in mav_comp
        assert "r_role_uav_kill_bonus" not in uav_comp
        assert "r_role_uav_death_penalty" not in uav_comp
        assert "r_role_uav_missile_warning" not in uav_comp
        env.close()

    def test_happo_ref_v0_still_works(self):
        """happo_ref_v0 is unaffected."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        env = HeteroUavCombatEnv(
            max_num_blue=2, max_num_red=3, max_steps=10,
            hetero_reward_mode="happo_ref_v0",
            observation_mode="mav_shared_geo",
            red_agent_types=["mav", "attack_uav", "attack_uav"],
            blue_agent_types=["attack_uav", "attack_uav"],
        )
        obs, info = env.reset(seed=0)
        actions = {aid: np.zeros(3, dtype=np.float32) for aid in env.agent_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        components = info.get("reward_components", {})
        mav_comp = components.get("red_0", {})
        assert "mav_survival" in mav_comp or "safety" in mav_comp
        env.close()
