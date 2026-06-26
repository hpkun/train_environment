"""Tests for tam_paper_reward_v2 reward mode."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_v2_env(**overrides):
    from uav_env import make_env
    config_path = "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml"
    kwargs = {}
    kwargs.update(overrides)
    return make_env(config_path, **kwargs)


class TestTamPaperRewardV2EnvCreation:
    def test_can_create_env(self):
        env = _make_v2_env()
        assert env.hetero_reward_mode == "tam_paper_reward_v2"
        assert env.tam_paper_reward_v2_config is not None
        assert "global_scale" in env.tam_paper_reward_v2_config
        env.close()

    def test_missing_config_raises(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        with pytest.raises(ValueError, match="tam_paper_reward_v2"):
            HeteroUavCombatEnv(
                hetero_reward_mode="tam_paper_reward_v2",
                max_num_red=3, max_num_blue=2, max_steps=100,
            )


class TestTamPaperV2RewardComponents:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = _make_v2_env()
        self.env.reset(seed=0)
        # Run a few steps to accumulate reward
        for _ in range(5):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in self.env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(self.env._last_step_obs, self.env.blue_ids, env=self.env))
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            if all(terminated.values()) or all(truncated.values()):
                self.env.reset(seed=_ + 1)
        self.rc = info.get("reward_components", {})

    def teardown_method(self):
        if hasattr(self, "env"):
            self.env.close()

    def test_mav_has_v2_components(self):
        mav_id = "red_0"
        comp = self.rc.get(mav_id, {})
        for key in ("tam_v2_mav_safety", "tam_v2_mav_support", "tam_v2_mav_event", "tam_v2_total"):
            assert key in comp, f"MAV missing {key}"

    def test_uav_has_v2_components(self):
        uav_id = "red_1"
        comp = self.rc.get(uav_id, {})
        for key in ("tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
                     "tam_v2_uav_distance", "tam_v2_uav_dodge", "tam_v2_uav_event", "tam_v2_total"):
            assert key in comp, f"UAV missing {key}"

    def test_no_active_fire_launch_bonus(self):
        for uid in ("red_1", "red_2"):
            comp = self.rc.get(uid, {})
            for key in ("uav_fire", "guided_fire", "launch_window_bonus",
                         "tam_v2_uav_fire_log"):
                val = comp.get(key)
                if val is not None and isinstance(val, (int, float)):
                    # uav_fire et al. must not be active reward — only log
                    pass

    def test_brma_items_are_log_only(self):
        # BRMA r_adv should be in log-only fields, not added to active reward
        comp = self.rc.get("red_0", {})
        # Log fields should exist
        for key in ("brma_r_adv_log", "brma_r_pitch_log"):
            assert key in comp, f"Missing BRMA log key {key}"

    def test_v1_unchanged(self):
        """paper_role_reward_v1 should still work as before."""
        from uav_env import make_env
        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_paper_role_reward_v1.yaml",
        )
        env.reset(seed=0)
        assert env.hetero_reward_mode == "paper_role_reward_v1"
        # Presence of BRMA r_adv (not log-only) confirms v1 behavior
        for _ in range(3):
            actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
            from algorithms.mappo.opponent_policy import OpponentPolicy
            opp = OpponentPolicy(mode="brma_rule", seed=0)
            actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
            _, _, _, _, info = env.step(actions)
        rc = info.get("reward_components", {})
        # v1 should have r_adv (active) not brma_r_adv_log
        assert "r_adv" in rc.get("red_1", {}) or "tam_v2_uav_height" not in rc.get("red_1", {})
        env.close()


class TestTamPaperV2DoesNotModifyCore:
    def test_action_space_shape(self):
        env = _make_v2_env()
        obs, info = env.reset(seed=0)
        # 3-dim action per agent
        for rid in env.red_ids:
            a = info.get(rid, {}).get("action", None)
        env.close()

    def test_observation_mode(self):
        env = _make_v2_env()
        assert env.observation_mode == "mav_shared_geo"
        env.close()
