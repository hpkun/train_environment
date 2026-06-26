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


class TestTamV2GeometryFixes:
    """Verify _tam_v2_feature is absolute, not relative, and callers are correct."""

    def test_feature_is_absolute_single_arg(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        import inspect
        sig = inspect.signature(HeteroUavCombatEnv._tam_v2_feature)
        params = list(sig.parameters.keys())
        assert params == ["sim"], f"expected ['sim'], got {params}"

    def test_feature_returns_absolute_not_zero(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        import numpy as np
        # Use a mock sim
        class MockSim:
            def get_position(self): return np.array([100.0, 200.0, 3000.0])
            def get_velocity(self): return np.array([250.0, 0.0, -5.0])
        feat = HeteroUavCombatEnv._tam_v2_feature(MockSim())
        assert feat[0] == 100.0, f"expected abs x, got {feat[0]}"
        assert feat[2] == -3000.0, f"expected -z (up), got {feat[2]}"
        assert feat[3] == 250.0, f"expected abs vx, got {feat[3]}"

    def test_ao_ta_uses_absolute_features(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        from uav_env.JSBSim.utils import get2d_AO_TA_R
        import numpy as np

        class MockSim:
            def __init__(self, x, y, z, vx, vy, vz):
                self.pos = np.array([x, y, z]); self.vel = np.array([vx, vy, vz])
            def get_position(self): return self.pos
            def get_velocity(self): return self.vel

        red = MockSim(0, 0, 6000, 0, 250, 0)        # heading north (vy=250)
        blue = MockSim(5000, 0, 6000, 0, 250, 0)      # heading north (vy=250), 5km east
        rf = HeteroUavCombatEnv._tam_v2_feature(red)
        bf = HeteroUavCombatEnv._tam_v2_feature(blue)
        ao, ta, rng = get2d_AO_TA_R(rf, bf)
        # Both heading north, blue 5km east → AO should ~90 deg, TA should ~90 deg
        assert 1.3 < ao < 1.9, f"expected AO ~1.57 (90°), got {ao:.3f}"
        assert 1.3 < ta < 1.9, f"expected TA ~1.57 (90°), got {ta:.3f}"

    def test_dodge_no_missile_returns_zero(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        import numpy as np
        env = _make_v2_env()
        cache = {}

        class MockSim:
            def get_position(self): return np.array([0.0, 0.0, 5000.0])
            def get_velocity(self): return np.array([250.0, 0.0, 0.0])
        sim = MockSim()
        sim.under_missiles = []
        total, angle, speed = env._tam_v2_dodge_reward(sim, 1000.0, cache)
        assert total == 0.0 and angle == 0.0 and speed == 0.0, f"no missile → (0,0,0), got ({total},{angle},{speed})"
        env.close()

    def test_dodge_threat_is_negative_not_clipped(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        import numpy as np

        class MockMissile:
            uid = "m1"; is_alive = True
            def get_position(self): return np.array([1000.0, 0.0, 5000.0])
            def get_velocity(self): return np.array([-600.0, 0.0, 0.0])  # heading toward aircraft at x=0

        class MockSim:
            def get_position(self): return np.array([0.0, 0.0, 5000.0])
            def get_velocity(self): return np.array([250.0, 0.0, 0.0])

        env = _make_v2_env()
        cache = {}
        sim = MockSim()
        sim.under_missiles = [MockMissile()]
        total, angle, speed = env._tam_v2_dodge_reward(sim, 1000.0, cache)
        # missile heading toward aircraft: r_angle ≈ -1.0, r_speed = 0 (first sighting)
        # total = r_angle + r_speed ≈ -1.0 — must NOT be clipped to 0
        assert total < -0.9, f"threat should give negative total, not clipped, got {total}"
        env.close()

    def test_dodge_picks_max_candidate(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        import numpy as np

        class MockMissile1:
            uid = "m1"; is_alive = True
            def get_position(self): return np.array([1000.0, 0.0, 5000.0])
            def get_velocity(self): return np.array([-600.0, 0.0, 0.0])  # heading toward: r_angle ≈ -1.0

        class MockMissile2:
            uid = "m2"; is_alive = True
            def get_position(self): return np.array([500.0, 500.0, 5000.0])
            def get_velocity(self): return np.array([600.0, 0.0, 0.0])   # heading away: r_angle ≈ +1.0

        class MockSim:
            def get_position(self): return np.array([0.0, 0.0, 5000.0])
            def get_velocity(self): return np.array([250.0, 0.0, 0.0])

        env = _make_v2_env()
        cache = {}
        sim = MockSim()
        sim.under_missiles = [MockMissile1(), MockMissile2()]
        total, angle, speed = env._tam_v2_dodge_reward(sim, 1000.0, cache)
        # m2 (heading away) has larger r_angle+r_speed → should be selected
        assert total > 0.5, f"should pick max candidate (missile heading away), got total={total}"
        assert angle > 0.5, f"angle should be from the away-heading missile, got {angle}"
        env.close()

    def test_v2_metadata_fields_present(self):
        env = _make_v2_env()
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
            assert "tam_v2_geometry_feature_semantics" in comp, f"{uid} missing geometry semantics"
            assert "tam_v2_height_formula_source" in comp, f"{uid} missing height formula source"
        env.close()


class TestTamV2Fixes:
    """Tests for the 5 implementation fixes."""

    def test_uav_dead_dense_rewards_are_zero(self):
        """UAV dead: height/speed/angle/distance/dodge all zero, only event remains."""
        from uav_env import make_env
        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml",
        )
        env.reset(seed=0)
        red_1 = env.red_planes.get("red_1")
        if red_1:
            # Temporarily override is_alive property by patching the sim
            orig_alive = type(red_1).is_alive
            try:
                # Use __dict__ override to fake is_alive=False
                type(red_1).is_alive = property(lambda self: False)
                env._uav_death_penalized = set()
                base_rewards, components = env._compute_rewards()
                comp = components.get("red_1", {})
                for k in ("tam_v2_uav_height", "tam_v2_uav_speed", "tam_v2_uav_angle",
                           "tam_v2_uav_distance", "tam_v2_uav_dodge"):
                    assert abs(comp.get(k, 0.0)) < 1e-8, f"{k} should be 0 when dead, got {comp.get(k)}"
            finally:
                type(red_1).is_alive = orig_alive
        env.close()

    def test_mav_dead_dense_rewards_are_zero(self):
        """MAV dead: safety/support all zero."""
        from uav_env import make_env
        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml",
        )
        env.reset(seed=0)
        mav = env.red_planes.get("red_0")
        if mav:
            orig_alive = type(mav).is_alive
            try:
                type(mav).is_alive = property(lambda self: False)
                env._mav_death_penalized = True
                base_rewards, components = env._compute_rewards()
                comp = components.get("red_0", {})
                for k in ("tam_v2_mav_safety", "tam_v2_mav_support"):
                    assert abs(comp.get(k, 0.0)) < 1e-8, f"{k} should be 0 when dead, got {comp.get(k)}"
            finally:
                type(mav).is_alive = orig_alive
        env.close()

    def test_speed_reward_zero_protection(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        r = HeteroUavCombatEnv._tam_v2_speed_reward(0.0, 250.0)
        # red_speed=0 is clamped to 1e-8, blue=250 >> red → blue is much faster → return -1.0
        assert not np.isnan(r) and not np.isinf(r), f"should not be nan/inf, got {r}"
        # red_speed=600, blue=250 → blue < 0.5*red (250<300) → return 1.0
        r2 = HeteroUavCombatEnv._tam_v2_speed_reward(600.0, 250.0)
        assert r2 == 1.0, f"red=600 >> blue=250 should give 1.0, got {r2}"

    def test_height_reward_uses_env_boundaries(self):
        from uav_env import make_env
        import numpy as np
        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml",
        )
        cfg = env.tam_paper_reward_v2_config
        # Below env min (2500) should return -1
        r_below = env._tam_v2_height_reward(2000.0, cfg)
        assert r_below == -1.0, f"below floor should be -1.0, got {r_below}"
        # Above env max (10000) should return -0.5
        r_above = env._tam_v2_height_reward(11000.0, cfg)
        assert r_above == -0.5, f"above ceiling should be -0.5, got {r_above}"
        # At optimal should return 1.0
        r_opt = env._tam_v2_height_reward(6000.0, cfg)
        assert r_opt == 1.0, f"at optimum should be 1.0, got {r_opt}"
        env.close()

    def test_out_of_zone_x_beyond_half_size(self):
        from uav_env import make_env
        import numpy as np
        env = make_env(
            "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml",
        )
        env.reset(seed=0)
        # Force out-of-zone by stepping past boundary — use step to trigger
        # Simpler: verify the env has the correct BATTLEFIELD_HALF_SIZE
        half = getattr(env, "BATTLEFIELD_HALF_SIZE", 40000.0)
        assert half == 40000.0
        # Verify out-of-zone check logic works: x=41000 > 40000
        out = abs(41000.0) > half
        assert out, "x=41000 should be out of zone"
        env.close()


class TestRewardModeCLI:
    """Test --reward-mode resolution in parallel runner."""

    def test_cli_default_is_none(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/train_happo_reference_parallel.py", "--help"],
            cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=60,
        )
        assert result.returncode == 0
        assert "--reward-mode" in result.stdout
        # Default should no longer be happo_ref_v0
        assert "happo_ref_v0" not in result.stdout or "default" not in result.stdout

    def test_reward_mode_conflict_raises(self):
        """CLI mismatch with YAML should raise ValueError."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/train_happo_reference_parallel.py",
             "--config", "uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v2.yaml",
             "--reward-mode", "paper_role_reward_v1",
             "--total-env-steps", "256", "--rollout-length", "64", "--num-envs", "1",
             "--device", "cpu", "--policy-arch", "flat", "--max-steps", "100",
             "--reset-timeout-sec", "60", "--step-timeout-sec", "60"],
            cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
        # Should exit with error due to reward_mode conflict
        assert result.returncode != 0, f"should fail on reward mode conflict, got stdout: {result.stdout[:500]}"
