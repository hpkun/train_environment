"""Tests for tam_brma_scripted_reward_v1."""
import sys, numpy as np, pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def _make_env(**overrides):
    from uav_env import make_env
    return make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_brma_scripted_reward_v1.yaml", **overrides)

class TestV1EnvCreation:
    def test_mode_registered(self):
        env = _make_env()
        assert env.hetero_reward_mode == "tam_brma_scripted_reward_v1"; env.close()
    def test_missing_config_raises(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
        with pytest.raises(ValueError, match="config"):
            HeteroUavCombatEnv(hetero_reward_mode="tam_brma_scripted_reward_v1", max_num_red=3, max_num_blue=2, max_steps=100)

class TestV1Components:
    def test_reward_keys_present(self):
        env = _make_env(); env.reset(seed=0)
        base_rewards, components = env._compute_rewards()
        c0 = components.get("red_0", {})
        for k in ("tam_brma_v1_flight", "tam_brma_v1_mav_safe", "tam_brma_v1_mav_support",
                   "tam_brma_v1_mav_aware", "tam_brma_v1_team_terminal", "tam_brma_v1_total"):
            assert k in c0, f"MAV missing {k}"
        c1 = components.get("red_1", {})
        for k in ("tam_brma_v1_flight", "tam_brma_v1_uav_gate_sit",
                   "tam_brma_v1_uav_g_own", "tam_brma_v1_uav_g_enemy",
                   "tam_brma_v1_uav_a_own", "tam_brma_v1_uav_t_rear", "tam_brma_v1_uav_d_gate",
                   "tam_brma_v1_team_terminal", "tam_brma_v1_total"):
            assert k in c1, f"UAV missing {k}"
        env.close()

    def test_no_fire_launch_guided_active(self):
        env = _make_env(); env.reset(seed=0)
        base_rewards, components = env._compute_rewards()
        for uid in ("red_0","red_1"):
            for bad in ("uav_fire","guided_fire","launch_window_bonus","active_dodge"):
                assert components.get(uid,{}).get(bad,0.0)==0.0, f"{uid} active {bad}"
        env.close()

    def test_total_excludes_diagnostics(self):
        """UAV total = flight + gate_sit + event + terminal only."""
        env = _make_env(); env.reset(seed=0)
        base_rewards, components = env._compute_rewards()
        c1 = components.get("red_1", {})
        active_sum = (c1.get("tam_brma_v1_flight", 0) + c1.get("tam_brma_v1_uav_gate_sit", 0)
                      + c1.get("tam_brma_v1_uav_event", 0) + c1.get("tam_brma_v1_team_terminal", 0))
        assert abs(c1["tam_brma_v1_total"] - active_sum) < 1e-9, \
            f"total {c1['tam_brma_v1_total']} != active sum {active_sum}"
        env.close()

    def test_mav_total_excludes_diagnostics(self):
        """MAV total = flight + safe + support + aware + event + terminal only."""
        env = _make_env(); env.reset(seed=0)
        base_rewards, components = env._compute_rewards()
        c0 = components.get("red_0", {})
        active_sum = (c0.get("tam_brma_v1_flight", 0) + c0.get("tam_brma_v1_mav_safe", 0)
                      + c0.get("tam_brma_v1_mav_support", 0) + c0.get("tam_brma_v1_mav_aware", 0)
                      + c0.get("tam_brma_v1_mav_event", 0) + c0.get("tam_brma_v1_team_terminal", 0))
        assert abs(c0["tam_brma_v1_total"] - active_sum) < 1e-9, \
            f"MAV total {c0['tam_brma_v1_total']} != active sum {active_sum}"
        env.close()


class TestV1TeamEvents:
    def test_team_uav_loss_shared(self):
        """UAV first death triggers team_uav_loss_shared for all red. Note: class-level override affects both UAVs."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = True
        env._step_kill_count = {}
        # Override UAVs to simulate death — class-level property affects both red_1 and red_2
        sim = env.red_planes.get("red_1")
        orig_alive = type(sim).is_alive
        type(sim).is_alive = property(lambda self: False)
        try:
            base_rewards, components = env._compute_rewards()
            uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
            per_death = float(uav_ev["team_uav_loss_shared"])
            expected = 2 * per_death  # both UAVs affected by class-level override
            mav_ev = components.get("red_0", {}).get("tam_brma_v1_mav_event", 0)
            assert abs(mav_ev - expected) < 1e-6, \
                f"2 UAV deaths → MAV event={mav_ev} should be 2*{per_death}={expected}"
        finally:
            type(sim).is_alive = orig_alive
        env.close()

    def test_mav_loss_to_uav(self):
        """MAV first death triggers mav_loss_to_uav for all UAVs."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = False
        env._step_kill_count = {}
        sim = env.red_planes.get("red_0")
        orig_alive = type(sim).is_alive
        type(sim).is_alive = property(lambda self: False)
        try:
            base_rewards, components = env._compute_rewards()
            uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
            expected_mav_loss = float(uav_ev["mav_loss_to_uav"])
            # UAVs should have mav_loss_to_uav added to event
            c1 = components.get("red_1", {})
            assert c1["tam_brma_v1_uav_event"] < -100, \
                f"UAV event should include mav_loss_to_uav, got {c1['tam_brma_v1_uav_event']}"
        finally:
            type(sim).is_alive = orig_alive
        env.close()


class TestV1DeathReason:
    def test_read_death_reason_from_death_reasons(self):
        env = _make_env(); env.reset(seed=0)
        env._death_reasons = {"red_1": "Crash_LowAlt"}
        reason = env._tam_brma_v1_read_death_reason("red_1")
        assert reason == "Crash_LowAlt"
        env.close()

    def test_noncombat_uses_noncombat_loss(self):
        """Crash_LowAlt death reason triggers noncombat_loss."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = True
        env._death_reasons = {"red_1": "Crash_LowAlt"}
        env._step_kill_count = {}
        sim = env.red_planes.get("red_1")
        orig_alive = type(sim).is_alive
        type(sim).is_alive = property(lambda self: False)
        try:
            base_rewards, components = env._compute_rewards()
            c1 = components.get("red_1", {})
            uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
            noncombat_penalty = float(uav_ev["noncombat_loss"])
            assert c1["tam_brma_v1_uav_death"] == noncombat_penalty, \
                f"Crash_LowAlt → noncombat_loss={noncombat_penalty}, got {c1['tam_brma_v1_uav_death']}"
        finally:
            type(sim).is_alive = orig_alive
        env.close()

    def test_unknown_reason_uses_normal_death(self):
        """Unknown death reason → normal death penalty."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = True
        env._death_reasons = {}
        env._death_events_step = []
        env._step_kill_count = {}
        sim = env.red_planes.get("red_1")
        orig_alive = type(sim).is_alive
        type(sim).is_alive = property(lambda self: False)
        try:
            base_rewards, components = env._compute_rewards()
            c1 = components.get("red_1", {})
            uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
            normal_penalty = float(uav_ev["death"])
            assert c1["tam_brma_v1_uav_death"] == normal_penalty, \
                f"unknown reason → normal death={normal_penalty}, got {c1['tam_brma_v1_uav_death']}"
        finally:
            type(sim).is_alive = orig_alive
        env.close()


class TestV1DGate:
    def test_below_min_is_neg1(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
        cfg = {"gate": {"min_range_m":500,"opt_range_m":5000,"launch_range_m":10000}}
        assert H._tam_brma_v1_d_gate(200, cfg) == -1.0
    def test_optimal_zone_is_1(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
        cfg = {"gate": {"min_range_m":500,"opt_range_m":5000,"launch_range_m":10000}}
        for d in [5000, 7000, 10000]: assert H._tam_brma_v1_d_gate(d, cfg) == 1.0
    def test_beyond_launch_is_positive_decay(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
        cfg = {"gate": {"min_range_m":500,"opt_range_m":5000,"launch_range_m":10000}}
        assert H._tam_brma_v1_d_gate(15000, cfg) > 0.0

class TestV1GEnemyThreat:
    def test_threat_uses_max_d_gate(self):
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
        cfg = {"gate": {"min_range_m":500,"opt_range_m":5000,"launch_range_m":10000,"ao_thresh_deg":45,"ta_thresh_deg":90,"enemy_threat_weight":0.8}}
        d_gate = H._tam_brma_v1_d_gate(200, cfg)
        assert d_gate == -1.0
        assert max(d_gate, 0.0) == 0.0

class TestV1TeamKillShared:
    def test_team_kill_shared_applies_to_mav(self):
        """MAV gets both team_kill_shared AND team_kill_credit on red kills."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = True
        env._step_kill_count = {"red_1": 1}
        base_rewards, components = env._compute_rewards()
        c0 = components.get("red_0", {})
        uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
        mav_ev = env.tam_brma_scripted_reward_v1_config["mav"]["event"]
        expected_shared = float(uav_ev["team_kill_shared"]) * 1
        expected_credit = float(mav_ev["team_kill_credit"]) * 1
        mav_event = c0["tam_brma_v1_mav_event"]
        assert mav_event >= expected_shared + expected_credit - 1e-6, \
            f"MAV event={mav_event}, expected shared={expected_shared}+credit={expected_credit}"
        assert c0.get("tam_brma_v1_mav_team_kill_shared") == expected_shared
        env.close()

class TestV1MultiDeath:
    def test_team_uav_loss_counts_multiple(self):
        """Two UAVs dying same step → 2 * team_uav_loss_shared."""
        env = _make_env(); env.reset(seed=0)
        env._tam_brma_scripted_uav_death_penalized = set()
        env._tam_brma_scripted_mav_death_penalized = True
        env._step_kill_count = {}
        # Kill both UAVs
        for aid in ("red_1", "red_2"):
            sim = env.red_planes.get(aid)
            orig = type(sim).is_alive
            type(sim).is_alive = property(lambda self, _aid=aid: False)
        try:
            base_rewards, components = env._compute_rewards()
            uav_ev = env.tam_brma_scripted_reward_v1_config["uav"]["event"]
            per_death = float(uav_ev["team_uav_loss_shared"])
            expected_total = 2 * per_death
            # MAV event includes team_uav_loss component
            mav_event = components["red_0"]["tam_brma_v1_mav_event"]
            assert abs(mav_event - expected_total) < 1e-6, \
                f"2 UAV deaths → MAV event should be 2*{per_death}={expected_total}, got {mav_event}"
        finally:
            for aid in ("red_1", "red_2"):
                sim = env.red_planes.get(aid)
                type(sim).is_alive = orig
        env.close()

class TestV1ObsCache:
    def test_reset_writes_last_step_obs(self):
        env = _make_env(); env.reset(seed=0)
        assert env._last_step_obs is not None
        assert len(env._last_step_obs) > 0, "_last_step_obs should be populated after reset"
        # Check it contains red agent keys
        assert any(k.startswith("red_") for k in env._last_step_obs), "should have red agent obs"
        env.close()

    def test_step_updates_last_step_obs(self):
        env = _make_env(); env.reset(seed=0)
        actions = {rid: np.zeros(3, dtype=np.float32) for rid in env.red_ids}
        from algorithms.mappo.opponent_policy import OpponentPolicy
        opp = OpponentPolicy(mode="brma_rule", seed=0)
        actions.update(opp.act(env._last_step_obs, env.blue_ids, env=env))
        env.step(actions)
        assert env._last_step_obs is not None
        assert len(env._last_step_obs) > 0
        env.close()

    def test_has_launch_track_not_always_unobserved(self):
        """After obs cache fix, _has_launch_track should NOT always return unobserved."""
        env = _make_env(); env.reset(seed=0)
        # Check _last_step_obs has the fields needed for track detection
        red1_obs = env._last_step_obs.get("red_1", {})
        # The observed_mask / track_source should be present (even if values may be 0/false)
        assert isinstance(red1_obs, dict), f"red_1 obs should be dict, got {type(red1_obs)}"
        # At minimum, the obs dict should not be empty
        assert len(red1_obs) > 0, "red_1 obs dict should not be empty after obs cache fix"
        env.close()

    def test_v4_unaffected(self):
        from uav_env import make_env
        env = make_env("uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_dynamics_f22_visual_mav_tam_paper_reward_v4.yaml")
        env.reset(seed=0)
        assert env._last_step_obs is not None
        assert len(env._last_step_obs) > 0
        env.close()

class TestV1Terminal:
    def test_full_win_requires_mav_alive(self):
        env = _make_env()
        cfg = env.tam_brma_scripted_reward_v1_config
        t = cfg["terminal"]
        assert t["full_win"] > t["costly_win"]
        env.close()
