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
        """d<500 gives d_gate=-1; max(-1,0)=0, so g_enemy=0 — no negative threat exploit."""
        from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv as H
        cfg = {"gate": {"min_range_m":500,"opt_range_m":5000,"launch_range_m":10000,"ao_thresh_deg":45,"ta_thresh_deg":90,"enemy_threat_weight":0.8}}
        d_gate = H._tam_brma_v1_d_gate(200, cfg)
        assert d_gate == -1.0
        assert max(d_gate, 0.0) == 0.0  # threat clamped to zero

class TestV1Terminal:
    def test_full_win_requires_mav_alive(self):
        env = _make_env()
        cfg = env.tam_brma_scripted_reward_v1_config
        t = cfg["terminal"]
        assert t["full_win"] > t["costly_win"], "full_win must be higher than costly_win"
        env.close()
