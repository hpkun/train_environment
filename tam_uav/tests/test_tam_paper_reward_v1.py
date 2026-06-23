"""Test tam_paper_reward_v1 reward components."""
from __future__ import annotations
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np


def test_env_creates_with_tam_paper_reward_v1():
    from uav_env import make_env
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"),
                   env_type="jsbsim_hetero", hetero_reward_mode="tam_paper_reward_v1")
    assert env.hetero_reward_mode == "tam_paper_reward_v1"
    env.close()


def test_reward_components_finite():
    from uav_env import make_env
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"),
                   env_type="jsbsim_hetero", hetero_reward_mode="tam_paper_reward_v1")
    obs, info = env.reset(seed=42)
    for _ in range(10):
        actions = {rid: np.random.randint(0, 40, 4).astype(np.int64) for rid in env.red_ids}
        obs, rewards, term, trunc, info = env.step(actions)
        for rid in env.red_ids:
            r = rewards.get(rid, 0.0)
            assert np.isfinite(r), f"Non-finite reward for {rid}"
        if all(term.values()) or all(trunc.values()):
            obs, info = env.reset(seed=42)
    env.close()


def test_uav_angle_favors_launch_geometry():
    """AO=0, TA=pi (tail chase) should have higher angle reward than AO=0, TA=0."""
    from uav_env.JSBSim.utils import get2d_AO_TA_R
    ego = np.array([0, 0, 0, 250, 0, 0], dtype=np.float64)
    enm_tail = np.array([1000, 0, 0, 250, 0, 0], dtype=np.float64)
    ao_tail, ta_tail, _ = get2d_AO_TA_R(ego, enm_tail)
    aa_tail = np.pi - ta_tail
    r_tail = 1.0 - (ao_tail + aa_tail) / np.pi
    enm_head = np.array([1000, 0, 0, -250, 0, 0], dtype=np.float64)
    ao_head, ta_head, _ = get2d_AO_TA_R(ego, enm_head)
    aa_head = np.pi - ta_head
    r_head = 1.0 - (ao_head + aa_head) / np.pi
    assert r_tail > r_head, f"Tail chase {r_tail:.4f} > head-on {r_head:.4f}"


def test_uav_angle_no_longer_rewards_small_ta():
    """With AA=pi-TA, small TA -> AA~pi -> low reward."""
    from uav_env.JSBSim.utils import get2d_AO_TA_R
    ego = np.array([0, 0, 0, 250, 0, 0], dtype=np.float64)
    enm = np.array([1000, 0, 0, -250, 0, 0], dtype=np.float64)
    ao, ta, _ = get2d_AO_TA_R(ego, enm)
    old_r = 1.0 - (ao + ta) / np.pi
    new_r = ta / np.pi - ao / np.pi
    assert old_r > new_r, f"Old gave {old_r:.4f} for head-on; new gives {new_r:.4f}"


def test_uav_distance_reward():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv.__new__(HeteroUavCombatEnv)
    env.tam_paper_reward_config = {
        "geometry": {"min_altitude_m": 750, "optimal_altitude_m": 6000, "max_altitude_m": 12000,
                      "combat_zone_radius_m": 50000, "missile_range_m": 14000, "max_speed_mps": 400},
        "uav": {"reward_weights": {"height": 10, "speed": 10, "angle": 15, "distance": 10, "dodge": 30},
                "event": {"kill_enemy": 200, "death": -200, "out_of_zone": -100}, "v_norm_mps": 1000},
    }
    r5 = env._tam_paper_uav_distance_reward(5000.0)
    r12 = env._tam_paper_uav_distance_reward(12000.0)
    assert r5 > r12, f"5km {r5:.4f} > 12km {r12:.4f}"


def test_height_penalizes_below_crash_floor():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv.__new__(HeteroUavCombatEnv)
    env.tam_paper_reward_config = {
        "geometry": {"min_altitude_m": 750, "optimal_altitude_m": 6000, "max_altitude_m": 12000,
                      "combat_zone_radius_m": 50000, "missile_range_m": 14000, "max_speed_mps": 400},
    }
    env.BATTLEFIELD_ALTITUDE_MIN = 2500.0
    r2000 = env._tam_paper_height_reward(2000.0)
    r6000 = env._tam_paper_height_reward(6000.0)
    assert r2000 == -1.0, f"2000m below floor -> -1, got {r2000}"
    assert r6000 > 0, f"6000m > 0, got {r6000}"


def test_mav_death_one_shot():
    from uav_env.JSBSim.envs.hetero_uav_combat_env import HeteroUavCombatEnv
    env = HeteroUavCombatEnv.__new__(HeteroUavCombatEnv)
    env.tam_paper_reward_config = {
        "global_scale": 1.0,
        "geometry": {"min_altitude_m": 750, "optimal_altitude_m": 6000, "max_altitude_m": 12000,
                      "combat_zone_radius_m": 50000, "missile_range_m": 14000, "max_speed_mps": 400},
        "mav": {"d_danger_m": 5000, "d_safe_m": 14000, "d_opt_m": 10000, "d_max_m": 30000,
                "death_penalty": 200, "team_kill_bonus": 200, "team_kill_bonus_cap": 200,
                "safety_weights": {"dist": 0.5, "threat": 0.3, "aspect": 0.2},
                "support_weights": {"pos": 0.6, "aware": 0.4}},
    }
    env._mav_death_penalized = False
    env.red_ids = ["red_0", "red_1", "red_2"]
    env._step_kill_count = {}
    env.agent_roles = {"red_0": "mav"}
    dead_mav = type('obj',(object,),{'is_alive':False,'get_position':lambda:np.zeros(3)})()
    r1, v1 = env._tam_paper_mav_reward("red_0", dead_mav, [])
    assert v1["tam_mav_death"] == -200, f"First death -> -200, got {v1['tam_mav_death']}"
    r2, v2 = env._tam_paper_mav_reward("red_0", dead_mav, [])
    assert v2["tam_mav_death"] == 0.0, f"Second death -> 0, got {v2['tam_mav_death']}"


def test_kill_reward_scale():
    from uav_env import make_env
    env = make_env(str(ROOT / "uav_env/JSBSim/configs/tam_happo_f22_3v2_direct.yaml"),
                   env_type="jsbsim_hetero", hetero_reward_mode="tam_paper_reward_v1")
    rcfg = env.tam_paper_reward_config
    assert rcfg["uav"]["event"]["kill_enemy"] == 200
    assert rcfg["uav"]["event"]["death"] == -200
    assert rcfg["uav"]["event"]["out_of_zone"] == -100
    env.close()
