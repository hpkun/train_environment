import inspect
import numpy as np
import pytest
from gymnasium import spaces
from aircombat_env_v1.combat import action_to_targets
from aircombat_env_v1.config import load_config
from aircombat_env_v1.paper_missile import PaperMissile
from aircombat_env_v1.simple_env import INFERRED,PUBLISHED,SimplePIDAircraft,SimpleTAMCombatEnv

def test_simple_env_creates_1v1_and_2v2():
    for mode,count in (("simple_paper_1v1",2),("simple_paper_2v2",4)):
        env=SimpleTAMCombatEnv(mode);env.reset(seed=1);assert len(env.agents)==count;env.close()

def test_simple_action_spaces_are_three_dimensional_boxes_without_fire():
    one=SimpleTAMCombatEnv("simple_paper_1v1");two=SimpleTAMCombatEnv("simple_paper_2v2")
    assert set(one.action_space)=={"red_0"} and set(two.action_space)=={"red_0","red_1"}
    assert all(isinstance(space,spaces.Box) and space.shape==(3,) for space in list(one.action_space.values())+list(two.action_space.values()))
    assert "fire" not in inspect.getsource(SimpleTAMCombatEnv.step);one.close();two.close()

def test_action_to_targets_mapping_is_unchanged():
    p,h,s=action_to_targets(np.array([1.,1.,1.]),0.);assert np.rad2deg(p)==pytest.approx(20) and np.rad2deg(h)==pytest.approx(60) and s==300
    p,h,s=action_to_targets(np.array([-1.,-1.,-1.]),0.);assert np.rad2deg(p)==pytest.approx(-20) and np.rad2deg(h)==pytest.approx(-60) and s==200

def test_pid_reads_existing_configuration_exactly():
    config=load_config();a=SimplePIDAircraft("a","red",120.,60.,0.,config)
    try:
        assert a.autopilot.roll_pid.kp==config["pid"]["roll"]["kp"]
        assert a.autopilot.pitch_pid.kd==config["pid"]["pitch"]["kd"]
        assert a.autopilot.speed_pid.ki==config["pid"]["speed"]["ki"]
    finally:a.close()

def test_pid_runs_each_of_twelve_physics_frames(monkeypatch):
    env=SimpleTAMCombatEnv("simple_paper_1v1","all",max_steps=1,weapon_enabled_agent_ids=set());env.reset(seed=1);counts={a.agent_id:0 for a in env.agents}
    for a in env.agents:
        original=a.autopilot.step
        def wrapped(*args,original=original,aid=a.agent_id,**kwargs):counts[aid]+=1;return original(*args,**kwargs)
        monkeypatch.setattr(a.autopilot,"step",wrapped)
    env.step({aid:np.zeros(3,np.float32) for aid in env.controlled_ids});assert set(counts.values())=={12};env.close()

def test_default_and_diagnostic_weapon_enablement():
    assert SimpleTAMCombatEnv().weapon_enabled_agent_ids is None
    env=SimpleTAMCombatEnv("simple_paper_1v1","all",max_steps=1,weapon_enabled_agent_ids={"red_0"});env.reset(seed=1);red,blue=env.by_id["red_0"],env.by_id["blue_0"];blue.position=red.position+np.array([1000.,0,0])
    _,_,_,_,info=env.step(env.build_rule_actions());assert info["missiles_fired"]==1 and blue.missile_left==2;env.close()

def test_paper_weapon_limits_are_unchanged():
    assert PUBLISHED["maximum_attack_range_m"]==14000 and PUBLISHED["launch_interval_s"]==25
    assert PUBLISHED["maximum_overload_g"]==30 and PUBLISHED["navigation_gain_y"]==PUBLISHED["navigation_gain_z"]==3
    assert INFERRED["missile_initial_speed_mps"]==500 and INFERRED["hit_radius_m"]==60

def test_flat_observation_dimensions_and_finiteness():
    for mode,dim in (("simple_paper_1v1",61),("simple_paper_2v2",73)):
        env=SimpleTAMCombatEnv(mode);obs,_=env.reset(seed=1);assert all(x.shape==(dim,) and x.dtype==np.float32 and np.isfinite(x).all() for x in obs.values());env.close()

def test_rewards_are_finite():
    env=SimpleTAMCombatEnv("simple_paper_1v1",max_steps=1,weapon_enabled_agent_ids=set());env.reset(seed=1);_,rewards,_,_,_=env.step({"red_0":np.zeros(3,np.float32)});assert np.isfinite(list(rewards.values())).all();env.close()

def test_nearest_live_target_and_reselection():
    env=SimpleTAMCombatEnv("simple_paper_2v2","all");env.reset(seed=1);assert env.current_targets["red_0"]=="blue_0"
    env.by_id["blue_0"].kill("shotdown");env._update_targets();assert env.current_targets["red_0"]=="blue_1" and env.valid_target_reselections>=1;env.close()

@pytest.mark.parametrize("reason",["crash","boundary","numerical_invalid"])
def test_non_missile_deaths_do_not_count_as_missile_kills(reason):
    env=SimpleTAMCombatEnv("simple_paper_1v1","all");env.reset(seed=1);env.by_id["blue_0"].kill(reason);info=env._info(None,None)
    assert info["red_missile_kills"]==0 and info["missile_hits"]==0;env.close()

def test_same_frame_bilateral_hits_are_draw():
    env=SimpleTAMCombatEnv("simple_paper_1v1","all",max_steps=2);env.reset(seed=1);red,blue=env.by_id["red_0"],env.by_id["blue_0"];cfg={**PUBLISHED,**INFERRED}
    env.weapon.missiles=[PaperMissile("m1","red_0","blue_0",blue.position.copy(),np.array([1,0,0]),cfg),PaperMissile("m2","blue_0","red_0",red.position.copy(),np.array([1,0,0]),cfg)]
    _,_,terminated,_,info=env.step(env.build_rule_actions());assert terminated and info["winner"]=="draw" and info["simultaneous_kills"]==1;env.close()

def test_configured_episode_limit_produces_timeout():
    env=SimpleTAMCombatEnv("simple_paper_1v1","all",max_steps=1,weapon_enabled_agent_ids=set());env.reset(seed=1);_,_,terminated,truncated,info=env.step(env.build_rule_actions())
    assert not terminated and truncated and info["termination_reason"]=="timeout";env.close()

@pytest.mark.integration
def test_single_aircraft_pid_level_1000_is_finite_and_alive():
    from aircombat_env_v1.scripts.check_simple_environment import run_action
    result=run_action([0,0,0],1000);assert result["completed_steps"]==1000 and not result["crash"] and not result["numerical_invalid"]

@pytest.mark.integration
def test_basic_action_directions_are_opposite_and_finite():
    from aircombat_env_v1.scripts.check_simple_environment import run_checks
    result=run_checks();assert all(result["direction_checks"].values())
    assert not any(x["numerical_invalid"] for x in result["basic_actions_50"].values())
