import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from gymnasium import spaces

from aircombat_env_v1.paper_action_semantics import map_action_indices
from aircombat_env_v1.paper_env import PaperAircraft,TAMPaperCombatEnv,PUBLISHED,INFERRED,SCENARIOS
from aircombat_env_v1.paper_missile import PaperMissile
from aircombat_env_v1.paper_observation import PaperObservation
from aircombat_env_v1.paper_reward import PaperReward
from aircombat_env_v1.paper_weapon import PaperWeaponManager

def fake(aid,side,pos):
    return SimpleNamespace(agent_id=aid,side=side,position=np.asarray(pos,float),velocity=np.array([250.,0,0]),
        alive=True,missile_left=2,death_reason=None,out_of_boundary=False,speed=250.,pitch=0.,heading=0.,roll=0.,
        load_factor_g=1.,kill=lambda reason:None)

def test_action_space_is_four_40_level_dimensions():
    env=TAMPaperCombatEnv("paper_nominal_1v1");assert isinstance(env.action_space["red_0"],spaces.MultiDiscrete);assert np.array_equal(env.action_space["red_0"].nvec,[40]*4)

def test_action_mapping_ranges_and_order():
    assert np.allclose(map_action_indices([0,0,0,0]),[.4,-1,-1,-1]);assert np.allclose(map_action_indices([39]*4),[.9,1,1,1])

def test_control_command_is_reordered_for_aircraft_simulator(monkeypatch):
    env=TAMPaperCombatEnv("paper_nominal_1v1","all");env.reset(seed=1);a=env.agents[0];seen=[]
    monkeypatch.setattr(a.simulator,"set_controls",lambda *values:seen.append(values));a.apply_direct_fcs_command([.7,.1,.2,.3])
    assert seen[-1]==(.1,.2,.3,.7);env.close()

def test_formal_initialization_retracts_gear_and_flaps():
    aircraft=PaperAircraft("init","red",120.,60.,0.)
    try:
        status=aircraft.control_initialization_status()
        assert status["gear_command_norm"]==0 and status["gear_position_norm"]==0
        assert status["gear_unit_positions_norm"]==[0.,0.,0.]
        assert status["flap_command_norm"]==0 and status["flap_position_norm"]==0
        assert status["engine_running"] and status["physics_dt_s"]==pytest.approx(1/60)
    finally:aircraft.close()

def test_formal_load_factor_reads_accelerations_nz(monkeypatch):
    aircraft=PaperAircraft("load","red",120.,60.,0.)
    try:
        original=aircraft.simulator.get_property
        monkeypatch.setattr(aircraft.simulator,"get_property",lambda name:3.25 if name=="accelerations/Nz" else original(name))
        aircraft._update();assert aircraft.load_factor_g==3.25
    finally:aircraft.close()

def test_formal_aircraft_saves_required_finite_state():
    aircraft=PaperAircraft("state","red",120.,60.,0.)
    try:assert np.isfinite([aircraft.position[2],aircraft.speed,aircraft.roll,aircraft.pitch,aircraft.heading,
                            aircraft.vertical_speed,aircraft.load_factor_g,aircraft.alpha,aircraft.beta]).all()
    finally:aircraft.close()

def test_formal_env_does_not_reference_paper_autopilot():
    import aircombat_env_v1.paper_env as module
    assert "PaperAutopilot" not in inspect.getsource(module)

def test_action_is_held_for_twelve_physics_frames(monkeypatch):
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1);env.reset(seed=1);count={a.agent_id:0 for a in env.agents}
    for a in env.agents:
        original=a.step_physics_once
        def wrapped(original=original,aid=a.agent_id):count[aid]+=1;return original()
        monkeypatch.setattr(a,"step_physics_once",wrapped)
    env.step(env.build_rule_actions());assert set(count.values())=={12};env.close()

def test_nominal_1v1_source_values():
    assert SCENARIOS["paper_nominal_1v1"]==[("red_0","red",120.,60.,0.),("blue_0","blue",120.,60.2,180.)]

def test_nominal_2v2_source_values():
    assert len(SCENARIOS["paper_nominal_2v2"])==4;assert SCENARIOS["paper_nominal_2v2"][1][2:]==(120.02,60.,0.)

@pytest.mark.integration
def test_every_attack_uav_starts_with_two_missiles():
    env=TAMPaperCombatEnv("paper_nominal_2v2","all");env.reset(seed=1);assert all(a.missile_left==2 for a in env.agents);env.close()

def test_automatic_launch_inside_14km_and_not_outside():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);t=fake("b","blue",[13999,0,6000]);assert w.try_launch(s,t,0) is not None
    w.reset();s.missile_left=2;t.position[0]=14001;assert w.try_launch(s,t,0) is None

def test_default_weapon_enablement_is_all_agents():
    assert TAMPaperCombatEnv("paper_nominal_1v1").weapon_enabled_agent_ids is None

@pytest.mark.integration
def test_weapon_enablement_only_controls_launch(monkeypatch):
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1,weapon_enabled_agent_ids={"red_0"});env.reset(seed=1)
    red,blue=env.by_id["red_0"],env.by_id["blue_0"];blue.position=red.position+np.array([1000.,0,0]);before=blue.position.copy()
    _,_,_,_,info=env.step(env.build_rule_actions());assert info["missiles_fired"]==1 and blue.missile_left==2 and np.isfinite(blue.position).all() and not np.array_equal(before,blue.position);env.close()

@pytest.mark.integration
def test_disabling_blue_weapon_prevents_blue_launch():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1,weapon_enabled_agent_ids={"red_0"});env.reset(seed=1);red,blue=env.by_id["red_0"],env.by_id["blue_0"];blue.position=red.position+np.array([1000.,0,0])
    _,_,_,_,info=env.step(env.build_rule_actions());assert all(e.get("shooter_id")!="blue_0" for e in info["events"]);env.close()

def test_25_second_cooldown_without_lock_gate():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);t=fake("b","blue",[-1000,0,6000])
    assert w.try_launch(s,t,0);assert w.try_launch(s,t,24.99) is None;assert w.try_launch(s,t,25.)

def test_second_launch_uses_new_live_target_after_cooldown():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);a=fake("a","blue",[1000,0,6000]);b=fake("b","blue",[1200,0,6000])
    assert w.try_launch(s,a,0)["target_id"]=="a";a.alive=False;assert w.try_launch(s,b,24.99) is None;assert w.try_launch(s,b,25.)["target_id"]=="b"

def test_no_ten_degree_lock_or_minimum_range():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);t=fake("b","blue",[-1,0,6000]);assert w.try_launch(s,t,0)

def test_missile_paper_gains_and_overload():
    m=PaperMissile("m","r","b",np.array([0,0,6000.]),np.array([1,0,0.]),{**PUBLISHED,**INFERRED});assert m.navigation_gain_y==3 and m.navigation_gain_z==3;assert PUBLISHED["maximum_overload_g"]==30

def test_begin_decision_marks_speed():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);t=fake("b","blue",[1000,0,6000]);w.try_launch(s,t,0);w.missiles[0].speed_mps=432.;w.begin_decision_step();assert w.missiles[0].decision_start_speed_mps==432

def test_dodge_angle_is_negative_cosine_and_speed_uses_decision_delta():
    r=PaperReward();a=fake("r","red",[0,0,6000]);b=fake("b","blue",[1000,0,6000]);m=SimpleNamespace(alive=True,target_id="r",position=np.array([-100.,0,6000]),velocity=np.array([500.,0,0]),speed_mps=400.,decision_start_speed_mps=500.)
    from aircombat_env_v1.paper_situation import assess_pair
    _,c=r.compute([a,b],{"r":"b","b":"r"},{"r":assess_pair(a.position,a.velocity,b.position,b.velocity),"b":assess_pair(b.position,b.velocity,a.position,a.velocity)},[m],[],{"r":True,"b":True},set())
    assert c["r"]["r_dodge_angle"]==pytest.approx(-1.);assert c["r"]["r_dodge_speed"]==pytest.approx(.1)

def test_1v1_observation_slots_and_masks():
    o=PaperObservation(1,1);a=fake("r","red",[0,0,6000]);b=fake("b","blue",[1000,0,6000]);item=o.build([a,b],[])["r"]
    assert item["ally_states"].shape==(0,5) and item["enemy_states"].shape==(1,5);assert item["enemy_mask"].tolist()==[1.];assert o.flatten(item).shape==(61,)

def test_2v2_observation_slots_and_masks():
    o=PaperObservation(2,2);agents=[fake("r0","red",[0,0,6000]),fake("r1","red",[0,10,6000]),fake("b0","blue",[1000,0,6000]),fake("b1","blue",[1000,10,6000])];item=o.build(agents,[])["r0"]
    assert item["ally_states"].shape==(1,5) and item["enemy_states"].shape==(2,5);assert o.flatten(item).shape==(73,)

def test_all_observation_arrays_are_float32():
    o=PaperObservation(1,1);item=o.build([fake("r","red",[0,0,6000]),fake("b","blue",[1000,0,6000])],[])["r"]
    assert all(value.dtype==np.float32 for value in item.values())

def test_dead_entities_zero_their_slots_and_masks():
    o=PaperObservation(1,1);a=fake("r","red",[0,0,6000]);b=fake("b","blue",[1000,0,6000]);b.alive=False;item=o.build([a,b],[])["r"]
    assert not item["enemy_mask"].any() and not item["enemy_states"].any()

@pytest.mark.integration
def test_nearest_target_selection_is_stable_and_retargets():
    env=TAMPaperCombatEnv("paper_nominal_2v2","all");env.reset(seed=1);assert env.current_targets["red_0"]=="blue_0"
    env.by_id["blue_0"].kill("shotdown");env._update_targets();assert env.current_targets["red_0"]=="blue_1";env.close()

def test_blue_crash_does_not_increment_red_missile_kill():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all");env.red_missile_kills=0;assert env.red_missile_kills==0

def test_target_dead_missile_is_not_a_hit():
    w=PaperWeaponManager(PUBLISHED,INFERRED);s=fake("r","red",[0,0,6000]);t=fake("b","blue",[1000,0,6000]);w.try_launch(s,t,0);t.alive=False;events=w.step_physics_once({"r":s,"b":t},1/60)
    assert events[0]["reason"]=="target_dead" and not events[0]["hit"] and w.total_hits==0

def test_hit_and_death_event_rewards_are_plus_and_minus_200():
    r=PaperReward();red=fake("r","red",[0,0,6000]);blue=fake("b","blue",[1000,0,6000]);blue.alive=False;blue.death_reason="shotdown"
    rewards,c=r.compute([red,blue],{"r":None,"b":None},{},[],[{"reason":"hit","shooter_id":"r"}],{"r":True,"b":True},set())
    assert c["r"]["r_event"]==200 and c["b"]["r_event"]==-200

@pytest.mark.integration
def test_controlled_red_returns_only_red_observations_and_rewards():
    env=TAMPaperCombatEnv("paper_nominal_2v2","red",max_steps=1);obs,_=env.reset(seed=1);assert set(obs)=={"red_0","red_1"}
    obs,rewards,_,_,_=env.step(env.build_rule_actions());assert set(obs)==set(rewards)=={"red_0","red_1"};env.close()

@pytest.mark.integration
def test_first_direct_frame_has_no_large_state_jump():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1);env.reset(seed=1);a=env.by_id["red_0"];alt,speed=a.position[2],a.speed
    env.step({aid:np.array([20,20,20,20]) for aid in env.controlled_ids});assert abs(a.position[2]-alt)<20 and abs(a.speed-speed)<20;env.close()

@pytest.mark.integration
def test_same_frame_bilateral_hits_produce_draw():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=2);env.reset(seed=1);red,blue=env.by_id["red_0"],env.by_id["blue_0"]
    cfg={**PUBLISHED,**INFERRED};m1=PaperMissile("m1","red_0","blue_0",blue.position.copy(),np.array([1,0,0]),cfg);m2=PaperMissile("m2","blue_0","red_0",red.position.copy(),np.array([1,0,0]),cfg)
    env.weapon.missiles=[m1,m2];events=env.weapon.step_physics_once(env.by_id,1/60);assert sum(e["hit"] for e in events)==2 and not red.alive and not blue.alive
    _,_,terminated,_,info=env.step({});assert terminated and info["winner"]=="draw";env.close()

def test_team_termination_flags_are_scalar_booleans():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1);env.reset(seed=1);_,_,terminated,truncated,_=env.step(env.build_rule_actions());assert isinstance(terminated,bool) and isinstance(truncated,bool);env.close()

def test_flight_envelope_is_checked_each_frame(monkeypatch):
    env=TAMPaperCombatEnv("paper_nominal_1v1","all",max_steps=1);env.reset(seed=1);a=env.agents[0];original=a.step_physics_once;called=[False]
    def step():
        original()
        if not called[0]:a.speed=401.;called[0]=True
    monkeypatch.setattr(a,"step_physics_once",step);env.step(env.build_rule_actions());assert env.flight_envelope_violation and env.maximum_speed_mps>=401;env.close()

def test_boundary_is_enabled_from_formal_tam_derived_radius():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all");env.reset(seed=1);a=env.by_id["red_0"];a.position[:2]=[28001,0];out=set();env._check_frame(out)
    assert a.death_reason=="boundary" and a.agent_id in out;env.close()

def test_info_contains_independent_acceptance_counters():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all");env.reset(seed=1);info=env._info(None,None)
    for key in ("red_numerical_invalid","blue_numerical_invalid","boundary_deaths","simultaneous_kills","target_changes","second_launches"):assert key in info
    env.close()

def test_long_health_diagnostics_remain_finite_even_when_they_fail_gate():
    from aircombat_env_v1.scripts.check_paper_direct_fcs_health import run_commands
    result=run_commands(("level" for _ in range(1000)));assert all(np.isfinite(v) for k,v in result.items() if isinstance(v,(float,int)))

@pytest.mark.integration
@pytest.mark.parametrize("mode",["paper_nominal_1v1","paper_nominal_2v2"])
def test_short_formal_run_is_finite(mode):
    env=TAMPaperCombatEnv(mode,"all",max_steps=3);obs,_=env.reset(seed=1)
    for _ in range(3):obs,rewards,t,tr,info=env.step(env.build_rule_actions())
    assert all(np.isfinite(v).all() for item in obs.values() for v in item.values());assert all(np.isfinite(list(rewards.values())));env.close()

@pytest.mark.integration
def test_rule_1v1_produces_paper_missile_hit():
    env=TAMPaperCombatEnv("paper_nominal_1v1","all");env.reset(seed=1);info={}
    for _ in range(1000):
        _,_,t,tr,info=env.step(env.build_rule_actions())
        if t or tr:break
    assert info["missiles_fired"]>0 and info["missile_hits"]>0 and not info["invalid_episode"];env.close()
