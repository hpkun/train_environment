import json
from types import SimpleNamespace
import numpy as np
import pytest
from aircombat_env_v1.simple_env import HETERO_SCENARIO,SimpleTAMCombatEnv
from aircombat_env_v1.simple_hetero_reward import (LegacySimpleMAVReward,PaperTable1MAVReward,
  MAV_DANGER_DISTANCE_M,MAV_SAFE_DISTANCE_M,MAV_SUPPORT_OPTIMAL_DISTANCE_M,MAV_SUPPORT_MAX_DISTANCE_M,build_mav_reward)

def aircraft(agent_id,side,role,position,velocity=(1,0,0),alive=True,death_reason=None):
    return SimpleNamespace(agent_id=agent_id,side=side,role=role,position=np.asarray(position,float),
      velocity=np.asarray(velocity,float),alive=alive,death_reason=death_reason)

def reward_state():
    mav=aircraft("red_mav_0","red","mav",(0,0,0))
    red=aircraft("red_uav_0","red","uav",(4000,0,0))
    blue=aircraft("blue_0","blue","uav",(6000,0,0),(-1,0,0))
    return mav,[mav,red,blue]

def compute(reward,mav,agents,missiles=(),events=(),alive_start=None,perception=None,out=()):
    return reward.compute(mav,agents,list(missiles),list(events),alive_start or {mav.agent_id:mav.alive},set(out),perception or {})

def test_reward_modes_and_environment_default_validation():
    assert isinstance(build_mav_reward("legacy_v1"),LegacySimpleMAVReward)
    assert isinstance(build_mav_reward("paper_table1_v2"),PaperTable1MAVReward)
    with pytest.raises(ValueError):build_mav_reward("bad")
    env=SimpleTAMCombatEnv(HETERO_SCENARIO);_,info=env.reset(seed=1)
    try:assert info["hetero_reward_mode"]=="paper_table1_v2"
    finally:env.close()
    with pytest.raises(ValueError,match="only configurable"):
        SimpleTAMCombatEnv("simple_paper_1v1",hetero_reward_mode="legacy_v1")

def test_legacy_v1_fixed_numeric_regression():
    mav,agents=reward_state();total,c=compute(LegacySimpleMAVReward(),mav,agents)
    assert c==pytest.approx({"r_safety_distance":-.4,"r_safety_threat":0.,"r_safety_aspect":-1.,"r_safety":-.4,
      "r_support_position":1.,"r_support_awareness":.3,"r_support":.72,"r_event":0.,"total":3.2})
    assert total==pytest.approx(3.2)

@pytest.mark.parametrize("distance,expected",[(0,-1),(7000,-.5),(13999,-1/14000),(14000,-.5),(21000,-.25),(28000,.2),(30000,.2)])
def test_distance_reward_all_table1_segments(distance,expected):
    assert PaperTable1MAVReward.distance_reward(distance)==pytest.approx(expected)

@pytest.mark.parametrize("distance,expected",[(0,-1),(7000,-.5),(14000,1),(21000,.5),(28000,-.5),(30000,-.5)])
def test_position_reward_all_table1_segments(distance,expected):
    assert PaperTable1MAVReward.position_reward(distance)==pytest.approx(expected)

def test_threat_and_aspect_sum_over_two_enemies():
    mav=aircraft("red_mav_0","red","mav",(0,0,0));blue0=aircraft("blue_0","blue","uav",(10000,0,0),(-1,0,0));blue1=aircraft("blue_1","blue","uav",(0,10000,0),(0,-1,0))
    _,c=compute(PaperTable1MAVReward(),mav,[mav,blue0,blue1]);assert c["r_safety_threat"]==0 and c["r_safety_aspect"]==pytest.approx(-2.)
    missile=SimpleNamespace(alive=True,target_id=mav.agent_id)
    _,c=compute(PaperTable1MAVReward(),mav,[mav,blue0,blue1],[missile]);assert c["r_safety_threat"]==-1

def test_battlefield_center_uses_alive_combatants_and_excludes_mav():
    mav=aircraft("red_mav_0","red","mav",(100,0,0));agents=[mav,aircraft("red_uav_0","red","uav",(0,0,0)),aircraft("red_uav_1","red","uav",(2,0,0)),aircraft("blue_0","blue","uav",(4,0,0)),aircraft("blue_1","blue","uav",(6,0,0))]
    assert np.array_equal(PaperTable1MAVReward.battlefield_center(agents),[3,0,0])
    agents[-1].position=np.array([10.,0,0]);assert np.array_equal(PaperTable1MAVReward.battlefield_center(agents),[4,0,0])
    agents[-1].alive=False;assert np.array_equal(PaperTable1MAVReward.battlefield_center(agents),[2,0,0])
    for a in agents[1:]:a.alive=False
    assert PaperTable1MAVReward.battlefield_center(agents) is None

def test_awareness_sums_detected_targets_and_relay_is_diagnostic_only():
    mav=aircraft("red_mav_0","red","mav",(0,0,0),(1,0,0));blue0=aircraft("blue_0","blue","uav",(1000,0,0));blue1=aircraft("blue_1","blue","uav",(2000,0,0));agents=[mav,blue0,blue1]
    reward=PaperTable1MAVReward();_,none=compute(reward,mav,agents,perception={"mav_detected_enemy_ids":[]})
    reward.reset();_,one=compute(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0"]})
    reward.reset();_,two=compute(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0","blue_1"],"relay_only_track_count":0})
    reward.reset();total_relay,relay=compute(reward,mav,agents,perception={"mav_detected_enemy_ids":["blue_0","blue_1"],"relay_only_track_count":99})
    assert none["r_support_awareness"]==0 and one["r_support_awareness"]==pytest.approx(.3) and two["r_support_awareness"]==pytest.approx(.6)
    assert relay["relay_only_track_count_log"]==99 and total_relay==pytest.approx(two["total"])

def hit(shooter="red_uav_0",target="blue_0",reason="hit"):
    return {"reason":reason,"shooter_id":shooter,"target_id":target}

def test_event_team_credit_cap_reset_and_event_filtering():
    mav=aircraft("red_mav_0","red","mav",(0,0,0));red=aircraft("red_uav_0","red","uav",(0,0,0));blue=aircraft("blue_0","blue","uav",(0,0,0));agents=[mav,red,blue];reward=PaperTable1MAVReward()
    assert compute(reward,mav,agents,events=[hit()])[1]["r_event_team_contribution"]==100
    assert compute(reward,mav,agents,events=[hit()])[1]["r_event_team_contribution"]==100
    assert compute(reward,mav,agents,events=[hit()])[1]["r_event_team_contribution"]==0
    reward.reset();assert reward.team_credit_awarded_so_far==0
    assert compute(reward,mav,agents,events=[hit(reason="miss"),hit("blue_0","red_uav_0")])[1]["r_event_team_contribution"]==0

def test_mav_death_is_once_and_boundary_has_no_extra_penalty():
    mav,agents=reward_state();reward=PaperTable1MAVReward();mav.alive=False;mav.death_reason="boundary"
    _,first=compute(reward,mav,agents,alive_start={mav.agent_id:True},out={mav.agent_id})
    _,second=compute(reward,mav,agents,alive_start={mav.agent_id:True},out={mav.agent_id})
    assert first["r_event_death"]==-200 and first["r_event"]==-200 and second["r_event_death"]==0

def test_total_identity_no_top_level_multiplier_and_components_finite():
    mav,agents=reward_state();total,c=compute(PaperTable1MAVReward(),mav,agents,perception={"mav_detected_enemy_ids":["blue_0"]})
    assert c["total_dense"]==pytest.approx(c["r_safety"]+c["r_support"])
    assert total==pytest.approx(c["r_safety"]+c["r_support"]+c["r_event"])
    assert np.isfinite([float(value) for value in c.values()]).all()

def test_environment_uses_post_step_mav_perception_for_awareness():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=1,weapon_enabled_agent_ids=set());env.reset(seed=1);base=env.perception_result
    class Sequence:
        calls=0
        def build(self,agents):
            self.calls+=1;result={key:(dict(value) if isinstance(value,dict) else list(value) if isinstance(value,list) else value) for key,value in base.items()}
            if self.calls==2:result["mav_detected_enemy_ids"]=[]
            return result
    env.hetero_perception=Sequence()
    try:
        _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.controlled_ids})
        assert info["mav_detected_enemy_ids"]==[] and info["reward_components"]["red_mav_0"]["r_support_awareness"]==0
    finally:env.close()

def test_uav_reward_contract_and_info_json_remain_valid():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,max_steps=1,weapon_enabled_agent_ids=set());env.reset(seed=1)
    try:
        _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.controlled_ids});json.dumps(info)
        assert set(info["reward_components"]["red_uav_0"])=={"r_height","r_speed","r_angle","r_distance","r_dodge_angle","r_dodge_speed","r_dodge","r_event","total"}
        assert info["hetero_reward_mode"]=="paper_table1_v2" and info["hetero_reward_contract_version"]=="heterogeneous_reward_v2"
    finally:env.close()
