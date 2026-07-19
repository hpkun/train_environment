from types import SimpleNamespace
import numpy as np
from aircombat_env_v1.hetero_perception import HeterogeneousPerceptionSystem,MAV_DETECTION_RANGE_M,UAV_DIRECT_DETECTION_RANGE_M

def agent(agent_id,side,role,position,alive=True):
    return SimpleNamespace(agent_id=agent_id,side=side,role=role,position=np.asarray(position,float),alive=alive)

def formation(blue_positions=((15000,0,0),(13000,0,0)),blue_alive=(True,True)):
    return [agent("red_uav_1","red","uav",(0,1000,0)),agent("red_mav_0","red","mav",(0,0,0)),
      agent("red_uav_0","red","uav",(0,-1000,0)),agent("blue_1","blue","uav",blue_positions[0],blue_alive[0]),
      agent("blue_0","blue","uav",blue_positions[1],blue_alive[1])]

def test_paper_fused_direct_shared_visible_and_relay_only_are_stable():
    result=HeterogeneousPerceptionSystem("paper_fused").build(formation())
    assert result["mav_detected_enemy_ids"]==["blue_0","blue_1"]
    assert result["direct_enemy_ids_by_agent"]["red_uav_0"]==["blue_0"]
    assert result["shared_enemy_ids_by_agent"]["red_uav_0"]==["blue_0","blue_1"]
    assert result["visible_enemy_ids_by_agent"]["red_uav_0"]==["blue_0","blue_1"]
    assert result["relay_only_tracks_by_uav"]["red_uav_0"]==["blue_1"]
    assert result["relay_only_track_count"]==2 and result["mav_support_active"] is True

def test_uav_only_ablation_disables_only_mav_sharing():
    result=HeterogeneousPerceptionSystem("uav_only_ablation").build(formation())
    assert result["mav_detected_enemy_ids"]==["blue_0","blue_1"]
    assert result["shared_enemy_ids_by_agent"]["red_uav_0"]==[]
    assert result["visible_enemy_ids_by_agent"]["red_uav_0"]==["blue_0"]
    assert result["relay_only_track_count"]==0 and result["mav_support_active"] is False

def test_mav_and_uav_detection_ranges_are_inclusive():
    agents=formation(((MAV_DETECTION_RANGE_M,0,0),(UAV_DIRECT_DETECTION_RANGE_M,-1000,0)))
    result=HeterogeneousPerceptionSystem().build(agents)
    assert result["mav_detected_enemy_ids"]==["blue_0","blue_1"]
    assert result["direct_enemy_ids_by_agent"]["red_uav_0"]==["blue_0"]

def test_dead_enemies_are_not_detected_and_ids_remain_sorted():
    result=HeterogeneousPerceptionSystem().build(formation(((1000,0,0),(2000,0,0)),(False,True)))
    assert result["mav_detected_enemy_ids"]==["blue_0"]
    assert result["visible_enemy_ids_by_agent"]["red_uav_0"]==["blue_0"]
