from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from uav_env.make_env import make_env
from uav_env.JSBSim.formal_v1.contract import ACTOR_OBS_DIM, CRITIC_STATE_DIM, require_action
from uav_env.JSBSim.formal_v1.geometry import combat_geometry
from uav_env.JSBSim.formal_v1.missile import FormalMissile
from uav_env.JSBSim.formal_v1.observation import build_actor_observation
from uav_env.JSBSim.formal_v1.sensing import red_track_sources
from uav_env.JSBSim.formal_v1.targeting import fire_gate, select_target, target_score
from uav_env.JSBSim.formal_v1 import reward as reward_module
from uav_env.JSBSim.formal_v1.reward import (
    angle_situation_score, compute_role_rewards, distance_situation_score,
    dodge_score, flight_safety_score, mav_dense_components,
    shared_information_score, speed_situation_score,
)
from uav_env.JSBSim.formal_v1.opponent import PaperGreedyOpponent
from algorithms.pure_happo.trainer import _alive_before_team_mean
import torch

CFG = "uav_env/JSBSim/configs/hetero_3v2_pure_happo_v1.yaml"


class FakeAircraft:
    def __init__(self, uid, position, velocity=(250, 0, 0), alive=True, missiles=2):
        self.uid=uid; self._position=np.asarray(position,dtype=float); self._velocity=np.asarray(velocity,dtype=float)
        self._rpy=np.zeros(3); self.is_alive=alive; self.num_missiles=missiles; self.num_left_missiles=missiles
    def get_position(self): return self._position
    def get_velocity(self): return self._velocity
    def get_rpy(self): return self._rpy
    def shotdown(self): self.is_alive=False


def fake_env():
    aircraft={
        "red_0":FakeAircraft("red_0",(-1000,0,6500),missiles=0),
        "red_1":FakeAircraft("red_1",(0,0,6000)), "red_2":FakeAircraft("red_2",(0,500,6000)),
        "blue_0":FakeAircraft("blue_0",(8000,0,6000)), "blue_1":FakeAircraft("blue_1",(9000,1000,6000)),
    }
    return SimpleNamespace(aircraft=aircraft,red_ids=["red_0","red_1","red_2"],blue_ids=["blue_0","blue_1"],
        roles={"red_0":"mav","red_1":"attack_uav","red_2":"attack_uav","blue_0":"attack_uav","blue_1":"attack_uav"},
        mav_detection_range_m=80000.,uav_detection_range_m=10000.,missiles=[],sim_time_sec=30.,
        last_launch_time={x:0. for x in aircraft},attack_interval_sec=25.,attack_range_m=14000.,
        launch_ata_rad=np.deg2rad(60),launch_ta_rad=np.deg2rad(90),
        previous_missile_risk={"red_0":0.,"red_1":0.,"red_2":0.},
        newly_dead=set(),death_reasons={},max_steps=1000)


def test_config_constructs_and_dimensions():
    env=make_env(CFG)
    obs,info=env.reset(seed=0)
    assert env.action_dim==3 and env.actor_obs_dim==ACTOR_OBS_DIM==68
    assert env.critic_state_dim==CRITIC_STATE_DIM==204
    assert all(value["flat"].shape==(68,) for value in obs.values())
    assert info["critic_state"].shape==(204,)
    env.close()


def test_reset_reuses_jsbsim_instances_and_restores_contract():
    env=make_env(CFG); env.reset(seed=0)
    identities={aid:id(sim) for aid,sim in env.aircraft.items()}
    env.aircraft["red_0"].shotdown(); env.reset(seed=1)
    assert identities=={aid:id(sim) for aid,sim in env.aircraft.items()}
    assert all(sim.is_alive for sim in env.aircraft.values())
    assert env.aircraft["red_0"].num_left_missiles==0
    assert env.aircraft["red_1"].num_left_missiles==2
    env.close()


def test_actions_are_strict_and_not_padded():
    assert require_action([0,0,0],"red_0").shape==(3,)
    with pytest.raises(ValueError): require_action([0,0,0,0],"red_0")
    with pytest.raises(ValueError): require_action([0,np.nan,0],"red_0")


def test_mav_shared_sensing_lifecycle_and_zero_hidden_truth():
    env=fake_env(); env.aircraft["blue_0"]._position=np.array([15000.,0,6000.])
    tracks=red_track_sources(env,"red_1")
    assert tracks["blue_0"]["mav_shared"] and not tracks["blue_0"]["direct"]
    env.aircraft["red_0"].is_alive=False
    assert not red_track_sources(env,"red_1")["blue_0"]["observable"]
    obs=build_actor_observation(env,"red_1")
    assert np.all(obs["enemies"][0]==0)
    env.aircraft["blue_0"]._position=np.array([9000.,0,6000.])
    assert red_track_sources(env,"red_1")["blue_0"]["direct"]


def test_target_assessment_and_duplicate_fire_gate():
    env=fake_env()
    assert select_target(env,"red_1") in env.blue_ids
    target=select_target(env,"red_1")
    allowed,diag=fire_gate(env,"red_1",target)
    assert allowed and diag["allowed"]
    env.missiles=[SimpleNamespace(is_launched=True,target_id=target)]
    allowed,diag=fire_gate(env,"red_2",target)
    assert not allowed and diag["duplicate_target_blocked"]
    assert select_target(env,"red_2")==target
    env.missiles[0].is_launched=False
    assert fire_gate(env,"red_2",target)[0]
    assert not fire_gate(env,"red_0",target)[0]


def test_real_fire_control_blocks_duplicate_but_keeps_tracking():
    env=make_env(CFG); env.reset(seed=2)
    for aid,pos in {"red_1":(0,0,6000),"red_2":(0,500,6000),
                    "blue_0":(8000,0,6000),"blue_1":(30000,0,6000)}.items():
        env.aircraft[aid]._position[:]=pos
        env.aircraft[aid]._velocity[:]=(250,0,0)
    env.selected_targets.update({"red_0":None,"red_1":"blue_0","red_2":"blue_0",
                                 "blue_0":None,"blue_1":None})
    records=env._automatic_fire()
    red_records=[x for x in records if x["shooter_id"].startswith("red")]
    assert len(red_records)==1 and red_records[0]["target_id"]=="blue_0"
    assert env.selected_targets["red_2"]=="blue_0"
    env.close()


def test_deterministic_missile_hit_and_target_dead():
    target=FakeAircraft("blue_0",(1000,0,0),velocity=(0,0,0))
    missile=FormalMissile("m0","red_1","blue_0",np.zeros(3),np.array([600.,0,0]),
                          hit_radius_m=100.,arming_time_sec=0.)
    event=None
    for _ in range(200):
        event=missile.step(1/60,target)
        if event: break
    assert event["event"]=="hit" and not target.is_alive
    dead=FakeAircraft("blue_1",(1000,0,0),alive=False)
    second=FormalMissile("m1","red_2","blue_1",np.zeros(3),np.array([600.,0,0]))
    assert second.step(1/60,dead)["reason"]=="target_dead"


def test_obvious_miss_times_out_and_stays_finite():
    target=FakeAircraft("blue_0",(10000,0,0),velocity=(1000,0,0))
    missile=FormalMissile("m0","red_1","blue_0",np.zeros(3),np.array([600.,0,0]),
                          max_flight_time_sec=0.5)
    event=None
    for _ in range(60):
        event=missile.step(1/60,target)
        target._position += target._velocity/60
        assert np.isfinite(np.r_[missile.position,missile.velocity]).all()
        if event: break
    assert event["event"]=="miss" and event["reason"]=="timeout"


def test_paper_greedy_actions_are_finite_and_bounded():
    env=fake_env(); env.missiles=[]
    actions=PaperGreedyOpponent().actions(env,"blue")
    assert set(actions)==set(env.blue_ids)
    assert all(a.shape==(3,) and np.isfinite(a).all() and np.max(np.abs(a))<=1 for a in actions.values())


def test_action_authority_and_mav_death_not_team_done():
    env=make_env(CFG); env.reset(seed=1)
    actions={aid:np.array([0.1 if aid=="red_0" else -0.1,0.2,-0.2],np.float32) for aid in env.red_ids}
    _,_,_,_,info=env.step(actions)
    assert info["control_targets"]["red_0"] != info["control_targets"]["red_1"]
    env.aircraft["red_0"].shotdown()
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert not info["team_done"] and info["active_mask"][0]==0
    env.close()


def test_reward_is_finite_and_death_event_is_once():
    env=fake_env(); env.aircraft["red_1"].is_alive=False
    env.newly_dead={"red_1"}; env.death_reasons={"red_1":"missile_hit"}
    env.previous_missile_risk["red_1"]=0.9
    reward,detail=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_0"},[])
    assert all(np.isfinite(list(reward.values())))
    dead=detail["per_agent"]["red_1"]
    assert dead["event"]==reward_module.EVENT_REWARDS["uav_death"]
    assert all(dead[key]==0 for key in (
        "flight","speed","angle","distance","dodge","missile_risk","dense"))
    first=reward["red_1"]
    env.newly_dead=set()
    second,_=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_0"},[])
    assert first==reward_module.EVENT_REWARDS["uav_death"]
    assert second["red_1"]==0


def test_reward_component_directions_and_bands():
    assert flight_safety_score(6000,250) > flight_safety_score(150,250)
    assert flight_safety_score(6000,250) > flight_safety_score(6000,70)
    assert flight_safety_score(6000,250) > flight_safety_score(6000,500)
    assert speed_situation_score(280,250) > speed_situation_score(100,250)
    assert angle_situation_score(0,np.pi) > angle_situation_score(np.pi,0)
    assert distance_situation_score(7000) > distance_situation_score(200)
    assert distance_situation_score(7000) > distance_situation_score(20000)
    assert dodge_score(0,0)==0
    assert dodge_score(.8,.4)>0 and dodge_score(.2,.6)<0


def test_shared_information_uses_all_live_uav_target_pairs():
    env=fake_env()
    env.aircraft["red_2"]._position=np.array([6000.,0,6000.])
    env.aircraft["blue_0"]._position=np.array([15000.,0,6000.])
    env.aircraft["blue_1"]._position=np.array([5000.,0,6000.])
    assert shared_information_score(env)==pytest.approx(0.25)
    env.aircraft["red_2"]._position=np.array([0.,500.,6000.])
    assert shared_information_score(env)==pytest.approx(0.5)
    env.aircraft["blue_1"]._position=np.array([20000.,0,6000.])
    assert shared_information_score(env)==1.0
    env.aircraft["blue_0"]._position=np.array([9000.,0,6000.])
    assert shared_information_score(env)==pytest.approx(0.5)
    env.aircraft["red_0"].is_alive=False
    assert shared_information_score(env)==0
    env.aircraft["red_0"].is_alive=True
    env.aircraft["red_1"].is_alive=env.aircraft["red_2"].is_alive=False
    assert shared_information_score(env)==0
    env.aircraft["red_1"].is_alive=True
    env.aircraft["blue_0"].is_alive=env.aircraft["blue_1"].is_alive=False
    assert shared_information_score(env)==0


def test_initial_mav_state_is_not_maximum_reward():
    env=fake_env()
    value=mav_dense_components(env,0.0)["dense"]
    assert -1 <= value < 0.9


def test_global_scale_preserves_dense_event_ratio_and_max_steps_is_irrelevant(monkeypatch):
    env=fake_env(); env.newly_dead={"red_1"}; env.death_reasons={"red_1":"missile_hit"}
    first,detail=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_0"},[])
    env.max_steps=37
    same,_=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_0"},[])
    assert first==same
    monkeypatch.setattr(reward_module,"GLOBAL_REWARD_SCALE",0.5)
    scaled,_=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_0"},[])
    assert scaled["red_1"]==pytest.approx(first["red_1"]*0.5)


def test_alive_before_team_mean_matches_manual_and_dead_agent_no_longer_dilutes():
    rewards=torch.tensor([[1.,2.,3.],[99.,4.,6.]])
    active=torch.tensor([[1.,1.,1.],[0.,1.,1.]])
    result=_alive_before_team_mean(rewards,active)
    assert torch.allclose(result,torch.tensor([2.,5.]))


def test_target_score_prefers_better_smooth_geometry():
    shooter=FakeAircraft("red_1",(0,0,6000),velocity=(280,0,0))
    favorable=FakeAircraft("blue_0",(8000,0,5800),velocity=(250,0,0))
    poor=FakeAircraft("blue_1",(-18000,0,7000),velocity=(450,0,0))
    assert target_score(shooter,favorable)>target_score(shooter,poor)


def test_paper_greedy_returns_exhaustive_maximum():
    env=fake_env(); policy=PaperGreedyOpponent()
    rows=policy.scored_candidates(env,"blue_0")
    decision=policy.decisions(env,"blue")["blue_0"]
    best=max(rows,key=lambda row:row["score"])
    assert len(rows)==len(env.red_ids)*18
    assert np.allclose(decision["action"],best["action"])
    assert decision["target_id"]==best["target_id"]
    assert decision["score"]==pytest.approx(best["score"])


def test_environment_uses_blue_greedy_target_for_motion_and_fire(monkeypatch):
    env=make_env(CFG); env.reset(seed=12)
    old_target=select_target(env,"blue_0")
    forced_target=next(aid for aid in env.red_ids if aid!=old_target)
    original=env.blue_policy.decisions(env)
    original["blue_0"]={**original["blue_0"],"target_id":forced_target,"score":2.0}
    monkeypatch.setattr(env.blue_policy,"decisions",lambda _env,team="blue":original)
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert old_target!=forced_target
    assert info["selected_targets"]["blue_0"]==forced_target
    assert info["fire_gates"]["blue_0"]["target_id"]==forced_target
    env.close()


def test_blue_without_live_red_target_is_stable_and_does_not_fire():
    env=make_env(CFG); env.reset(seed=13)
    for aid in env.red_ids:
        env.aircraft[aid].shotdown()
    decisions=env.blue_policy.decisions(env)
    assert all(row["target_id"] is None for row in decisions.values())
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert all(info["selected_targets"][aid] is None for aid in env.blue_ids)
    assert not any(event["event"]=="launch" and event["shooter_id"].startswith("blue")
                   for event in info["step_events"])
    env.close()


def test_greedy_avoidance_has_no_world_heading_sign_bonus_and_tracks_missile_direction():
    env=fake_env(); policy=PaperGreedyOpponent()
    rows=policy.scored_candidates(env,"blue_0")
    left=next(x for x in rows if x["target_id"]=="red_1" and
              np.isclose(x["action"][0],0) and np.isclose(x["action"][1],-1/6) and x["action"][2]>0)
    right=next(x for x in rows if x["target_id"]=="red_1" and
               np.isclose(x["action"][0],0) and np.isclose(x["action"][1],1/6) and x["action"][2]>0)
    assert left["components"]["dodge"]==right["components"]["dodge"]==0
    missile=lambda p,v: SimpleNamespace(is_launched=True,target_id="blue_0",
                                         position=np.asarray(p,float),velocity=np.asarray(v,float))
    env.missiles=[missile((8500,1000,6000),(0,-600,0))]
    from_north=policy.actions(env,"blue")["blue_0"]
    env.missiles=[missile((8500,-1000,6000),(0,600,0))]
    from_south=policy.actions(env,"blue")["blue_0"]
    assert np.sign(from_north[1])==-np.sign(from_south[1])


def test_event_scale_is_40_marked_dense_steps_and_roles_are_distinct():
    assert abs(reward_module.EVENT_REWARDS["red_kill"] / 0.20)==40
    env=fake_env()
    rewards,_=compute_role_rewards(env,{"red_1":"blue_0","red_2":"blue_1"},[])
    assert len(set(round(value,6) for value in rewards.values()))>1


def test_terminal_contract_red_elimination_and_timeout_are_exclusive():
    env=make_env(CFG); env.reset(seed=3)
    for aid in env.red_ids: env.aircraft[aid].shotdown()
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert info["team_done"] and info["outcome"]=="blue_win" and info["end_reason"]=="red_eliminated"
    env.reset(seed=4); env.step_count=env.max_steps-1
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert info["team_done"] and info["outcome"]=="draw" and info["end_reason"]=="timeout"
    env.close()


def test_simultaneous_elimination_is_mutual_draw():
    env=make_env(CFG); env.reset(seed=5)
    for aircraft in env.aircraft.values(): aircraft.shotdown()
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert info["team_done"] and info["outcome"]=="mutual_elimination"
    assert info["end_reason"]=="mutual_elimination"
    env.close()


def test_same_decision_last_aircraft_missile_hits_are_mutual_elimination():
    env=make_env(CFG); env.reset(seed=6)
    for aid in ("red_0","red_2","blue_1"): env.aircraft[aid].shotdown()
    red=env.aircraft["red_1"]; blue=env.aircraft["blue_0"]
    red_missile=FormalMissile("rm","red_1","blue_0",blue.get_position()-np.array([10.,0,0]),
                              np.array([600.,0,0]),hit_radius_m=100.,arming_time_sec=0.)
    blue_missile=FormalMissile("bm","blue_0","red_1",red.get_position()+np.array([10.,0,0]),
                               np.array([-600.,0,0]),hit_radius_m=100.,arming_time_sec=0.)
    red_missile.flight_time_sec=blue_missile.flight_time_sec=0.2
    env.missiles=[red_missile,blue_missile]
    _,_,_,_,info=env.step({aid:np.zeros(3,np.float32) for aid in env.red_ids})
    assert info["outcome"]=="mutual_elimination" and info["team_done"]
    assert sum(e["event"]=="hit" for e in info["step_events"])==2
    env.close()
