import numpy as np
import pytest
import torch

from aircombat_env_v1.env import AirCombat1v1Env
from aircombat_env_v1.missile import (Missile, PAPER_MISSILE_PARAMETERS,
    PROJECT_MISSILE_ASSUMPTIONS, compute_paper_eq9_overloads)
from aircombat_env_v1.observation import build_observation
from aircombat_env_v1.opponent import height_reward, paper_structured_engineering_score
from aircombat_env_v1.recurrent_ppo import RecurrentActor, RecurrentCritic, RecurrentRolloutBuffer
from aircombat_env_v1.reward import terminal_reward
from aircombat_env_v1.scenario import make_scenario
from aircombat_env_v1.seeds import *


def state(north=0., speed=250.):
    # At the equator-ish origin, latitude shift is sufficient for these tests.
    return {"longitude":120.,"latitude":60.+north/111000.,"altitude":6000.,
        "true_airspeed":speed,"roll":0.,"pitch":0.,"heading":0.,
        "v_north":speed,"v_east":0.,"v_down":0.,"load_factor":1.}


def test_paper_minimum_altitude_is_750_not_250():
    assert height_reward(750.) > height_reward(749.)
    assert "250" not in height_reward.__doc__ if height_reward.__doc__ else True

def test_nominal_pair_matches_official_tam_mapping():
    red,blue=make_scenario("paper_nominal_1v1",np.random.default_rng(1))
    assert (red.altitude_m,red.speed_mps,red.heading_deg)==(6000.,250.,0.)
    assert (blue.latitude_deg,blue.heading_deg)==(60.2,180.)

def test_candidate_score_depends_on_candidate_velocity():
    own,target=state(),state(6500.)
    a=paper_structured_engineering_score(own,target,np.array([0,0,1.]))
    b=paper_structured_engineering_score(own,target,np.array([0,0,-1.]))
    assert a!=b

def test_6500m_observation_distance_is_not_saturated():
    obs=build_observation(state(),state(6500.),{})
    assert .4 < obs[12] < .6

def test_observation_is_finite_float32_and_compact():
    obs=build_observation(state(),state(6500.),{})
    assert obs.dtype==np.float32 and obs.shape[0]<=28 and np.isfinite(obs).all()

def test_paper_pn_gains_and_limits():
    assert PAPER_MISSILE_PARAMETERS["navigation_gain_y"]==3
    assert PAPER_MISSILE_PARAMETERS["navigation_gain_z"]==3
    assert PAPER_MISSILE_PARAMETERS["maximum_overload_g"]==30

def test_paper_range_and_interval():
    assert PAPER_MISSILE_PARAMETERS["maximum_attack_range_m"]==14000
    assert PAPER_MISSILE_PARAMETERS["launch_interval_s"]==25

def test_eq9_is_finite():
    assert np.isfinite(compute_paper_eq9_overloads([1000,10,0],[-300,0,0],[500,0,0])).all()

def test_missile_hit_terminates():
    m=Missile(np.zeros(3),np.array([1,0,0]),"red","blue",0)
    assert m.step(np.array([10,0,0]),np.zeros(3),True,1/60)=="hit"

def test_missile_overload_is_capped():
    m=Missile(np.array([0,0,1000]),np.array([1,0,0]),"red","blue",0)
    m.step(np.array([100,100,2000]),np.zeros(3),True,1/60)
    assert m.overload<=30+1e-6

def test_event_rewards():
    assert terminal_reward("red_hit")==200 and terminal_reward("blue_hit")==-200
    assert terminal_reward("red_crash")==-200

def test_formal_action_space_is_mixed():
    env=AirCombat1v1Env(max_steps=1); assert set(env.action_space.spaces)=={"maneuver","fire"}; env.close()

def test_combined_log_probability_and_entropy():
    actor=RecurrentActor(); obs=torch.zeros(2,26); h=actor.initial_hidden(2)
    _,_,lp,en,_=actor.act(obs,h,torch.ones(2,1)); assert lp.shape==(2,) and en.shape==(2,)

def test_gru_hidden_reset_mask():
    actor=RecurrentActor(); obs=torch.ones(1,26); h=torch.ones(1,1,128)
    _,_,_,_,a=actor.act(obs,h,torch.ones(1,1)); _,_,_,_,b=actor.act(obs,torch.zeros_like(h),torch.ones(1,1))
    assert torch.allclose(a,b)

def test_critic_gru_hidden_reset_mask():
    critic=RecurrentCritic(); obs=torch.ones(1,26); h=torch.ones(1,1,128)
    a,_=critic(obs,h,torch.ones(1,1)); b,_=critic(obs,torch.zeros_like(h),torch.ones(1,1)); assert torch.allclose(a,b)

def test_buffer_makes_sequences_not_flat_steps():
    b=RecurrentRolloutBuffer(4,1)
    for i in range(4): b.add(observations=np.zeros((1,26)),maneuvers=np.zeros((1,3)),fire=[0],log_probs=[0],values=[0],next_values=[0],rewards=[0],terminated=[False],truncated=[False],episode_starts=[i==0],actor_hidden=np.zeros((1,128)),critic_hidden=np.zeros((1,128)))
    chunks=b.sequences(2); assert len(chunks)==2 and chunks[0]["observations"].shape==(2,26)

def test_seed_sets_reproducible_and_disjoint():
    assert NOMINAL_VALIDATION_SEEDS==tuple(range(1000,1050))
    assert not set(NOMINAL_VALIDATION_SEEDS)&set(GENERALIZATION_LOW_SEEDS)

def test_perturbation_same_seed_reproducible_and_different_seed_differs():
    a=make_scenario("generalization_low",np.random.default_rng(7)); b=make_scenario("generalization_low",np.random.default_rng(7)); c=make_scenario("generalization_low",np.random.default_rng(8))
    assert a==b and a!=c

@pytest.mark.integration
def test_no_geometric_dwell_hit_without_fire():
    env=AirCombat1v1Env(scenario_mode="fixed_tail_chase",opponent_policy="straight",max_steps=3); env.reset(seed=1)
    for _ in range(3): _,_,term,trunc,info=env.step({"maneuver":np.zeros(3,np.float32),"fire":0})
    assert info["red_launch_count"]==0 and info["event"]!="red_hit"; env.close()

@pytest.mark.integration
def test_pursuit_fire_launches_and_is_finite():
    from aircombat_env_v1.combat import pursuit_action
    env=AirCombat1v1Env(max_steps=200); obs,_=env.reset(seed=1)
    for _ in range(200):
        obs,r,t,tr,info=env.step({"maneuver":pursuit_action(env.red_state,env.blue_state),"fire":1})
        assert np.isfinite(obs).all() and np.isfinite(r)
        if t or tr: break
    assert info["red_launch_count"]>=1; env.close()
