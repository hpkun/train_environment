import json
import numpy as np
from gymnasium import spaces
from aircombat_env_v1.paper_reward import PaperReward
from aircombat_env_v1.paper_situation import paper_situation_score
from aircombat_env_v1.simple_env import HETERO_SCENARIO,SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

def make_env(**kwargs):
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,**kwargs);obs,info=env.reset(seed=1);return env,obs,info

def zero_actions(env):return {aid:np.zeros(3,np.float32) for aid in env.controlled_ids if env.by_id[aid].alive}

def test_entities_roles_initial_state_and_weapons():
    env,_,info=make_env()
    try:
        assert len(env.agents)==5 and info["agent_roles"]=={"red_uav_0":"uav","red_uav_1":"uav","red_mav_0":"mav","blue_0":"uav","blue_1":"uav"}
        expected={"red_uav_0":(120.,60.,0.),"red_uav_1":(120.04,60.,0.),"red_mav_0":(120.02,59.98,0.),"blue_0":(120.,60.2,180.),"blue_1":(120.04,60.2,180.)}
        for aid,(lon,lat,heading) in expected.items():
            a=env.by_id[aid];assert a.longitude==lon and a.latitude==lat and a.initial_heading_deg==heading
            assert abs(a.state["altitude"]-6000)<1 and abs(a.speed-250)<1 and abs(np.rad2deg(a.heading)-heading)<1e-3
        assert env.by_id["red_mav_0"].missile_left==0
        assert all(env.by_id[aid].missile_left==2 for aid in ("red_uav_0","red_uav_1","blue_0","blue_1"))
    finally:env.close()

def test_action_observation_roles_and_adapter_shapes():
    env,obs,_=make_env()
    try:
        assert all(isinstance(space,spaces.Box) and space.shape==(3,) for space in env.action_space.values())
        assert all(value.shape==(81,) and value.dtype==np.float32 and np.isfinite(value).all() for value in obs.values())
        assert np.array_equal(obs["red_mav_0"][-2:],[1,0]) and all(np.array_equal(obs[aid][-2:],[0,1]) for aid in ("red_uav_0","red_uav_1"))
        adapter=SimpleMAPPOAdapter(env);actor,state,_=adapter.reset(seed=1)
        assert adapter.num_agents==3 and actor.shape==(3,81) and state.shape==(243,) and adapter.action_dim==3
    finally:env.close()

def test_initial_fused_and_ablation_observations_follow_visibility_masks():
    fused,fused_obs,fused_info=make_env();ablation=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode="uav_only_ablation")
    try:
        ablation_obs,ablation_info=ablation.reset(seed=1)
        for aid in ("red_uav_0","red_uav_1"):
            assert fused_info["direct_enemy_ids_by_agent"][aid]==[]
            assert fused_info["shared_enemy_ids_by_agent"][aid]==["blue_0","blue_1"]
            assert fused_info["visible_enemy_ids_by_agent"][aid]==["blue_0","blue_1"]
            assert np.array_equal(fused_obs[aid][69:71],[1,1])
            assert ablation_info["visible_enemy_ids_by_agent"][aid]==[]
            assert np.all(ablation_obs[aid][17:27]==0) and np.array_equal(ablation_obs[aid][69:71],[0,0])
            assert fused_obs[aid].shape==ablation_obs[aid].shape==(81,) and np.array_equal(fused_obs[aid][-2:],[0,1])
        assert fused_info["relay_only_track_count"]==4
    finally:fused.close();ablation.close()

def test_non_heterogeneous_scenario_rejects_ablation_mode():
    import pytest
    with pytest.raises(ValueError,match="only configurable"):
        SimpleTAMCombatEnv("simple_paper_1v1",hetero_perception_mode="uav_only_ablation")

def test_paper_situation_score_weights_and_heterogeneous_target_selection():
    env,_,_=make_env(weapon_enabled_agent_ids=set())
    try:
        ego=env.by_id["red_uav_0"];near=env.by_id["blue_0"];far=env.by_id["blue_1"]
        near.position=ego.position+np.array([1000.,0.,3000.]);near.velocity=ego.velocity.copy()
        far.position=ego.position+np.array([5000.,0.,-3000.]);far.velocity=ego.velocity.copy()
        assert np.linalg.norm(near.position-ego.position)<np.linalg.norm(far.position-ego.position)
        assert paper_situation_score(ego.position,ego.velocity,far.position,far.velocity)>paper_situation_score(ego.position,ego.velocity,near.position,near.velocity)
        env._update_perception();env._update_targets(False)
        assert ego.current_target=="blue_1"
        far.position=near.position.copy();far.velocity=near.velocity.copy();env._update_perception();env._update_targets(False)
        assert ego.current_target=="blue_0"
    finally:env.close()

def test_targets_visibility_fire_control_and_mav_weapon_invariants():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,hetero_perception_mode="uav_only_ablation",max_steps=2);env.reset(seed=1)
    try:
        assert all(env.by_id[aid].current_target is None for aid in ("red_uav_0","red_uav_1"))
        _,_,_,_,info=env.step(zero_actions(env));assert info["red_uav_launches_using_direct_track"]==0
        uav=env.by_id["red_uav_0"];blue=env.by_id["blue_0"];blue.position=uav.position+np.array([1000.,0.,0.])
        env._update_perception();env._update_targets(False)
        assert uav.current_target=="blue_0" and env.target_selection_source[uav.agent_id]=="direct"
        _,_,_,_,info=env.step(zero_actions(env))
        launchers=[event["shooter_id"] for event in info["events"] if event["event_type"]=="missile_launch"]
        assert "red_uav_0" in launchers and "red_mav_0" not in launchers
        assert info["red_uav_launches_using_direct_track"]>=1 and env.by_id["red_mav_0"].missile_left==0
    finally:env.close()

def test_heterogeneous_info_is_json_serializable_and_targets_are_visible():
    env,_,info=make_env()
    try:
        json.dumps(info)
        assert set(info["target_selection_source"].values())<={"direct","mav_shared","none","global_rule_opponent"}
        assert all(info["red_uav_targets_visible"].values())
        for aid in ("red_uav_0","red_uav_1"):
            assert info["current_targets"][aid] in info["visible_enemy_ids_by_agent"][aid]
        assert env.by_id["red_mav_0"].current_target in info["mav_detected_enemy_ids"]
    finally:env.close()

def test_dead_entity_slot_zero_but_role_is_retained():
    env,_,_=make_env();env.by_id["red_uav_0"].kill("shotdown");obs=env._observations()["red_mav_0"]
    try:assert np.all(obs[7:12]==0) and obs[67]==0 and np.array_equal(obs[-2:],[1,0])
    finally:env.close()

def test_mav_never_launches_and_uavs_keep_paper_reward():
    env,_,_=make_env(max_steps=1)
    try:
        mav=env.by_id["red_mav_0"]
        for blue in (env.by_id["blue_0"],env.by_id["blue_1"]):blue.position=mav.position+np.array([1000.,0.,0.])
        _,_,_,_,info=env.step(zero_actions(env));launchers=[e["shooter_id"] for e in info["events"] if e["event_type"]=="missile_launch"]
        assert "red_mav_0" not in launchers and mav.missile_left==0 and isinstance(env.reward_model,PaperReward)
        assert set(info["reward_components"]["red_uav_0"])=={"r_height","r_speed","r_angle","r_distance","r_dodge_angle","r_dodge_speed","r_dodge","r_event","total"}
    finally:env.close()

def test_mav_reward_components_are_present_and_finite():
    env,_,_=make_env(max_steps=1,weapon_enabled_agent_ids=set())
    try:
        _,rewards,_,_,info=env.step(zero_actions(env));components=info["reward_components"]["red_mav_0"]
        assert set(components)=={"r_safety_distance","r_safety_threat","r_safety_aspect","r_safety","r_support_position","r_support_awareness","r_support","r_event","total"}
        assert np.isfinite(list(components.values())).all() and np.isfinite(rewards["red_mav_0"])
    finally:env.close()

def terminal_after(killed):
    env,_,_=make_env(max_steps=2,weapon_enabled_agent_ids=set())
    try:
        for aid in killed:env.by_id[aid].kill("shotdown")
        return env.step(zero_actions(env))[2:5]
    finally:env.close()

def test_heterogeneous_mission_termination_rules():
    terminated,_,info=terminal_after(["red_mav_0"]);assert terminated and info["winner"]=="blue" and info["red_team_failed_by_mav_loss"]
    terminated,_,info=terminal_after(["red_uav_0","red_uav_1"]);assert terminated and info["winner"]=="blue" and info["red_team_failed_by_uav_loss"]
    terminated,_,info=terminal_after(["blue_0","blue_1"]);assert terminated and info["winner"]=="red"
    terminated,_,info=terminal_after(["blue_0","blue_1","red_mav_0"]);assert terminated and info["winner"]=="draw"

def test_existing_observation_dimensions_are_unchanged():
    for scenario,dim in (("simple_paper_1v1",61),("simple_paper_2v2",73)):
        env=SimpleTAMCombatEnv(scenario);obs,_=env.reset(seed=1);assert all(x.shape==(dim,) for x in obs.values());env.close()

def test_heterogeneous_random_rollout_64_steps_is_finite():
    env,obs,_=make_env(max_steps=64,weapon_enabled_agent_ids=set())
    try:
        for _ in range(64):
            obs,rewards,terminated,truncated,info=env.step({aid:env.action_space[aid].sample() for aid in env.controlled_ids if env.by_id[aid].alive})
            assert all(np.isfinite(x).all() for x in obs.values()) and np.isfinite(list(rewards.values())).all()
            if terminated or truncated:break
        assert info["numerical_invalid"]==0
    finally:env.close()

def test_heterogeneous_completes_one_mappo_update():
    env=SimpleTAMCombatEnv(HETERO_SCENARIO,weapon_enabled_agent_ids=set());adapter=SimpleMAPPOAdapter(env);obs,state,_=adapter.reset(seed=1)
    model=SharedMAPPOActorCritic(81,243);buffer=RolloutBuffer(16,3,81,243);active=np.ones(3,np.float32)
    try:
        for _ in range(16):
            with torch.no_grad():actions,logp,value,_=model.act(torch.tensor(obs),torch.tensor(state))
            next_obs,next_state,rewards,done,next_active,_=adapter.step(actions.numpy());buffer.store(obs,state,actions.numpy(),logp.numpy(),rewards,done,value.item(),active);obs,state,active=next_obs,next_state,next_active
        stats=MAPPOTrainer(model,ppo_epochs=1).update(buffer,model.value(torch.tensor(state)).item());assert np.isfinite(list(stats.values())).all()
    finally:env.close()
