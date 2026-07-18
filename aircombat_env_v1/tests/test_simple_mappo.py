import copy
import numpy as np
import pytest
from aircombat_env_v1.simple_env import SimpleTAMCombatEnv
from aircombat_env_v1.simple_mappo import MAPPOTrainer,RolloutBuffer,SharedMAPPOActorCritic,SimpleMAPPOAdapter
import torch

def make_adapter(mode):
    env=SimpleTAMCombatEnv(mode,"red",weapon_enabled_agent_ids=set());adapter=SimpleMAPPOAdapter(env);obs,state,_=adapter.reset(seed=1);return env,adapter,obs,state

@pytest.mark.parametrize("mode,oshape,sshape,ashape",[("simple_paper_1v1",(1,61),(61,),(1,3)),("simple_paper_2v2",(2,73),(146,),(2,3))])
def test_adapter_shapes(mode,oshape,sshape,ashape):
    env,a,obs,state=make_adapter(mode);assert obs.shape==oshape and state.shape==sshape
    result=a.step(np.zeros(ashape,np.float32));assert result[0].shape==oshape and result[1].shape==sshape and result[2].shape==(ashape[0],);env.close()

def test_adapter_agent_order_matches_environment():
    env,a,_,_=make_adapter("simple_paper_2v2");assert a.agent_ids==env.controlled_ids;env.close()

def test_actor_is_one_shared_network_and_critic_is_scalar_team_value():
    model=SharedMAPPOActorCritic(73,146);assert not isinstance(model.actor,torch.nn.ModuleList)
    assert model.value(torch.zeros(146)).shape==(1,) and sum(isinstance(x,torch.nn.Linear) for x in model.actor)==3

def test_actions_are_finite_bounded_and_deterministic_repeats():
    model=SharedMAPPOActorCritic(61,61);obs=torch.randn(1,61);state=torch.randn(61)
    action,logp,_,_=model.act(obs,state);assert torch.isfinite(action).all() and torch.isfinite(logp).all() and (action.abs()<=1).all()
    a1=model.act(obs,state,True)[0];a2=model.act(obs,state,True)[0];assert torch.equal(a1,a2)

def test_squashed_action_log_prob_matches_evaluation():
    torch.manual_seed(2);model=SharedMAPPOActorCritic(61,61);obs=torch.randn(1,61);state=torch.randn(61)
    action,logp,_,_=model.act(obs,state);evaluated=model.evaluate_actions(obs,state.unsqueeze(0),action)[0];assert torch.allclose(logp,evaluated,atol=1e-5)

def test_rollout_buffer_shapes_and_clear():
    b=RolloutBuffer(4,2,73,146);sample=(np.zeros((2,73),np.float32),np.zeros(146,np.float32),np.zeros((2,3),np.float32),np.zeros(2,np.float32),np.ones(2,np.float32),False,0.,np.ones(2,np.float32))
    b.store(*sample);d=b.to_tensors();assert d["actor_obs"].shape==(1,2,73) and d["critic_state"].shape==(1,146) and d["actions"].shape==(1,2,3);b.clear();assert b.length==0

def gae_data(rewards,dones,values):
    return {"rewards":torch.tensor(rewards,dtype=torch.float32)[:,None],"active_masks":torch.ones(len(rewards),1),"team_dones":torch.tensor(dones,dtype=torch.float32),"values":torch.tensor(values,dtype=torch.float32)}

def test_gae_stops_at_team_done_and_bootstraps_true_next_value():
    trainer=MAPPOTrainer(SharedMAPPOActorCritic(2,2),gamma=1.,gae_lambda=1.,ppo_epochs=1)
    advantages,_=trainer.compute_gae(gae_data([1,1],[1,0],[0,0]),5.);assert torch.allclose(advantages,torch.tensor([1.,6.]))
    advantages,_=trainer.compute_gae(gae_data([1],[0],[0]),5.);assert advantages.item()==pytest.approx(6.)

def test_dead_red_mask_zero_does_not_set_team_done():
    env,a,_,_=make_adapter("simple_paper_2v2");env.by_id["red_0"].kill("shotdown")
    _,_,_,done,mask,_=a.step(np.zeros((2,3),np.float32));assert mask.tolist()==[0.,1.] and not done;env.close()

def build_buffer(model,inactive_action=0.,inactive_logp=0.):
    b=RolloutBuffer(4,2,3,6);obs=np.arange(24,dtype=np.float32).reshape(4,2,3)/24;state=obs.reshape(4,6);actions=np.zeros((4,2,3),np.float32)
    with torch.no_grad():logp=model.evaluate_actions(torch.tensor(obs.reshape(8,3)),torch.tensor(state),torch.tensor(actions.reshape(8,3)))[0].reshape(4,2).numpy();values=model.value(torch.tensor(state)).numpy()
    actions[:,1]=inactive_action;logp[:,1]=inactive_logp
    for t in range(4):b.store(obs[t],state[t],actions[t],logp[t],[1,999],False,values[t],[1,0])
    return b

def test_inactive_samples_do_not_change_actor_update():
    torch.manual_seed(3);m1=SharedMAPPOActorCritic(3,6);m2=copy.deepcopy(m1);t1=MAPPOTrainer(m1,ppo_epochs=1);t2=MAPPOTrainer(m2,ppo_epochs=1)
    t1.update(build_buffer(m1,0,0),0);t2.update(build_buffer(m2,.9,999),0)
    for p1,p2 in zip(list(m1.actor.parameters())+[m1.log_std],list(m2.actor.parameters())+[m2.log_std]):assert torch.allclose(p1,p2,atol=1e-7)

def random_buffer(model,T,N,D,S):
    b=RolloutBuffer(T,N,D,S)
    for _ in range(T):
        obs=np.random.randn(N,D).astype(np.float32);state=np.random.randn(S).astype(np.float32)
        with torch.no_grad():a,lp,v,_=model.act(torch.tensor(obs),torch.tensor(state))
        b.store(obs,state,a.numpy(),lp.numpy(),np.random.randn(N),False,v.item(),np.ones(N))
    return b

def test_ppo_update_is_finite_and_changes_parameters():
    torch.manual_seed(4);model=SharedMAPPOActorCritic(5,5);before=[p.detach().clone() for p in model.parameters()]
    stats=MAPPOTrainer(model,ppo_epochs=2).update(random_buffer(model,16,1,5,5),0.)
    assert np.isfinite(list(stats.values())).all() and any(not torch.equal(a,b) for a,b in zip(before,model.parameters())) and all(torch.isfinite(p).all() for p in model.parameters())

def test_checkpoint_reload_preserves_deterministic_action(tmp_path):
    model=SharedMAPPOActorCritic(61,61);obs=torch.randn(1,61);state=torch.randn(61);expected=model.act(obs,state,True)[0]
    path=tmp_path/"model.pt";torch.save(model.state_dict(),path);loaded=SharedMAPPOActorCritic(61,61);loaded.load_state_dict(torch.load(path,weights_only=True));assert torch.equal(expected,loaded.act(obs,state,True)[0])

@pytest.mark.parametrize("mode,N,D,S",[("simple_paper_1v1",1,61,61),("simple_paper_2v2",2,73,146)])
def test_environment_completes_64_step_rollout(mode,N,D,S):
    env,a,obs,state=make_adapter(mode);model=SharedMAPPOActorCritic(D,S)
    for _ in range(64):
        with torch.no_grad():actions=model.act(torch.tensor(obs),torch.tensor(state))[0].numpy()
        obs,state,rewards,done,mask,info=a.step(actions)
        assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        if done:obs,state,_=a.reset(seed=2)
    env.close()

def test_2v2_completes_one_ppo_update():
    env,a,obs,state=make_adapter("simple_paper_2v2");model=SharedMAPPOActorCritic(73,146);buffer=RolloutBuffer(16,2,73,146);active=np.ones(2,np.float32)
    for _ in range(16):
        with torch.no_grad():actions,lp,value,_=model.act(torch.tensor(obs),torch.tensor(state))
        next_obs,next_state,rewards,done,next_active,_=a.step(actions.numpy());buffer.store(obs,state,actions.numpy(),lp.numpy(),rewards,done,value.item(),active);obs,state,active=next_obs,next_state,next_active
    stats=MAPPOTrainer(model,ppo_epochs=1).update(buffer,model.value(torch.tensor(state)).item());assert np.isfinite(list(stats.values())).all();env.close()
